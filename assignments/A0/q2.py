import torch
import torch.nn as nn
import torch.optim as optim

def get_checkpoint_path():
    """Return the path to save the best performing model checkpoint.
    
    Returns:
        checkpoint_path (str)
            The path to save the best performing model checkpoint
    """
    return 'best_model_checkpoint.pth'

class LinearRegression(nn.Module):
    def __init__(self, input_dim=1, output_dim=1):
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        # TODO: Implement
        return self.linear(x)
        raise NotImplementedError

def create_loss_and_optimizer(model):
    """Create and return a loss function and optimizer.
    
    Parameters:
        model (torch.nn.Module)
            A neural network
    
    Returns:
        loss_fn (function)
            The loss function for the model
        optimizer (torch.optim.Optimizer)
            The optimizer for the model
    """
    # TODO: Implement
    loss_fn = nn.MSELoss()  # Mean Squared Error Loss
    optimizer = optim.SGD(model.parameters(), lr=0.01)  # Stochastic Gradient Descent
    return loss_fn, optimizer
    raise NotImplementedError

def train(x, y, model, loss_fn, optimizer, checkpoint_path, num_epochs=1000):
    """Train a model.
    
    Parameters:
        x (torch.Tensor)
            The input data
        y (torch.Tensor)
            The expected output data
        model (torch.nn.Module)
            A neural network
        loss_fn (function)
            The loss function
        optimizer (torch.optim.Optimizer)
            The optimizer for the model
        checkpoint_path (str)
            The path to save the best performing checkpoint
        num_epochs (int)
            The number of epochs to train for
    
    Side Effects:
        - Save the best performing model checkpoint to `checkpoint_path`
    """
    # TODO: Implement
    best_loss = float('inf')
    for epoch in range(num_epochs):
        model.train()  # Set model to training mode
        optimizer.zero_grad()  # Clear previous gradients
        
        # Forward pass
        predictions = model(x)
        # Compute loss, mean sqr error
        loss = loss_fn(predictions, y)
        # Backpropagation
        loss.backward()
        # Update weights
        optimizer.step()
        
        # Print loss every 100 epochs for tracking
        if (epoch + 1) % 100 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")
        
        if loss.item() < best_loss:
            best_loss = loss.item()
            torch.save(model.state_dict(), checkpoint_path)
            print(f"Model checkpoint saved at epoch {epoch+1}")
    
    print("Training completed.")
    return
    raise NotImplementedError

def load_model_checkpoint(checkpoint_path):
    """Load a model checkpoint from disk.

    Parameters:
        checkpoint_path (str)
            The path to load the checkpoint from
    
    Returns:
        model (torch.nn.Module)
            The model loaded from the checkpoint
    """
    # TODO: Implement
    model = LinearRegression(input_dim=1, output_dim=1)
    model.load_state_dict(torch.load(checkpoint_path))
    model.eval()
    
    print(f"Model checkpoint loaded from {checkpoint_path}")
    return model
    raise NotImplementedError
    