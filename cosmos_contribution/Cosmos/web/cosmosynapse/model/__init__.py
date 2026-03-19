"""
COSMOS 54D Model Package.

The 12D Cosmic Davis Hebbian Transformer (ver 4.2)
Architecture: 12D CST + 24D Hebbian Plasticity + 18D Chaos = 54D State Space

This package contains:
- CosmosConfig: Model hyperparameters and architecture configuration
- CosmosTransformer: The full 54D transformer with Hebbian learning
"""

from .cosmos_config import CosmosConfig
from .cosmos_model import CosmosTransformer

__all__ = ["CosmosConfig", "CosmosTransformer"]
