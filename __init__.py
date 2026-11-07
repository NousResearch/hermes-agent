"""model-doctor plugin package."""
from .model_doctor import register  # re-export for plugin loader

__all__ = ["register"]
