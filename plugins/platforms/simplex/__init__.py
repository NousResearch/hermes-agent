# Canonical name (not ``from .adapter``): the loader runs this package as ``hermes_plugins.platforms__simplex``;
# anything importing ``plugins.platforms.simplex.*`` would otherwise execute the adapter a second time.
from plugins.platforms.simplex.adapter import register

__all__ = ["register"]
