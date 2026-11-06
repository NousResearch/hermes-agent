# Canonical name (not ``from .adapter``): the loader runs this package as ``hermes_plugins.platforms__email``;
# anything importing ``plugins.platforms.email.*`` would otherwise execute the adapter a second time.
from plugins.platforms.email.adapter import register

__all__ = ["register"]
