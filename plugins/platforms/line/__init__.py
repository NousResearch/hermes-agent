# Canonical name (not ``from .adapter``): the loader runs this package as ``hermes_plugins.platforms__line``;
# anything importing ``plugins.platforms.line.*`` would otherwise execute the adapter a second time.
from plugins.platforms.line.adapter import register

__all__ = ["register"]
