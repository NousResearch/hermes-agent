# Canonical name (not ``from .adapter``): the loader runs this package as ``hermes_plugins.platforms__ntfy``;
# anything importing ``plugins.platforms.ntfy.*`` would otherwise execute the adapter a second time.
from plugins.platforms.ntfy.adapter import register

__all__ = ["register"]
