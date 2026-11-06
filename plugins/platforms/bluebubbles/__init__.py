# Canonical name (not ``from .adapter``): the loader runs this package as ``hermes_plugins.platforms__bluebubbles``;
# anything importing ``plugins.platforms.bluebubbles.*`` would otherwise execute the adapter a second time.
from plugins.platforms.bluebubbles.adapter import register

__all__ = ["register"]
