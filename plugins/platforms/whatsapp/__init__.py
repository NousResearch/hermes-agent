# Canonical name (not ``from .adapter``): the loader runs this package as ``hermes_plugins.platforms__whatsapp``;
# anything importing ``plugins.platforms.whatsapp.*`` would otherwise execute the adapter a second time.
from plugins.platforms.whatsapp.adapter import register

__all__ = ["register"]
