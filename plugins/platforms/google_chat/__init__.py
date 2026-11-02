# Canonical name (not ``from .adapter``): the loader runs this package as ``hermes_plugins.platforms__google_chat``;
# anything importing ``plugins.platforms.google_chat.*`` would otherwise execute the adapter a second time.
from plugins.platforms.google_chat.adapter import register

__all__ = ["register"]
