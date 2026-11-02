# Canonical name (not ``from .adapter``): the loader runs this package as ``hermes_plugins.platforms__irc``;
# anything importing ``plugins.platforms.irc.*`` would otherwise execute the adapter a second time.
from plugins.platforms.irc.adapter import register

__all__ = ["register"]
