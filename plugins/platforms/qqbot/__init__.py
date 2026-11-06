# Canonical name (not ``from .register``): the loader runs this package as ``hermes_plugins.platforms__qqbot``;
# anything importing ``plugins.platforms.qqbot.*`` would otherwise execute the adapter a second time.
from plugins.platforms.qqbot.register import register

__all__ = ["register"]
