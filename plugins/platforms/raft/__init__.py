# Canonical name (not ``from .adapter``): the loader runs this package as ``hermes_plugins.platforms__raft``;
# anything importing ``plugins.platforms.raft.*`` would otherwise execute the adapter a second time.
from plugins.platforms.raft.adapter import register

__all__ = ["register"]
