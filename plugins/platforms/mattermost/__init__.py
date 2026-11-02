# Canonical name (not ``from .adapter``): the loader runs this package as ``hermes_plugins.platforms__mattermost``;
# anything importing ``plugins.platforms.mattermost.*`` would otherwise execute the adapter a second time.
from plugins.platforms.mattermost.adapter import register

__all__ = ["register"]
