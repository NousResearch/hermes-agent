# Canonical name (not ``from .adapter``): the loader runs this package as ``hermes_plugins.platforms__weixin``;
# anything importing ``plugins.platforms.weixin.*`` would otherwise execute the adapter a second time.
from plugins.platforms.weixin.adapter import register

__all__ = ["register"]
