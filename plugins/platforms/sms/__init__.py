# Canonical name (not ``from .adapter``): the loader runs this package as ``hermes_plugins.platforms__sms``;
# anything importing ``plugins.platforms.sms.*`` would otherwise execute the adapter a second time.
from plugins.platforms.sms.adapter import register

__all__ = ["register"]
