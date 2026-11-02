# Absolute import (not ``from .adapter``): the plugin loader executes this package under a synthetic
# ``hermes_plugins.platforms__buzz`` name, and the adapter's own absolute sibling imports would then
# load (and execute) a second ``plugins.platforms.buzz.adapter``. One canonical name = one adapter class.
from plugins.platforms.buzz.adapter import register

__all__ = ["register"]
