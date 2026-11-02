# Absolute import (not ``from .adapter``): the plugin loader executes this package under a synthetic
# ``hermes_plugins.platforms__matrix`` name, and the adapter's own absolute sibling imports would then
# load (and execute) a second ``plugins.platforms.matrix.adapter``. One canonical name = one adapter class.
from plugins.platforms.matrix.adapter import register

__all__ = ["register"]
