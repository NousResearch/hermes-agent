# Absolute import (not ``from .adapter``): the plugin loader executes this package under a synthetic
# ``hermes_plugins.platforms__signal`` name, and the adapter's own absolute sibling imports would then
# load a second ``hermes_agent.plugins.platforms.signal.adapter``. One canonical name = one SignalAdapter class.
from hermes_agent.plugins.platforms.signal.register import register

__all__ = ["register"]
