# Absolute import (not ``from .adapter``): the plugin loader executes this package under a synthetic
# ``hermes_plugins.platforms__whatsapp_cloud`` name, and ``hermes whatsapp-cloud`` imports
# ``plugins.platforms.whatsapp_cloud.setup_wizard``, which would load (and execute) a second adapter.
from plugins.platforms.whatsapp_cloud.adapter import register

__all__ = ["register"]
