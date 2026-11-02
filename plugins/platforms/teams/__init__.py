# Absolute import (not ``from .adapter``): the plugin loader executes this package under a synthetic
# ``hermes_plugins.platforms__teams`` name, and the teams_pipeline plugin imports ``plugins.platforms.teams.summary_writer``,
# which would load (and execute) a second adapter. One canonical name = one adapter class.
from plugins.platforms.teams.adapter import register

__all__ = ["register"]
