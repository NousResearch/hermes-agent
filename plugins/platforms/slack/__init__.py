# Absolute import (not ``from .adapter``): the plugin loader executes this package under a synthetic
# ``hermes_plugins.platforms__slack`` name, and ``scripts/generate_conformance_vectors.py`` and evals import ``plugins.platforms.slack.adapter``,
# which would load (and execute) a second adapter. One canonical name = one adapter class.
from plugins.platforms.slack.adapter import register

__all__ = ["register"]
