"""GPU providers — each module self-registers on import.

To add a new provider, create a sibling module that calls
``register_provider(...)`` and import it here.
"""

from hermes_cli.gpu.providers import dcgm  # noqa: F401  — registers DCGM
