"""Compatibility exports for older code that imports hermes_cli.cli."""

from hermes_cli.moa_config import list_moac_available, resolve_moac_spec

__all__ = ["resolve_moac_spec", "list_moac_available"]
