"""Compatibility shim — bundled memory providers live in ``hermes_memory.plugins.memory``."""

from hermes_memory.plugins.memory import (  # noqa: F401
    _ProviderCollector,
    discover_memory_providers,
    discover_plugin_cli_commands,
    find_provider_dir,
    load_memory_provider,
)

__all__ = [
    "_ProviderCollector",
    "discover_memory_providers",
    "discover_plugin_cli_commands",
    "find_provider_dir",
    "load_memory_provider",
]
