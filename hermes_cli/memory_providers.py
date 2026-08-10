"""Compatibility shim — re-exports from plugins.memory.config_schema.

Upstream refactored hermes_cli/memory_providers into plugins/memory/config_schema
(commit 2a632807e).  This module keeps the old import paths working for fork
code (web_server.py Weixin QR + dashboard auth customizations) that hasn't been
rebased onto the new module layout yet.
"""

from plugins.memory.config_schema import (  # noqa: F401
    ProviderConfigSchema as MemoryProvider,
    ProviderField,
    get_provider_config_schema as get_memory_provider,
)