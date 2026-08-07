"""Catalog of installable standalone memory-provider plugins.

The catalog lets `hermes memory setup` offer providers that are distributed
outside the core repository. Entries point at standalone plugin repos; the
provider implementation still lives with its maintainer and is installed through
the normal `hermes plugins install` path.
"""

from __future__ import annotations

INSTALLABLE_MEMORY_PROVIDERS = [
    {
        "name": "openbrain",
        "label": "openbrain",
        "setup_hint": "install standalone plugin",
        "identifier": "longman391/hermes-openbrain-memory-provider",
        "description": "OpenBrain (OB1) over MCP Streamable HTTP",
        "project_url": "https://github.com/NateBJones-Projects/OB1",
        "plugin_url": "https://github.com/longman391/hermes-openbrain-memory-provider",
    },
]
