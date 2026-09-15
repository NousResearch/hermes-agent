"""MCP servers and runtime plugins, as things a NOVA agent can be granted.

Both are capabilities the runtime already has. NOVA's job here is the same as it was for
channels and toolsets: **discover what the runtime offers, let a tenant grant it per agent,
and compile the grant into the profile** — not to reimplement either.

Two facts shaped everything in this package.

**``nova apply`` rewrites the profile's ``config.yaml`` wholesale.** So a grant cannot live
in that file. It lives in the bundle, and materialization compiles it — the same reason
``SOUL.md`` is derived from ``bundle/prompts/`` rather than edited in place.

**An MCP server registers its tools as a toolset named ``mcp-<name>``**
(``tools/mcp_tool_registration.py``). An MCP server is therefore not a new governance
surface; it is a toolset, and the compiled policy decides its tools like any other.
"""

from nova.extensions.model import (
    AUTH_KINDS,
    ExtensionCatalogue,
    McpServer,
    PluginEntry,
    catalogue,
    set_discovery,
)

__all__ = [
    "AUTH_KINDS",
    "ExtensionCatalogue",
    "McpServer",
    "PluginEntry",
    "catalogue",
    "set_discovery",
]
