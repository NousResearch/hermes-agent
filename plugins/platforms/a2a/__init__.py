"""A2A (Agent2Agent) protocol server for Hermes Agent.

Exposes the Hermes ``AIAgent`` as an A2A-compliant remote agent so any
A2A-speaking client or peer agent can discover it (via the Agent Card) and
delegate tasks over JSON-RPC + SSE. Sibling to ``acp_adapter`` (editor
integration over stdio) and ``mcp_serve`` (tools over MCP).

Run it with ``hermes-a2a`` or ``python -m plugins.platforms.a2a``. See
``.plans/a2a-protocol.md`` for the design.
"""

from typing import Any


def register(ctx: Any) -> None:
    """Load the platform adapter only when plugin registration runs.

    Keeping this import lazy lets the standalone entry point execute its
    bootstrap and import-path hardening before gateway modules are imported.
    """
    from .adapter import register as register_adapter

    register_adapter(ctx)


__all__ = ["register"]
