"""Slash command dispatch for the multitenancy plugin.

Hermes' built-in command handlers (in ``GatewayRunner._handle_message``) only
fire when the gateway main flow handles a message. Since the multitenancy
plugin returns ``{"action": "skip"}`` from ``pre_gateway_dispatch``, the main
flow never runs — so we own these commands ourselves.

The plugin owns a small fixed set of operational commands:
``/stop``, ``/status``, ``/new``, ``/reset``, and ``/help``.
"""
from __future__ import annotations

from typing import Optional

# Commands the plugin owns. Anything else falls through to dispatch (where the
# router treats it as a normal user message).
_KNOWN_COMMANDS = frozenset({"stop", "status", "new", "reset", "help"})


def parse_command(text: str) -> Optional[tuple[str, str]]:
    """Parse a slash command. Returns (cmd, args) or None."""
    if not text or not text.startswith("/"):
        return None
    parts = text.split(maxsplit=1)
    raw = parts[0][1:].lower()
    args = parts[1] if len(parts) > 1 else ""
    # Reject paths and known-bad shapes
    if "/" in raw or not raw:
        return None
    if raw in _KNOWN_COMMANDS:
        return (raw, args)
    return None
