"""Remote 'node host': run the Meet bot on a different machine than the gateway.

    gateway (Linux)  ── ws://mac.local:18789 ──▶  node server (Mac) → process_manager → meet_bot

Why: Google sign-in + Chrome profile live on the user's laptop; running the bot there reuses
that profile without shipping credentials. NodeClient (gateway RPC), NodeServer (hosts the
bot), NodeRegistry (approved nodes: name → url+token), protocol (envelope helpers).
"""

from __future__ import annotations

from plugins.google_meet.node import protocol
from plugins.google_meet.node.client import NodeClient
from plugins.google_meet.node.protocol import (  # noqa: F401
    VALID_REQUEST_TYPES, decode, encode, make_error, make_request, make_response, validate_request)
from plugins.google_meet.node.registry import NodeRegistry
from plugins.google_meet.node.server import NodeServer

__all__ = [
    "NodeClient", "NodeServer", "NodeRegistry", "protocol", "make_request", "make_response",
    "make_error", "encode", "decode", "validate_request", "VALID_REQUEST_TYPES"]
