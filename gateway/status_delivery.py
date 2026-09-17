"""Shared status-line delivery (issue #30045: edit-in-place instead of appending bubbles).

One implementation, two consumers today: the gateway's own turn-status lane
(``run_turn_runner._status_callback_sync`` via ``gateway.run``) and the plugin-facing
``ctx.emit_status`` / ``ctx.platform_actions.send_status`` facade in ``hermes_cli.platform_actions``.
Public module, no imports from ``gateway.run`` — safe to import from either layer.
"""
from __future__ import annotations

from typing import Any, Dict, Optional


async def send_or_update_status(adapter, chat_id, status_key, content, metadata: Optional[Dict[str, Any]] = None):
    """Route a status through ``adapter.send_or_update_status`` when supported (edits the previous
    bubble for the same ``status_key`` instead of appending); otherwise fall back to plain send.

    See #30045. ``metadata`` may carry thread routing (``thread_id``/``message_thread_id``);
    adapters consume it, this helper passes it through untouched."""
    sender = getattr(adapter, "send_or_update_status", None)
    if callable(sender):
        return await sender(chat_id, status_key, content, metadata=metadata)
    return await adapter.send(chat_id, content, metadata=metadata)
