"""Ephemeral chat progress lines for RUNNING background delegations.

Deliberately separate from the completion pipeline: a progress line must never claim the durable
delivery row, wake the parent turn, or race the result. The gateway's async-delegation watcher
calls :func:`deliver_delegation_progress` for each ``delegation_progress`` event it drains; every
other consumer formats that type as ``None`` (see ``tools.process_registry_notifications``).
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from gateway.config import Platform

logger = logging.getLogger(__name__)


async def deliver_delegation_progress(runner: Any, evt: dict) -> None:
    """Send one progress line to the chat that spawned the delegation.

    Best-effort and terminal: no durable claim, no delivery row, no parent turn, and no retry. A
    line that cannot be routed is dropped — the completion result still arrives — because retrying
    would surface stale progress after the result.
    """
    text = str(evt.get("text") or "").strip()
    if not text:
        return
    try:
        from tools.environments.sandbox_scope import event_visible
        if not event_visible(evt):
            return
        source = await asyncio.to_thread(runner._build_process_event_source, evt)
        # A raw api_server route would need the durable delivery row completions use; progress
        # must not create one, so this stays chat-platforms-only.
        if source is None or source.platform == Platform.API_SERVER or not source.chat_id:
            return
        await runner._send_watcher_message(
            source.platform.value, source.chat_id, source.thread_id, text, evt)
    except Exception:
        logger.debug("Delegation progress delivery failed for %s", evt.get("delegation_id"),
                     exc_info=True)
