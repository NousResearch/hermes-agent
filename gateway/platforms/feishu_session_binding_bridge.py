"""Feishu Session Binding Bridge — sidecar for v2.10 multi-entry session binding.

Called from gateway/platforms/feishu.py after session_key is built but before
message dispatch.  This is an optional sidecar: if any step fails, the main
Feishu message flow continues uninterrupted.

The sidecar:
1. Normalizes the SessionSource into a minimal raw payload.
2. Uses FeishuEntryAdapter to produce a canonical EntryEvent.
3. Writes the workspace/session binding to SessionBinding store.
4. Does NOT create tasks, call agents, route, or write ledger events.
"""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from gateway.session import SessionSource

logger = logging.getLogger(__name__)


def record_feishu_session_binding(source: "SessionSource", session_key: str) -> None:
    """Record a Feishu session binding for v2.10 without affecting existing flow.

    Called from FeishuAdapter._process_inbound_message() after session_key
    is built but before _dispatch_inbound_event().  Failures are caught and
    logged; the main flow continues.

    Args:
        source: The SessionSource for the incoming message.
        session_key: The existing session key string from build_session_key().
    """
    try:
        from agent.managed_agents.feishu_entry_adapter import FeishuEntryAdapter
        from agent.managed_agents.session_binding import put_binding

        raw = {
            "chat_id": source.chat_id,
            "message_id": getattr(source, "message_id", None) or f"auto-{int(time.time())}",
            "open_id": source.user_id or "unknown",
            "content": "",
            "thread_id": source.thread_id,
            "session_key": session_key,
        }

        adapter = FeishuEntryAdapter()
        event = adapter.normalize_event(raw)

        put_binding(
            entrypoint="feishu",
            external_channel_id=source.chat_id,
            external_thread_id=source.thread_id,
            workspace_id=event.workspace_id,
            session_id=session_key,
        )
        logger.debug("Recorded session binding for Feishu session_key=%s", session_key)
    except Exception:
        logger.debug("Session binding sidecar failed (non-critical): %s", exc_info=True)
