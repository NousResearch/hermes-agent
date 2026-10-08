"""Emit ``hermes.compaction`` records as NeMo Relay marks on the owning session's scope stack.

The mark goes under the live turn of the same session when there is one, else under that session's
scope (gateway hygiene and other out-of-turn compactions). The session's own stack matters: Relay resets
LLM-history freshness only for the agent scope that owns a ``compaction`` mark, so the next LLM start
after a committed rewrite records the full history again. No live Relay host or session scope means
there is nothing to record. Relay exporters are configured by the user, so the mark adds no outbound
path of its own.
"""

from __future__ import annotations

import logging
from typing import Any

from agent import relay_runtime

logger = logging.getLogger(__name__)

COMPACTION_DATA_SCHEMA = {"name": "hermes.compaction", "version": "1"}


def _resolve_target(session_id: str) -> tuple[relay_runtime.RelayRuntime, relay_runtime.RelaySession, Any] | None:
    """``(host, session, parent handle)`` for ``session_id``: its live turn first, then its session scope."""
    turn = relay_runtime.active_turn(session_id)
    host = turn.lease.live_runtime() if turn is not None else None
    if host is not None:
        session = turn.lease.session
        return host, session, turn.handle or session.handle
    host = relay_runtime.HOST_REGISTRY.for_profile(create=False)
    if not isinstance(host, relay_runtime.RelayRuntime):
        return None
    session = host.get_session(session_id)
    if session is None or session.handle is None:
        return None
    return host, session, session.handle


def emit_compaction_mark(session_id: str, name: str, data: dict[str, Any]) -> bool:
    """Emit one mark; returns False when no Relay session owns ``session_id``."""
    target = _resolve_target(session_id) if session_id else None
    if target is None:
        logger.debug("no live Relay session for %s; %s mark not recorded", session_id or "none", name)
        return False
    host, session, handle = target
    host.run_in_session(
        session, host.relay.scope.event, name, handle=handle, data=data, data_schema=dict(COMPACTION_DATA_SCHEMA),
        metadata=relay_runtime.runtime_metadata(host.runtime_id),
    )
    return True
