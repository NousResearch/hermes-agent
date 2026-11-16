"""Emit ``hermes.compaction`` records as NeMo Relay marks on the owning session's scope stack.

The mark goes under the live turn of the same session when there is one, else under that session's
scope (gateway hygiene and other out-of-turn compactions), else under the live turn the compacting agent
started in (a child session a rotating commit created mid-turn), and never from a turn Relay does not
instrument. The session's own stack matters: Relay resets LLM-history freshness only for the agent scope
that owns a ``compaction`` mark, so the next LLM start after a committed rewrite records the full history
again. No live Relay host or session scope means
there is nothing to record. Relay exporters are configured by the user, so the mark adds no outbound
path of its own.
"""

from __future__ import annotations

import logging
from typing import Any

from agent import relay_runtime

logger = logging.getLogger(__name__)

COMPACTION_DATA_SCHEMA = {"name": "hermes.compaction", "version": "1"}
# Marks are observability: after one stalled publish, stop waiting on Relay for the rest of the process
# so a wedged pipeline costs one bounded timeout, not one per compaction under the compression lease.
_stalled = False


def _turn_target(session_id: str) -> tuple[relay_runtime.RelayRuntime, relay_runtime.RelaySession, Any] | None:
    turn = relay_runtime.active_turn(session_id)
    host = turn.lease.live_runtime() if turn is not None else None
    if host is None:
        return None
    session = turn.lease.session
    return host, session, turn.handle or session.handle


def _session_target(session_id: str) -> tuple[relay_runtime.RelayRuntime, relay_runtime.RelaySession, Any] | None:
    host = relay_runtime.HOST_REGISTRY.for_profile(create=False)
    if not isinstance(host, relay_runtime.RelayRuntime):
        return None
    session = host.get_session(session_id)
    if session is None or session.handle is None:
        return None
    return host, session, session.handle


def _resolve_target(
    session_id: str, turn_session_id: str | None = None,
) -> tuple[relay_runtime.RelayRuntime, relay_runtime.RelaySession, Any] | None:
    """``(host, session, parent handle)`` for ``session_id``: its live turn, then its session scope, then the
    live turn of ``turn_session_id``.

    ``turn_session_id`` is the session the compacting agent's turn started in. A rotating commit moves the
    agent to a child session that has no Relay scope until the next turn opens one, so a later rotation in
    the same turn is marked on that turn. When the child already has a scope, that scope owns the next LLM
    starts for the child and keeps the mark. A turn Relay does not instrument (a second concurrent turn on
    the session, such as a background-review fork) gets nothing: its compaction rewrote its own transcript,
    not the live session's."""
    if not relay_runtime.relay_instrumentation_enabled():
        return None
    target = _turn_target(session_id) or _session_target(session_id)
    if target is None and turn_session_id:
        target = _turn_target(turn_session_id)
    return target


def compaction_target_available(session_id: str, *, turn_session_id: str | None = None) -> bool:
    """Whether a mark for this session has a live Relay scope to receive it."""
    return bool(session_id and _resolve_target(session_id, turn_session_id))


def emit_compaction_mark(
    session_id: str, name: str, data: dict[str, Any], *, turn_session_id: str | None = None,
) -> bool:
    """Emit one mark; returns False when no Relay scope owns ``session_id`` (see ``_resolve_target``)."""
    target = _resolve_target(session_id, turn_session_id) if session_id else None
    if target is None:
        logger.debug("no live Relay session for %s; %s mark not recorded", session_id or "none", name)
        return False
    global _stalled
    if _stalled:
        return False
    host, session, handle = target
    try:
        host.run_in_session(
            session, host.relay.scope.event, name, handle=handle, data=data,
            data_schema=dict(COMPACTION_DATA_SCHEMA), metadata=relay_runtime.runtime_metadata(host.runtime_id),
            timeout=relay_runtime.SCOPE_OP_TIMEOUT,
        )
    except TimeoutError:
        _stalled = True
        raise
    return True
