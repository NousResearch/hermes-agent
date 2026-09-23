"""Accounting at completion of an exact heartbeat admission attempt.

Execution means entering the gateway's agent runner after turn preparation,
not reserving an adapter slot, successful model completion, or outbound delivery.
The done callback retains the watch's profile ContextVars and manager claim.
"""
import logging

logger = logging.getLogger("gateway.run")


async def resolve_heartbeat_owner(runner, event, entry):
    """Keep normal reset/topic resolution, then admit only the original lineage."""
    expected = getattr(event, "_heartbeat_session_id", None)
    if not expected:
        return True
    resolved = entry.session_id
    if resolved != expected:
        def compression_tip():
            return runner.session_store._db_for_key(entry.session_key).get_compression_tip(expected)

        tip = await runner._run_in_executor_with_context(compression_tip)
        if tip != resolved:
            return False
    # Keep a value, not the mutable routing entry: preparation and hooks can yield
    # to /new or /stop before the agent runner starts.
    event._heartbeat_resolved_session_id = resolved
    return heartbeat_owner_is_current(runner, event, entry.session_key)


def heartbeat_owner_is_current(runner, event, session_key):
    expected = getattr(event, "_heartbeat_resolved_session_id", None)
    if not expected:
        return True
    current = runner.session_store.lookup_by_session_key(session_key)
    return current is not None and not current.suspended and current.session_id == expected


def process_heartbeat_still_alive(event) -> bool:
    """Revalidate a background-process heartbeat against the live process registry.

    A ``terminal(background=true, notify=true, heartbeat=…)`` heartbeat is a snapshot
    taken while the process was running. Between the registry's ``_emit_heartbeat``
    put on ``completion_queue`` and the gateway's adapter admission, the process can
    exit; admitting the stale event would then promote a "still running" message into
    a fresh agent turn after the user already saw the completion notice (#120334).

    The synthetic ``MessageEvent`` carries the originating session identity stamped at
    injection time (``_process_heartbeat_session_id`` + ``_process_heartbeat_started_at``).
    We re-look-up the live process registry: the session must still be registered and
    non-exited, and the started_at epoch must match (a reuse-guard against recycled
    ids). Anything else means stale — the caller drops the event WITHOUT starting a
    turn. Returns True for events that are not background-process heartbeats (the
    scheduled /heartbeat path keeps its own provenance check)."""
    sid = getattr(event, "_process_heartbeat_session_id", None)
    if not sid:
        return True
    expected_started_at = getattr(event, "_process_heartbeat_started_at", None)
    try:
        from tools.process_registry import process_registry
        session = process_registry.get(sid)
    except Exception:
        logger.debug(
            "Process heartbeat liveness check raised for session %s — treating as stale", sid,
            exc_info=True,
        )
        return False
    if session is None:
        return False
    # Match the exact incarnation: if the id was recycled (e.g. a fresh spawn reused
    # the slot), started_at diverges and we treat the heartbeat as stale.
    if expected_started_at is not None and session.started_at != expected_started_at:
        return False
    return not session.exited


def settle_heartbeat_attempt(event, manager):
    if not getattr(event, "_heartbeat_execution_started", False):
        try:
            manager.abandon_fire()
        except Exception:
            logger.warning("Failed to refund unexecuted heartbeat", exc_info=True)
