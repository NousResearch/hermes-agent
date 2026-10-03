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
    from gateway.run_pinned_session import pinned_session_continues
    resolved = entry.session_id
    if not await pinned_session_continues(runner, entry, expected):
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


def settle_heartbeat_attempt(event, manager):
    if not getattr(event, "_heartbeat_execution_started", False):
        try:
            manager.abandon_fire()
        except Exception:
            logger.warning("Failed to refund unexecuted heartbeat", exc_info=True)
