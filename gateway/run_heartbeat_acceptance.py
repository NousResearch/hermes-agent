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


def settle_heartbeat_attempt(event, manager):
    if not getattr(event, "_heartbeat_execution_started", False):
        try:
            manager.abandon_fire()
        except Exception:
            logger.warning("Failed to refund unexecuted heartbeat", exc_info=True)


async def admit_heartbeat(runner, adapter, source, session_id, route):
    """Coalesce overdue ticks behind the canonical FIFO, not an adapter-local slot."""
    import json
    from gateway.platforms.event import MessageEvent
    from hermes_cli.heartbeat import HeartbeatManager
    from hermes_state_runtime import list_session_admissions

    from gateway.session_authorities import active_authority
    authority = active_authority(runner)
    if authority is None:
        return
    pending = list_session_admissions(authority.db, session_id=session_id)
    if any(row['payload'].get('native_text_v1', {}).get('automation', {}).get('heartbeat')
           for row in pending):
        return
    manager = HeartbeatManager(session_id)
    prompt = manager.due_prompt()
    if prompt is None:
        return
    state = manager.state
    identity = json.dumps(['heartbeat', session_id, state.created_at, state.fire_count], separators=(',', ':'))
    event = MessageEvent(text=prompt, source=source, internal=True, message_id=identity,
        metadata={'gateway_session_key': route, 'gateway_session_id': session_id})
    event._heartbeat_session_id = session_id
    try:
        await authority.admit_automation(adapter, event, identity)
    except BaseException:
        # Readiness is not a delivery attempt, and cancellation before commit is not ACK.
        if getattr(event, '_gateway_accepted', False) is not True:
            manager.abandon_fire()
        raise
