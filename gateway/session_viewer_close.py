"""The last viewer closing a session ends it on the owner, lifecycle included (#134889 LC-1/LC-2/LC-6).

Main's classic CLI and TUI ran the end of a conversation in their own process: ``end_session``,
the memory providers' ``on_session_end`` extraction and plugin ``on_session_finalize`` (which also
closes the Relay conversation scope). Under the gateway those clients are viewers: the agent, its
memory manager and the Relay scope live in the owner, so ``session.detach {end_reason}`` from the
last viewer is where the owner runs them. A close that lands while a turn is still queued or running
is remembered and completed when the FIFO goes idle, so exiting mid-turn still ends the row.
"""
import asyncio
import logging
from functools import partial

from hermes_state_local import local_receipt
from hermes_state_runtime import RuntimeStoreError

logger = logging.getLogger(__name__)

_BUSY_SQL = ("SELECT 1 FROM session_admissions WHERE target_session_id=? AND status IN ('queued','started','unknown') "
             "UNION ALL SELECT 1 FROM worker_executions WHERE session_id IN (?,?) "
             "AND status IN ('registered','running','unknown') LIMIT 1")


def _physical_target(authority, session_id):
    try:
        return local_receipt(authority.db, session_id)['entry']['session_id']
    except (RuntimeStoreError, KeyError, TypeError):
        return authority.db.get_compression_tip(session_id) or session_id


def _busy(authority, session_id, target):
    with authority.db._read_ctx() as conn:
        return conn.execute(_BUSY_SQL, (session_id, session_id, target)).fetchone() is not None


async def close_viewer_session(authority, session_id, reason):
    """``session.detach {end_reason}`` left *session_id* without viewers: close it now, or once idle."""
    authority.pending_closes[session_id] = reason
    await settle_viewer_close(authority, session_id)


async def settle_viewer_close(authority, session_id):
    """Complete a pending viewer close once *session_id* has no viewer and an idle FIFO: stamp the
    row with the viewer's reason, commit the cached agent's memory and fire ``on_session_finalize``.
    The drain calls this at every idle point; a viewer attaching meanwhile cancels the close."""
    reason = authority.pending_closes.get(session_id)
    live = authority.sessions.get(session_id)
    if reason is None or live is None or live.subscribers:
        authority.pending_closes.pop(session_id, None)
        return
    from gateway.session_acp_lifecycle import end_idle_viewer_session, reopen_if_reattached
    from gateway.session_runtime_workers import tracked_write
    await tracked_write(authority, partial(end_idle_viewer_session, authority, session_id, reason),
                        then=partial(reopen_if_reattached, authority, session_id))
    target = _physical_target(authority, session_id)
    if live.subscribers or authority.pending_closes.get(session_id) != reason:
        return
    if await asyncio.to_thread(_busy, authority, session_id, target):
        return  # the drain's next idle point completes it
    authority.pending_closes.pop(session_id, None)
    await _run_close_lifecycle(authority, live, target, reason)


async def _run_close_lifecycle(authority, live, target, reason):
    """What main's client ran at exit and ``/new``, in the process that owns the session: the
    memory providers' end-of-session commit (the cached agent is evicted at a boundary, as a
    gateway ``/new`` does) and the finalize hooks under the owning profile's scope."""
    runner = authority.runner
    evict = getattr(runner, '_evict_cached_agent_at_boundary', None)
    if evict is not None:
        evict(live.route)
    finalize = getattr(runner, '_finalize_session_off_loop', None)
    if finalize is None:
        return
    row = await asyncio.to_thread(authority.db.get_session, target)
    platform = (row or {}).get('source') or (live.source.platform.value if live.source else '')
    messages = await finalize(session_id=target, platform=platform, reason=reason, session_key=live.route)
    if messages:
        # No viewer is left to show them (session.detach carries no messages yet).
        logger.info('Session-end plugin message(s) for %s dropped: no viewer', target)
