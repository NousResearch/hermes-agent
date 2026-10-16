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


# The viewer's /new: the classic CLI's ``new_session``; Ink's ``tui_close`` (/new or a resume switch,
# both of which main's TUI followed with on_session_reset for the session it opened next).
_RESET_REASONS = frozenset({'new_session', 'tui_close'})


class _PendingClose:
    """One viewer close; identity, not the reason, tells a settler its close is still the pending one."""
    __slots__ = ('reason',)

    def __init__(self, reason):
        self.reason = reason


async def close_viewer_session(authority, session_id, reason):
    """``session.detach {end_reason}`` left *session_id* without viewers: close it now, or once idle.
    Its own async delegations end now, mid-turn too, as main's CLI exit and TUI close ended them
    (#134889 LC-7); each returns as an interrupted completion, and ones its still-running turn starts
    later are refused until a viewer attaches again. Other sessions' work is untouched."""
    live = authority.sessions[session_id]
    if live.subscribers:
        return  # a viewer attached while the detach was off the loop: nothing closes or ends
    from tools.async_delegation import interrupt_for_session, refuse_for_session
    route = live.route
    parents = tuple(dict.fromkeys((session_id, _physical_target(authority, session_id))))
    refuse_for_session((authority.profile_id, session_id), session_key=route, parent_session_ids=parents)
    for parent in parents:
        interrupt_for_session(session_key=route, parent_session_id=parent, reason=f'viewer_close:{reason}')
    authority.pending_closes[session_id] = _PendingClose(reason)
    await settle_viewer_close(authority, session_id)


async def settle_viewer_close(authority, session_id):
    """Complete a pending viewer close once *session_id* has no viewer and an idle FIFO: stamp the
    row with the viewer's reason, commit the cached agent's memory and fire ``on_session_finalize``.
    The drain calls this at every idle point; a viewer attaching meanwhile cancels the close."""
    close = authority.pending_closes.get(session_id)
    live = authority.sessions.get(session_id)
    if close is None or live is None or live.subscribers:
        authority.pending_closes.pop(session_id, None)
        return
    target = _physical_target(authority, session_id)
    # Every await below can let a viewer reattach (attach drops the pending close) or another
    # settler run, so the close is re-checked after each one.
    if await asyncio.to_thread(_busy, authority, session_id, target):
        return  # the drain's next idle point completes it
    row = await asyncio.to_thread(authority.db.get_session, target)
    if not _still_pending(authority, session_id, live, close):
        return
    # Claimed with no await since the check: exactly one settler stamps the row and runs the
    # lifecycle, and a close cancelled before its claim never wrote ``ended_at``.
    del authority.pending_closes[session_id]
    from gateway.session_acp_lifecycle import end_idle_viewer_session, reopen_if_reattached
    from gateway.session_runtime_workers import tracked_write
    ended = await tracked_write(authority, partial(end_idle_viewer_session, authority, session_id, close.reason),
                                then=partial(reopen_if_reattached, authority, session_id))
    if not ended:
        await _requeue_if_busy(authority, session_id, live, target, close)
        return
    if live.subscribers:
        return  # a viewer reattached during the stamp; reopen_if_reattached reopened the row
    platform = (row or {}).get('source') or (live.source.platform.value if live.source else '')
    await _run_close_lifecycle(authority, live, target, platform, close.reason)


def _still_pending(authority, session_id, live, close):
    return not live.subscribers and authority.pending_closes.get(session_id) is close


async def _requeue_if_busy(authority, session_id, live, target, close):
    """The stamp was refused. If input was admitted after the idle check, the close goes back to the
    drain's next idle point; any other refusal (row already ended) leaves nothing to finish."""
    if not await asyncio.to_thread(_busy, authority, session_id, target):
        return
    if live.subscribers or session_id in authority.pending_closes:
        return  # a reattach cancelled it, or a newer close replaced it
    authority.pending_closes[session_id] = close
    # The drain may have passed its idle point while the close was out of the map.
    await settle_viewer_close(authority, session_id)


async def _run_close_lifecycle(authority, live, target, platform, reason):
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
    messages = await finalize(session_id=target, platform=platform, reason=reason, session_key=live.route)
    if messages:
        # No viewer is left to show them (session.detach carries no messages yet).
        logger.info('Session-end plugin message(s) for %s dropped: no viewer', target)
    reset = getattr(runner, '_reset_session_off_loop', None)
    if reason in _RESET_REASONS and reset is not None:
        # The gateway's own /new arguments. The viewer opens its next session only after this
        # detach, so the new id is not known here.
        await reset(session_key=live.route, session_id=None, reason='new_session', platform=platform,
                    old_session_id=target, new_session_id=None)
