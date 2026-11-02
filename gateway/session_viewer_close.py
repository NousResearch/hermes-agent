"""The last viewer closing a session ends it on the owner, lifecycle included (#134889 LC-1/LC-2/LC-6).

Main's classic CLI and TUI ran the end of a conversation in their own process: ``end_session``,
the memory providers' ``on_session_end`` extraction and plugin ``on_session_finalize`` (which also
closes the Relay conversation scope). Under the gateway those clients are viewers: the agent, its
memory manager and the Relay scope live in the owner, so ``session.detach {end_reason}`` from the
last viewer is where the owner runs them. A close that lands while a turn is still queued or running
is remembered and completed when the FIFO goes idle, so exiting mid-turn still ends the row.
The pending close is also kept in ``state_meta`` (``CLOSE_PREFIX + session_id``): a gateway stop or
restart before the FIFO goes idle would otherwise drop it, and nothing else ends a local row. Boot
recovery re-arms it (``rearm_persisted_closes``), so the restarted owner's first idle point ends it.
Each stored close carries its own operation id, and every delete is compare-and-delete on that exact
record under the owner-epoch fence: a cancelled older close, a reattach or a stale owner can only
remove the record it saw, never a newer close stored since.
"""
import asyncio
import contextlib
import json
import logging
import time
import uuid
from functools import partial
from pathlib import Path

from gateway.session_history_mount import is_history_mount
from hermes_state_local import local_receipt
from hermes_state_mutation_retirement import VIEWER_CLOSE_PREFIX
from hermes_state_runtime import RuntimeStoreError

logger = logging.getLogger(__name__)

CLOSE_PREFIX = VIEWER_CLOSE_PREFIX  # route retirement drops it too

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
# Ends nobody asked for (#143547 LC-35/LC-37): the last viewer vanished without closing (a killed
# terminal, a dropped socket), or the previous gateway stopped under it. Both stay resumable.
ORPHAN_REASONS = frozenset({'ws_orphan_reap', 'startup_orphan_reap'})
# The legacy startup sweep's session TTL (tui_gateway ``_SESSION_TTL_S``): see arm_startup_orphan_closes.
STARTUP_IDLE_S = 6 * 3600.0


class _PendingClose:
    """One viewer close; identity, not the reason, tells a settler its close is still the pending one.
    ``record`` is its ``state_meta`` value: a unique operation id plus the reason."""
    __slots__ = ('reason', 'record')

    def __init__(self, reason, record=None):
        self.reason = reason
        self.record = record or json.dumps({'op': uuid.uuid4().hex, 'reason': reason}, sort_keys=True)

    @classmethod
    def from_record(cls, record):
        try:
            reason = json.loads(record)['reason']
        except (ValueError, TypeError, KeyError):
            reason = record  # a bare-reason record from before operation ids
        return cls(reason, record)


async def close_viewer_session(authority, session_id, reason):
    """``session.detach {end_reason}`` left *session_id* without viewers: close it now, or once idle.
    Its own async delegations end now, mid-turn too, as main's CLI exit and TUI close ended them
    (#134889 LC-7); each returns as an interrupted completion, and ones its still-running turn starts
    later are refused until a viewer attaches again. Other sessions' work is untouched."""
    live = authority.sessions[session_id]
    if live.subscribers:
        return  # a viewer attached while the detach was off the loop: nothing closes or ends
    if is_history_mount(live):
        return  # stored history: no route, agent or delegation of its own
    from tools.async_delegation import interrupt_for_session, refuse_for_session
    route = live.route
    parents = tuple(dict.fromkeys((session_id, _physical_target(authority, session_id))))
    refuse_for_session((authority.profile_id, session_id), session_key=route, parent_session_ids=parents)
    for parent in parents:
        interrupt_for_session(session_key=route, parent_session_id=parent, reason=f'viewer_close:{reason}')
    authority.pending_closes[session_id] = close = _PendingClose(reason)
    await _store_and_settle(authority, session_id, live, close)


async def _store_and_settle(authority, session_id, live, close):
    """Persist *close* (already pending in memory), then complete it at once if the FIFO is idle."""
    await asyncio.to_thread(_persist_close, authority, session_id, close.record)
    if not _still_pending(authority, session_id, live, close):
        # A viewer attached during the write: its own clear may have run before this commit. Only
        # this close's record goes; a newer close stored since keeps its own.
        await asyncio.to_thread(forget_persisted_close, authority, session_id, close.record)
        return
    await settle_viewer_close(authority, session_id)


def _persist_close(authority, session_id, record):
    from hermes_state_runtime import _epoch

    def write(conn):
        _epoch(conn, authority.epoch)
        conn.execute('INSERT INTO state_meta(key,value) VALUES(?,?) ON CONFLICT(key) DO UPDATE SET value=excluded.value',
                     (CLOSE_PREFIX + session_id, record))
    authority.db._execute_write(write)


def forget_persisted_close(authority, session_id, record):
    """Blocking: drop *session_id*'s stored close only if it is still exactly *record* (that close
    completed or was cancelled), and only while this authority owns the store. Returns whether a
    row went."""
    from hermes_state_runtime import _epoch

    def write(conn):
        _epoch(conn, authority.epoch)
        return conn.execute('DELETE FROM state_meta WHERE key=? AND value=?',
                            (CLOSE_PREFIX + session_id, record)).rowcount
    try:
        return bool(authority.db._execute_write(write))
    except RuntimeStoreError as exc:
        logger.info('Stored close for %s left in place: %s', session_id, exc)
        return False


def cancel_persisted_close(authority, session_id):
    """Blocking, for a viewer attach: drop the stored close unless it is the one now pending in
    memory (a detach that ran after the attach stored it; it is that close's own to settle)."""
    record = authority.db.get_meta(CLOSE_PREFIX + session_id)
    if record is None:
        return
    pending = authority.pending_closes.get(session_id)
    if pending is not None and pending.record == record:
        return
    forget_persisted_close(authority, session_id, record)


def rearm_persisted_closes(authority):
    """Boot/serve recovery, after the local sessions are restored: every stored close whose session
    is mounted with no viewer goes back to ``pending_closes``. The caller schedules the drain, whose
    idle point (after any queued turn; never past an ``unknown`` one) completes it. Returns the ids."""
    with authority.db._read_ctx() as conn:
        stored = conn.execute('SELECT key, value FROM state_meta WHERE key LIKE ?', (CLOSE_PREFIX + '%',)).fetchall()
    armed = []
    for key, record in stored:
        session_id = key[len(CLOSE_PREFIX):]
        live = authority.sessions.get(session_id)
        if live is None or is_history_mount(live):
            continue  # not restored (paused, retired): left stored for a later recovery
        if live.subscribers:
            forget_persisted_close(authority, session_id, record)
            continue
        authority.pending_closes.setdefault(session_id, _PendingClose.from_record(record))
        armed.append(session_id)
    return armed


def _seconds(raw, default):
    with contextlib.suppress(ValueError, TypeError):
        return max(0.0, float(raw))
    return default


def _section(config, key):
    value = config.get(key)
    return value if isinstance(value, dict) else {}


def orphan_close_grace(authority, *, startup=False):
    """Blocking: ``sessions.orphan_end_grace_s`` of the profile that owns *authority*'s store (0 = never).
    Unset, the legacy backend's switches for the same reaping still opt out: ``dashboard.
    ws_orphan_reap_grace_s: 0`` (park forever) and, for *startup* arms, ``startup_orphan_sweep: false``."""
    from hermes_cli.config_defaults import DEFAULT_CONFIG
    from hermes_cli.config_effective import load_user_config_effective
    from gateway.session_local import profile_env_scope
    from utils import is_truthy_value
    default = DEFAULT_CONFIG['sessions']['orphan_end_grace_s']
    home = Path(authority.db.db_path).resolve().parent
    # A dropped socket detaches outside the RPC's owner scope: expand ``${VAR}`` against this profile.
    with profile_env_scope(home):
        config = load_user_config_effective(home / 'config.yaml')
    sessions, dashboard = _section(config, 'sessions'), _section(config, 'dashboard')
    if 'orphan_end_grace_s' in sessions:
        return _seconds(sessions['orphan_end_grace_s'], default)
    legacy_off = _seconds(dashboard.get('ws_orphan_reap_grace_s'), default) == 0 or (
        startup and not is_truthy_value(dashboard.get('startup_orphan_sweep'), default=True))
    return 0.0 if legacy_off else default


def orphan_candidate(authority, session_id):
    """Blocking: *session_id* is a local session with an open row. ACP ends on its own at detach."""
    try:
        receipt = local_receipt(authority.db, session_id)
        row = authority.db.get_session(receipt['entry']['session_id'])
    except (RuntimeStoreError, KeyError, TypeError):
        return False  # not a local session
    return receipt.get('policy', {}).get('source') != 'acp' and row is not None and row['ended_at'] is None


def _call_later(loop, delay, callback):
    # The loop's monotonic clock: time the host spends asleep does not count toward the grace.
    return loop.call_later(delay, callback)


def _stopping(authority):
    return authority.retiring or authority.retired or authority.runner._draining


def arm_orphan_close(authority, session_id, reason, grace):
    """*session_id* has no viewer and nobody closed it: unless one attaches within *grace* seconds,
    close it as a viewer would, without stopping its turn or its async delegations (#106742). A
    stopping owner arms nothing; the next boot does (``recover_local_sessions``)."""
    live = authority.sessions.get(session_id)
    if not grace or live is None or live.subscribers or is_history_mount(live) or _stopping(authority):
        return
    cancel_orphan_close(authority, session_id)
    authority.orphan_closes[session_id] = _call_later(
        asyncio.get_running_loop(), grace, partial(_fire_orphan_close, authority, session_id, reason))


async def arm_detached_orphan_close(authority, session_id):
    """``session.detach`` with no ``end_reason`` (a killed terminal, a dropped socket) left no viewer."""
    def grace():
        return orphan_close_grace(authority) if orphan_candidate(authority, session_id) else 0
    arm_orphan_close(authority, session_id, 'ws_orphan_reap', await asyncio.to_thread(grace))


def _idle_s(authority, session_id):
    from hermes_state_common import _sql_session_last_active
    # Resolve the target BEFORE opening a read context: ``_physical_target`` reads the receipt, and
    # a read that lands on the degraded path (non-WAL, or every read permit spent) waits on the
    # writer's non-reentrant lock. Nested inside this block it would wait on the lock this thread
    # already holds — a boot-time self-deadlock the watchdog can only answer by killing the gateway.
    target = _physical_target(authority, session_id)
    with authority.db._read_ctx() as conn:
        row = conn.execute(f'SELECT {_sql_session_last_active("s")} FROM sessions s WHERE s.id = ?',
                           (target,)).fetchone()
    return time.time() - row[0] if row and row[0] else 0.0


def arm_startup_orphan_closes(authority, session_ids):
    """Boot, after the stored closes are re-armed: the previous owner stopped under these sessions
    (idle, or with input it left queued), so no viewer is left to close them (#143547 LC-37). One
    that reattaches keeps it open; a fired close still waits for the queued input to settle.
    An idle classic CLI only redials on its next call, so a session idle less than the legacy
    sweep's TTL waits out the rest of it, as that sweep did, before its grace starts to count."""
    grace = orphan_close_grace(authority, startup=True)
    for session_id in session_ids:
        if grace and session_id not in authority.pending_closes and orphan_candidate(authority, session_id):
            delay = max(grace, STARTUP_IDLE_S - _idle_s(authority, session_id))
            arm_orphan_close(authority, session_id, 'startup_orphan_reap', delay)


def cancel_orphan_close(authority, session_id):
    handle = authority.orphan_closes.pop(session_id, None)
    if handle is not None:
        handle.cancel()


def _fire_orphan_close(authority, session_id, reason):
    authority.orphan_closes.pop(session_id, None)
    live = authority.sessions.get(session_id)
    if live is None or live.subscribers or session_id in authority.pending_closes or _stopping(authority):
        return
    # The persisted pending close a viewer's own close uses: it completes once the turn settles, and
    # survives a restart. Never ``close_viewer_session``: that would interrupt the delegations.
    authority.pending_closes[session_id] = close = _PendingClose(reason)
    from gateway.session_runtime_workers import track_mutation
    track_mutation(authority, _store_and_settle(authority, session_id, live, close))


async def settle_viewer_close(authority, session_id):
    """Complete a pending viewer close once *session_id* has no viewer and an idle FIFO: stamp the
    row with the viewer's reason, commit the cached agent's memory and fire ``on_session_finalize``.
    The drain calls this at every idle point; a viewer attaching meanwhile cancels the close."""
    close = authority.pending_closes.get(session_id)
    live = authority.sessions.get(session_id)
    if close is None or live is None or live.subscribers:
        authority.pending_closes.pop(session_id, None)
        return
    if not await _resolve_abandoned_unknown(authority, session_id, live, close):
        return  # left paused (a viewer came back, or the resolution was refused)
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
    await asyncio.to_thread(forget_persisted_close, authority, session_id, close.record)
    if live.subscribers:
        return  # a viewer reattached during the stamp; reopen_if_reattached reopened the row
    platform = (row or {}).get('source') or (live.source.platform.value if live.source else '')
    from gateway.session_vault_prompt import release_unlocks
    release_unlocks(session_id)
    await _run_close_lifecycle(authority, live, target, platform, close.reason)


async def _resolve_abandoned_unknown(authority, session_id, live, close):
    """A turn the previous owner lost (hard kill) recovers ``unknown`` and pauses the FIFO until
    someone Discards it, but the last viewer closed the session: nobody is left to. The viewer's
    close is that decision, so the owner resolves it as Discard does (interrupted, never replayed,
    transcript boundary written) and the drain moves on; a queued successor still runs before the
    close. A viewer back first cancels the close and keeps the decision. Returns whether the
    close may go on to its idle check (no ``unknown`` turn is left)."""
    from hermes_state_runtime import list_session_admissions
    rows = await asyncio.to_thread(list_session_admissions, authority.db, session_id=session_id)
    unknown = [row for row in rows if row['status'] == 'unknown']
    if not unknown:
        return True
    if close.reason in ORPHAN_REASONS:
        return False  # nobody closed it: the turn waits for its user's own Discard
    from gateway.session_contract import SessionRef
    ref = SessionRef(authority.profile_id, session_id)
    for row in unknown:
        if not _still_pending(authority, session_id, live, close):
            return False
        try:
            await authority.resolve_unknown_as_owner(ref, row['admission_id'], row['generation'])
        except RuntimeStoreError as exc:
            logger.warning('Unknown turn %s on closed session %s left paused: %s',
                           row['admission_id'], session_id, exc.reason)
            return False
        logger.info('Unknown turn %s resolved interrupted: its session %s was closed by its last viewer',
                    row['admission_id'], session_id)
    # The resolution also rescheduled the drain; the idle check below and its next idle point
    # are both safe (the close is claimed exactly once).
    return _still_pending(authority, session_id, live, close)


def _still_pending(authority, session_id, live, close):
    return not live.subscribers and authority.pending_closes.get(session_id) is close


async def _requeue_if_busy(authority, session_id, live, target, close):
    """The stamp was refused. If input was admitted after the idle check, the close goes back to the
    drain's next idle point; any other refusal (row already ended) leaves nothing to finish."""
    if not await asyncio.to_thread(_busy, authority, session_id, target):
        if session_id not in authority.pending_closes:
            await asyncio.to_thread(forget_persisted_close, authority, session_id, close.record)  # already ended
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
