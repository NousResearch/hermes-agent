"""Local startup recovery: blocking preparation in workers, publication on the owner loop."""
import asyncio
import logging

from gateway.config import Platform
from gateway.session_contract import SessionRef
from hermes_state_runtime import RuntimeStoreError, list_session_admissions

logger = logging.getLogger(__name__)


def _route_snapshot(store, route):
    entry = store._entries.get(route)
    return None if entry is None else (entry.session_id, entry.origin.to_dict() if entry.origin else None)


def _prepare(authority, schedule):
    from gateway.session_local_recovery import (
        _CatchUpPending, _ReceiptRefused, _stray_route_target, local_receipt_ids, prepare_local_session,
    )
    from gateway.session_viewer_close import persisted_closes, startup_orphan_delays
    store, prepared, pending, retry = authority.runner.session_store, {}, {}, []
    for sid in local_receipt_ids(authority):
        try:
            item = prepare_local_session(authority, sid, catch_up=False)
            receipt, _, route, _, _ = item
            with store._lock:
                store._ensure_loaded_locked()
                snapshot = _route_snapshot(store, route)
                stray = snapshot is not None and _stray_route_target(authority, receipt, snapshot[0])
            prepared[sid] = item, (snapshot, stray)
            pending[sid] = list_session_admissions(authority.db, session_id=sid)
        except _CatchUpPending:
            retry.append(sid)
        except _ReceiptRefused:
            continue
        except RuntimeStoreError as exc:
            logger.warning('Local session %s paused: %s', sid, exc.reason)
    closes = persisted_closes(authority) if schedule else {}
    orphans = [sid for sid, rows in pending.items() if not any(r['status'] == 'unknown' for r in rows)]
    delays = startup_orphan_delays(authority, orphans) if schedule else {}
    return prepared, pending, closes, delays, retry


def _stopping(authority):
    return authority.retiring or authority.retired or getattr(authority.runner, '_draining', False)


def _adapter(authority):
    """Inspect without creating a secondary adapter map during suspended preparation."""
    registry = getattr(authority.runner, 'session_authorities', None)
    name = registry.profile_name(authority) if registry is not None else None
    adapters = (authority.runner.adapters if name is None else
                getattr(authority.runner, '_profile_adapters', {}).get(name, {}))
    return adapters.get(Platform.LOCAL)


def _publish_sessions(authority, prepared, before):
    from gateway.session_local_recovery import publish_local_session
    published = set()
    for sid, (item, proof) in prepared.items():
        # A retire/delete or a concurrent restore won while the reads were suspended.
        if before.get(sid) is not None and authority.sessions.get(sid) is not before[sid]:
            continue
        try:
            publish_local_session(authority, sid, item, route_proof=proof)
            published.add(sid)
        except RuntimeStoreError as exc:
            logger.warning('Local session %s paused: %s', sid, exc.reason)
    return published


def _publish_closes(authority, published, pending, closes, delays, previous_closes, viewers):
    from gateway.session_history_mount import is_history_mount
    from gateway.session_runtime_workers import track_mutation
    from gateway.session_viewer_close import (
        CLOSE_PREFIX, _PendingClose, arm_orphan_close, forget_persisted_close, settle_viewer_close,
    )
    for sid in published:
        rows = pending[sid]
        unknown = any(row['status'] == 'unknown' for row in rows)
        if rows and not unknown:
            authority._schedule(SessionRef(authority.profile_id, sid))
        live = authority.sessions[sid]
        if live.viewer_revision != viewers.get(sid, 0):
            continue  # attach cancelled this snapshot, including a subsequent plain detach
        record = closes.get(CLOSE_PREFIX + sid)
        if record is not None and not is_history_mount(live):
            if live.subscribers:
                # Compare-and-delete is a tracked, epoch-fenced write, never a loop-side DB call.
                track_mutation(authority, asyncio.to_thread(forget_persisted_close, authority, sid, record))
            elif authority.pending_closes.get(sid) is previous_closes.get(sid):
                authority.pending_closes.setdefault(sid, _PendingClose.from_record(record))
                authority._schedule(SessionRef(authority.profile_id, sid))
                if unknown:
                    track_mutation(authority, settle_viewer_close(authority, sid))
        if sid in delays and sid not in authority.pending_closes and sid not in authority.orphan_closes:
            arm_orphan_close(authority, sid, 'startup_orphan_reap', delays[sid])


def _publish(authority, bundle, before, adapter, previous_closes, viewers, schedule):
    """False means a worker changed the routing snapshot: prepare again, never wait here."""
    current_adapter = _adapter(authority)
    if (_stopping(authority)
            or (adapter is not None and current_adapter is not adapter)
            or (current_adapter is not None and getattr(current_adapter, 'authority', None) is not authority)):
        return True
    prepared, pending, closes, delays, _ = bundle
    store = authority.runner.session_store
    if not store._lock.acquire(blocking=False):
        return False
    try:
        if any(_route_snapshot(store, item[2]) != proof[0] for item, proof in prepared.values()):
            return False
        published = _publish_sessions(authority, prepared, before)
    finally:
        store._lock.release()
    if schedule:
        _publish_closes(authority, published, pending, closes, delays, previous_closes, viewers)
    return True


async def _prepare_operation(authority, schedule, initial):
    from gateway.session_local_adoption import adopt_before_restore, adopt_local_compressions
    # This whole operation is tracked (or joined before an unpublished build can unwind).
    # Compression adoption retains the expected-receipt and owner-epoch fences.
    await adopt_local_compressions(authority, tracked=not initial)
    failed = set()
    while not _stopping(authority):
        bundle = await asyncio.to_thread(_prepare, authority, schedule)
        retry = set(bundle[4]) - failed
        if not retry:
            return bundle
        for sid in retry:
            if not await adopt_before_restore(authority, sid, tracked=not initial):
                failed.add(sid)
    return {}, {}, {}, {}, []


async def _prepare_joined(authority, schedule, initial):
    from gateway.session_runtime_workers import track_mutation
    operation = _prepare_operation(authority, schedule, initial)
    task = asyncio.create_task(operation) if initial else track_mutation(authority, operation)
    try:
        return await asyncio.shield(task)
    except asyncio.CancelledError:
        if initial:
            # No registry entry exists for retirement to join: do not release its DB/reservation
            # while the preparation's compression write (or a reader using that DB) is still live.
            while not task.done():
                try:
                    await asyncio.shield(task)
                except asyncio.CancelledError:
                    continue
                except Exception:
                    logger.exception('Local recovery preparation failed while joining a cancelled build')
                    break  # preserve the caller's cancellation, not a late preparation error
            if not task.cancelled():
                task.exception()
        raise


async def recover_local_sessions_async(authority, *, schedule=False, initial=False):
    """Prepare without live mutations; publish in the caller's step, with no await afterwards.

    Initial construction calls this BEFORE registering the authority. Nothing yields between
    its loop-side publication and the initializer's remaining registration steps.
    """
    from gateway.session_authorities import async_owner_scope
    async with async_owner_scope(authority):
        while not _stopping(authority):
            before, adapter = dict(authority.sessions), _adapter(authority)
            previous_closes = dict(authority.pending_closes)
            viewers = {sid: live.viewer_revision for sid, live in before.items()}
            bundle = await _prepare_joined(authority, schedule, initial)
            if _publish(authority, bundle, before, adapter, previous_closes, viewers, schedule):
                return
