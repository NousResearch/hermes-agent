"""A local receipt's compression catch-up (#140060), committed off the owner loop.

``adopt_untracked_compression`` takes the writer lock, up to ``_WRITE_PATIENCE_S`` when contended. On
the owner loop that stalls every profile it serves (streams, controls, admissions), including during
a hot-profile activation. So the async paths that restore a receipt (boot and hot-profile recovery, a
socket's first read) commit the catch-up in a worker first, and the restore that follows only reads.
A restore reached synchronously on the loop hands it to a tracked worker (``schedule_adoption``).
"""
import asyncio
import logging
from functools import partial

from persistence.core.errors import AuthorityRefusal

logger = logging.getLogger(__name__)


def on_owner_loop():
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return False
    return True


def _binding(receipt):
    """The stored receipt *receipt* as the ``SessionBinding`` the bindings verbs compare."""
    from persistence.authority.bindings import SessionBinding
    return SessionBinding(
        profile_id=receipt['profile_id'], principal_id=receipt['principal_id'], request_id=receipt['request_id'],
        session_id=receipt['session_id'], route=receipt['route'], entry=receipt['entry'], policy=receipt['policy'],
        lineage=tuple(receipt.get('lineage', [receipt['session_id']])), legacy_session_id=receipt.get('legacy_session_id'))


def _due(authority, sid):
    """*sid*'s stored receipt when a catch-up would commit for it on this owner, else None. Read-only."""
    from gateway.session_local_recovery import owned_receipt
    from gateway.session_persistence import authority_persistence
    try:
        receipt = owned_receipt(authority, sid)[0]
        return receipt if authority_persistence(authority).bindings.compression_adoptable(_binding(receipt)) else None
    except (KeyError, TypeError, ValueError, AttributeError):
        return None  # the restore refuses this receipt and logs the check that failed


def _adopt(authority, sid, receipt):
    from gateway.session_persistence import authority_persistence
    authority_persistence(authority).bindings.adopt_untracked_compression(
        epoch=authority.epoch, session_id=sid, expected=_binding(receipt))


async def adopt_before_restore(authority, sid, *, tracked=True):
    """Commit *sid*'s due catch-up in a worker, so the restore that follows only reads.

    ``tracked=False``: an authority still being built, not yet published for retirement to join."""
    receipt = _due(authority, sid)
    if receipt is None:
        return
    write = partial(_adopt, authority, sid, receipt)
    try:
        if tracked:
            from gateway.session_runtime_workers import tracked_write
            await tracked_write(authority, write)
        else:
            await asyncio.to_thread(write)
    except AuthorityRefusal as exc:  # the restore that follows refuses the receipt and logs it
        logger.warning('Local session %s compression catch-up failed: %s', sid, exc.reason)


async def adopt_local_compressions(authority):
    """Boot and hot-profile recovery: every receipt's due catch-up, before ``recover_local_sessions``."""
    from gateway.session_local_recovery import local_receipt_ids
    for sid in local_receipt_ids(authority):
        await adopt_before_restore(authority, sid, tracked=False)


def schedule_adoption(authority, sid, receipt):
    """A sync restore on the owner loop found *receipt*'s catch-up due: commit it in a tracked worker,
    then publish the session on the loop. One worker per session at a time."""
    from gateway.session_runtime_workers import track_mutation
    pending = getattr(authority, '_local_adoptions', None)
    if pending is None:
        pending = authority._local_adoptions = {}
    if sid in pending and not pending[sid].done():
        return
    logger.info('Local session %s: committing its compression catch-up off the owner loop', sid)

    async def adopt_then_publish():
        try:
            await asyncio.to_thread(_adopt, authority, sid, receipt)
            if not getattr(authority, 'retiring', False):
                from gateway.session_local_recovery import restore_local_session
                restore_local_session(authority, sid)
        except AuthorityRefusal as exc:
            logger.warning('Local session %s paused after its compression catch-up: %s', sid, exc.reason)
    pending[sid] = track_mutation(authority, adopt_then_publish())
