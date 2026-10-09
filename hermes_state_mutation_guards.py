"""Transcript mutation guards evaluated on the receipt's writer connection."""
from hermes_state_common import _ENDED_ROW_SQL, _ended_by_compression
from hermes_state_errors import SessionCompressionInProgressError, SessionTurnLeaseLostError
from hermes_state_runtime import RuntimeStoreError

# What ``require_idle``'s transcript-guard check raises when a live turn lease or a
# compression lock protects a target: the same retryable "busy" verdict as a running
# admission, so mutation surfaces map it to 409 rather than letting it escape as a 500.
MUTATION_GUARD_REFUSALS = (SessionTurnLeaseLostError, SessionCompressionInProgressError)


def require_not_executing(conn, session_ids):
    """Refuse while a turn is running (or its outcome is unknown) on any of ``session_ids``.
    Queued admissions are allowed: a follower waits on the logical owner and simply runs
    against whatever physical target the mutation publishes. Workers have no queued state
    (registered/running/unknown are all live), so any non-terminal worker is executing."""
    for sid in session_ids:
        admissions = conn.execute("SELECT status FROM session_admissions WHERE target_session_id=? AND status IN ('started','unknown')", (sid,)).fetchall()
        workers = conn.execute("SELECT status FROM worker_executions WHERE session_id=? AND status!='terminal'", (sid,)).fetchall()
        states = {row[0] for row in [*admissions, *workers]}
        if 'unknown' in states:
            raise RuntimeStoreError('unknown_execution')
        if states:
            raise RuntimeStoreError('session_busy')


def _require_transcript_unleased(db, conn, sid):
    # A logical owner closed by compression is an ancestor, not a transcript
    # target; its live successor (also in session_ids) carries the lease/lock.
    if _ended_by_compression(conn.execute(_ENDED_ROW_SQL, (sid,)).fetchone()):
        return
    db._check_transcript_write_guards(conn, sid, None,
        reject_active_turn_lease=True, reject_active_compression_lock=True)


def require_target_advanceable(db, conn, session_ids):
    """Fence for writes that publish a new physical target (reset, compress, model):
    refuse executing/unknown work and live transcript leases, never a queued follower,
    which waits on the logical owner and runs against whatever target is published."""
    require_not_executing(conn, session_ids)
    for sid in session_ids:
        _require_transcript_unleased(db, conn, sid)


def require_idle(db, conn, session_ids):
    for sid in session_ids:
        admissions = conn.execute("SELECT status FROM session_admissions WHERE target_session_id=? AND status!='terminal'", (sid,)).fetchall()
        workers = conn.execute("SELECT status FROM worker_executions WHERE session_id=? AND status!='terminal'", (sid,)).fetchall()
        states = {row[0] for row in [*admissions, *workers]}
        if 'unknown' in states:
            raise RuntimeStoreError('unknown_execution')
        if states:
            raise RuntimeStoreError('session_busy')
        _require_transcript_unleased(db, conn, sid)


def delete_targets(conn, session_id):
    from hermes_state_sessions import _collect_delegate_child_ids, _expand_compression_lineage_ids
    import json
    from hermes_state_compression import _CHAIN_CAP
    from hermes_state_local import POLICY_PREFIX
    from hermes_state_local_lineage import validate_local_lineage
    targets = {session_id}
    saved = conn.execute('SELECT value FROM state_meta WHERE key=?',
                         (POLICY_PREFIX + session_id,)).fetchone()
    if saved is not None:
        receipt = json.loads(saved[0])
        validate_local_lineage(conn, receipt)
        targets.update(receipt.get('lineage', [session_id]))
    # Canonical admissions bind to the compression root for every producer, not only
    # local receipts: every physical continuation of a target goes with it, or the next
    # message on the route re-admits the "deleted" conversation through the surviving child.
    # The walk also runs BACKWARD to the root (#57543): a sidebar row carries the chain tip's
    # id, and a surviving root re-projects as the "deleted" conversation on the next reload.
    # Every walked row must belong to the requested row's principal domain: only that row was
    # authorized, so a parent link into another principal's chain (an imported or forged edge)
    # stops the walk instead of deleting their conversation.
    from hermes_state_mutation_binding import same_history_owner
    frontier = list(targets)
    for _ in range(_CHAIN_CAP):
        frontier = [sid for sid in _expand_compression_lineage_ids(conn, frontier)
                    if sid not in targets and same_history_owner(conn, session_id, sid)]
        if not frontier:
            break
        targets.update(frontier)
    targets.update(_collect_delegate_child_ids(conn, targets))
    return [session_id, *sorted(targets - {session_id})]
