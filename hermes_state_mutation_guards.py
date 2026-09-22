"""Transcript mutation guards evaluated on the receipt's writer connection."""
from hermes_state_common import _ENDED_ROW_SQL, _ended_by_compression
from hermes_state_runtime import RuntimeStoreError


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


def require_idle(db, conn, session_ids):
    for sid in session_ids:
        admissions = conn.execute("SELECT status FROM session_admissions WHERE target_session_id=? AND status!='terminal'", (sid,)).fetchall()
        workers = conn.execute("SELECT status FROM worker_executions WHERE session_id=? AND status!='terminal'", (sid,)).fetchall()
        states = {row[0] for row in [*admissions, *workers]}
        if 'unknown' in states:
            raise RuntimeStoreError('unknown_execution')
        if states:
            raise RuntimeStoreError('session_busy')
        # A logical owner closed by compression is an ancestor, not a transcript
        # target; its live successor (also in session_ids) carries the lease/lock.
        if _ended_by_compression(conn.execute(_ENDED_ROW_SQL, (sid,)).fetchone()):
            continue
        db._check_transcript_write_guards(conn, sid, None,
            reject_active_turn_lease=True, reject_active_compression_lock=True)


def delete_targets(conn, session_id):
    from hermes_state_sessions import _collect_delegate_child_ids
    import json
    from hermes_state_compression import _CHAIN_STEP_SQL
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
    frontier = list(targets)
    while frontier:
        row = conn.execute(_CHAIN_STEP_SQL, (frontier.pop(),)).fetchone()
        if row is not None and row[0] not in targets:
            targets.add(row[0])
            frontier.append(row[0])
    targets.update(_collect_delegate_child_ids(conn, targets))
    return [session_id, *sorted(targets - {session_id})]
