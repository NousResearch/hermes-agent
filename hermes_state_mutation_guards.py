"""Transcript mutation guards evaluated on the receipt's writer connection."""
from hermes_state_common import _ENDED_ROW_SQL, _ended_by_compression
from hermes_state_runtime import RuntimeStoreError


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


def _compression_children(conn, parent_ids):
    from hermes_state_sessions import SessionSessionsMixin
    # Use the existing parent-bound fork predicate: compressed children inherit
    # their parent's older branch/delegate markers without becoming new forks.
    children = set()
    for parent in parent_ids:
        rows = conn.execute('''
            SELECT child.id FROM sessions parent
            JOIN sessions child ON child.parent_session_id=parent.id
            WHERE parent.id=? AND parent.end_reason='compression'
            ''' + SessionSessionsMixin._NON_CONTINUATION_CHILD_FILTER_SQL.format(alias='child.')
            + ' LIMIT 2', (parent, parent, parent)).fetchall()
        if len(rows) > 1:
            raise RuntimeStoreError('admission_conflict')
        children.update(row[0] for row in rows)
    return children


def delete_targets(conn, session_id):
    from hermes_state_sessions import _collect_delegate_child_ids
    import json
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
    frontier = set(targets)
    while frontier:
        found = _compression_children(conn, frontier)
        found.update(_collect_delegate_child_ids(conn, frontier))
        frontier = found - targets
        targets.update(frontier)
    return [session_id, *sorted(targets - {session_id})]
