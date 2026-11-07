"""Worker exclusion follows the logical conversation and its physical transcript lineage."""
import json


def runtime_lineage(conn, session_id):
    from hermes_state_local import POLICY_PREFIX
    from hermes_state_local_lineage import local_physical_target
    from hermes_state_sessions import _expand_compression_lineage_ids
    from hermes_state_compression import _CHAIN_CAP
    from hermes_state_runtime import RuntimeStoreError
    targets = {session_id, local_physical_target(conn, session_id)}
    row = conn.execute('SELECT chat_id FROM sessions WHERE id=?', (session_id,)).fetchone()
    if row and row['chat_id']:
        saved = conn.execute('SELECT value FROM state_meta WHERE key=?', (POLICY_PREFIX + row['chat_id'],)).fetchone()
        if saved is not None and session_id in json.loads(saved[0]).get('lineage', [row['chat_id']]):
            targets.update({row['chat_id'], local_physical_target(conn, row['chat_id'])})
    frontier = list(targets)
    for _ in range(_CHAIN_CAP):
        frontier = [sid for sid in _expand_compression_lineage_ids(conn, frontier) if sid not in targets]
        if not frontier:
            return targets
        targets.update(frontier)
    raise RuntimeStoreError('storage_unavailable')


def worker_states(conn, session_id):
    # One indexed probe for the whole lineage (json_each: no bound-parameter limit on long chains).
    return {row[0] for row in conn.execute(
        "SELECT DISTINCT status FROM worker_executions WHERE session_id IN (SELECT value FROM json_each(?)) "
        "AND status IN ('registered','running','unknown')", (json.dumps(sorted(runtime_lineage(conn, session_id))),))}


def discard_orphan_workers_on_reset(conn, session_ids):
    """An explicit reset revokes unlinked unknown compute; it never replays its side effects.

    Admission-linked work still requires its own Discard receipt. A running/registered worker
    remains a hard reset refusal, and transaction rollback preserves all fences on refusal.
    """
    for sid in session_ids:
        orphans = conn.execute("SELECT execution_id FROM worker_executions WHERE session_id=? AND status='unknown' "
                               "AND execution_id NOT LIKE 'admission-worker:%'", (sid,)).fetchall()
        for (execution_id,) in orphans:
            conn.execute("UPDATE worker_executions SET status='terminal' WHERE execution_id=?", (execution_id,))
            compact_terminal_receipts(conn, execution_id)


RETIRED_RECEIPT = {'retired': True}
# A receipt answers one exact (execution, sequence, digest) re-send — a lost response. A terminal
# execution can only be re-sent its newest publications (the closing ``execution.finish`` almost
# always), so it keeps this many and no more; a live one keeps every receipt, they are the retries
# in flight. A store held 149,564 receipts for 328 terminal executions before this bound existed.
TERMINAL_RECEIPT_KEEP = 32

# A delegated child's execution id is its parent's plus this and a random suffix, so every
# descendant of an execution is one prefix probe away (gateway/session_worker_child.py).
CHILD_SEPARATOR = '/child:'


def retire_child_executions(conn, execution_id):
    """A child never outlives the execution that reserved it: every live descendant goes terminal
    with it, its read receipts shrink and its open session row is ended. A late child write is
    then refused ``stale_generation`` like any write after retirement."""
    _retire_children(conn, conn.execute("SELECT execution_id,session_id FROM worker_executions "
                                        "WHERE execution_id>=? AND execution_id<? AND status!='terminal'",
                                        descendant_range(execution_id)).fetchall())


def descendant_range(execution_id):
    """``(low, high)`` bounds every descendant id of ``execution_id`` on the primary key index
    (';' sorts right after the separator's ':'), so a no-child finish costs one index probe."""
    return execution_id + CHILD_SEPARATOR, execution_id + CHILD_SEPARATOR[:-1] + ';'


def retire_stale_children(conn, epoch):
    """Owner restart: a child dies with its worker, which dies with its owner, so it is never
    adoptable; only its root execution can be recovered as ``unknown``."""
    _retire_children(conn, conn.execute("SELECT execution_id,session_id FROM worker_executions WHERE kind='child' "
                                        "AND status!='terminal' AND owner_epoch!=?", (epoch,)).fetchall())


def _retire_children(conn, rows):
    for child, _session_id in rows:
        conn.execute("UPDATE worker_executions SET status='terminal' WHERE execution_id=?", (child,))
        _compact(conn, child)
        _end_child_row(conn, child)


def _end_child_row(conn, execution_id):
    import time
    conn.execute("UPDATE sessions SET ended_at=?,end_reason='agent_close' WHERE ended_at IS NULL "
                 "AND id=(SELECT session_id FROM worker_executions WHERE execution_id=?)", (time.time(), execution_id))


def compact_terminal_receipts(conn, execution_id):
    """Every terminal transition calls this, so it also retires the execution's delegated children.
    Once an execution is terminal its read receipts are dead weight: a ``compression.history``
    read stores the whole transcript and a context read a full session row, so keeping them leaves
    one full-history copy per managed turn and ``state.db`` grows quadratically. They shrink to a
    marker in the closing transaction (an exact late replay is refused ``stale_generation``, as
    after retirement); ``payload_digest`` stays, so a late duplicate with other content still
    conflicts. Mutation acknowledgements (counts, row annotations, assignments) stay replayable."""
    retire_child_executions(conn, execution_id)
    _compact(conn, execution_id)
    if CHILD_SEPARATOR in execution_id:
        # A child that finished itself (an early close: failed constructor, refused sibling) never
        # reaches _retire_children, which only selects live children: its row ends here.
        _end_child_row(conn, execution_id)


def _compact(conn, execution_id):
    conn.execute("UPDATE worker_receipts SET result_json=? WHERE execution_id=? AND json_valid(result_json) "
                 "AND (json_type(result_json, '$.messages') IS NOT NULL "
                 "OR json_type(result_json, '$.session') IS NOT NULL)",
                 (json.dumps(RETIRED_RECEIPT), execution_id))


def retired_receipt(result_json):
    return json.loads(result_json) == RETIRED_RECEIPT


def reclaim_worker_receipts(conn, *, older_than_days):
    """Reclaim the receipts no retry can still resolve, without touching an execution in flight.

    ``compact_terminal_receipts`` already drops a terminal execution's read *payloads*; the rows
    stayed, one per publication, forever — session retirement deletes only the executions of a
    session it is already deleting, and nothing else ages this table. A terminal execution keeps
    its newest ``TERMINAL_RECEIPT_KEEP`` receipts (the closing one still replays byte-exact); one
    whose session is inactive beyond ``older_than_days`` keeps none. ``registered``/``running``/
    ``unknown`` executions are skipped: their receipts are the retries in flight.

    ``older_than_days`` is the caller's retention window, not a second policy: the session that
    owns an execution is what prune measures, so its receipts die on the same clock.
    """
    import time
    from hermes_state_common import _sql_session_last_active_by_id
    conn.execute('DELETE FROM worker_receipts WHERE NOT EXISTS (SELECT 1 FROM worker_executions e '
                 'WHERE e.execution_id=worker_receipts.execution_id)')
    if older_than_days is not None:
        cutoff = time.time() - float(older_than_days) * 86400
        active = _sql_session_last_active_by_id('e.session_id')
        conn.execute(f"DELETE FROM worker_receipts WHERE execution_id IN (SELECT e.execution_id FROM "
                     f"worker_executions e WHERE e.status='terminal' AND COALESCE({active}, 0) < ?)", (cutoff,))
    over = conn.execute(
        "SELECT e.execution_id, MAX(r.sequence) FROM worker_executions e JOIN worker_receipts r "
        "ON r.execution_id=e.execution_id WHERE e.status='terminal' "
        "GROUP BY e.execution_id HAVING MAX(r.sequence) > ?", (TERMINAL_RECEIPT_KEEP,)).fetchall()
    for execution_id, last in over:
        conn.execute('DELETE FROM worker_receipts WHERE execution_id=? AND sequence<=?',
                     (execution_id, last - TERMINAL_RECEIPT_KEEP))
