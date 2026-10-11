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


def compact_terminal_receipts(conn, execution_id):
    """Once an execution is terminal its read receipts are dead weight: a ``compression.history``
    read stores the whole transcript and a context read a full session row, so keeping them leaves
    one full-history copy per managed turn and ``state.db`` grows quadratically. They shrink to a
    marker in the closing transaction (an exact late replay is refused ``stale_generation``, as
    after retirement); ``payload_digest`` stays, so a late duplicate with other content still
    conflicts. Mutation acknowledgements (counts, row annotations, assignments) stay replayable."""
    conn.execute("UPDATE worker_receipts SET result_json=? WHERE execution_id=? AND json_valid(result_json) "
                 "AND (json_type(result_json, '$.messages') IS NOT NULL "
                 "OR json_type(result_json, '$.session') IS NOT NULL)",
                 (json.dumps(RETIRED_RECEIPT), execution_id))


def retired_receipt(result_json):
    return json.loads(result_json) == RETIRED_RECEIPT
