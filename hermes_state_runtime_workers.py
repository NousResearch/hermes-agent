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
    return {row[0] for sid in runtime_lineage(conn, session_id) for row in conn.execute(
        "SELECT status FROM worker_executions WHERE session_id=? AND status!='terminal'", (sid,))}


def discard_orphan_workers_on_reset(conn, session_ids):
    """An explicit reset revokes unlinked unknown compute; it never replays its side effects.

    Admission-linked work still requires its own Discard receipt. A running/registered worker
    remains a hard reset refusal, and transaction rollback preserves all fences on refusal.
    """
    for sid in session_ids:
        conn.execute("UPDATE worker_executions SET status='terminal' WHERE session_id=? AND status='unknown' "
                     "AND execution_id NOT LIKE 'admission-worker:%'", (sid,))
