"""Worker exclusion follows the logical conversation and its physical transcript lineage."""

import json
import os

from persistence.core.errors import AuthorityRefusal
from persistence.workers.executions import DetachedWorkerReservation
from persistence.workers.vocabulary import CHILD_SEPARATOR

from ..authority._runtime import _epoch


DETACHED_RECORD_PREFIX = 'gateway.detached_worker.v1.'


def declare_detached_worker(conn, scope, admission_id, socket_path):
    """Write only the reservation identity and its listener, never the worker secret."""
    record = {key: scope[key] for key in ('pid', 'birth', 'execution_id', 'generation', 'session_id')}
    record.update(admission_id=admission_id, socket=socket_path)
    conn.execute('INSERT INTO state_meta(key,value) VALUES(?,?)',
                 (DETACHED_RECORD_PREFIX + scope['execution_id'], json.dumps(record)))


def _detached_link(conn, key, record):
    if not isinstance(record, dict) or set(record) != {
        'admission_id', 'execution_id', 'session_id', 'generation', 'pid', 'birth', 'socket',
    }:
        return None
    if type(record['pid']) is not int or record['pid'] <= 0 or type(record['birth']) not in (int, float):
        return None
    if type(record['generation']) is not int or not all(
        isinstance(record[field], str) and record[field] for field in ('session_id', 'socket', 'admission_id')
    ):
        return None
    execution_id = 'admission-worker:' + record['admission_id']
    if record['execution_id'] != execution_id or key != DETACHED_RECORD_PREFIX + execution_id:
        return None
    # Compression can move the worker to a physical successor without changing the admission.
    return conn.execute(
        "SELECT a.owner_epoch FROM session_admissions a JOIN worker_executions w "
        "ON w.execution_id=? AND w.generation=a.generation AND w.owner_epoch=a.owner_epoch "
        "JOIN sessions s ON s.id=w.session_id AND s.runtime_generation=w.generation "
        "JOIN sessions logical ON logical.id=a.target_session_id AND logical.runtime_generation=a.generation "
        "WHERE a.admission_id=? AND a.generation=? AND a.status='started' "
        "AND w.kind='compute' AND w.status IN ('registered','running') "
        "AND w.session_id=json_extract(a.lineage_json,'$[#-1]') "
        "AND EXISTS (SELECT 1 FROM json_each(a.lineage_json) WHERE value=?)",
        (execution_id, record['admission_id'], record['generation'], record['session_id']),
    ).fetchone()


def _detached_worker_alive(pid, birth):
    if os.name != 'posix' or pid == os.getpid():
        return False
    import psutil
    from hermes_cli.process_identity import worker_birth

    # One fresh observation for an unreadable identity, never for a known mismatch or exit.
    # Do not wait under the recovery write lock or transfer on an inconclusive observation.
    for _ in range(2):
        try:
            proc = psutil.Process(pid)
            return (proc.uids().real == psutil.Process().uids().real and worker_birth(proc) == birth
                    and proc.status() != psutil.STATUS_ZOMBIE and proc.is_running())
        except psutil.NoSuchProcess:
            return False
        except (psutil.Error, OSError):
            continue
    return False


def _detached_records(conn):
    records = conn.execute('SELECT key,value FROM state_meta WHERE key>=? AND key<?',
                           (DETACHED_RECORD_PREFIX, DETACHED_RECORD_PREFIX[:-1] + '/')).fetchall()
    for key, raw in records:
        try:
            record = json.loads(raw)
        except (ValueError, TypeError):
            record = None
        link = _detached_link(conn, key, record)
        yield key, record, link


def read_detached_reservations(conn, epoch) -> tuple[DetachedWorkerReservation, ...]:
    """Return only this owner's linked active declarations, without acquiring a worker."""
    _epoch(conn, epoch)
    return tuple(record for _, record, link in _detached_records(conn)
                 if link is not None and link[0] == epoch)


def transfer_detached_workers(conn, epoch):
    """After the owner fence rotates, carry the original pair inside the recovery write."""
    for key, record, link in _detached_records(conn):
        if link is not None and link[0] == epoch:
            continue  # Repeated recovery must not discard a still-current reservation.
        if (link is None or not 0 < link[0] < epoch
                or not _detached_worker_alive(record['pid'], record['birth'])):
            conn.execute('DELETE FROM state_meta WHERE key=?', (key,))
            continue
        moved_worker = conn.execute('UPDATE worker_executions SET owner_epoch=? WHERE execution_id=?',
                                    (epoch, record['execution_id'])).rowcount
        moved_admission = conn.execute('UPDATE session_admissions SET owner_epoch=? WHERE admission_id=?',
                                       (epoch, record['admission_id'])).rowcount
        if moved_worker != 1 or moved_admission != 1:
            raise AuthorityRefusal('stale_generation')  # Roll back the entire recovery write.
        # Child execution IDs encode their reserving ancestry; only live descendants from
        # this same owner survive with the proven root. Other stale children are retired below.
        conn.execute("UPDATE worker_executions SET owner_epoch=? WHERE execution_id>=? AND execution_id<? "
                     "AND kind='child' AND status IN ('registered','running') AND owner_epoch=?",
                     (epoch, *descendant_range(record['execution_id']), link[0]))


def runtime_lineage(conn, session_id):
    from ..authority._local import local_physical_target
    from ..conversation._store.local import POLICY_PREFIX
    from ..conversation._store.compression import _CHAIN_CAP
    from ..conversation._store.sessions import _expand_compression_lineage_ids
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
    raise AuthorityRefusal('storage_unavailable')


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
    """Retire stale children not transferred with a proven surviving detached root."""
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
    conn.execute('DELETE FROM state_meta WHERE key=?', (DETACHED_RECORD_PREFIX + execution_id,))
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
