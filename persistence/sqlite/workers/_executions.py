"""Worker registration and adoption."""

from persistence.core.errors import AuthorityRefusal

from ..authority._runtime import (
    _epoch, _linked_worker_admission, _secret_digest, _session, _text, _worker_assignment,
    _worker_public,
)


def register_worker_execution(holder, *, epoch: int, execution_id: str, session_id: str,
                              generation: int, kind: str, adoption_secret: str, require_idle: bool = False) -> dict:
    for value in (execution_id, session_id):
        _text(value)
    if kind not in ('cron', 'child', 'compute', 'kanban'):
        raise AuthorityRefusal('invalid_params')
    digest = _secret_digest(adoption_secret)
    def write(conn):
        _epoch(conn, epoch)
        session = _session(conn, session_id)
        if type(generation) is not int or session['runtime_generation'] != generation:
            raise AuthorityRefusal('stale_generation')
        old = conn.execute('SELECT * FROM worker_executions WHERE execution_id=?', (execution_id,)).fetchone()
        if old is not None:
            if (old['session_id'], old['generation'], old['kind'], old['owner_epoch'], old['adoption_digest']) != (session_id, generation, kind, epoch, digest):
                raise AuthorityRefusal('admission_conflict')
            return _worker_public(old)
        from ._runtime_workers import runtime_lineage, worker_states
        targets = runtime_lineage(conn, session_id)
        if require_idle and any(conn.execute("SELECT 1 FROM session_admissions WHERE target_session_id=? AND status IN ('queued','started','unknown')", (sid,)).fetchone() for sid in targets):
            raise AuthorityRefusal('stale_generation')
        if worker_states(conn, session_id):
            raise AuthorityRefusal('stale_generation')
        if any(conn.execute("SELECT 1 FROM session_admissions WHERE target_session_id=? AND status='unknown'", (sid,)).fetchone() for sid in targets):
            raise AuthorityRefusal('unknown_execution')
        conn.execute("""INSERT INTO worker_executions(execution_id,session_id,kind,owner_epoch,generation,status,adoption_digest)
            VALUES(?,?,?,?,?,'registered',?)""", (execution_id, session_id, kind, epoch, generation, digest))
        return _worker_public(_worker_assignment(conn, execution_id, session_id, generation))
    return holder._execute_write(write)


def adopt_worker_execution(holder, *, epoch: int, execution_id: str, session_id: str,
                           generation: int, adoption_secret: str) -> dict:
    """Authority first verifies the original producer claim and live worker proof."""
    import hmac
    digest = _secret_digest(adoption_secret)
    def write(conn):
        _epoch(conn, epoch)
        row = _worker_assignment(conn, execution_id, session_id, generation)
        if row['status'] == 'terminal':
            raise AuthorityRefusal('stale_generation')
        if not hmac.compare_digest(row['adoption_digest'], digest):
            raise AuthorityRefusal('permission_denied')
        # An admission-reserved worker reports to the reserving owner over that owner's pipes and
        # dies with them, so only its same-epoch handshake adopts it. A late handshake reaching a
        # restarted owner would re-arm the recovered `unknown` admission as `started` with nobody
        # left to settle it (and Discard refuses `started`); refuse it so recovery's verdict stands.
        if execution_id.startswith('admission-worker:') and row['owner_epoch'] != epoch:
            raise AuthorityRefusal('stale_epoch')
        linked = _linked_worker_admission(
            conn, session_id, generation, row['owner_epoch'], ('started', 'unknown'))
        if linked is not None:
            changed = conn.execute("""UPDATE session_admissions SET owner_epoch=?,status='started'
                WHERE admission_id=? AND generation=? AND owner_epoch=?
                AND status IN ('started','unknown')""",
                (epoch, linked['admission_id'], generation, row['owner_epoch']))
            if changed.rowcount != 1:
                raise AuthorityRefusal('stale_generation')
        changed = conn.execute(
            "UPDATE worker_executions SET owner_epoch=?,status='running' WHERE execution_id=?",
            (epoch, execution_id))
        if changed.rowcount != 1:
            raise AuthorityRefusal('stale_generation')
        return _worker_public(_worker_assignment(conn, execution_id, session_id, generation))
    return holder._execute_write(write)
