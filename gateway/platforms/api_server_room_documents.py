"""Document admission recovery: a reserved Run is not an accepted canonical input."""
import json

from hermes_state_runtime import RuntimeStoreError


def recover_unaccepted(adapter, record, *, scope, key, fingerprint, session_id):
    """True only after retiring a dead owner's demonstrably unaccepted reservation.

    A live/unknown owner or incomplete canonical evidence is uncertainty, never
    permission to ask for new bytes or run again. Accepted/retired input wins.
    """
    from gateway.session_authorities import active_authority
    from hermes_state_terminal import identity_key
    authority = active_authority(adapter.gateway_runner)
    if authority is None:
        raise RuntimeStoreError('storage_unavailable')
    authority._require_admission_open()
    run_id = record['run_id']
    with authority.db._read_ctx() as conn:
        rows = conn.execute("SELECT target_session_id FROM session_admissions WHERE principal_id='api' AND request_id=?", (run_id,)).fetchall()
        retired = conn.execute('SELECT value FROM state_meta WHERE key=?',
                               (identity_key('api', session_id, run_id),)).fetchone()
        projected = conn.execute("SELECT 1 FROM logical_attempts WHERE principal_id='api' AND request_id=?", (run_id,)).fetchone()
        if projected and not rows and not retired:
            raise RuntimeStoreError('room_document_outcome_unknown')
        if rows or retired:
            if rows and (len(rows) != 1 or rows[0][0] != session_id):
                raise RuntimeStoreError('admission_conflict')
            return False
    if record.get('status', {}).get('status') not in {'queued', 'interrupted'}:
        raise RuntimeStoreError('room_document_outcome_unknown')
    pid, started = record.get('owner_pid'), record.get('owner_started')
    if type(pid) is not int or pid <= 0 or type(started) is not int or started <= 0:
        raise RuntimeStoreError('room_document_outcome_unknown')
    from gateway.status import _pid_exists, get_process_start_time, start_time_fingerprints_match
    try:
        exists = _pid_exists(pid)
        current = get_process_start_time(pid) if exists else None
        dead = not exists or (current and not start_time_fingerprints_match(started, current))
    except Exception:
        dead = False
    if not dead:
        raise RuntimeStoreError('room_document_preparing')
    if not adapter._run_idempotency_store.forget_unaccepted(scope, key, fingerprint, record):
        raise RuntimeStoreError('room_document_outcome_unknown')
    return True


def accepted_document_run(adapter, *, run_id, session_id, dispatch, scope):
    """A missing/expired HTTP receipt cannot turn accepted or retired input into a new run."""
    from gateway.session_authorities import active_authority
    from hermes_state_terminal import identity_key
    authority = active_authority(adapter.gateway_runner)
    if authority is None:
        raise RuntimeStoreError('storage_unavailable')
    authority._require_admission_open()
    with authority.db._read_ctx() as conn:
        rows = conn.execute("SELECT * FROM session_admissions WHERE principal_id='api' AND request_id=?", (run_id,)).fetchall()
        retired = conn.execute('SELECT value FROM state_meta WHERE key=?',
                               (identity_key('api', session_id, run_id),)).fetchone()
        projected = conn.execute("SELECT 1 FROM logical_attempts WHERE principal_id='api' AND request_id=?", (run_id,)).fetchone()
        if not rows:
            if retired or projected:
                raise RuntimeStoreError('room_document_outcome_unknown')
            return None
        if len(rows) != 1 or rows[0]['target_session_id'] != session_id:
            raise RuntimeStoreError('admission_conflict')
        row = rows[0]
        try:
            data = json.loads(row['payload_json'])['api_turn_v1']
            if data['settings']['room_dispatch'] != dispatch or data['run_owner_scope'] != scope:
                raise RuntimeStoreError('admission_conflict')
        except (ValueError, KeyError, TypeError) as exc:
            if isinstance(exc, RuntimeStoreError):
                raise
            raise RuntimeStoreError('room_document_outcome_unknown') from exc
        return ({'started': 'running', 'unknown': 'interrupted'}.get(row['status'], row['status'])
                if row['status'] != 'terminal' else row['outcome'])
