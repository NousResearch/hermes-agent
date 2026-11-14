"""Host-owned worker identities for goal barriers (never model-supplied IDs or counts)."""
LIVE = frozenset({'running', 'stalling', 'finalizing'})
TERMINAL = frozenset({'completed', 'failed', 'interrupted', 'cancelled'})


def workers(session_id):
    from hermes_cli.goals import _get_session_db
    db = _get_session_db()
    if db is None:
        return []
    lineage = db.get_compression_lineage(session_id) or [session_id]
    with db._read_ctx() as conn:
        marks = ','.join('?' for _ in lineage)
        rows = conn.execute('SELECT delegation_id,dispatched_at,state FROM async_delegations '
                            f'WHERE parent_session_id IN ({marks})', tuple(lineage)).fetchall()
    return [{'id': row[0], 'dispatched_at': row[1]} for row in rows if row[2] in LIVE]


def completed(session_id, evidence):
    from hermes_cli.goals import _get_session_db
    db = _get_session_db()
    if db is None:
        return set()
    lineage = db.get_compression_lineage(session_id) or [session_id]
    found = set()
    with db._read_ctx() as conn:
        for worker in evidence:
            row = conn.execute('SELECT parent_session_id,dispatched_at,state FROM async_delegations '
                               'WHERE delegation_id=?', (worker['id'],)).fetchone()
            if row is not None and row[0] in lineage and row[1] == worker['dispatched_at'] and row[2] in TERMINAL:
                found.add(worker['id'])
    return found


def park_workers(manager, seconds, reason):
    evidence = getattr(manager, '_judge_workers', [])
    if not evidence or completed(manager.session_id, evidence):
        return False  # no work at snapshot time, or it returned during the judge
    manager._park(reason, waiting_until=_now() + seconds, waiting_workers=evidence,
                  waiting_on_delegations=len(evidence))
    return True


def completion_ids(session_id):
    """The latest canonical turn's host-verified completion descriptor, shared by CLI/gateway judges."""
    import json
    from hermes_cli.goals import _get_session_db
    from hermes_state_local import local_lineage_owner
    db = _get_session_db()
    if db is None:
        return set()
    lineage = db.get_compression_lineage(session_id) or [session_id]
    with db._read_ctx() as conn:
        owner = local_lineage_owner(conn, lineage[0])
        row = conn.execute('SELECT payload_json,status,outcome FROM session_admissions WHERE target_session_id=? '
                           "AND status IN ('started','terminal') ORDER BY seq DESC LIMIT 1", (owner,)).fetchone()
    if row is None or (row[1] == 'terminal' and row[2] != 'completed'):
        return set()
    payload = json.loads(row[0])
    descriptor = payload.get('local_automation_v1') or payload.get('native_text_v1', {}).get('automation', {})
    identities = descriptor.get('identities') or [descriptor.get('identity')]
    result = set()
    for raw in identities:
        if not isinstance(raw, str) or not raw:
            continue
        try:
            identity = json.loads(raw)
        except json.JSONDecodeError:
            continue  # generic automation identities are opaque strings, not completion evidence
        if (isinstance(identity, list) and len(identity) == 3 and identity[0] == 'async_delegation'
                and isinstance(identity[1], str) and identity[1] and identity[2] == ''):
            result.add(identity[1])
    return result


def release_completion(manager):
    state = manager.state
    if state is None or state.status != 'active' or not state.waiting_workers:
        return
    if completed(manager.session_id, state.waiting_workers) & completion_ids(manager.session_id):
        manager.stop_waiting()


def _now():
    from hermes_cli import goals
    return goals.time.time()
