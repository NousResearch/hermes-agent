"""The canonical FIFO's hosted attempt keys. Lookup and admission serialize on the same profile writer."""
from __future__ import annotations

import json

from persistence.core.errors import AuthorityRefusal
from session_rules.hosted_tasks import parse_hosted_attempt
from session_rules.legacy_local import same_owner


def closed_attempt_prefix(session_id):
    return 'gateway.hosted.closed_attempt.v1:' + json.dumps([session_id], separators=(',', ':'))[:-1] + ','


def closed_attempt_key(session_id, task_id, generation):
    return closed_attempt_prefix(session_id) + json.dumps([task_id, generation], separators=(',', ':'))[1:]


def find_or_close_attempt(conn, *, session_id, room_id, task_id, generation, principal, owner):
    # Retirement may follow the gateway's authorization. A retired key must never be recreated.
    if not conn.execute('SELECT 1 FROM sessions WHERE id=?', (session_id,)).fetchone():
        raise AuthorityRefusal('not_found')
    rows = conn.execute(
        "SELECT admission_id, request_id, principal_id, status, outcome FROM session_admissions "
        "WHERE target_session_id=? AND instr(request_id, ?) > 0 ORDER BY seq",
        (session_id, json.dumps(task_id)))
    for row in rows:
        if row['request_id'].startswith('hosted:') and same_owner(row['principal_id'], principal, owner):
            identity, attempt = parse_hosted_attempt(row['request_id'], room_id)
            if identity.task_id == task_id and attempt == generation:
                return {name: row[name] for name in ('admission_id', 'status', 'outcome')}
    conn.execute('INSERT OR IGNORE INTO state_meta(key,value) VALUES(?,?)',
                 (closed_attempt_key(session_id, task_id, generation), 'closed'))
    return None
