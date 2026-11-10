"""Durable membership and reply anchors for native parallel work.

The native admission transaction owns capacity. Reservations without an admission
are inert; cancellation never frees a slot while a physical worker still owns it.
"""
from hermes_state_runtime import RuntimeStoreError


_ACTIVE = """(EXISTS (SELECT 1 FROM session_admissions a
    WHERE a.target_session_id=w.session_id AND a.status IN ('queued','started','unknown'))
    OR EXISTS (SELECT 1 FROM worker_executions e
    WHERE e.session_id=w.session_id AND e.status!='terminal')
    OR EXISTS (SELECT 1 FROM parallel_work_physical p WHERE p.work_id=w.work_id))"""


def commit_work(conn, binding, session_id, message_id, *, limit):
    values = (binding.work_id, binding.scope, binding.route, binding.requester)
    old = conn.execute('SELECT * FROM parallel_work WHERE work_id=?', (binding.work_id,)).fetchone()
    active = conn.execute(f'SELECT w.work_id,w.scope FROM parallel_work w WHERE {_ACTIVE}').fetchall()
    if not any(row[0] == binding.work_id for row in active):
        if len(active) >= 8 or sum(row[1] == binding.scope for row in active) >= limit:
            raise RuntimeStoreError('queue_full', field='parallel_work', detail='parallel work capacity reached; retry after a work finishes')
    if old is not None:
        if (old['scope'], old['route'], old['requester'], old['session_id']) != (*values[1:], session_id):
            raise RuntimeStoreError('admission_conflict', field='parallel_work')
    else:
        conn.execute('INSERT INTO parallel_work(work_id,scope,route,requester,session_id) VALUES(?,?,?,?,?)',
                     (*values, session_id))
    _record_anchor(conn, binding, message_id)


def _record_anchor(conn, binding, message_id, prompt=None):
    if not message_id:
        return
    old = conn.execute('SELECT work_id FROM parallel_work_anchors WHERE scope=? AND message_id=?',
                       (binding.scope, str(message_id))).fetchone()
    if old is not None and old[0] != binding.work_id:
        raise RuntimeStoreError('admission_conflict', field='parallel_anchor')
    conn.execute('INSERT OR IGNORE INTO parallel_work_anchors(scope,message_id,work_id) VALUES(?,?,?)',
                 (binding.scope, str(message_id), binding.work_id))
    if prompt is not None:
        conn.execute('UPDATE parallel_work_anchors SET prompt_kind=?,prompt_id=?,prompt_generation=? '
                     'WHERE scope=? AND message_id=?', (*prompt, binding.scope, str(message_id)))


def record_output(db, binding, result, *, prompt=None):
    if isinstance(result, (list, tuple)):
        for item in result:
            record_output(db, binding, item, prompt=prompt)
        return
    if not getattr(result, 'success', False):
        return
    raw = getattr(result, 'raw_response', None)
    raw_ids = raw.get('message_ids', ()) if isinstance(raw, dict) else ()
    ids = dict.fromkeys((getattr(result, 'message_id', None),
                         *(getattr(result, 'continuation_message_ids', None) or ()),
                         *(raw_ids or ())))

    def write(conn):
        if conn.execute('SELECT 1 FROM parallel_work WHERE work_id=? AND scope=? AND route=?',
                        (binding.work_id, binding.scope, binding.route)).fetchone() is None:
            raise RuntimeStoreError('not_found', field='parallel_work')
        for mid in ids:
            _record_anchor(conn, binding, mid, prompt)
    db._execute_write(write)


def by_anchor(db, scope, message_id):
    with db._read_ctx() as conn:
        row = conn.execute('''SELECT w.*,a.prompt_kind,a.prompt_id,a.prompt_generation
            FROM parallel_work w JOIN parallel_work_anchors a
            ON a.work_id=w.work_id WHERE a.scope=? AND a.message_id=?''', (scope, str(message_id))).fetchone()
        return dict(row) if row is not None else None


def by_route(db, route):
    with db._read_ctx() as conn:
        row = conn.execute('SELECT * FROM parallel_work WHERE route=?', (route,)).fetchone()
        return dict(row) if row is not None else None


def active_work(db, scope):
    with db._read_ctx() as conn:
        return [dict(row) for row in conn.execute(
            f'SELECT w.* FROM parallel_work w WHERE w.scope=? AND {_ACTIVE}', (scope,)).fetchall()]


def binding_values(row):
    return {name: row[name] for name in ('work_id', 'scope', 'route', 'requester')}


def reconcile_rebooted_physical(db, machine_id, boot_id):
    """Release only local bodies proven gone by a same-machine kernel reboot.

    Process death alone is insufficient: terminal children may survive their
    owner. Unknown admissions remain unknown, never requeued or discarded here.
    Unreadable identities and foreign-machine leases retain capacity.
    """
    if not machine_id or not boot_id:
        return 0
    return db._execute_write(lambda conn: conn.execute(
        "DELETE FROM parallel_work_physical WHERE machine_id=? AND boot_id!='' AND boot_id!=?",
        (machine_id, boot_id)).rowcount)
