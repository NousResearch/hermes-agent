"""Lower-owner bounded room inventory; durable revisions fence negative proofs."""
import json
from gateway.hosted_room_driver import _task_from_row
from hermes_state_runtime import RuntimeStoreError

SCAN_PREFIX = 'gateway.hosted.output_scan.v1:'
REV_PREFIX = 'gateway.hosted.task_revision.v1:'
_INSTALLED = 'gateway.hosted.task_scan.installed.v1'
BUDGET = 32


def initialize(conn):
    names = {'hosted_task_scan_' + op for op in ('insert', 'update', 'delete')}
    existing = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='trigger'")}
    if conn.execute('SELECT 1 FROM state_meta WHERE key=?', (_INSTALLED,)).fetchone():
        if not names <= existing:
            raise RuntimeStoreError('storage_unavailable')
        return
    # Startup only, before publication consumers run. Reuse the driver's schema
    # owner rather than adding a task-table copy to the Output consumer.
    from gateway.hosted_room_driver import _initialize_schema
    _initialize_schema(conn)
    for op in ('insert', 'update', 'delete'):
        row = 'OLD' if op == 'delete' else 'NEW'
        conn.execute(f"""CREATE TRIGGER hosted_task_scan_{op} AFTER {op.upper()} ON hosted_room_driver_tasks
            BEGIN INSERT INTO state_meta(key,value) VALUES('{REV_PREFIX}' || {row}.room_id,'1')
            ON CONFLICT(key) DO UPDATE SET value=CAST(value AS INTEGER)+1; END""")
    conn.execute('INSERT INTO state_meta(key,value) VALUES(?,?)', (_INSTALLED, '1'))


def revision(conn, room_id):
    row = conn.execute('SELECT value FROM state_meta WHERE key=?', (REV_PREFIX + room_id,)).fetchone()
    return int(row[0]) if row else 0


def require_initialized(conn):
    if conn.execute('SELECT value FROM state_meta WHERE key=?', (_INSTALLED,)).fetchone() is None:
        raise RuntimeStoreError('storage_unavailable')
    names = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='trigger'")}
    if not {'hosted_task_scan_' + op for op in ('insert', 'update', 'delete')} <= names:
        raise RuntimeStoreError('storage_unavailable')


def scan_state(conn, room_id):
    row = conn.execute('SELECT value FROM state_meta WHERE key=?', (SCAN_PREFIX + room_id,)).fetchone()
    return json.loads(row[0]) if row else None


def pending(conn, room_id):
    saved = scan_state(conn, room_id)
    if saved is None and conn.execute('SELECT 1 FROM state_meta WHERE key=?', (_INSTALLED,)).fetchone():
        require_initialized(conn)
        return conn.execute('SELECT 1 FROM hosted_room_driver_tasks WHERE room_id=? LIMIT 1', (room_id,)).fetchone() is not None
    if saved is not None:
        require_initialized(conn)
    return saved is not None and (saved['pending'] or saved['revision'] != revision(conn, room_id))


def page(conn, room_id):
    """One frozen indexed task-id range; generation lives in each task snapshot.

    Mutations behind the cursor invalidate the completed revision and force a new
    pass. No cursor advancement until the caller finishes the whole batch.
    """
    require_initialized(conn)
    saved = scan_state(conn, room_id)
    if saved is None or not saved['pending'] or saved['after'] == saved['highwater']:
        high = conn.execute('SELECT task_id FROM hosted_room_driver_tasks WHERE room_id=? ORDER BY task_id DESC LIMIT 1',
                            (room_id,)).fetchone()
        saved = dict(version=1, room_id=room_id, revision=revision(conn, room_id),
                     after='', highwater=high[0] if high else '', pending=True)
        conn.execute('INSERT INTO state_meta(key,value) VALUES(?,?) ON CONFLICT(key) DO UPDATE SET value=excluded.value',
                     (SCAN_PREFIX + room_id, json.dumps(saved, sort_keys=True)))
    rows = conn.execute('SELECT * FROM hosted_room_driver_tasks WHERE room_id=? AND task_id>? AND task_id<=? '
                        'ORDER BY task_id LIMIT ?', (room_id, saved['after'], saved['highwater'], BUDGET)).fetchall()
    return saved, [_task_from_row(r) for r in rows]


def finish(conn, room_id, saved, tasks):
    require_initialized(conn)
    if scan_state(conn, room_id) != saved:
        raise RuntimeStoreError('output_cleanup_pending')
    after = tasks[-1]['identity'].task_id if tasks else saved['highwater']
    more = conn.execute('SELECT 1 FROM hosted_room_driver_tasks WHERE room_id=? AND task_id>? AND task_id<=? LIMIT 1',
                        (room_id, after, saved['highwater'])).fetchone()
    done = dict(saved, after=after if more else saved['highwater'],
                pending=bool(more) or revision(conn, room_id) != saved['revision'])
    conn.execute('UPDATE state_meta SET value=? WHERE key=?',
                 (json.dumps(done, sort_keys=True), SCAN_PREFIX + room_id))
