"""Forward-only TUI subscription protocol in the native board database."""
import uuid
from pathlib import Path


def migrate(conn):
    columns = {r[1] for r in conn.execute('PRAGMA table_info(kanban_notify_subs)')}
    for name, ddl in (("delivery_version", "INTEGER NOT NULL DEFAULT 0"),
                      ("subscription_generation", "TEXT"), ("cutover_event_id", "INTEGER")):
        if name not in columns:
            conn.execute(f'ALTER TABLE kanban_notify_subs ADD COLUMN {name} {ddl}')
    conn.execute('CREATE TABLE IF NOT EXISTS kanban_delivery_meta (key TEXT PRIMARY KEY, value TEXT NOT NULL)')
    conn.execute("INSERT OR IGNORE INTO kanban_delivery_meta VALUES ('generation', ?)", (uuid.uuid4().hex,))


def read_pending(conn, sub):
    from hermes_cli import kanban_db as kb
    from hermes_cli.kanban_db_notify import _SUB_KEY_WHERE, _sub_key
    key = _sub_key(sub['task_id'], sub['platform'], sub['chat_id'], sub.get('thread_id'))
    with kb.write_txn(conn):
        conn.execute('UPDATE kanban_notify_subs SET delivery_version=1, subscription_generation=?, '
                     'cutover_event_id=last_event_id ' + _SUB_KEY_WHERE + ' AND delivery_version=0',
                     (uuid.uuid4().hex, *key))
        row = conn.execute('SELECT * FROM kanban_notify_subs ' + _SUB_KEY_WHERE, key).fetchone()
        if row is None:
            return None, []
        sub = dict(row)
        if sub['delivery_version'] != 1 or not sub['subscription_generation']:
            raise ValueError('Unknown TUI subscription version/generation')
        sub['board_generation'] = conn.execute("SELECT value FROM kanban_delivery_meta WHERE key='generation'").fetchone()[0]
        sub['board_path'] = str(Path(conn.execute('PRAGMA database_list').fetchone()[2]).resolve())
        events = [kb.Event.from_row(r) for r in conn.execute(
            'SELECT * FROM task_events WHERE task_id=? AND id>? ORDER BY id LIMIT 32',
            (sub['task_id'], sub['last_event_id']))]
    return sub, events


def acknowledge(conn, sub, expected, event_id):
    from hermes_cli import kanban_db as kb
    from hermes_cli.kanban_db_notify import _SUB_KEY_WHERE, _sub_key
    if event_id <= expected:
        return False
    with kb.write_txn(conn):
        generation = conn.execute("SELECT value FROM kanban_delivery_meta WHERE key='generation'").fetchone()
        if not generation or generation[0] != sub['board_generation']:
            return False
        return conn.execute('UPDATE kanban_notify_subs SET last_event_id=? ' + _SUB_KEY_WHERE +
            ' AND delivery_version=1 AND subscription_generation=? AND last_event_id=?',
            (event_id, *_sub_key(sub['task_id'], sub['platform'], sub['chat_id'], sub.get('thread_id')),
             sub['subscription_generation'], expected)).rowcount == 1


def refuse_legacy_replay(sub, event_id):
    if sub.get('cutover_event_id') is None or event_id <= sub['cutover_event_id']:
        raise ValueError('Ambiguous legacy consumed range: replay refused; cursor retained')
