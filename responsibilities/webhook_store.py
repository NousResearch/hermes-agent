"""Durable declaration routes and bounded delivery queues on the native gateway."""
from contextlib import contextmanager
import json
import logging
import secrets
import sqlite3
import time
from hermes_constants import get_hermes_home
from responsibilities.common import get_responsibilities_root
from responsibilities.packages import scan_workspace_responsibilities


@contextmanager
def database():
    path = get_hermes_home() / 'webhooks' / 'responsibilities.db'
    path.parent.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(path, timeout=10)
    db.row_factory = sqlite3.Row
    try:
        db.executescript('''
        CREATE TABLE IF NOT EXISTS routes(token TEXT PRIMARY KEY, package TEXT, name TEXT, definition TEXT, hash TEXT, retired REAL);
        CREATE UNIQUE INDEX IF NOT EXISTS active_declaration ON routes(package,name) WHERE retired IS NULL;
        CREATE TABLE IF NOT EXISTS deliveries(id INTEGER PRIMARY KEY, token TEXT, stream TEXT, dedup TEXT, body TEXT, received REAL, state TEXT DEFAULT 'pending');
        CREATE UNIQUE INDEX IF NOT EXISTS delivery_dedup ON deliveries(token,dedup);
        CREATE TABLE IF NOT EXISTS receipts(token TEXT, dedup TEXT, received REAL, PRIMARY KEY(token,dedup));
        CREATE TABLE IF NOT EXISTS ingress(token TEXT, received REAL);
        ''')
        with db:
            db.execute("BEGIN IMMEDIATE")
            yield db
    finally:
        db.close()


def reconcile():
    scan = scan_workspace_responsibilities(get_responsibilities_root())
    valid = {e.name: e.to_dict() for e in scan.entries}
    with database() as db:
        active = db.execute('SELECT * FROM routes WHERE retired IS NULL').fetchall()
        for row in active:
            name, trigger = row['package'], row['name']
            entry = valid.get(name)
            if name in scan.package_errors:
                continue
            if entry and ('webhooks' in entry['webhook_errors'] or trigger + '.yaml' in entry['webhook_errors']):
                continue
            if not entry or trigger not in entry['webhooks']:
                db.execute('UPDATE routes SET retired=? WHERE token=?', (time.time(), row['token']))
        for name, entry in valid.items():
            for trigger, declaration in entry['webhooks'].items():
                row = db.execute('SELECT * FROM routes WHERE package=? AND name=? AND retired IS NULL', (name, trigger)).fetchone()
                if row is None:
                    candidates = db.execute('SELECT * FROM routes WHERE hash=? AND retired>? ORDER BY retired DESC', (declaration['content_hash'], time.time()-3600)).fetchall()
                    row = candidates[0] if len(candidates) == 1 else None
                token = row['token'] if row else secrets.token_urlsafe(32)
                db.execute('INSERT OR REPLACE INTO routes VALUES(?,?,?,?,?,NULL)', (token,name,trigger,json.dumps(declaration),declaration['content_hash']))
        db.execute("DELETE FROM deliveries WHERE received<? AND state!='running'", (time.time()-86400,))
    return scan


def route(token):
    path = get_hermes_home() / 'webhooks' / 'responsibilities.db'
    if not path.is_file():
        return None
    db = sqlite3.connect(path.as_uri() + '?mode=ro', uri=True, timeout=10)
    db.row_factory = sqlite3.Row
    try:
        row = db.execute('SELECT * FROM routes WHERE token=? AND retired IS NULL', (token,)).fetchone()
        return dict(row) if row else None
    finally:
        db.close()


def receipt(package, name):
    from hermes_cli.config import load_config_readonly
    base = load_config_readonly().get('webhook', {}).get('public_url', '').rstrip('/')
    from gateway.config import load_gateway_config, Platform
    ingress = load_gateway_config().platforms.get(Platform.WEBHOOK)
    if not base or not ingress or not ingress.enabled:
        return {'warning': 'Webhook declaration saved. Configure webhook.public_url and enable native webhook ingress before registering it with a provider.'}
    with database() as db:
        row = db.execute('SELECT token FROM routes WHERE package=? AND name=? AND retired IS NULL', (package,name)).fetchone()
    from hermes_cli.profiles import get_active_profile_name
    profile = get_active_profile_name()
    prefix = '/p/' + profile if profile not in ('default', 'custom') else ''
    return {'webhook_url': base + prefix + '/responsibilities/' + row['token']} if row else {}


def accept(token, stream, dedup, body):
    now = time.time()
    with database() as db:
        db.execute("DELETE FROM ingress WHERE received<?", (now-60,))
        db.execute("DELETE FROM receipts WHERE received<?", (now-86400,))
        db.execute("INSERT INTO ingress VALUES (?,?)", (token,now))
        if db.execute("SELECT count(*) FROM ingress WHERE token=?", (token,)).fetchone()[0] > 30:
            return "rate_limited"
        if db.execute('SELECT 1 FROM receipts WHERE token=? AND dedup=?', (token,dedup)).fetchone():
            return 'duplicate'
        db.execute('INSERT INTO receipts VALUES (?,?,?)', (token,dedup,now))
        db.execute('INSERT INTO deliveries(token,stream,dedup,body,received) VALUES(?,?,?,?,?)', (token,stream,dedup,json.dumps(body),now))
        dropped = db.execute("DELETE FROM deliveries WHERE id IN (SELECT id FROM deliveries WHERE token=? AND stream=? AND state='pending' ORDER BY id DESC LIMIT -1 OFFSET 50)", (token,stream)).rowcount
        if dropped:
            logging.getLogger(__name__).warning('Webhook pending buffer overflow dropped %d oldest deliveries', dropped)
    return 'accepted'


def streams():
    with database() as db:
        return [tuple(row) for row in db.execute("SELECT DISTINCT token,stream FROM deliveries WHERE state='pending'")]


def claim(token, stream):
    with database() as db:
        rows = db.execute("SELECT id,body FROM deliveries WHERE token=? AND stream=? AND state='pending' ORDER BY id", (token,stream)).fetchall()
        db.executemany("UPDATE deliveries SET state='running' WHERE id=?", ((row['id'],) for row in rows))
    return [row['id'] for row in rows], [json.loads(row['body']) for row in rows]


def complete(ids, state='done'):
    with database() as db:
        db.executemany('UPDATE deliveries SET state=? WHERE id=?', ((state,item) for item in ids))
