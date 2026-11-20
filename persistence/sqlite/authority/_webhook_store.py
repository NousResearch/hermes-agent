"""Webhook ledger transaction bodies; only the backend handles SQL and writers."""
from __future__ import annotations

import json
from pathlib import Path
import time

from persistence.authority.admissions import raise_gate
from persistence.authority.webhooks import StoreIncarnation, WebhookLedger, WebhookRecord
from persistence.core.errors import AuthorityRefusal
from ._runtime import _epoch
from ..conversation._store.terminal import ADMISSION_PREFIX
from ..gateway._routes import RETRY_HORIZON_S
from persistence.authority.webhook_policy import native_digest

CLOSED = frozenset({'settled', 'suppressed', 'refused', 'failed'})
ORIGIN_CLOSED = CLOSED | {'origin_unknown', 'linked_native_admission'}


def record(ledger, origin_id, raw):
    value = json.loads(raw)
    return WebhookRecord(ledger, origin_id, value.get('_revision', 0), value)


def source_retired(conn, ledger, value, native):
    ids = ([value['origin_id']] if ledger == WebhookLedger.FINAL else [])
    ids += [x for x in (value.get('source_admission_id'), native.get('webhook_root')) if x]
    for admission_id in dict.fromkeys(ids):
        if not conn.execute('SELECT 1 FROM session_admissions a JOIN sessions s ON s.id=a.target_session_id '
                            'WHERE a.admission_id=?', (admission_id,)).fetchone():
            row = conn.execute('SELECT value FROM state_meta WHERE key=?', (ADMISSION_PREFIX + admission_id,)).fetchone()
            return (json.loads(row[0]).get('retired_at') or 0) if row else 0
    return None


def _same_identity(prior, value):
    immutable = ('origin_id', 'producer_home', 'target', 'target_home', 'route', 'route_digest',
                 'principal', 'request_id', 'fingerprint')
    if any(prior.get(k) != value.get(k) for k in immutable):
        raise AuthorityRefusal('admission_conflict')
    old, new = prior.get('target_pin'), value.get('target_pin')
    # A first recipient store may be frozen exactly once, never rebound to another home/generation.
    if old != new and (not old or not new or old.get('store') is not None
                       or any(old.get(k) != new.get(k) for k in ('home', 'directory'))):
        raise AuthorityRefusal('admission_conflict')
    if prior.get('state') != 'preparing' and prior.get('delivery_id') != value.get('delivery_id'):
        raise AuthorityRefusal('admission_conflict')
    if prior.get('state') not in {'preparing', 'linked_native_admission'}:
        if any(prior.get(k) != value.get(k) for k in ('message', 'message_digest', 'category', 'author')):
            raise AuthorityRefusal('admission_conflict')


def publish(conn, *, epoch, change, native, claim, gate):
    _epoch(conn, epoch)
    raise_gate(gate)
    ledger = WebhookLedger(change.ledger)
    key = ledger.value + change.origin_id
    raw = conn.execute('SELECT value FROM state_meta WHERE key=?', (key,)).fetchone()
    value = dict(change.value)
    if value.get('origin_id') != change.origin_id:
        raise AuthorityRefusal('invalid_params')
    if raw:
        prior = json.loads(raw[0])
        if prior.get('payload_compacted_at'):
            raise AuthorityRefusal('permission_denied')
        if prior.get('_revision', 0) != change.revision:
            raise AuthorityRefusal('stale_generation')
        _same_identity(prior, value)
        for field in ('source_retired_at', 'created_at'):
            if field in prior:
                value[field] = prior[field]
        if prior.get('state') in CLOSED and prior.get('state') != value.get('state'):
            raise AuthorityRefusal('stale_generation')
    elif change.revision:
        raise AuthorityRefusal('stale_generation')
    if claim or not raw or prior.get('state') == 'preparing':
        retired = value.get('source_retired_at')
        if retired is None:
            retired = source_retired(conn, ledger, value, native or value.get('native') or {})
        if retired is not None:
            now = time.time()
            value.update(source_retired_at=retired, state='refused', reason='source_unauthorized',
                         uncertain=False, observed_at=now)
            if not raw and (not retired or retired < now - RETRY_HORIZON_S):
                captured = value.pop('native', None)
                if captured is not None:
                    value['native_digest'] = native_digest(captured)
                value.update(message='', payload_compacted_at=now)
    value['_revision'] = change.revision + 1
    conn.execute('INSERT INTO state_meta(key,value) VALUES(?,?) ON CONFLICT(key) DO UPDATE SET value=excluded.value',
                 (key, json.dumps(value, ensure_ascii=False, separators=(',', ':'))))
    return WebhookRecord(ledger, change.origin_id, value['_revision'], value)


def file_identity(path):
    stat = Path(path).stat()
    if not stat.st_ino:
        raise ValueError('store continuity unavailable')
    return (stat.st_dev, stat.st_ino)


def capture(holder, conn):
    home = Path(holder.db_path).resolve().parent
    directory, identity = file_identity(home), file_identity(holder.db_path)
    row = conn.execute("SELECT value FROM state_meta WHERE key='store_instance_id'").fetchone()
    if (identity != file_identity(holder.db_path) or directory != file_identity(home)
            or tuple(holder._db_file_identity or ()) != identity):
        raise AuthorityRefusal('admission_conflict')
    return StoreIncarnation(str(home), directory, identity, row[0] if row else None)


def require_incarnation(holder, conn, expected):
    try:
        if expected.file is None or capture(holder, conn) != expected:
            raise AuthorityRefusal('admission_conflict')
    except (OSError, ValueError, TypeError):
        raise AuthorityRefusal('admission_conflict') from None
