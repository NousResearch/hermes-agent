"""Webhook producer payload minimization, owned by the existing retirement pass.

Admission-backed copies share the retired result's horizon. A standalone preparation
has no result to retire, so its payload gets the same horizon from creation instead.
Identity/digests/outcomes survive; compacted records can never authorize another send.
"""
import hashlib
import json


FINAL_PREFIX = 'webhook.final.intent.v1.'
ORIGIN_PREFIX = 'webhook.origin.v1.'
_CLOSED = frozenset({'settled', 'suppressed', 'refused', 'failed', 'origin_unknown', 'linked_native_admission'})


def native_digest(native):
    """Keep exact duplicate comparisons without retaining the source snapshot forever."""
    proof = {key: value for key, value in native.items() if key != 'webhook_target'}
    return hashlib.sha256(json.dumps(proof, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def _records(conn):
    for prefix in (FINAL_PREFIX, ORIGIN_PREFIX):
        rows = conn.execute('SELECT key,value FROM state_meta WHERE key>=? AND key<?', (prefix, prefix + '~')).fetchall()
        for key, raw in rows:
            try:
                value = json.loads(raw)
            except (ValueError, TypeError):
                continue
            if isinstance(value, dict):
                yield key, value


def _write(conn, key, value):
    conn.execute('UPDATE state_meta SET value=? WHERE key=?',
                 (json.dumps(value, ensure_ascii=False, separators=(',', ':')), key))


def _associated(value, ids, requests):
    native = value.get('native') or {}
    return (value.get('origin_id') in ids or value.get('source_admission_id') in ids
            or native.get('webhook_root') in ids
            or (value.get('principal'), value.get('request_id')) in requests)


def retire_webhook_sources(conn, admissions):
    """Stamp associated copies before deleting the admission's source attribution."""
    if not admissions:
        return
    ids = {row['admission_id'] for row in admissions}
    requests = {(row['principal_id'], row['request_id']) for row in admissions}
    retired_at = min(row['retired_at'] for row in admissions)
    for key, value in _records(conn):
        if not value.get('payload_compacted_at') and _associated(value, ids, requests):
            value.setdefault('source_retired_at', retired_at)
            _write(conn, key, value)


def _compact(conn, key, value, now):
    native = value.pop('native', None)
    if native is not None:
        value['native_digest'] = native_digest(native)
    value.update(message='', payload_compacted_at=now)
    if value.get('state') not in _CLOSED:
        value.update(state='refused', reason='source_unavailable', observed_at=now)
    _write(conn, key, value)


def expire_webhook_payloads(conn, expired_admissions, *, now, cutoff):
    """Called in the result-expiry transaction, including copies from before stamping existed.

    Old standalone records without creation time age from this pass, just like old
    terminal admission tombstones. Active linked sources keep their result lifecycle.
    """
    ids = {row['admission_id'] for row in expired_admissions}
    requests = {(row['principal_id'], row['request_id']) for row in expired_admissions}
    for key, value in _records(conn):
        if value.get('payload_compacted_at'):
            continue
        retired_at = value.get('source_retired_at')
        if _associated(value, ids, requests) or (retired_at is not None and retired_at < cutoff):
            _compact(conn, key, value, now)
            continue
        native = value.get('native') or {}
        if key.startswith(FINAL_PREFIX) or value.get('source_admission_id') or native.get('webhook_root') or retired_at:
            continue
        # A preparing marker may have committed a real admission before linking it.
        if conn.execute('SELECT 1 FROM session_admissions WHERE principal_id=? AND request_id=? LIMIT 1',
                        (value.get('principal'), value.get('request_id'))).fetchone():
            continue
        created_at = value.get('created_at')
        if created_at is None:
            value['created_at'] = now
            _write(conn, key, value)
        elif created_at < cutoff:
            _compact(conn, key, value, now)
