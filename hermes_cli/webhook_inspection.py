"""Payload-free, read-only projection of the current profile's webhook ledgers.

The ledger contains private messages and provenance. Project only fixed fields and
allowlisted diagnostics, including when an old or malformed row is encountered.
"""
import re
import sqlite3

from hermes_constants import get_hermes_home, profile_name_for_home

_PREFIXES = ('webhook.final.intent.v1.', 'webhook.origin.v1.')
_PAGE_SIZE = 50
_ID = re.compile(r'(?:[a-f0-9]{32}|[a-f0-9]{64})\Z')
_STATES = frozenset({'preparing', 'origin_unknown', 'linked_native_admission', 'pending', 'queued',
                     'claimed', 'settled', 'suppressed', 'refused', 'failed', 'ambiguous'})
_KINDS = frozenset({'preparing', 'agent_final', 'direct_template', 'hook_reply', 'completion_final',
                    'process_final', 'linked_native_admission'})
_OUTCOMES = frozenset({'unknown', 'admitted', 'completed', 'failed', 'interrupted', 'cancelled'})
_NATIVE = frozenset({'queued', 'claimed', 'settled', 'failed', 'cancelled', 'ambiguous'})
_REASONS = frozenset({'source_unknown', 'source_policy', 'empty_or_silent', 'source_result_unavailable',
    'redaction_failed', 'native_failed', 'native_refused', 'native_ambiguous', 'missing_native_evidence',
    'legacy_unpinned', 'owner_unavailable', 'handoff_unverified', 'target_or_receipt_mismatch',
    'source_unauthorized', 'source_unavailable', 'preparation_interrupted'})
_FIELDS = ('origin_id', 'origin_kind', 'delivery_id', 'source_outcome', 'target', 'state',
           'native_status', 'native_admission_id', 'source_admission_id', 'reason')


def _identifier(value):
    return value if isinstance(value, str) and _ID.fullmatch(value) else None


def _target_profile(target):
    from gateway.webhook_target import target_profile
    try:
        return target_profile(target) or profile_name_for_home(get_hermes_home()) or 'default'
    except (ValueError, TypeError):
        return None


def _projection(row):
    return dict(namespace=row['key'].rsplit('.', 1)[0], origin_id=_identifier(row['origin_id']),
        origin_kind=row['origin_kind'] if row['origin_kind'] in _KINDS else 'unknown',
        delivery_id=_identifier(row['delivery_id']),
        source_outcome=row['source_outcome'] if row['source_outcome'] in _OUTCOMES else 'unknown',
        target_profile=_target_profile(row['target']),
        state=row['state'] if row['state'] in _STATES else 'unknown',
        native_status=row['native_status'] if row['native_status'] in _NATIVE else None,
        native_admission_id=_identifier(row['native_admission_id']),
        source_admission_id=(_identifier(row['origin_id']) if row['origin_kind'] in {'agent_final', 'completion_final'}
                             else _identifier(row['source_admission_id'])),
        reason=row['reason'] if row['reason'] in _REASONS else ('unrecognized' if row['reason'] else None))


def delivery_states(*, cursor='', routes=None):
    """One bounded keyset page. A caller may further restrict its authorized routes.

    HTTP scopes/authorizes the profile before calling; CLI and MCP inherit their
    existing profile scope. No recipient store, message, or filesystem path is read.
    """
    if cursor and not any(cursor.startswith(prefix) and _identifier(cursor[len(prefix):]) for prefix in _PREFIXES):
        raise ValueError('Invalid webhook delivery cursor')
    empty = dict(deliveries=[], next_cursor=None)
    path = get_hermes_home() / 'state.db'
    if not path.is_file() or routes == ():
        return empty
    columns = ','.join(f"json_extract(value,'$.{field}') AS {field}" for field in _FIELDS)
    parts, params = [], []
    for prefix in _PREFIXES:
        parts.append('(key>=? AND key<? AND length(substr(key,?)) IN (32,64) '
                     "AND substr(key,?) NOT GLOB '*[^a-f0-9]*')")
        params.extend((prefix, prefix + '~', len(prefix) + 1, len(prefix) + 1))
    query = 'SELECT key,' + columns + ' FROM state_meta WHERE (' + ' OR '.join(parts) + ')'
    query += ' AND key>? AND json_valid(value)'
    params.append(cursor)
    if routes is not None:
        query += " AND json_extract(value,'$.route') IN (SELECT value FROM json_each(?))"
        import json
        params.append(json.dumps(routes))
    query += ' ORDER BY key LIMIT ?'
    params.append(_PAGE_SIZE + 1)
    with sqlite3.connect(path.resolve().as_uri() + '?mode=ro', uri=True, timeout=1) as conn:
        conn.row_factory = sqlite3.Row
        if not conn.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='state_meta'").fetchone():
            return empty
        rows = conn.execute(query, params).fetchall()
    return dict(deliveries=[_projection(row) for row in rows[:_PAGE_SIZE]],
                next_cursor=rows[_PAGE_SIZE - 1]['key'] if len(rows) > _PAGE_SIZE else None)
