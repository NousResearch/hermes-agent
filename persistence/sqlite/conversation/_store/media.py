"""Retired media and spillover candidates recorded inside the deletion fence."""


import hashlib
import json


PREFIX = 'gateway.media.retired.v1.'


SPILLOVER_PREFIX = 'gateway.spillover.retired.v1.'


def retire_media(conn, payload, request_id='', *, media_root):
    from session_rules.media import admission_media_references, hosted_document_references
    hosted = hosted_document_references(request_id, payload, media_root) if media_root is not None else []
    references = [*admission_media_references(payload), *hosted,
                  *payload.get('api_turn_v1', {}).get('media', [])]
    for reference in references:
        key = PREFIX + hashlib.sha256(reference['path'].encode()).hexdigest()
        conn.execute('INSERT OR IGNORE INTO state_meta(key,value) VALUES(?,?)',
                     (key, json.dumps(reference)))


def retire_spillover(conn, session_ids):
    """Record the spillover archives the doomed sessions' messages point at, inside the delete
    transaction (the pointers vanish with the rows), as durable candidates (#126351)."""
    from persistence.sqlite.conversation._store.sql_common import _id_chunks, _placeholders
    from persistence.sqlite.conversation._store.rules import SPILLOVER_SQL_FILTER, spillover_names_in_text
    for chunk in _id_chunks(list(session_ids)):
        for content, api_content in conn.execute(
                f"SELECT content, api_content FROM messages WHERE session_id IN ({_placeholders(chunk)})"
                f" AND {SPILLOVER_SQL_FILTER}", chunk).fetchall():
            conn.executemany('INSERT OR IGNORE INTO state_meta(key,value) VALUES(?,?)',
                             [(SPILLOVER_PREFIX + n, json.dumps(n)) for n in spillover_names_in_text(content, api_content)])
