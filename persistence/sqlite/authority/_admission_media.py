"""The admission ledger's media reads and writes: the held paths, the transcript mentions, the settled media lists
and the durable retirement candidates. File checks and unlinks stay with the owner, after these return.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from hermes_state_keys import prefix_range
from session_rules.media import hosted_document_references

__all__ = [
    "drop_retired_candidates",
    "history_mentions",
    "record_retired_media",
    "retired_candidates",
    "settled_media",
    "unheld_media",
]


# The paths still held: each live admission's media and hosted documents, and every API turn's images.
def _held_media_paths(conn, media_root):
    # Project only references, not potentially large inline-image/history payloads; a live hosted
    # row's prompt is projected too, because its documents are held by text, not by a reference.
    rows = conn.execute("""SELECT status, request_id, json_extract(payload_json,
            '$.attachments_v1.media', '$.native_text_v1.media', '$.api_turn_v1.media'),
            CASE WHEN status!='terminal' AND request_id LIKE 'hosted:%'
                 THEN json_extract(payload_json, '$.text') END
            FROM session_admissions WHERE status!='terminal'
            OR json_type(payload_json, '$.api_turn_v1.media') IS NOT NULL""").fetchall()
    held = set()
    for status, request_id, encoded, text in rows:
        attachments, native, api = json.loads(encoded)
        # API images remain canonical history context after the turn completes.
        references = list(api or ())
        if status != 'terminal':
            references.extend(attachments or ())
            references.extend(native or ())
            references.extend(hosted_document_references(request_id, {'text': text}, media_root))
        held.update(reference['path'] for reference in references)
    return held


def _history_mentions(conn, references, after_id=0):
    """Candidate paths any transcript row (id > after_id) mentions raw or JSON-escaped, in ONE
    messages pass. Rows are prefiltered in SQL on the literal text before each candidate's digest
    directory, so only rows that could name a candidate reach Python's exact substring test."""
    # Matched by the content-addressed tail ``<sha256>/<name>`` with either separator, so every
    # spelling a note can carry (the host path, a sandbox backend's ``to_agent_visible_cache_path``)
    # counts as a mention.
    needles = {}
    for reference in references:
        raw = reference['path']
        for tail in (reference['sha256'] + sep + Path(raw).name for sep in ('/', '\\')):
            needles[tail] = needles[json.dumps(tail)[1:-1]] = raw
    prefixes = sorted({reference['sha256'] for reference in references})
    # The digests are ONE bound JSON array, never one ``instr`` term each: the statement's size
    # must not grow with the candidate count (SQLite's expression-depth limit fails at ~1000 terms).
    query = ('WITH digests(value) AS MATERIALIZED (SELECT value FROM json_each(?)) SELECT content FROM messages '
             'WHERE EXISTS (SELECT 1 FROM digests WHERE instr(content, digests.value))' + (' AND id>?' if after_id else ''))
    mentioned = set()
    for (content,) in conn.execute(query, (json.dumps(prefixes), *((after_id,) if after_id else ()))):
        text = content.decode('utf-8', 'replace') if isinstance(content, bytes) else str(content)
        mentioned.update(raw for needle, raw in needles.items() if needle in text)
    return mentioned


def history_mentions(holder, references):
    """``(watermark, mentioned)``: the highest ``messages.id`` and the mentioned paths, from one read snapshot."""
    with holder._read_ctx() as conn:
        conn.execute('BEGIN')  # one snapshot: the id watermark and the scan agree
        try:
            last_id = conn.execute('SELECT coalesce(max(id), 0) FROM messages').fetchone()[0]
            seen = _history_mentions(conn, references)
        finally:
            if conn.in_transaction:
                conn.execute('ROLLBACK')
    return last_id, seen


def unheld_media(holder, references, *, after_id, retain_history, media_root):
    """``(candidates, held)`` under the writer: a new admission cannot commit while the held set is computed."""
    def collect(conn):
        held = _held_media_paths(conn, media_root)
        if retain_history:
            held |= _history_mentions(conn, references, after_id=after_id)
        return [reference for reference in references if reference['path'] not in held], held
    return holder._execute_write(collect)


# Each settled admission's attachment and native media, one list per row.
def settled_media(holder):
    with holder._read_ctx() as conn:
        rows = conn.execute("""SELECT json_extract(payload_json, '$.attachments_v1.media', '$.native_text_v1.media')
            FROM session_admissions WHERE status='terminal' AND (json_type(payload_json, '$.attachments_v1.media')
            IS NOT NULL OR json_type(payload_json, '$.native_text_v1.media') IS NOT NULL)""").fetchall()
    media = []
    for (encoded,) in rows:
        attachments, native = json.loads(encoded)
        media.append((attachments or []) + (native or []))
    return media


# Records the references as durable retirement candidates, read back by ``retired_candidates``.
def record_retired_media(holder, references):
    from ..conversation._store.media import retire_media
    # No request id names a hosted prompt, so the media root is never read.
    holder._execute_write(lambda conn: retire_media(conn, {'attachments_v1': {'media': list(references)}}, media_root=None))


def retired_candidates(holder, kind):
    from ..conversation._store.media import PREFIX, SPILLOVER_PREFIX
    with holder._read_ctx() as conn:
        # Primary-key range, not LIKE: the common no-candidate case costs one index probe.
        rows = conn.execute('SELECT key,value FROM state_meta WHERE key>=? AND key<?',
                            prefix_range(PREFIX if kind == 'media' else SPILLOVER_PREFIX)).fetchall()
    return [(key, json.loads(raw)) for key, raw in rows]


def drop_retired_candidates(holder, keys):
    done = [(key,) for key in keys]
    holder._execute_write(lambda conn: conn.executemany('DELETE FROM state_meta WHERE key=?', done))
