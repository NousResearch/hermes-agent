"""Main's ``gateway/session_api_media.py`` bodies (as of the pinned main), the legacy side of the backend parity checks."""

import base64
import hashlib


_MIME_EXT = {'image/png': '.png', 'image/jpeg': '.jpg', 'image/gif': '.gif', 'image/webp': '.webp'}


def _data_url_bytes(url):
    """``(mime, bytes)`` for a committable ``data:image/...;base64,`` URL, else ``None``."""
    header, _, encoded = url.partition(',')
    mime = header[len('data:'):].split(';', 1)[0].strip().lower()
    if not url.lower().startswith('data:') or mime not in _MIME_EXT or ';base64' not in header.lower():
        return None
    try:
        return mime, base64.b64decode(encoded, validate=True)
    except ValueError:
        return None


_RETAINED = 'hermes-retained:'


def _compacted(content, media):
    """``content`` with every committed inline image reduced to its digest, or ``None`` when
    nothing changes. Only bytes this admission committed are compacted (same order as capture)."""
    if not isinstance(content, list) or not media:
        return None
    digests, parts, changed = iter(reference['sha256'] for reference in media), [], False
    for part in content:
        url = (part.get('image_url') or {}).get('url', '') if isinstance(part, dict) and part.get('type') == 'image_url' else ''
        decoded = _data_url_bytes(url) if url else None
        digest = next(digests, None) if decoded is not None else None
        if digest is not None and hashlib.sha256(decoded[1]).hexdigest() == digest:
            part = {**part, 'image_url': {**part['image_url'], 'url': _RETAINED + digest}}
            changed = True
        parts.append(part)
    return parts if changed else None


def compact_settled_api_payloads(db, admission_id=None):
    """Drop what a TERMINAL API admission row no longer needs: one row at settlement, or
    (``admission_id=None``) the startup sweep for rows a crash left between settlement and this pass.

    - The redundant base64 copy of committed images in ``text``: the bytes stay in content-addressed
      ``native-inputs`` (held by ``api_turn_v1.media`` as history context).
    - The caller-supplied ``api_turn_v1.history`` (every turn carries the whole conversation, inline
      images included, so a session's ledger grew quadratically). Execution read it; a terminal row
      is read only by exact-retry projections. Responses rows keep it: their terminal replay rebuilds
      the ``previous_response_id`` snapshot from it.

    ``payload_digest`` is untouched, so an exact retry still matches. Queued/started/unknown rows
    are never rewritten: they may still execute."""
    import json
    from hermes_state_runtime import _json
    # LIKE, not instr(): SQLite LIKE is ASCII case-insensitive, as ``_data_url_bytes`` is.
    candidates = ("SELECT admission_id FROM session_admissions WHERE status='terminal' AND principal_id='api' "
                  "AND ((json_type(payload_json, '$.api_turn_v1.media') IS NOT NULL "
                  "AND payload_json LIKE '%data:image/%') OR (json_type(payload_json, '$.api_turn_v1.history')='array' "
                  "AND request_id NOT LIKE 'responses:%'))")
    if admission_id is None:
        # The scan runs on a read snapshot; a write transaction opens only when there is work.
        with db._read_ctx() as conn:
            ids = [row[0] for row in conn.execute(candidates)]
    else:
        ids = [admission_id]

    def write(conn):
        compacted = 0
        for row_id in ids:
            row = conn.execute(candidates.replace('SELECT admission_id', 'SELECT payload_json, request_id')
                               + ' AND admission_id=?', (row_id,)).fetchone()
            if row is None:
                continue
            payload = json.loads(row[0])
            data = payload['api_turn_v1']
            text = _compacted(payload.get('text'), data.get('media'))
            drop = isinstance(data.get('history'), list) and not row[1].startswith('responses:')
            if text is not None or drop:
                payload = {**payload, 'text': payload.get('text') if text is None else text,
                           'api_turn_v1': {**data, 'history': None} if drop else data}
                conn.execute('UPDATE session_admissions SET payload_json=? WHERE admission_id=?',
                             (_json(payload), row_id))
                compacted += 1
        return compacted
    return db._execute_write(write) if ids else 0
