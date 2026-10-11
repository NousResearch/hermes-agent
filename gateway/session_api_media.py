"""API image parts become committed media references that outlive the transient turn."""
import base64
import hashlib
import os
from pathlib import Path

_MIME_EXT = {'image/png': '.png', 'image/jpeg': '.jpg', 'image/gif': '.gif', 'image/webp': '.webp'}
# Same per-admission count as ``prompt.submit`` attachments (``session_ingress_media._ATTACHMENT_LIMIT``).
_IMAGE_LIMIT = 10


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


def _image_urls(content):
    for part in content:
        if isinstance(part, dict) and part.get('type') == 'image_url':
            yield (part.get('image_url') or {}).get('url', '')


def commit_api_images(content):
    """Stage every inline ``data:`` image deterministically (same bytes -> same path, so an
    exact retry keeps its admission digest) and commit the bytes as immutable media."""
    from gateway.platforms.base import get_image_cache_dir
    from gateway.platforms.base import get_inbound_media_max_bytes
    from gateway.session_ingress_media import capture_native_media, sniff_image_mime
    from hermes_state_runtime import RuntimeStoreError
    images = [decoded for decoded in map(_data_url_bytes, _image_urls(content)) if decoded is not None]
    # Validate the whole batch before one byte lands on disk: a refused request must not
    # leave staged or committed bytes behind, and the declared type must be the real one.
    limit = max(0, get_inbound_media_max_bytes())
    if (len(images) > _IMAGE_LIMIT or (limit and sum(len(data) for _, data in images) > limit)
            or any(sniff_image_mime(data) != mime for mime, data in images)):
        raise RuntimeStoreError('invalid_params')
    staged = []
    for mime, data in images:
        path = Path(get_image_cache_dir()).resolve() / ('api_' + hashlib.sha256(data).hexdigest()[:32] + _MIME_EXT[mime])
        if not path.exists():
            temporary = path.with_name(f'{path.name}.{os.getpid()}.tmp')
            temporary.write_bytes(data)
            os.replace(temporary, path)
        staged.append(path)
    return capture_native_media(staged)


def restore_api_images(content, media):
    """The transient content keeps its pixels; its text gains the validated committed reference
    per inline image (remote URLs keep their address) so the persisted user row records what
    was seen, exactly like native image turns."""
    from gateway.session_ingress_media import restore_native_media
    paths = iter(restore_native_media(media))
    hints = []
    for url in _image_urls(content):
        if _data_url_bytes(url) is not None:
            hints.append(f'[Image attached at: {next(paths)}]')
        else:
            hints.append(f'[Image attached: {url}]')
    if not hints:
        return content
    parts = [dict(part) for part in content]
    text = next((part for part in parts if part.get('type') == 'text'), None)
    if text is None:
        text = {'type': 'text', 'text': 'What do you see in this image?'}
        parts.insert(0, text)
    text['text'] = text['text'] + '\n\n' + '\n'.join(hints)
    return parts


# A settled admission's inline ``data:`` image is replaced by this reference to its committed bytes.
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


def rehydrate_api_images(content, media):
    """The original request content of a compacted settled row, for terminal replay projections;
    an image whose committed bytes are gone becomes a text notice rather than a dangling URL."""
    if not isinstance(content, list):
        return content
    from gateway.session_ingress_media import restore_native_media
    from hermes_state_runtime import RuntimeStoreError
    by_digest = {reference['sha256']: reference for reference in media or ()}
    mimes = {ext: mime for mime, ext in _MIME_EXT.items()}
    parts = []
    for part in content:
        url = (part.get('image_url') or {}).get('url', '') if isinstance(part, dict) and part.get('type') == 'image_url' else ''
        reference = by_digest.get(url[len(_RETAINED):]) if url.startswith(_RETAINED) else None
        if url.startswith(_RETAINED):
            try:
                path = Path(restore_native_media([reference])[0]) if reference else None
            except RuntimeStoreError:
                path = None
            if path is None:
                part = {'type': 'text', 'text': '[Image no longer retained]'}
            else:
                data = base64.b64encode(path.read_bytes()).decode()
                part = {**part, 'image_url': {**part['image_url'],
                        'url': f'data:{mimes.get(path.suffix, "image/png")};base64,{data}'}}
        parts.append(part)
    return parts
