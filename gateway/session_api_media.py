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
    if mime == 'image/jpg':
        mime = 'image/jpeg'
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


def _api_images(content, *, include_aliases=True):
    from gateway.session_ingress_media import sniff_image_mime
    for url in _image_urls(content):
        if not include_aliases and _is_jpeg_alias(url):
            continue
        decoded = _data_url_bytes(url)
        if decoded is not None:
            yield sniff_image_mime(decoded[1]), decoded[1]


def _api_image_reference(mime, data):
    """The same immutable name for fingerprint projection and actual capture."""
    from gateway.session_ingress_media import native_inputs_root
    digest = hashlib.sha256(data).hexdigest()
    name = 'api_' + digest[:32] + _MIME_EXT[mime]
    return {'path': str(native_inputs_root() / digest / name), 'sha256': digest, 'size': len(data)}


def commit_api_images(content, *, include_aliases=True):
    """Stage every inline ``data:`` image deterministically (same bytes -> same path, so an
    exact retry keeps its admission digest) and commit the bytes as immutable media."""
    from gateway.platforms.base import get_image_cache_dir
    from gateway.platforms.base import get_inbound_media_max_bytes
    from gateway.session_ingress_media import capture_native_media
    from hermes_state_runtime import RuntimeStoreError
    # The bytes decide the committed type: SDKs often label JPEG bytes ``data:image/png`` and
    # main forwarded those to the provider; only bytes that are no supported image are refused.
    images = list(_api_images(content, include_aliases=include_aliases))
    # Validate the whole batch before one byte lands on disk: a refused request must not
    # leave staged or committed bytes behind.
    limit = max(0, get_inbound_media_max_bytes())
    if (len(images) > _IMAGE_LIMIT or (limit and sum(len(data) for _, data in images) > limit)
            or any(mime is None for mime, _ in images)):
        raise RuntimeStoreError('invalid_params')
    staged = []
    for mime, data in images:
        path = Path(get_image_cache_dir()).resolve() / Path(_api_image_reference(mime, data)['path']).name
        if not path.exists():
            temporary = path.with_name(f'{path.name}.{os.getpid()}.tmp')
            temporary.write_bytes(data)
            os.replace(temporary, path)
        staged.append(path)
    return capture_native_media(staged)


def _project_api_images(payload, *, include_aliases):
    """Fingerprint candidate without publishing bytes or applying fresh-admission caps."""
    images = list(_api_images(payload['text'], include_aliases=include_aliases))
    if any(mime is None for mime, _ in images):
        return None
    data = payload['api_turn_v1']
    media = [_api_image_reference(mime, image) for mime, image in images]
    return {**payload, 'api_turn_v1': {**data, 'media': media}}


def _is_jpeg_alias(url):
    return (url.lower().startswith('data:')
            and url.split(';', 1)[0][len('data:'):].strip().lower() == 'image/jpg')


def api_image_retry_payload(db, *, epoch, session_id, request_id, payload, client_key):
    """Use the old capture projection only for an exact existing live or retired identity.

    The normal admission path rechecks this fingerprint and epoch. This projection must never
    create work; the caller fences new writes and captures only the historical subset on a match.
    """
    import json
    from gateway.session_api_turn import _one_target
    from hermes_state_runtime import ADMITTED_REQUEST_CONFLICT, RuntimeStoreError, _epoch, _text, admission_fingerprint
    from hermes_state_terminal import identity_key, terminal_admission
    content = payload['text']
    if not isinstance(content, list) or not any(_is_jpeg_alias(url) for url in _image_urls(content)):
        return payload
    _text(session_id)
    _text(request_id)
    with db._read_ctx() as conn:
        _epoch(conn, epoch)
        _one_target(session_id, request_id, conn, client_key=client_key)
        row = conn.execute("SELECT payload_digest FROM session_admissions WHERE principal_id='api' "
                           'AND target_session_id=? AND request_id=?', (session_id, request_id)).fetchone()
        if row is None:
            saved = conn.execute('SELECT value FROM state_meta WHERE key=?',
                                 (identity_key('api', session_id, request_id),)).fetchone()
            row = terminal_admission(conn, json.loads(saved[0])) if saved else None
    if row is None:
        return payload
    for include_aliases in (False, True):
        candidate = _project_api_images(payload, include_aliases=include_aliases)
        if candidate is not None and row['payload_digest'] == admission_fingerprint(
                canonical_target=session_id, payload={'input': candidate, 'intent': 'queue'}):
            # Modern retries still take normal capture; only an exact historical digest may
            # omit newly supported aliases. Changed requests conflict before capture can refuse.
            return payload if include_aliases else candidate
    raise RuntimeStoreError('admission_conflict', **ADMITTED_REQUEST_CONFLICT)


def restore_api_images(content, media):
    """The transient content keeps its pixels; its text gains the validated committed reference
    per inline image (remote URLs keep their address) so the persisted user row records what
    was seen, exactly like native image turns."""
    from gateway.session_ingress_media import restore_native_media
    # Legacy rows skipped aliases: positional consumption would steal the next image's path.
    paths = {reference['sha256']: path for reference, path in zip(media, restore_native_media(media), strict=True)}
    hints = []
    for url in _image_urls(content):
        decoded = _data_url_bytes(url)
        path = paths.get(hashlib.sha256(decoded[1]).hexdigest()) if decoded is not None else None
        if path is not None:
            hints.append(f'[Image attached at: {path}]')
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
    nothing changes. Only bytes this admission committed are compacted, including legacy rows
    whose unretained aliases have no corresponding media entry."""
    if not isinstance(content, list) or not media:
        return None
    digests, parts, changed = {reference['sha256'] for reference in media}, [], False
    for part in content:
        url = (part.get('image_url') or {}).get('url', '') if isinstance(part, dict) and part.get('type') == 'image_url' else ''
        decoded = _data_url_bytes(url) if url else None
        digest = hashlib.sha256(decoded[1]).hexdigest() if decoded is not None else None
        if digest in digests:
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
