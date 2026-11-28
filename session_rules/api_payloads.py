"""The API turn payload rules: the bytes an inline ``data:`` image URL carries, and what a settled API admission's stored payload keeps, which the store applies inside its compaction write.

Standard library only.
"""

from __future__ import annotations

import base64
import hashlib


MIME_EXT = {'image/png': '.png', 'image/jpeg': '.jpg', 'image/gif': '.gif', 'image/webp': '.webp'}
# A settled admission's inline ``data:`` image is replaced by this reference to its committed bytes.
RETAINED = 'hermes-retained:'


def data_url_bytes(url):
    """``(mime, bytes)`` for a committable ``data:image/...;base64,`` URL, else ``None``."""
    header, _, encoded = url.partition(',')
    mime = header[len('data:'):].split(';', 1)[0].strip().lower()
    if not url.lower().startswith('data:') or mime not in MIME_EXT or ';base64' not in header.lower():
        return None
    try:
        return mime, base64.b64decode(encoded, validate=True)
    except ValueError:
        return None


def _compacted(content, media):
    """``content`` with every committed inline image reduced to its digest, or ``None`` when
    nothing changes. Only bytes this admission committed are compacted (same order as capture)."""
    if not isinstance(content, list) or not media:
        return None
    digests, parts, changed = iter(reference['sha256'] for reference in media), [], False
    for part in content:
        url = (part.get('image_url') or {}).get('url', '') if isinstance(part, dict) and part.get('type') == 'image_url' else ''
        decoded = data_url_bytes(url) if url else None
        digest = next(digests, None) if decoded is not None else None
        if digest is not None and hashlib.sha256(decoded[1]).hexdigest() == digest:
            part = {**part, 'image_url': {**part['image_url'], 'url': RETAINED + digest}}
            changed = True
        parts.append(part)
    return parts if changed else None


def compacted_api_payload(payload, request_id):
    """A TERMINAL API admission's stored payload without what it no longer needs, or ``None`` when
    nothing changes:

    - The redundant base64 copy of committed images in ``text``: the bytes stay in content-addressed
      ``native-inputs`` (held by ``api_turn_v1.media`` as history context).
    - The caller-supplied ``api_turn_v1.history`` (every turn carries the whole conversation, inline
      images included, so a session's ledger grew quadratically). Execution read it; a terminal row
      is read only by exact-retry projections. Responses rows keep it: their terminal replay rebuilds
      the ``previous_response_id`` snapshot from it."""
    data = payload['api_turn_v1']
    text = _compacted(payload.get('text'), data.get('media'))
    drop = isinstance(data.get('history'), list) and not request_id.startswith('responses:')
    if text is None and not drop:
        return None
    return {**payload, 'text': payload.get('text') if text is None else text,
            'api_turn_v1': {**data, 'history': None} if drop else data}
