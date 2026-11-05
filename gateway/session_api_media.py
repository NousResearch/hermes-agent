"""API image parts become committed media references that outlive the transient turn."""
import base64
import hashlib
import os
from pathlib import Path

from session_rules.api_payloads import MIME_EXT, RETAINED, data_url_bytes

# Same per-admission count as ``prompt.submit`` attachments (``session_ingress_media._ATTACHMENT_LIMIT``).
_IMAGE_LIMIT = 10


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
    from persistence.core.errors import AuthorityRefusal
    # The bytes decide the committed type: SDKs often label JPEG bytes ``data:image/png`` and
    # main forwarded those to the provider; only bytes that are no supported image are refused.
    images = [(sniff_image_mime(decoded[1]), decoded[1])
              for decoded in map(data_url_bytes, _image_urls(content)) if decoded is not None]
    # Validate the whole batch before one byte lands on disk: a refused request must not
    # leave staged or committed bytes behind.
    limit = max(0, get_inbound_media_max_bytes())
    if (len(images) > _IMAGE_LIMIT or (limit and sum(len(data) for _, data in images) > limit)
            or any(mime is None for mime, _ in images)):
        raise AuthorityRefusal('invalid_params')
    staged = []
    for mime, data in images:
        path = Path(get_image_cache_dir()).resolve() / ('api_' + hashlib.sha256(data).hexdigest()[:32] + MIME_EXT[mime])
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
        if data_url_bytes(url) is not None:
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


def compact_settled_api_payloads(store, admission_id=None):
    """Drop what a TERMINAL API admission row no longer needs (``session_rules.api_payloads``'s
    ``compacted_api_payload``): one row at settlement, or (``admission_id=None``) the startup sweep
    for rows a crash left between settlement and this pass. Each row is re-read and rewritten inside
    one write.

    ``payload_digest`` is untouched, so an exact retry still matches. Queued/started/unknown rows
    are never rewritten: they may still execute. *store* is a storage interface handle."""
    # The scan runs on a read snapshot; a write transaction opens only when there is work.
    ids = store.admissions.api_compaction_candidates() if admission_id is None else [admission_id]
    return store.admissions.compact_api_payloads(admission_ids=ids) if ids else 0


def rehydrate_api_images(content, media):
    """The original request content of a compacted settled row, for terminal replay projections;
    an image whose committed bytes are gone becomes a text notice rather than a dangling URL."""
    if not isinstance(content, list):
        return content
    from gateway.session_ingress_media import restore_native_media
    from persistence.core.errors import AuthorityRefusal
    by_digest = {reference['sha256']: reference for reference in media or ()}
    mimes = {ext: mime for mime, ext in MIME_EXT.items()}
    parts = []
    for part in content:
        url = (part.get('image_url') or {}).get('url', '') if isinstance(part, dict) and part.get('type') == 'image_url' else ''
        reference = by_digest.get(url[len(RETAINED):]) if url.startswith(RETAINED) else None
        if url.startswith(RETAINED):
            try:
                path = Path(restore_native_media([reference])[0]) if reference else None
            except AuthorityRefusal:
                path = None
            if path is None:
                part = {'type': 'text', 'text': '[Image no longer retained]'}
            else:
                data = base64.b64encode(path.read_bytes()).decode()
                part = {**part, 'image_url': {**part['image_url'],
                        'url': f'data:{mimes.get(path.suffix, "image/png")};base64,{data}'}}
        parts.append(part)
    return parts
