"""API image parts become committed media references that outlive the transient turn."""
import base64
import hashlib
import os
from pathlib import Path

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


def _image_urls(content):
    for part in content:
        if isinstance(part, dict) and part.get('type') == 'image_url':
            yield (part.get('image_url') or {}).get('url', '')


def commit_api_images(content):
    """Stage every inline ``data:`` image deterministically (same bytes -> same path, so an
    exact retry keeps its admission digest) and commit the bytes as immutable media."""
    from gateway.platforms.base import get_image_cache_dir
    from gateway.session_ingress_media import capture_native_media
    staged = []
    for url in _image_urls(content):
        decoded = _data_url_bytes(url)
        if decoded is None:
            continue
        mime, data = decoded
        path = Path(get_image_cache_dir()).resolve() / ('api_' + hashlib.sha256(data).hexdigest()[:32] + _MIME_EXT[mime])
        if not path.exists():
            temporary = path.with_name(path.name + '.%d.tmp' % os.getpid())
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
            hints.append('[Image attached at: %s]' % next(paths))
        else:
            hints.append('[Image attached: %s]' % url)
    if not hints:
        return content
    parts = [dict(part) for part in content]
    text = next((part for part in parts if part.get('type') == 'text'), None)
    if text is None:
        text = {'type': 'text', 'text': 'What do you see in this image?'}
        parts.insert(0, text)
    text['text'] = text['text'] + '\n\n' + '\n'.join(hints)
    return parts
