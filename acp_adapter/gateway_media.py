"""Editor prompt content -> ``prompt.submit`` text and staged image attachments (ACP gateway bridge)."""
from __future__ import annotations

import asyncio
import logging
import os
from contextlib import suppress

from hermes_cli.gateway_client import GatewayClientError

logger = logging.getLogger(__name__)


async def _fetch_remote_image(url):
    """Bytes of an editor's remote image URL, or None when it cannot be fetched.

    The in-process server passed such a URL to the model as an image part; admission carries only
    staged bytes, so the adapter downloads it through the gateway's SSRF-guarded image fetcher
    (pre-flight and per-redirect address checks, the inbound media size cap) and stages the bytes
    like any other image (#134889 ACP-16)."""
    from gateway.platforms.base import cache_image_from_url
    try:
        path = await cache_image_from_url(url, ".img", retries=0)
    except Exception:
        # Boundary: an unsafe address, an HTTP error, a non-image body or a transport failure
        # degrades to the URL as text, never a failed prompt.
        logger.info("ACP remote image not fetched; sent as text", exc_info=True)
        return None
    return await asyncio.to_thread(_take, path)


def _take(path):
    """The fetched file's bytes; the cache copy is removed (staging writes its own sniffed-type file)."""
    try:
        with open(path, "rb") as fetched:
            return fetched.read()
    finally:
        with suppress(OSError):
            os.unlink(path)


async def stage_user_content(content):
    """Shared-converter output -> ``(text, attachments)`` for ``prompt.submit``.

    Text-only prompts stay a plain string. Media parts (image ``data:`` URLs from
    image blocks, image resource links and embedded blobs, and remote image URLs once
    fetched) are staged as bytes in the profile image cache; the authority commits them
    at admission. A remote image URL that cannot be fetched is kept as text so the model
    still sees it. Only a remote image awaits (its download); every other part stages inline."""
    if isinstance(content, str):
        return content, []
    import base64
    import binascii
    from gateway.platforms.base import cache_image_from_bytes
    from gateway.session_ingress_media import _ATTACHMENT_LIMIT, _IMAGE_EXT, sniff_image_mime
    texts, attachments = [], []
    for part in content:
        if part.get('type') == 'text':
            texts.append(part['text'])
            continue
        url = part['image_url']['url']
        if len(attachments) >= _ATTACHMENT_LIMIT:
            raise GatewayClientError('acp_content_invalid')
        if url.startswith('data:'):
            try:
                raw = base64.b64decode(url.partition(',')[2], validate=True)
            except (binascii.Error, ValueError) as exc:
                raise GatewayClientError('acp_content_invalid') from exc
        else:
            raw = await _fetch_remote_image(url) if url.startswith(('http://', 'https://')) else None
            if raw is None:
                texts.append(f"[Image attached: {url}]")
                continue
        # The sniffed bytes are authoritative for the staged extension and admitted type; the
        # client's declared type (a file suffix, or image/png by default) is never trusted. A
        # part admission cannot carry (bmp/svg, over the media cap) degrades to a text note so
        # the user's text still runs, as main's in-process server did for oversize images.
        mime = sniff_image_mime(raw)
        if mime is None:
            texts.append("[Image omitted: unsupported image format]")
            continue
        try:
            path = cache_image_from_bytes(raw, _IMAGE_EXT[mime])
        except ValueError as exc:
            texts.append(f"[Image omitted: {exc}]")
            continue
        attachments.append({'path': path, 'mime': mime})
    return "\n".join(texts), attachments


def submit_fingerprint(session_id, content):
    """Identity of one logical prompt as the editor sent it (text, image ``data:`` bytes, remote image
    URLs), taken before staging: a retry of the same prompt matches without re-fetching a remote image,
    whose second download could fail or return other bytes and so change what is submitted."""
    import hashlib
    digest = hashlib.sha256()
    for part in [content] if isinstance(content, str) else content:
        kind, value = ('text', part) if isinstance(part, str) else \
            ('text', part['text']) if part.get('type') == 'text' else ('image', part['image_url']['url'])
        digest.update(kind.encode() + len(value).to_bytes(8, 'big') + value.encode())
    return session_id, digest.hexdigest()
