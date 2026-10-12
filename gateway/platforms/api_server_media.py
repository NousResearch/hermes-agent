"""``MEDIA:<path>`` image tags in API-server replies -> inline base64 data URLs.

Remote OpenAI-compatible frontends (Open WebUI, mobile apps) cannot read server paths, so image
tags in a final response are inlined before crossing the HTTP boundary. On a remote terminal
backend the image lives in the turn's sandbox and is fetched through the same fallback the
messaging lane uses (#466), keyed by the session the turn ran under (#130875).
"""
import base64
import re
from pathlib import Path
from typing import Optional

from gateway.platforms.base import MEDIA_TAG_CLEANUP_RE, _terminal_sentinel_start, validate_media_delivery_path

_MEDIA_MIME = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".gif": "image/gif",
               ".webp": "image/webp", ".bmp": "image/bmp"}
_MEDIA_IMG_EXT = set(_MEDIA_MIME)
_MEDIA_DATA_URL_MAX_BYTES = 5 * 1024 * 1024  # skip images larger than 5MB


def _api_turn_session_key(session_id: Optional[str], gateway_session_key: Optional[str]) -> str:
    """The session key ``_run_agent`` binds a turn under, and so the key a remote terminal backend
    registers the turn's sandbox by. Media resolution runs after the turn has cleared it."""
    return gateway_session_key or session_id or ""


def _resolve_media_to_data_urls(text: str, session_key: str = "") -> str:
    """Replace ``MEDIA:<path>`` tags with inline base64 data URLs (remote frontends can't read
    server paths); non-image/unreadable paths stay untouched. Security: the shared
    ``MEDIA_TAG_CLEANUP_RE`` anchor + ``validate_media_delivery_path`` denylist — a bare-token
    match would let a traversal path in the reply exfiltrate any readable image. A path this host
    cannot see is fetched from the turn's remote sandbox (``session_key``, see
    ``_api_turn_session_key``), like the messaging lane. Blocks on that fetch: call off the loop."""
    if not text or "MEDIA:" not in text:
        return text

    def _to_data_url(path_str: str) -> Optional[str]:
        # validate_media_delivery_path() strips wrapping quotes/trailing punctuation itself.
        safe_path = validate_media_delivery_path(path_str)
        if not safe_path:
            from gateway.media_fetch import fetch_remote_media
            safe_path = fetch_remote_media(path_str, session_key)
        p = Path(safe_path) if safe_path else None
        suffix = p.suffix.lower() if p else ""
        if suffix not in _MEDIA_IMG_EXT:
            return None
        try:
            if p.stat().st_size > _MEDIA_DATA_URL_MAX_BYTES:
                return None
            b64 = base64.b64encode(p.read_bytes()).decode()
        except OSError:
            return None
        return f"![image](data:{_MEDIA_MIME[suffix]};base64,{b64})"

    def _repl(m: re.Match[str]) -> str:
        return _to_data_url(m.group("path")) or m.group(0)
    try:
        # A leaked terminal <|eos|> glued to the last tag is not a path terminator (#111046):
        # scan without it, and drop it (control token, never content) only when a tag resolved.
        sentinel_start = _terminal_sentinel_start(text)
        scan = text[:sentinel_start] if sentinel_start >= 0 else text
        resolved = MEDIA_TAG_CLEANUP_RE.sub(_repl, scan)
        return text if resolved == scan else resolved
    except Exception:
        return text
