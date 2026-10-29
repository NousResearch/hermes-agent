"""Relay media client — gateway↔connector media plane (Phase 2). EXPERIMENTAL.

The relay wire carries media BY REFERENCE: inbound ``media_urls`` name connector
re-hosts (``{connector}/relay/media/{id}``) and an outbound ``send_media`` op names
a ``source_url`` the connector resolves back to bytes. ``download(url)`` GETs a
re-hosted attachment to a local temp file (vision/file tools consume LOCAL paths,
like every native adapter); ``upload(path)`` POSTs local bytes to ``/relay/media``
and returns the reference for a subsequent ``send_media`` op. Both present the
per-gateway signed bearer the WS upgrade uses; stdlib ``urllib`` in a thread executor.
"""

from __future__ import annotations

import asyncio
import email.message
import email.utils
import http.client
import json
import logging
import mimetypes
import os
import tempfile
import urllib.error
import urllib.request
from pathlib import Path
from typing import Optional

from gateway.platforms.event import clean_attachment_filename
from gateway.relay.auth import make_upgrade_token

logger = logging.getLogger(__name__)

# Mirrors the connector's MEDIA_MAX_BYTES (mediaStore.ts): fail fast here
# instead of round-tripping to a connector 413.
MEDIA_MAX_BYTES = 25 * 1024 * 1024

_REQUEST_TIMEOUT_S = 30.0

# Upload retry: the delays BEFORE each extra attempt (3 attempts in all). A transient
# hop failure (an egress proxy's own 403, a reset, a connector store outage) usually
# clears within seconds, and a dropped attachment is worse than a short wait. Worst
# case per file is 3 x _REQUEST_TIMEOUT_S + 7s, well inside cron's per-attachment
# media-send timeout (300s default).
_UPLOAD_RETRY_DELAYS_S = (2.0, 5.0)
# Statuses worth another attempt. The connector itself never answers 403 on
# POST /relay/media (401 bad bearer, 413 too large, 400 client fault, 503 store
# down), so a 403 is an intermediary refusing on its own. 401/400/413/415 are final.
_UPLOAD_RETRY_STATUSES = frozenset({403, 408, 429, 500, 502, 503, 504})

# Discord's CDN (and other public hosts) 403 urllib's default UA, which
# silently killed every CDN pass-through download. Always send a descriptive UA.
_MEDIA_USER_AGENT = "HermesAgent-Relay/1.0 (+https://github.com/NousResearch/hermes-agent)"


def media_base_url(relay_dial_url: str) -> str:
    """Map the ``ws(s)://…/relay`` dial URL to the ``http(s)://…`` connector base."""
    raw = (relay_dial_url or "").strip().rstrip("/")
    if raw.startswith("ws://"):
        raw = "http://" + raw[len("ws://") :]
    elif raw.startswith("wss://"):
        raw = "https://" + raw[len("wss://") :]
    raw = raw.removesuffix("/relay")
    return raw


class RelayMediaClient:
    """Authenticated client for the connector's ``/relay/media`` routes."""

    def __init__(self, base_url: str, gateway_id: Optional[str], secret: Optional[str]) -> None:
        self._base_url = base_url.rstrip("/")
        self._gateway_id = gateway_id or ""
        self._secret = secret or ""

    @property
    def enabled(self) -> bool:
        """True when the client can authenticate (per-gateway creds present)."""
        return bool(self._base_url and self._gateway_id and self._secret)

    def _bearer(self) -> str:
        return make_upgrade_token(self._gateway_id, self._secret)

    def is_relay_media_url(self, url: str) -> bool:
        """Is ``url`` a connector re-host reference (needs our bearer to GET)?"""
        return "/relay/media/" in (url or "")

    async def upload(
        self, file_path: str, *, mime: Optional[str] = None, filename: Optional[str] = None
    ) -> Optional[str]:
        """POST local file bytes to ``/relay/media``; return the reference URL or None on any failure."""
        return (await self.upload_with_error(file_path, mime=mime, filename=filename))[0]

    async def upload_with_error(
        self, file_path: str, *, mime: Optional[str] = None, filename: Optional[str] = None
    ) -> tuple[Optional[str], str]:
        """``upload`` that also says why it failed: ``(reference, "")`` or ``(None, reason)``.

        Transient failures (transport errors, ``_UPLOAD_RETRY_STATUSES``) are retried
        after ``_UPLOAD_RETRY_DELAYS_S``; a final relay answer is not.
        """
        if not self.enabled:
            return None, "relay media client has no credentials"
        path = Path(file_path)
        try:
            data = await asyncio.get_running_loop().run_in_executor(None, path.read_bytes)
        except OSError:
            logger.warning("relay media upload: cannot read %s", file_path)
            return None, "attachment file is unreadable"
        if not data or len(data) > MEDIA_MAX_BYTES:
            logger.warning(
                "relay media upload: %s size %d outside (0, %d]", file_path, len(data), MEDIA_MAX_BYTES
            )
            return None, f"attachment size {len(data)} outside (0, {MEDIA_MAX_BYTES}]"
        content_type = (
            mime or mimetypes.guess_type(filename or path.name)[0] or "application/octet-stream"
        )
        name = (filename or path.name)[:255]
        url = f"{self._base_url}/relay/media"

        def _post() -> tuple[Optional[str], str, bool]:
            """One attempt: ``(reference, failure reason, worth retrying)``."""
            headers = {
                "User-Agent": _MEDIA_USER_AGENT,
                # Minted per attempt: never retry with a token that aged during backoff.
                "Authorization": f"Bearer {self._bearer()}",
                "Content-Type": content_type,
                "X-Media-Filename": name,
            }
            req = urllib.request.Request(url, data=data, headers=headers, method="POST")
            try:
                with urllib.request.urlopen(req, timeout=_REQUEST_TIMEOUT_S) as resp:
                    media_id = json.loads(resp.read().decode("utf-8")).get("id")
                    if media_id:
                        return f"{self._base_url}/relay/media/{media_id}", "", False
                    return None, "connector answered without a media id", False
            except urllib.error.HTTPError as exc:
                from gateway.relay.http_diagnostics import describe_http_error, read_error_body

                return None, describe_http_error(exc, read_error_body(exc)), exc.code in _UPLOAD_RETRY_STATUSES
            except (urllib.error.URLError, OSError, http.client.HTTPException) as exc:
                return None, f"transport error: {getattr(exc, 'reason', None) or exc}", True
            except ValueError as exc:
                return None, f"unreadable connector response: {exc}", False

        loop = asyncio.get_running_loop()
        attempts = len(_UPLOAD_RETRY_DELAYS_S) + 1
        attempt, reason = 0, ""
        while attempt < attempts:
            attempt += 1
            ref, reason, retryable = await loop.run_in_executor(None, _post)
            if ref:
                if attempt > 1:
                    logger.info("relay media upload of %s succeeded on attempt %d", name, attempt)
                return ref, ""
            if not retryable or attempt == attempts:
                break
            delay = _UPLOAD_RETRY_DELAYS_S[attempt - 1]
            logger.warning(
                "relay media upload of %s failed (attempt %d/%d): %s; retrying in %.0fs",
                name, attempt, attempts, reason, delay,
            )
            await asyncio.sleep(delay)
        logger.warning("relay media upload of %s failed (attempt %d/%d): %s", name, attempt, attempts, reason)
        return None, reason

    async def download(self, url: str, *, suggested_name: Optional[str] = None) -> Optional[str]:
        """GET an attachment to a local temp file; return its path or None on any failure."""
        result = await self.download_with_name(url, suggested_name=suggested_name)
        return result[0] if result else None

    async def download_with_name(
        self, url: str, *, suggested_name: Optional[str] = None
    ) -> Optional[tuple[str, str]]:
        """GET an attachment to a local temp file; return ``(path, original_name)`` or None.

        ``original_name`` is ``suggested_name`` when given, else the response's
        ``Content-Disposition`` filename, cleaned for display; ``""`` when neither names it.
        The file is always a random ``relay_media_*`` temp name (the name only picks the
        extension), so identical names never collide.
        The bearer is presented only for connector re-host URLs; public URLs
        (e.g. a Discord CDN pass-through) are fetched without it.
        """
        if not url:
            return None
        needs_auth = self.is_relay_media_url(url)
        if needs_auth and not self.enabled:
            return None
        headers = {"User-Agent": _MEDIA_USER_AGENT}
        if needs_auth:
            headers["Authorization"] = f"Bearer {self._bearer()}"

        def _get() -> Optional[tuple[str, str]]:
            req = urllib.request.Request(url, headers=headers)
            try:
                with urllib.request.urlopen(req, timeout=_REQUEST_TIMEOUT_S) as resp:
                    if int(resp.headers.get("Content-Length") or 0) > MEDIA_MAX_BYTES:
                        logger.warning("relay media download too large: %s", url)
                        return None
                    data = resp.read(MEDIA_MAX_BYTES + 1)
                    if not data or len(data) > MEDIA_MAX_BYTES:
                        return None
                    # Extension matters: vision/file tools sniff by extension.
                    # Prefer suggested/content-disposition name, then mime, then .bin.
                    name = clean_attachment_filename(suggested_name) or _content_disposition_filename(
                        resp.headers.get("Content-Disposition") or ""
                    )
                    ext = Path(name).suffix if name else ""
                    if not ext.isascii() or not ext[1:].isalnum() or len(ext) > 16:
                        ext = ""
                    if not ext:
                        mime = (resp.headers.get("Content-Type") or "").split(";")[0]
                        ext = mimetypes.guess_extension(mime) or ".bin"
                    fd, tmp_path = tempfile.mkstemp(prefix="relay_media_", suffix=ext)
                    with os.fdopen(fd, "wb") as fh:
                        fh.write(data)
                    return tmp_path, name
            except (urllib.error.URLError, ValueError, OSError) as exc:
                logger.warning("relay media download failed for %s: %s", url, exc)
                return None

        return await asyncio.get_running_loop().run_in_executor(None, _get)


def _content_disposition_filename(header: str) -> str:
    """The filename a ``Content-Disposition`` header names, cleaned for display; ``""`` if none.

    RFC 6266: an RFC 5987 ``filename*`` (``UTF-8''r%C3%A9sum%C3%A9.pdf``) wins over the plain
    ``filename``; quoted values and escapes are handled by the stdlib MIME parser."""
    if not header:
        return ""
    msg = email.message.Message()
    try:
        msg["Content-Disposition"] = header
        params = msg.get_params(header="content-disposition") or []
    except (ValueError, TypeError, IndexError):
        return ""
    plain, extended = "", ""
    for key, value in params[1:]:
        if key.lower() != "filename":
            continue
        if isinstance(value, tuple):
            extended = extended or email.utils.collapse_rfc2231_value(value)
        else:
            plain = plain or email.utils.unquote(value)
    return clean_attachment_filename(extended or plain)


__all__ = ["MEDIA_MAX_BYTES", "RelayMediaClient", "media_base_url"]
