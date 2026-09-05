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
import json
import logging
import mimetypes
import os
import tempfile
import urllib.error
import urllib.request
from pathlib import Path
from typing import Optional

from gateway.relay.auth import make_upgrade_token

logger = logging.getLogger(__name__)

# Mirrors the connector's MEDIA_MAX_BYTES (mediaStore.ts): fail fast here
# instead of round-tripping to a connector 413.
MEDIA_MAX_BYTES = 25 * 1024 * 1024

_REQUEST_TIMEOUT_S = 30.0

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
    if raw.endswith("/relay"):
        raw = raw[: -len("/relay")]
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
        if not self.enabled:
            return None
        path = Path(file_path)
        try:
            data = path.read_bytes()
        except OSError:
            logger.warning("relay media upload: cannot read %s", file_path)
            return None
        if not data or len(data) > MEDIA_MAX_BYTES:
            logger.warning(
                "relay media upload: %s size %d outside (0, %d]", file_path, len(data), MEDIA_MAX_BYTES
            )
            return None
        content_type = (
            mime or mimetypes.guess_type(filename or path.name)[0] or "application/octet-stream"
        )
        headers = {
            "User-Agent": _MEDIA_USER_AGENT,
            "Authorization": f"Bearer {self._bearer()}",
            "Content-Type": content_type,
            "X-Media-Filename": (filename or path.name)[:255],
        }
        url = f"{self._base_url}/relay/media"

        def _post() -> Optional[str]:
            req = urllib.request.Request(url, data=data, headers=headers, method="POST")
            try:
                with urllib.request.urlopen(req, timeout=_REQUEST_TIMEOUT_S) as resp:
                    media_id = json.loads(resp.read().decode("utf-8")).get("id")
                    return f"{self._base_url}/relay/media/{media_id}" if media_id else None
            except (urllib.error.URLError, ValueError, OSError) as exc:
                logger.warning("relay media upload failed: %s", exc)
                return None

        return await asyncio.get_running_loop().run_in_executor(None, _post)

    async def download(self, url: str, *, suggested_name: Optional[str] = None) -> Optional[str]:
        """GET an attachment to a local temp file; return its path or None on any failure.

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

        def _get() -> Optional[str]:
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
                    name = suggested_name or ""
                    if not name:
                        cd = resp.headers.get("Content-Disposition") or ""
                        if "filename=" in cd:
                            name = cd.split("filename=", 1)[1].strip().strip('"')
                    ext = Path(name).suffix if name else ""
                    if not ext:
                        mime = (resp.headers.get("Content-Type") or "").split(";")[0]
                        ext = mimetypes.guess_extension(mime) or ".bin"
                    fd, tmp_path = tempfile.mkstemp(prefix="relay_media_", suffix=ext)
                    with os.fdopen(fd, "wb") as fh:
                        fh.write(data)
                    return tmp_path
            except (urllib.error.URLError, ValueError, OSError) as exc:
                logger.warning("relay media download failed for %s: %s", url, exc)
                return None

        return await asyncio.get_running_loop().run_in_executor(None, _get)


__all__ = ["RelayMediaClient", "media_base_url", "MEDIA_MAX_BYTES"]
