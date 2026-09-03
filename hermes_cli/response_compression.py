"""Selective lossless compression for HTTP responses.

Only data formats that are normally text-heavy and safe to compress are
included. HTML is intentionally excluded because the dashboard HTML carries
session material and may contain attacker-influenced values; JSON API
responses and exports are the primary bandwidth target.
"""

from __future__ import annotations

import re
from typing import Any

from starlette.datastructures import Headers
from starlette.middleware.gzip import GZipMiddleware, GZipResponder


_COMPRESSIBLE_CONTENT_TYPES = {
    "application/json",
    "application/javascript",
    "application/xml",
    "image/svg+xml",
    "text/css",
    "text/plain",
}
_MEDIA_TYPE_RE = re.compile(
    r"^[!#$%&'*+.^_`|~0-9A-Za-z-]+/[!#$%&'*+.^_`|~0-9A-Za-z-]+$"
)
_QVALUE_RE = re.compile(r"^(?:0(?:\.[0-9]{0,3})?|1(?:\.0{0,3})?)$")

# Exact responses that can contain unredacted credentials or arbitrary local
# file/log content. Query strings are not part of ASGI ``scope['path']``.
_EXCLUDED_EXACT_PATHS = {
    "/api/env/reveal",
    "/api/files/download",
    "/api/files/read",
    "/api/fs/download",
    "/api/fs/read-data-url",
    "/api/fs/read-text",
    "/api/logs",
    "/api/media",
    "/api/ops/backup/download",
}
# Route families that contain one-time tokens, secrets, raw configuration, or
# authentication/session material. Match only the root or a slash descendant.
_EXCLUDED_PATH_TREES = (
    "/api/actions",
    "/api/auth",
    "/api/config",
    "/api/mcp/oauth",
    "/api/messaging/telegram/onboarding",
    "/api/messaging/whatsapp/onboarding",
    "/api/pairing",
    "/api/providers/oauth",
    "/api/webhooks",
    "/auth",
)


def _is_compressible_content_type(content_types: list[str]) -> bool:
    if len(content_types) != 1:
        return False
    content_type = content_types[0]
    media_type = content_type.split(";", 1)[0].strip().lower()
    if not _MEDIA_TYPE_RE.fullmatch(media_type):
        return False
    if media_type in _COMPRESSIBLE_CONTENT_TYPES:
        return True
    top_level, subtype = media_type.split("/", 1)
    return top_level == "application" and subtype.endswith("+json")


def _quality(parameters: list[str]) -> float:
    quality = 1.0
    seen = False
    for parameter in parameters:
        name, separator, raw_value = parameter.partition("=")
        if name.lower() != "q" or seen or not separator:
            return 0.0
        seen = True
        if not _QVALUE_RE.fullmatch(raw_value):
            return 0.0
        quality = float(raw_value)
    return quality


def _accepts_gzip(values: list[str]) -> bool:
    """Return whether ``Accept-Encoding`` permits gzip compression."""
    gzip_qualities: list[float] = []
    wildcard_qualities: list[float] = []
    for value in values:
        for item in value.split(","):
            parts = [part.strip() for part in item.split(";")]
            encoding = parts[0].lower()
            if not encoding:
                continue
            quality = _quality(parts[1:])
            if encoding == "gzip":
                gzip_qualities.append(quality)
            elif encoding == "*":
                wildcard_qualities.append(quality)
    if gzip_qualities:
        return all(quality > 0 for quality in gzip_qualities)
    return bool(wildcard_qualities) and all(
        quality > 0 for quality in wildcard_qualities
    )


def _is_excluded_path(path: str) -> bool:
    if path in _EXCLUDED_EXACT_PATHS:
        return True
    return any(
        path == root or path.startswith(f"{root}/")
        for root in _EXCLUDED_PATH_TREES
    )


class _SelectiveGZipResponder(GZipResponder):
    """Starlette's streaming gzip responder with a content-type allowlist."""

    async def _send_with_selective_compression(self, message: dict[str, Any]) -> None:
        if message["type"] == "http.response.start":
            # Delay response.start until the first body chunk, as the parent
            # responder does, but make the exclusion decision from our safer
            # allowlist rather than Starlette's event-stream-only default.
            self.initial_message = message
            headers = Headers(raw=self.initial_message["headers"])
            self.content_encoding_set = "content-encoding" in headers
            self.content_type_is_excluded = (
                message["status"] == 206
                or "content-range" in headers
                or "etag" in headers
                or not _is_compressible_content_type(
                    headers.getlist("content-type")
                )
            )
            return

        # Starlette renamed this hook from ``send_with_gzip`` to
        # ``send_with_compression``. Resolve the parent hook dynamically so the
        # middleware keeps its allowlist on both supported API generations.
        parent = super()
        send_hook = getattr(parent, "send_with_compression", None)
        if send_hook is None:
            send_hook = parent.send_with_gzip
        await send_hook(message)

    async def send_with_compression(self, message: dict[str, Any]) -> None:
        await self._send_with_selective_compression(message)

    async def send_with_gzip(self, message: dict[str, Any]) -> None:
        await self._send_with_selective_compression(message)


class SelectiveGZipMiddleware(GZipMiddleware):
    """Compress eligible JSON/text responses while preserving streaming."""

    async def __call__(self, scope, receive, send):
        if (
            scope["type"] != "http"
            or _is_excluded_path(scope.get("path", ""))
            or not _accepts_gzip(
                Headers(scope=scope).getlist("Accept-Encoding")
            )
        ):
            await self.app(scope, receive, send)
            return

        responder = _SelectiveGZipResponder(
            self.app,
            self.minimum_size,
            compresslevel=self.compresslevel,
        )
        await responder(scope, receive, send)
