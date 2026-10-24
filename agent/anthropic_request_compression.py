"""Gzip native Anthropic Messages bodies (Claude Code parity, 2.1.300+).

Lossless transport encoding: tokens and prompt caching are unchanged. A rejected
encoding disables gzip process-wide and resends the original body once.
Like Claude Code, custom-CA and proxied routes stay uncompressed.
"""

import gzip
import json
import logging
import os
import re
import secrets
from functools import cache, lru_cache

import httpx

NATIVE_HOST = "api.anthropic.com"
MIN_BODY_BYTES = 4096
GZIP_LEVEL = 3
_REJECTION_STATUSES = frozenset({400, 403, 415})
# Claude Code's classifier verbatim: its JS ``$`` never matches before a trailing newline, so ``\Z``.
_BAD_JSON = re.compile(r"^bad json: (?:invalid character |unexpected EOF\Z|unexpected end of JSON input\Z|EOF\Z)")
# What JavaScript's String.prototype.trimEnd removes (Python's rstrip() differs: it keeps U+FEFF
# and strips U+001C..U+001F and U+0085).
_JS_TRIM = " \t\n\v\f\r\u00a0\u1680\u2000\u2001\u2002\u2003\u2004\u2005\u2006\u2007\u2008\u2009\u200a\u2028\u2029\u202f\u205f\u3000\ufeff"
_latched_off: str | None = None
logger = logging.getLogger(__name__)


def native_gzip_allowed() -> bool:
    # Custom-CA middleboxes may reject gzip with resets or 5xx errors that the
    # classifier cannot see. Proxy routing is checked on the client at send time.
    return not any(
        os.environ.get(name)
        for name in ("SSL_CERT_FILE", "SSL_CERT_DIR", "REQUESTS_CA_BUNDLE", "CURL_CA_BUNDLE")
    )


def _padding() -> bytes:
    return b"\n" + "".join(secrets.choice(" \t") for _ in range(secrets.randbelow(257))).encode()


def _eligible(request: httpx.Request) -> bool:
    return (
        _latched_off is None
        and request.method == "POST"
        and request.url.host == NATIVE_HOST
        and request.url.port is None  # Claude Code's host check keeps a non-default port, so :8443 stays raw
        and request.url.path.startswith("/v1/messages")
        and "content-encoding" not in request.headers
    )


def _compressed(request: httpx.Request, raw: bytes) -> httpx.Request:
    body = gzip.compress(raw + _padding(), compresslevel=GZIP_LEVEL, mtime=0)
    headers = request.headers.copy()
    headers["Content-Encoding"] = "gzip"
    headers["Content-Length"] = str(len(body))
    return httpx.Request(request.method, request.url, headers=headers, content=body, extensions=request.extensions)


def _says_invalid_json(text: str) -> bool:
    """Claude Code 2.1.302's origin-400 test: a raw body is trimmed at the end only, an Anthropic
    error envelope needs ``type: "error"`` and string ``error.type``/``error.message``, and a
    ``bad_json`` error string is matched untrimmed."""
    try:
        data = json.loads(text)
    except ValueError:
        return _BAD_JSON.match(text.rstrip(_JS_TRIM)) is not None
    if not isinstance(data, dict):
        return False
    error = data.get("error")
    if data.get("type") == "error" and isinstance(error, dict) and isinstance(error.get("type"), str):
        message = error.get("message")
        if isinstance(message, str):
            return message.startswith((
                "The request body is not valid JSON",
                "Failed to parse request: could not parse request body as JSON",
            ))
    return (data.get("reason") == "bad_json" and isinstance(error, str)
            and _BAD_JSON.match(error) is not None)


def _error_fields(text: str) -> tuple[str, str | None]:
    try:
        data = json.loads(text)
    except ValueError:
        return "", None
    error = data.get("error") if isinstance(data, dict) else None
    if not isinstance(error, dict):
        return "", None
    message = error.get("message")
    details = error.get("details")
    error_code = details.get("error_code") if isinstance(details, dict) else None
    return message if isinstance(message, str) else "", error_code if isinstance(error_code, str) else None


def _rejection_kind(response: httpx.Response) -> str | None:
    if response.status_code not in _REJECTION_STATUSES:
        return None
    from_anthropic = response.headers.get("request-id", "").startswith("req_") or "cf-ray" in response.headers
    if not from_anthropic:
        return "non_anthropic_rejection"
    if response.status_code == 415:
        return "unsupported_encoding"
    if response.status_code == 400:
        text = response.content[:8192].decode("utf-8", "replace")
        message, error_code = _error_fields(text)
        if error_code == "request_body_encoding_unsupported" or message.startswith("Unsupported Content-Encoding"):
            return "unsupported_encoding"
        if _says_invalid_json(text):
            return "invalid_json_at_origin"
    return None


def _latch(kind: str, status: int) -> None:
    global _latched_off
    if _latched_off is None:
        _latched_off = kind
        logger.warning("Anthropic request gzip disabled: %s (HTTP %s)", kind, status)


class _GzipRequestSend:
    def send(self, request: httpx.Request, **kwargs) -> httpx.Response:
        # Use the client's actual route, including system proxies and NO_PROXY
        # mount semantics, rather than guessing from environment variables.
        if not _eligible(request) or self._transport_for_url(request.url) is not self._transport:
            return super().send(request, **kwargs)
        raw = request.read()
        if len(raw) < MIN_BODY_BYTES:
            return super().send(request, **kwargs)
        response = super().send(_compressed(request, raw), **kwargs)
        if response.status_code not in _REJECTION_STATUSES:
            return response
        response.read()
        kind = _rejection_kind(response)
        if kind is None:
            return response
        _latch(kind, response.status_code)
        response.close()
        return super().send(request, **kwargs)


@cache
def anthropic_gzip_client_cls(sdk) -> type | None:
    base = getattr(sdk, "DefaultHttpxClient", None)
    if not isinstance(base, type):
        return None

    class GzipHttpxClient(_GzipRequestSend, base):
        pass

    return GzipHttpxClient


@lru_cache(maxsize=1)
def gzip_httpx_client_cls() -> type:
    class GzipHttpxClient(_GzipRequestSend, httpx.Client):
        pass

    return GzipHttpxClient
