"""Host-side fetcher for provider-sanctioned quota counters.

Currently the only implemented source is OpenRouter's documented key-info
endpoint (``official_key_limits``). Everything else is reported as unavailable:
no endpoint is guessed, no quota is invented, and failure is never silent.

Hard boundaries (non-negotiable):

- runs with the host-held credential of the *pool entry bound to the requested
  opaque scope* — never a different account, never a credential the plugin
  supplied;
- 2 second wall-clock bound including failure, no retries;
- response body capped (64 KiB) and only the documented counter object is read;
- redirects off the documented host are not followed;
- the credential is never returned, logged, or placed in an exception message;
- the result carries counters only — remaining + used == limit, or unavailable.
"""

from __future__ import annotations

import json
import logging
import math
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from typing import Any, Optional

logger = logging.getLogger(__name__)

OPENROUTER_KEY_URL = "https://openrouter.ai/api/v1/key"
OFFICIAL_KEY_LIMITS = "official_key_limits"
DEFAULT_TIMEOUT = 2.0
MAX_TIMEOUT = 2.0
MAX_BODY_BYTES = 64 * 1024
OBSERVATION_TTL = 120.0  # matches the consumer's accepted observation freshness

_OPENROUTER_HOSTS = {"openrouter.ai", "www.openrouter.ai"}


def _unavailable(reason: str) -> dict:
    return {"status": "unavailable", "provider_id": "openrouter", "quota_source": OFFICIAL_KEY_LIMITS,
            "reason": reason}


class _RefuseRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):  # noqa: ANN001
        return None


_opener = urllib.request.build_opener(_RefuseRedirect())


class _DeadlineExceeded(Exception):
    """Internal: the single total operation budget expired."""


def _http_get_json(url: str, token: str, timeout: float) -> tuple[int, Any]:
    """GET *url* with *token*; returns (status, parsed-json|None).

    ``timeout`` is a TOTAL monotonic budget covering connect, response headers
    and every body read. urllib's socket timeout is only an inactivity bound:
    bytes that trickle in can outlast it, so the deadline is enforced here as
    well and the socket timeout is refreshed to the remaining budget before
    each read. No worker thread is started, so a refusal leaves no background
    request behind.
    """
    deadline = time.monotonic() + max(float(timeout), 0.0)
    req = urllib.request.Request(
        url, method="GET",
        headers={"Authorization": f"Bearer {token}", "Accept": "application/json"})
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise _DeadlineExceeded()
    with _opener.open(req, timeout=remaining) as resp:
        host = urllib.request.urlparse(resp.geturl()).hostname or ""
        if host.lower() not in _OPENROUTER_HOSTS:
            return 502, None
        sock = getattr(getattr(getattr(resp, "fp", None), "raw", None), "_sock", None)
        chunks: list[bytes] = []
        total = 0
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise _DeadlineExceeded()
            if sock is not None:
                try:
                    sock.settimeout(remaining)
                except Exception:  # health: allow BLE001 -- best effort, monotonic check still bounds us
                    pass
            want = min(4096, MAX_BODY_BYTES + 1 - total)
            # http.client's read(n) loops internally until n bytes arrive, so a
            # slow trickle can outlast our per-iteration deadline check. read1
            # returns at most one raw read, letting the monotonic check run
            # between network reads.
            read1 = getattr(getattr(resp, "fp", None), "read1", None)
            chunk = read1(want) if callable(read1) else resp.read(min(1, want))
            if not chunk:
                break
            chunks.append(chunk)
            total += len(chunk)
            if total > MAX_BODY_BYTES:
                return 502, None
        raw = b"".join(chunks)
        try:
            return int(resp.status), json.loads(raw.decode("utf-8", "replace"))
        except (ValueError, UnicodeDecodeError):
            return int(resp.status), None


def _int_ge_zero(value: Any) -> Optional[int]:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    if not math.isfinite(value) or value < 0 or value != int(value):
        return None
    return int(value)


def fetch_openrouter_key_limits(token: str, *, credential_scope_id: str,
                                timeout: float = DEFAULT_TIMEOUT) -> dict:
    """The one implemented quota source. Counters only; anything off-contract is unavailable."""
    effective_timeout = min(max(float(timeout) if timeout else DEFAULT_TIMEOUT, 0.5), MAX_TIMEOUT)
    started = time.time()
    try:
        status, body = _http_get_json(OPENROUTER_KEY_URL, token, effective_timeout)
    except (_DeadlineExceeded, TimeoutError):
        # socket.timeout is a TimeoutError; the refreshed per-read socket
        # timeout is the mechanism that enforces the total monotonic budget.
        logger.debug("quota_fetch: openrouter key endpoint exceeded the total deadline")
        return _unavailable("fetch_timeout")
    except Exception:  # health: allow BLE001 -- network/TLS/HTTP failures are all 'unavailable'; token never logged
        logger.debug("quota_fetch: openrouter key endpoint unavailable")
        return _unavailable("fetch_failed")
    if status != 200:
        return _unavailable("endpoint_status_%d" % status)
    data = body.get("data") if isinstance(body, dict) else None
    row = data.get("free_model_daily_requests") if isinstance(data, dict) else None
    limit = _int_ge_zero(row.get("limit")) if isinstance(row, dict) else None
    remaining = _int_ge_zero(row.get("remaining")) if isinstance(row, dict) else None
    used = _int_ge_zero(row.get("used")) if isinstance(row, dict) else None
    if limit is None or remaining is None or used is None or remaining + used != limit:
        return _unavailable("unexpected_response_shape")
    now = time.time()
    next_utc_midnight = (datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)
                         .timestamp() + 24 * 3600)
    return {
        "status": "ok",
        "provider_id": "openrouter",
        "credential_scope_id": credential_scope_id,
        "quota_source": OFFICIAL_KEY_LIMITS,
        "observed_at": now,
        "expires_at": min(started + OBSERVATION_TTL, next_utc_midnight),
        "limit": limit,
        "remaining": remaining,
        "used": used,
    }
