"""Hermes-side webhook deployment guard for the Feishu adapter.

This module owns the aiohttp server lifecycle plus the rate-limit, anomaly
tracker, body-size, Content-Type, body-read-timeout, verification token,
signature, URL-verification challenge, and encrypted-payload guards. Event
dispatch is delegated to the SDK via ``channel.handle_webhook_request``.

The public API (``start_webhook_server`` plus the two dataclasses) is the
contract relied on by ``FeishuAdapter._connect_webhook`` and the contract
test suite.
"""

import asyncio
import hashlib
import hmac
import json
import logging
import time
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Dict, Mapping, Tuple

logger = logging.getLogger(__name__)

# aiohttp is part of the messaging extra (pyproject.toml); same try/except
# pattern as the adapter top-level.
try:
    import aiohttp
    from aiohttp import web
except ImportError:  # pragma: no cover - import guard
    aiohttp = None  # type: ignore[assignment]
    web = None  # type: ignore[assignment]

WEBHOOK_AVAILABLE = aiohttp is not None

_RUNNER_HANDLERS: Dict[int, Callable[[Any], Awaitable[Any]]] = {}


def _normalize_sdk_response_status(status_code: int, response_body: bytes) -> int:
    """Normalize SDK-auth failures that are reported as 5xx.

    lark-oapi 1.6.0 returns HTTP 500 with ``{"msg":"invalid verification_token"}``.
    That is an authentication failure in the incoming request, so expose it
    as 401 while leaving the SDK response body unchanged.
    """
    if status_code < 500:
        return status_code
    try:
        body_text = response_body.decode("utf-8", errors="replace").lower()
    except Exception:
        return status_code
    if (
        "invalid verification_token" in body_text
        or "signature verification failed" in body_text
    ):
        return 401
    return status_code


_SDK_CANONICAL_HEADER_NAMES = (
    "X-Lark-Request-Timestamp",
    "X-Lark-Request-Nonce",
    "X-Lark-Signature",
)


def _headers_for_sdk(headers: Mapping[str, str]) -> Dict[str, str]:
    out = dict(headers) if headers else {}
    lower_index = {str(k).lower(): v for k, v in out.items()}
    for canonical in _SDK_CANONICAL_HEADER_NAMES:
        if canonical in out:
            continue
        value = None
        try:
            value = headers.get(canonical)  # type: ignore[arg-type]
        except Exception:
            value = None
        if value is None:
            value = lower_index.get(canonical.lower())
        if value is not None:
            out[canonical] = value
    return out


def _payload_token(payload: Dict[str, Any]) -> str:
    header = payload.get("header") if isinstance(payload, dict) else None
    if isinstance(header, dict):
        token = header.get("token")
        if token:
            return str(token)
    token = payload.get("token") if isinstance(payload, dict) else None
    return str(token or "")


def _signature_valid(
    headers: Mapping[str, str],
    body_bytes: bytes,
    *,
    encrypt_key: str,
) -> bool:
    canonical = _headers_for_sdk(headers)
    timestamp = str(canonical.get("X-Lark-Request-Timestamp", "") or "")
    nonce = str(canonical.get("X-Lark-Request-Nonce", "") or "")
    signature = str(canonical.get("X-Lark-Signature", "") or "")
    if not (timestamp and nonce and signature):
        return False
    try:
        body_str = body_bytes.decode("utf-8", errors="replace")
        expected = hashlib.sha256(
            f"{timestamp}{nonce}{encrypt_key}{body_str}".encode("utf-8")
        ).hexdigest()
        return hmac.compare_digest(expected, signature)
    except Exception:
        logger.debug("[Feishu] Signature verification raised", exc_info=True)
        return False

# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------
# These match the historical openclaw webhook anomaly + rate-limit profile
# and are locked down by the contract tests in
# tests/gateway/feishu/test_webhook_security.py — change in lock-step with
# those tests.

MAX_BODY_BYTES: int = 1 * 1024 * 1024            # 1 MB body limit
RATE_WINDOW_SECONDS: int = 60                    # sliding window for rate limiter
RATE_LIMIT_MAX: int = 120                        # max requests per window per composite key
RATE_MAX_KEYS: int = 4096                        # max tracked rate-limit keys (LRU evict-on-full)
BODY_TIMEOUT_SECONDS: int = 30                   # max seconds to read request body
ANOMALY_THRESHOLD: int = 25                      # consecutive error responses before WARNING log
ANOMALY_TTL_SECONDS: int = 6 * 60 * 60           # anomaly tracker TTL (6 hours)


# ---------------------------------------------------------------------------
# Public dataclasses
# ---------------------------------------------------------------------------

@dataclass
class RateLimit:
    """Rate-limit configuration for the webhook server."""

    window_seconds: int
    max_requests: int


@dataclass
class WebhookAnomaly:
    """Single anomaly observation passed to ``on_anomaly`` hook.

    ``status_code`` is the integer HTTP status the request received;
    sub-classification (e.g. "401-token" vs "401-sig") is logged via
    ``logger.warning`` inside the tracker but is not part of this hook
    payload.
    """

    remote_ip: str
    status_code: int
    count: int
    first_seen_ts: float


# ---------------------------------------------------------------------------
# Internal classes (rate limiter + anomaly tracker)
# ---------------------------------------------------------------------------

class _RateLimiter:
    """Sliding-window rate limiter, scoped by composite key.

    Composite key is "{app_id}:{path}:{remote_ip}" -- built by the caller
    and passed via ``allow(rate_key)``. Sliding window with LRU-on-full
    eviction of stale keys.
    """

    def __init__(
        self,
        *,
        window_seconds: int,
        max_requests: int,
        max_keys: int = RATE_MAX_KEYS,
    ) -> None:
        self._window_seconds = window_seconds
        self._max_requests = max_requests
        self._max_keys = max_keys
        # rate_key -> (count, window_start_ts)
        self._counts: Dict[str, Tuple[int, float]] = {}

    def allow(self, rate_key: str) -> bool:
        """Return False when the composite rate_key has exceeded max_requests
        within the current sliding window."""
        now = time.time()

        # Fast path: existing entry within current window.
        entry = self._counts.get(rate_key)
        if entry is not None:
            count, window_start = entry
            if now - window_start < self._window_seconds:
                if count >= self._max_requests:
                    return False
                self._counts[rate_key] = (count + 1, window_start)
                return True

        # New window for an existing key, or a brand-new key -- prune stale first.
        if len(self._counts) >= self._max_keys:
            stale_keys = [
                k for k, (_, ws) in self._counts.items()
                if now - ws >= self._window_seconds
            ]
            for k in stale_keys:
                del self._counts[k]
            # If still at capacity after pruning, fail closed -- caller emits 429.
            # Fail-open here lets an attacker cycle through 4096+ source keys to
            # disable rate limiting for everyone in the same window.
            if rate_key not in self._counts and len(self._counts) >= self._max_keys:
                return False

        self._counts[rate_key] = (1, now)
        return True


class _AnomalyTracker:
    """Per-IP anomaly counter -- emits WARNING every ANOMALY_THRESHOLD hits.

    TTL: 6 h, threshold: 25, "increment-or-reset on TTL" semantics:

    - ``on_anomaly`` hook called every time ``record(...)`` increments;
      the callback receives a fresh ``WebhookAnomaly`` snapshot.
    - Sub-classification (e.g. "401-token" vs "401-sig") is captured via the
      ``note`` parameter to ``record(...)`` and surfaced in the WARNING
      log; it is not part of the dataclass payload.
    """

    def __init__(
        self,
        *,
        on_anomaly: Callable[[WebhookAnomaly], None],
        threshold: int = ANOMALY_THRESHOLD,
        ttl_seconds: int = ANOMALY_TTL_SECONDS,
    ) -> None:
        self._on_anomaly = on_anomaly
        self._threshold = threshold
        self._ttl_seconds = ttl_seconds
        # remote_ip -> (count, last_status_note, first_seen_ts)
        self._counts: Dict[str, Tuple[int, str, float]] = {}

    def record(self, remote_ip: str, status_code: int, *, note: str = "") -> None:
        """Increment anomaly counter; emit WARNING every ``threshold`` hits.

        ``note`` is a free-form sub-classification (e.g. "401-token",
        "401-sig", "400-encrypted", "415", "413", "408", "400") -- included
        in the log message and stored on the entry for reference, but not
        surfaced via the public ``WebhookAnomaly`` payload.
        """
        now = time.time()
        entry = self._counts.get(remote_ip)
        if entry is not None:
            count, _last_note, first_seen = entry
            if now - first_seen < self._ttl_seconds:
                count += 1
                if count % self._threshold == 0:
                    logger.warning(
                        "[Feishu] Webhook anomaly: %d consecutive error responses (%d%s) "
                        "from %s over the last %.0fs",
                        count,
                        status_code,
                        f"/{note}" if note else "",
                        remote_ip,
                        now - first_seen,
                    )
                self._counts[remote_ip] = (count, note, first_seen)
                # Snapshot for hook
                self._fire_hook(remote_ip, status_code, count, first_seen)
                return
        # First occurrence or TTL expired -- start fresh.
        self._counts[remote_ip] = (1, note, now)
        self._fire_hook(remote_ip, status_code, 1, now)

    def clear(self, remote_ip: str) -> None:
        """Reset the anomaly counter after a successful request."""
        self._counts.pop(remote_ip, None)

    def _fire_hook(
        self, remote_ip: str, status_code: int, count: int, first_seen_ts: float
    ) -> None:
        try:
            self._on_anomaly(WebhookAnomaly(
                remote_ip=remote_ip,
                status_code=status_code,
                count=count,
                first_seen_ts=first_seen_ts,
            ))
        except Exception as exc:
            logger.warning(
                "[Feishu] webhook_guard on_anomaly hook raised: %s", exc, exc_info=True
            )


# ---------------------------------------------------------------------------
# Internal aiohttp request handler
# ---------------------------------------------------------------------------

def _build_aiohttp_handler(
    *,
    app_id: str,
    path: str,
    handle_request: Callable[[Mapping[str, str], bytes], Awaitable[Tuple[int, bytes]]],
    rate_limiter: _RateLimiter,
    anomaly_tracker: _AnomalyTracker,
    encrypt_key: str = "",
    verification_token: str = "",
) -> Callable[[Any], Awaitable[Any]]:
    """Construct the aiohttp request handler with deployment guards baked in.

    Hermes preserves the historical outer auth/encryption guards before
    delegating event dispatch to ``handle_request`` -- typically
    ``channel.handle_webhook_request``.

    Note: empty ``encrypt_key`` AND empty ``verification_token`` together
    disable authentication — this combination is explicitly rejected by
    ``start_webhook_server`` before this function is called. Callers that
    bypass ``start_webhook_server`` (e.g. unit tests, custom integrations)
    are responsible for ensuring at least one secret is set in production.
    """

    async def _handler(request: Any) -> Any:
        remote_ip: str = (getattr(request, "remote", None) or "unknown")

        # Rate limiting on the composite key keyed by app_id+path+remote_ip.
        rate_key = f"{app_id}:{path}:{remote_ip}"
        if not rate_limiter.allow(rate_key):
            logger.warning("[Feishu] Webhook rate limit exceeded for %s", remote_ip)
            anomaly_tracker.record(remote_ip, 429, note="rate-limit")
            return web.Response(status=429, text="Too Many Requests")

        # Feishu webhooks always send application/json.
        headers = getattr(request, "headers", {}) or {}
        content_type = str(headers.get("Content-Type", "") or "").split(";")[0].strip().lower()
        if content_type and content_type != "application/json":
            logger.warning(
                "[Feishu] Webhook rejected: unexpected Content-Type %r from %s",
                content_type, remote_ip,
            )
            anomaly_tracker.record(remote_ip, 415, note="content-type")
            return web.Response(status=415, text="Unsupported Media Type")

        # Early reject when Content-Length already exceeds the limit.
        content_length = getattr(request, "content_length", None)
        if content_length is not None and content_length > MAX_BODY_BYTES:
            logger.warning(
                "[Feishu] Webhook body too large (%d bytes) from %s",
                content_length, remote_ip,
            )
            anomaly_tracker.record(remote_ip, 413, note="content-length")
            return web.Response(status=413, text="Request body too large")

        # Bound the body read so a stalled peer can't pin a worker forever.
        try:
            body_bytes: bytes = await asyncio.wait_for(
                request.read(),
                timeout=BODY_TIMEOUT_SECONDS,
            )
        except asyncio.TimeoutError:
            logger.warning(
                "[Feishu] Webhook body read timed out after %ds from %s",
                BODY_TIMEOUT_SECONDS, remote_ip,
            )
            anomaly_tracker.record(remote_ip, 408, note="read-timeout")
            return web.Response(status=408, text="Request Timeout")
        except Exception:
            anomaly_tracker.record(remote_ip, 400, note="read-error")
            return web.json_response({"code": 400, "msg": "failed to read body"}, status=400)

        # Re-check body size for chunked transfers / responses without Content-Length.
        if len(body_bytes) > MAX_BODY_BYTES:
            logger.warning(
                "[Feishu] Webhook body exceeds limit (%d bytes) from %s",
                len(body_bytes), remote_ip,
            )
            anomaly_tracker.record(remote_ip, 413, note="actual-size")
            return web.Response(status=413, text="Request body too large")

        # Reject malformed bodies before invoking the SDK handler.
        try:
            payload = json.loads(body_bytes.decode("utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError):
            anomaly_tracker.record(remote_ip, 400, note="bad-json")
            return web.json_response({"code": 400, "msg": "invalid json"}, status=400)
        if not isinstance(payload, dict):
            anomaly_tracker.record(remote_ip, 400, note="bad-json")
            return web.json_response({"code": 400, "msg": "invalid json"}, status=400)

        if payload.get("type") == "url_verification":
            return web.json_response({"challenge": payload.get("challenge", "")})

        if verification_token:
            incoming_token = _payload_token(payload)
            if not incoming_token or not hmac.compare_digest(
                incoming_token, verification_token,
            ):
                anomaly_tracker.record(remote_ip, 401, note="token")
                return web.Response(status=401, text="Invalid verification token")

        if encrypt_key and not _signature_valid(
            headers, body_bytes, encrypt_key=encrypt_key,
        ):
            anomaly_tracker.record(remote_ip, 401, note="sig")
            return web.Response(status=401, text="Invalid signature")

        if payload.get("encrypt"):
            anomaly_tracker.record(remote_ip, 400, note="encrypted")
            return web.json_response(
                {"code": 400, "msg": "encrypted webhook payloads are not supported"},
                status=400,
            )

        # Delegate signature / verification_token / URL verification / decrypt /
        # dispatch to SDK ``handle_request`` (typically channel.handle_webhook_request).
        try:
            status_code, response_body = await handle_request(
                _headers_for_sdk(headers), body_bytes,
            )
        except Exception as exc:
            logger.error(
                "[Feishu] Webhook handle_request raised for %s: %s",
                remote_ip, exc, exc_info=True,
            )
            anomaly_tracker.record(remote_ip, 500, note="handler-exception")
            return web.json_response({"code": 500, "msg": "internal error"}, status=500)

        status_code = _normalize_sdk_response_status(status_code, response_body)

        # Track anomalies: any 4xx/5xx counts; success clears the IP's counter.
        if status_code >= 400:
            anomaly_tracker.record(remote_ip, status_code, note="sdk-rejected")
        else:
            anomaly_tracker.clear(remote_ip)

        # Pass SDK status + body back to the HTTP response unchanged.
        return web.Response(
            status=status_code,
            body=response_body,
            content_type="application/json",
        )

    return _handler


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

async def start_webhook_server(
    *,
    host: str,
    port: int,
    path: str,
    app_id: str,
    handle_request: Callable[[Mapping[str, str], bytes], Awaitable[Tuple[int, bytes]]],
    rate_limit: RateLimit,
    on_anomaly: Callable[[WebhookAnomaly], None],
    encrypt_key: str = "",
    verification_token: str = "",
) -> "web.AppRunner":
    """Start an aiohttp webhook server bound to ``handle_request``.

    Returns the ``aiohttp.web.AppRunner`` so the caller can shut it down
    via ``await runner.cleanup()``.

    Args:
        host: Interface to bind (e.g. "127.0.0.1" or "0.0.0.0").
        port: TCP port.
        path: URL path to mount (e.g. "/feishu/webhook").
        app_id: Feishu app_id -- used as part of the rate-limit composite key
            ``{app_id}:{path}:{remote_ip}`` so multiple feishu apps in one
            process don't collide.
        handle_request: Async callable receiving (headers, body_bytes) and
            returning (status_code, response_body_bytes). Typically
            ``channel.handle_webhook_request`` from lark_oapi.channel.
        rate_limit: RateLimit dataclass with window_seconds + max_requests.
        on_anomaly: Sync callback invoked every time the anomaly counter
            increments. Hermes injects ``_hermes_log_webhook_anomaly``.
        encrypt_key: Feishu encrypt key used for local signature checks.
        verification_token: Feishu verification token checked before dispatch.

    Raises:
        RuntimeError: When aiohttp is not installed (WEBHOOK_AVAILABLE False),
            or when both ``encrypt_key`` and ``verification_token`` are empty
            (refusing to expose an unauthenticated endpoint).
    """
    if not WEBHOOK_AVAILABLE:
        raise RuntimeError("aiohttp not installed; webhook mode unavailable")

    if not encrypt_key and not verification_token:
        raise RuntimeError(
            "Refusing to start Feishu webhook server: both FEISHU_ENCRYPT_KEY "
            "and FEISHU_VERIFICATION_TOKEN are empty. Configure at least one "
            "secret so inbound events are authenticated before reaching the SDK."
        )

    rate_limiter = _RateLimiter(
        window_seconds=rate_limit.window_seconds,
        max_requests=rate_limit.max_requests,
    )
    anomaly_tracker = _AnomalyTracker(on_anomaly=on_anomaly)

    handler = _build_aiohttp_handler(
        app_id=app_id,
        path=path,
        handle_request=handle_request,
        rate_limiter=rate_limiter,
        anomaly_tracker=anomaly_tracker,
        encrypt_key=encrypt_key,
        verification_token=verification_token,
    )

    app = web.Application()
    app.router.add_post(path, handler)
    runner = web.AppRunner(app)
    await runner.setup()
    # Keep the partial-bound aiohttp handler reachable to
    # ``FeishuAdapter._handle_webhook_request`` without relying on aiohttp
    # runner objects supporting dynamic attributes.
    _RUNNER_HANDLERS[id(runner)] = handler
    site = web.TCPSite(runner, host, port)
    await site.start()
    logger.info(
        "[Feishu] Webhook server started: %s:%d%s (rate_limit=%d/%ds, anomaly_threshold=%d)",
        host, port, path, rate_limit.max_requests, rate_limit.window_seconds, ANOMALY_THRESHOLD,
    )
    return runner


def get_webhook_handler(runner: Any) -> Callable[[Any], Awaitable[Any]] | None:
    """Return the handler associated with a runner created by this module."""

    return _RUNNER_HANDLERS.get(id(runner))


def clear_webhook_handler(runner: Any) -> None:
    """Forget the handler associated with a runner after cleanup."""

    _RUNNER_HANDLERS.pop(id(runner), None)
