"""Outbound "call me back" — place a Teams call via the worker's HTTP endpoint.

The worker exposes a loopback HTTP endpoint (`POST {worker_base_url}/api/calls`,
default port 9440) that places an outbound 1:1 Teams call. It is HMAC-authenticated
exactly like the WS bridge, except the signed payload is ``"{ts}.{userObjectId}"``
(the callee's AAD object id) rather than the callId.

Contract (worker `OutboundCallController.cs`):
  headers: X-OpenClawTeamsBridge-Timestamp, X-OpenClawTeamsBridge-Signature
  body:    {"userObjectId": <aad id>, "tenantId": <tenant>}
  200 ->   {"callId": ..., "scenarioId": ...}
"""

from __future__ import annotations

import asyncio
import logging
import time
from urllib.parse import urlparse

from . import hmac_auth
from .config import HEADER_SIGNATURE, HEADER_TIMESTAMP

logger = logging.getLogger(__name__)

_LOOPBACK_HOSTS = {"127.0.0.1", "localhost", "::1"}


class OutboundError(RuntimeError):
    """Raised when an outbound place-call request fails."""


async def place_call(
    *,
    user_object_id: str,
    tenant_id: str,
    shared_secret: str,
    worker_base_url: str = "http://127.0.0.1:9440",
    timeout_s: float = 15.0,
    allow_remote: bool = False,
) -> dict:
    """Place an outbound Teams call to ``user_object_id`` in ``tenant_id``.

    Returns the worker's ``{"callId": ..., "scenarioId": ...}`` on success.
    Raises :class:`OutboundError` on validation/auth/transport failure.

    SSRF guard: refuses a non-loopback ``worker_base_url`` unless ``allow_remote``
    — otherwise a misconfigured host would receive the HMAC-signed request.
    """
    import aiohttp

    if not user_object_id or not tenant_id:
        raise OutboundError("user_object_id and tenant_id are required")
    if not shared_secret:
        raise OutboundError("shared secret not configured")
    host = (urlparse(worker_base_url).hostname or "").lower()
    if not allow_remote and host not in _LOOPBACK_HOSTS:
        raise OutboundError(
            f"refusing outbound place-call to non-loopback worker '{host}' "
            "(set allow_remote_worker to override)"
        )

    ts = int(time.time() * 1000)
    signature = hmac_auth.sign(shared_secret, ts, user_object_id)
    headers = {
        HEADER_TIMESTAMP: str(ts),
        HEADER_SIGNATURE: signature,
        "Content-Type": "application/json",
    }
    body = {"userObjectId": user_object_id, "tenantId": tenant_id}
    url = f"{worker_base_url.rstrip('/')}/api/calls"

    try:
        timeout = aiohttp.ClientTimeout(total=timeout_s)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.post(url, json=body, headers=headers) as resp:
                text = await resp.text()
                if resp.status != 200:
                    raise OutboundError(f"worker returned {resp.status}: {text}")
                try:
                    return await resp.json()
                except (aiohttp.ContentTypeError, ValueError):
                    return {"raw": text}
    except (aiohttp.ClientError, asyncio.TimeoutError) as exc:
        raise OutboundError(f"transport error placing call: {exc}") from exc
