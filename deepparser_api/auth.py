from __future__ import annotations

import hmac
import time
from collections import defaultdict
from typing import Annotated

import aiosqlite
from fastapi import Header, HTTPException, Request

from . import db
from .db import fetchone
from .config import (
    AUTH_FAIL_MAX,
    AUTH_FAIL_WINDOW_SECS,
    KEYS_PER_IP_MAX,
    KEYS_PER_IP_WINDOW_SECS,
)

# In-memory rate-limit stores.  { ip: [(timestamp, ...), ...] }
_auth_failures: dict[str, list[float]] = defaultdict(list)
_key_registrations: dict[str, list[float]] = defaultdict(list)


def _client_ip(request: Request) -> str:
    forwarded = request.headers.get("X-Forwarded-For")
    if forwarded:
        return forwarded.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


def _check_rate_limit(
    store: dict[str, list[float]],
    ip: str,
    window: int,
    max_hits: int,
) -> bool:
    """Return True if limit exceeded, False if OK.  Prunes old entries in-place."""
    now = time.time()
    hits = store[ip]
    store[ip] = [t for t in hits if now - t < window]
    if len(store[ip]) >= max_hits:
        return True
    return False


def _record_hit(store: dict[str, list[float]], ip: str) -> None:
    store[ip].append(time.time())


async def require_api_key(
    request: Request,
    x_api_key: Annotated[str | None, Header()] = None,
) -> dict:
    """FastAPI dependency — returns the api_keys row or raises 401/403/429."""
    ip = _client_ip(request)

    if _check_rate_limit(_auth_failures, ip, AUTH_FAIL_WINDOW_SECS, AUTH_FAIL_MAX):
        raise HTTPException(
            status_code=429,
            detail={
                "code": "RATE_LIMITED",
                "message": "Too many failed auth attempts. Try again in 60 seconds.",
                "doc_url": "https://github.com/ysh145/hermes-agent/tree/main/deepparser",
            },
        )

    if not x_api_key:
        _record_hit(_auth_failures, ip)
        raise HTTPException(
            status_code=401,
            detail={
                "code": "MISSING_API_KEY",
                "message": "Include your API key in the X-API-Key header.",
                "doc_url": "https://github.com/ysh145/hermes-agent/tree/main/deepparser",
            },
        )

    async with db.connect() as conn:
        conn.row_factory = aiosqlite.Row
        row = await fetchone(conn, "SELECT * FROM api_keys WHERE key = ?", (x_api_key,))

    if row is None or not hmac.compare_digest(x_api_key, row["key"]):
        _record_hit(_auth_failures, ip)
        raise HTTPException(
            status_code=401,
            detail={
                "code": "INVALID_API_KEY",
                "message": "API key not recognized.",
                "doc_url": "https://github.com/ysh145/hermes-agent/tree/main/deepparser",
            },
        )

    if row["revoked"]:
        raise HTTPException(
            status_code=403,
            detail={
                "code": "REVOKED_API_KEY",
                "message": "This API key has been revoked. Re-register at POST /keys.",
                "doc_url": "https://github.com/ysh145/hermes-agent/tree/main/deepparser",
            },
        )

    return dict(row)


def check_keys_rate_limit(request: Request) -> None:
    """Call at the start of POST /keys to enforce 5 registrations/IP/hour."""
    ip = _client_ip(request)
    if _check_rate_limit(_key_registrations, ip, KEYS_PER_IP_WINDOW_SECS, KEYS_PER_IP_MAX):
        raise HTTPException(
            status_code=429,
            detail={
                "code": "RATE_LIMITED",
                "message": "Too many key registrations from this IP. Try again in 1 hour.",
                "doc_url": "https://github.com/ysh145/hermes-agent/tree/main/deepparser",
            },
        )
    _record_hit(_key_registrations, ip)
