"""Yuanbao sign-token acquisition, caching, signing and retry.

Moved out of ``yuanbao.py`` (the adapter facade) by the fix for #134367's secondary, which tripped
the file-size ratchet: the repo's remedy for file growth is to offset it by moving an existing
topic into a topical sibling. ``yuanbao`` re-exports ``SignManager`` so every existing import path
and test seam (``gateway.platforms.yuanbao.SignManager``) keeps working.
"""

from __future__ import annotations

import asyncio
import hashlib
import hmac
import logging
import secrets
import sys
import time
from datetime import datetime, timedelta, timezone
from typing import Any

import httpx

from hermes_cli.version_info import get_version_info
from gateway.platforms.yuanbao_proto import HERMES_INSTANCE_ID

logger = logging.getLogger(__name__)

# AUTH_BIND / sign-token header values
APP_VERSION = BOT_VERSION = get_version_info().base_version
YUANBAO_INSTANCE_ID = str(HERMES_INSTANCE_ID)
OPERATION_SYSTEM = sys.platform


class SignManager:
    """Sign-token acquisition, caching, signing and retry. All state is class-level so one
    shared client serves the whole process."""
    TOKEN_PATH = "/api/v5/robotLogic/sign-token"
    RETRYABLE_CODE = 10099
    MAX_RETRIES = 3
    RETRY_DELAY_S = 1.0
    CACHE_REFRESH_MARGIN_S = 60  # treat as expiring this many seconds early
    HTTP_TIMEOUT_S = 10.0
    _cache: dict[str, dict[str, Any]] = {}  # app_key → {"token", "bot_id", "expire_ts", ...}
    # Per-app_key refresh locks, created lazily from async context so they bind to the running
    # loop; disconnect() clears them to avoid stale locks across reconnects.
    _locks: dict[str, asyncio.Lock] = {}

    @classmethod
    def get_refresh_lock(cls, app_key: str) -> asyncio.Lock:
        """Per-app_key refresh lock (create on demand). Call only from a running event loop."""
        if app_key not in cls._locks:
            cls._locks[app_key] = asyncio.Lock()
        return cls._locks[app_key]

    @staticmethod
    def compute_signature(nonce: str, timestamp: str, app_key: str, app_secret: str) -> str:
        """HMAC-SHA256(key=app_secret, msg=nonce+timestamp+app_key+app_secret).hexdigest()."""
        plain = nonce + timestamp + app_key + app_secret
        return hmac.new(app_secret.encode(), plain.encode(), hashlib.sha256).hexdigest()

    @staticmethod
    def build_timestamp() -> str:
        """Beijing-time ISO-8601 timestamp without milliseconds (2006-01-02T15:04:05+08:00)."""
        return datetime.now(tz=timezone(timedelta(hours=8))).strftime("%Y-%m-%dT%H:%M:%S+08:00")

    @classmethod
    def is_cache_valid(cls, entry: dict[str, Any]) -> bool:
        return entry["expire_ts"] - time.time() > cls.CACHE_REFRESH_MARGIN_S

    @classmethod
    def clear_locks(cls) -> None:
        cls._locks.clear()

    @classmethod
    def purge_expired(cls) -> int:
        """Drop expired token-cache entries; returns count purged."""
        now = time.time()
        expired_keys = [k for k, v in cls._cache.items() if now - v.get("expire_ts", 0) > 0]
        for k in expired_keys:
            cls._cache.pop(k, None)
        return len(expired_keys)

    @classmethod
    async def fetch(cls, app_key: str, app_secret: str, api_domain: str, route_env: str = "") -> dict[str, Any]:
        """POST sign-token, retrying RETRYABLE_CODE up to MAX_RETRIES times."""
        url = f"{api_domain.rstrip('/')}{cls.TOKEN_PATH}"
        async with httpx.AsyncClient(timeout=cls.HTTP_TIMEOUT_S) as client:
            for attempt in range(cls.MAX_RETRIES + 1):
                nonce = secrets.token_hex(16)
                timestamp = cls.build_timestamp()
                payload = {"app_key": app_key, "nonce": nonce,
                           "signature": cls.compute_signature(nonce, timestamp, app_key, app_secret), "timestamp": timestamp}
                headers = {"Content-Type": "application/json", "X-AppVersion": APP_VERSION, "X-OperationSystem": OPERATION_SYSTEM,
                           "X-Instance-Id": YUANBAO_INSTANCE_ID, "X-Bot-Version": BOT_VERSION}
                if route_env:
                    headers["X-Route-Env"] = route_env
                logger.info("Sign token request: url=%s%s", url, f" (retry {attempt}/{cls.MAX_RETRIES})" if attempt > 0 else "")
                response = await client.post(url, json=payload, headers=headers)
                if response.status_code != 200:
                    raise RuntimeError(f"Sign token API returned {response.status_code}: {response.text[:200]}")
                try:
                    result_data: dict[str, Any] = response.json()
                except Exception as exc:
                    raise ValueError(f"Sign token response parse error: {exc}") from exc
                code = result_data.get("code")
                if code == 0:
                    data = result_data.get("data")
                    if not isinstance(data, dict):
                        raise ValueError(f"Sign token response missing 'data' field: {result_data}")
                    logger.info("Sign token success: bot_id=%s", data.get("bot_id"))
                    return data
                if code != cls.RETRYABLE_CODE or attempt >= cls.MAX_RETRIES:
                    raise RuntimeError(f"Sign token error: code={code}, msg={result_data.get('msg', '')}")
                logger.warning("Sign token retryable: code=%s, retrying in %ss (attempt=%d/%d)",
                               code, cls.RETRY_DELAY_S, attempt + 1, cls.MAX_RETRIES)
                await asyncio.sleep(cls.RETRY_DELAY_S)
        raise RuntimeError("Sign token failed: max retries exceeded")

    @classmethod
    async def _fetch_into_cache(cls, app_key: str, app_secret: str, api_domain: str, route_env: str) -> None:
        data = await cls.fetch(app_key, app_secret, api_domain, route_env)
        duration: int = data.get("duration", 0)
        cls._cache[app_key] = {
            "token": data.get("token", ""), "bot_id": data.get("bot_id", ""), "duration": duration,
            "product": data.get("product", ""), "source": data.get("source", ""),
            "expire_ts": time.time() + (duration if duration > 0 else 3600),
        }

    @classmethod
    async def get_token(cls, app_key: str, app_secret: str, api_domain: str, route_env: str = "") -> dict[str, Any]:
        """WS auth token, served from cache while valid (with CACHE_REFRESH_MARGIN_S)."""
        cls.purge_expired()
        cached = cls._cache.get(app_key)
        if cached and cls.is_cache_valid(cached):
            logger.info("Using cached token (%ds remaining)", int(cached["expire_ts"] - time.time()))
            return dict(cached)
        async with cls.get_refresh_lock(app_key):
            cached = cls._cache.get(app_key)
            if cached and cls.is_cache_valid(cached):
                return dict(cached)
            await cls._fetch_into_cache(app_key, app_secret, api_domain, route_env)
        return dict(cls._cache[app_key])

    @classmethod
    async def force_refresh(cls, app_key: str, app_secret: str, api_domain: str, route_env: str = "") -> dict[str, Any]:
        """Clear the cached token and re-sign."""
        logger.warning("[force-refresh] Clearing cache and re-signing token: app_key=****%s", app_key[-4:])
        async with cls.get_refresh_lock(app_key):
            cls._cache.pop(app_key, None)
            await cls._fetch_into_cache(app_key, app_secret, api_domain, route_env)
        return dict(cls._cache[app_key])
