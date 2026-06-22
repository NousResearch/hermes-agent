"""Cube credential acquisition — A12 Token API with static-key fallback."""

from __future__ import annotations

import json
import logging
import os
import urllib.request
from contextlib import contextmanager
from typing import Iterator

logger = logging.getLogger(__name__)


def _token_api_url() -> str:
    return (
        os.getenv("SANDBOX_TOKEN_API_URL", "").strip()
        or os.getenv("CUBE_TOKEN_API_URL", "").strip()
    )


def _static_api_key() -> str:
    return (
        os.getenv("CUBE_API_KEY", "").strip()
        or os.getenv("E2B_API_KEY", "").strip()
    )


def acquire_api_key(task_id: str | None, tier: str) -> str:
    """Return a Cube API key — Token API first, then static env fallback."""
    url = _token_api_url()
    if url:
        try:
            key = _fetch_token(url, task_id=task_id, tier=tier)
            if key:
                return key
        except Exception as exc:
            logger.warning("Cube token API failed (tier=%s): %s", tier, exc)
            if os.getenv("CUBE_TOKEN_FAIL_CLOSED", "").strip().lower() in {
                "1",
                "true",
                "yes",
            }:
                raise

    static = _static_api_key()
    if static:
        return static

    raise RuntimeError(
        "Cube credentials unavailable: set SANDBOX_TOKEN_API_URL or CUBE_API_KEY"
    )


def _fetch_token(url: str, *, task_id: str | None, tier: str) -> str:
    payload = {
        "task_id": task_id or "default",
        "tier": tier,
    }
    headers = {"Content-Type": "application/json"}
    auth = os.getenv("SANDBOX_TOKEN_API_AUTH", "").strip()
    if auth:
        headers["Authorization"] = auth

    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers=headers,
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        body = json.loads(resp.read().decode("utf-8"))

    for key in ("api_key", "token", "cube_api_key"):
        val = body.get(key)
        if isinstance(val, str) and val.strip():
            return val.strip()

    data = body.get("data")
    if isinstance(data, dict):
        for key in ("api_key", "token", "cube_api_key"):
            val = data.get(key)
            if isinstance(val, str) and val.strip():
                return val.strip()

    raise RuntimeError(f"Token API response missing api_key field: {body!r}")


@contextmanager
def cube_credentials(task_id: str | None, tier: str) -> Iterator[str]:
    """Temporarily inject Cube API keys for one high-risk tool call."""
    api_key = acquire_api_key(task_id, tier)
    saved = {
        "CUBE_API_KEY": os.environ.get("CUBE_API_KEY"),
        "E2B_API_KEY": os.environ.get("E2B_API_KEY"),
        "TERMINAL_ENV": os.environ.get("TERMINAL_ENV"),
    }
    os.environ["CUBE_API_KEY"] = api_key
    os.environ["E2B_API_KEY"] = api_key
    os.environ["TERMINAL_ENV"] = "cube_sandbox"
    try:
        yield api_key
    finally:
        for name, value in saved.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value
