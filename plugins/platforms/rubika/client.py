"""Thin async HTTP wrapper over Rubika's Bot API (https://rubika.ir/botapi).

Base URL shape: POST https://botapi.rubika.ir/v3/{token}/{method}, JSON body,
JSON response of the form {"status": "OK"|<error>, "data": {...}}.
"""

import logging
from typing import Any, Dict

import httpx

logger = logging.getLogger(__name__)

BASE_URL = "https://botapi.rubika.ir/v3"


class RubikaAPIError(Exception):
    """Raised when the Rubika Bot API returns a non-OK status."""

    def __init__(self, message: str, status: str):
        super().__init__(message)
        self.status = status


class RubikaClient:
    """One instance per bot token. Not thread-safe across event loops; create
    per-adapter, not shared globally."""

    def __init__(self, token: str, *, timeout: float = 30.0):
        self._token = token
        self._timeout = timeout

    async def call(self, method: str, **params: Any) -> Dict[str, Any]:
        """POST to /v3/{token}/{method} with params as the JSON body; return
        the "data" field on success, raise RubikaAPIError otherwise."""
        url = f"{BASE_URL}/{self._token}/{method}"
        async with httpx.AsyncClient(timeout=self._timeout) as http_client:
            response = await http_client.post(url, json=params)
            response.raise_for_status()
            body = response.json()
        status = body.get("status")
        if status != "OK":
            message = f"Rubika API error calling {method}: status={status}"
            logger.warning(message)
            raise RubikaAPIError(message, status=str(status))
        return body.get("data") or {}
