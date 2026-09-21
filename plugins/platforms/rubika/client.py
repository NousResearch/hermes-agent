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
        the "data" field on success, raise RubikaAPIError on any error (HTTP or API)."""
        url = f"{BASE_URL}/{self._token}/{method}"
        async with httpx.AsyncClient(timeout=self._timeout) as http_client:
            response = await http_client.post(url, json=params)
            try:
                response.raise_for_status()
            except httpx.HTTPStatusError as exc:
                message = f"Rubika API HTTP error calling {method}: {exc}"
                logger.warning(message)
                raise RubikaAPIError(message, status=str(response.status_code)) from exc
            body = response.json()
        status = body.get("status")
        if status != "OK":
            message = f"Rubika API error calling {method}: status={status}"
            logger.warning(message)
            raise RubikaAPIError(message, status=str(status))
        return body.get("data", {})

    async def upload_file(self, file_path: str, file_type: str) -> str:
        """Two-step upload: requestSendFile -> POST bytes to upload_url -> file_id.
        file_type is one of Rubika's requestSendFile type strings (e.g. "Image",
        "Video", "Voice", "Music", "File", "Gif")."""
        request_data = await self.call("requestSendFile", type=file_type)
        upload_url = request_data.get("upload_url")
        if not upload_url:
            raise RubikaAPIError("requestSendFile returned no upload_url", status="NO_UPLOAD_URL")
        with open(file_path, "rb") as fh:
            file_bytes = fh.read()
        async with httpx.AsyncClient(timeout=self._timeout) as http_client:
            response = await http_client.post(
                upload_url, files={"file": (file_path.rsplit("/", 1)[-1], file_bytes)})
            try:
                response.raise_for_status()
            except httpx.HTTPStatusError as exc:
                message = f"Rubika file upload HTTP error: {exc}"
                logger.warning(message)
                raise RubikaAPIError(message, status=str(response.status_code)) from exc
            body = response.json()
        if body.get("status") != "OK":
            raise RubikaAPIError(
                f"File upload failed: status={body.get('status')}", status=str(body.get("status")))
        file_id = (body.get("data") or {}).get("file_id")
        if not file_id:
            raise RubikaAPIError("Upload response missing file_id", status="NO_FILE_ID")
        return file_id
