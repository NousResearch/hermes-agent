"""Shared QQ Bot REST API client.

Used by both the live ``QQAdapter`` (through its own internal methods) and
the standalone sender (``_standalone_send``).  Provides:

* Token acquisition and refresh
* Authenticated API requests
* Target resolution from chat_id prefixes (standalone, not dependent on
  the live adapter's inbound ``_chat_type_map``)
* File upload via ``ChunkedUploader``
* Text and media message sending

The adapter still owns its own lifecycle (WebSocket, reconnect, HTTP client
session); the standalone sender creates a temporary ``QQApiClient`` with an
``httpx.AsyncClient`` that lives only for the duration of the send.
"""

from __future__ import annotations

import asyncio
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from gateway.platforms.qqbot.chunked_upload import (
    ChunkedUploader,
    UploadDailyLimitExceededError,
    UploadFileTooLargeError,
)
from gateway.platforms.qqbot.constants import (
    API_BASE,
    DEFAULT_API_TIMEOUT,
    FILE_UPLOAD_TIMEOUT,
    MAX_MESSAGE_LENGTH,
    MEDIA_TYPE_FILE,
    MEDIA_TYPE_IMAGE,
    MEDIA_TYPE_VIDEO,
    MEDIA_TYPE_VOICE,
    MSG_TYPE_MEDIA,
    MSG_TYPE_TEXT,
    TOKEN_URL,
)
from gateway.platforms.qqbot.utils import build_user_agent

logger = logging.getLogger(__name__)

# ── File-type classification ───────────────────────────────────────────

_IMAGE_EXTS: set[str] = {".jpg", ".jpeg", ".png", ".webp", ".gif"}
_VIDEO_EXTS: set[str] = {".mp4", ".mov", ".avi", ".mkv", ".3gp"}
_VOICE_EXTS: set[str] = {".silk", ".wav", ".mp3", ".flac", ".ogg", ".opus"}


def classify_media_type(ext: str, *, force_document: bool = False) -> int:
    """Classify a file extension into a QQ Bot ``file_type`` constant.

    Returns one of ``MEDIA_TYPE_IMAGE``, ``MEDIA_TYPE_VIDEO``,
    ``MEDIA_TYPE_VOICE``, or ``MEDIA_TYPE_FILE``.
    """
    if force_document:
        return MEDIA_TYPE_FILE
    ext = ext.lower()
    if ext in _IMAGE_EXTS:
        return MEDIA_TYPE_IMAGE
    if ext in _VIDEO_EXTS:
        return MEDIA_TYPE_VIDEO
    if ext in _VOICE_EXTS:
        return MEDIA_TYPE_VOICE
    return MEDIA_TYPE_FILE


# ── Target resolution (standalone — does NOT depend on _chat_type_map) ──

def resolve_target(chat_id: str) -> Tuple[str, str]:
    """Resolve a QQBot chat_id into ``(target_type, target_id)``.

    Handles explicit prefixes:
      ``c2c:<openid>`` → ``('c2c', '<openid>')``
      ``user:<openid>`` → ``('c2c', '<openid>')``
      ``group:<openid>`` → ``('group', '<openid>')``
      ``guild:<id>``     → ``('guild', '<id>')``

    Raw OpenIDs (no prefix) are returned as ``('c2c', '<openid>')``
    — the caller may probe and fall back to group on 404.
    """
    raw = str(chat_id)
    if ":" in raw:
        prefix, rest = raw.split(":", 1)
        prefix = prefix.lower()
        if prefix in {"c2c", "user"}:
            return "c2c", rest
        if prefix == "group":
            return "group", rest
        if prefix == "guild":
            return "guild", rest
    # Raw OpenID — default to C2C, caller may probe group on 404
    if raw.strip():
        return "c2c", raw.strip()
    return "unknown", raw


# ── Shared API client ──────────────────────────────────────────────────

class QQApiClient:
    """Authenticated QQ Bot REST client.

    Manages access token lifecycle and provides authenticated API request
    and file upload methods.  Can be backed by either an adapter-owned
    persistent session or a temporary standalone client.
    """

    def __init__(
        self,
        app_id: str,
        client_secret: str,
        http_client: Any,
        *,
        log_tag: str = "QQBot",
    ):
        self._app_id = app_id
        self._client_secret = client_secret
        self._http_client = http_client  # httpx.AsyncClient or aiohttp.ClientSession
        self._access_token: Optional[str] = None
        self._token_expires_at: float = 0.0
        self._log_tag = log_tag

    # ── Token ──────────────────────────────────────────────────────────

    async def ensure_token(self) -> str:
        """Return a valid access token, refreshing if necessary."""
        now = time.time()
        if self._access_token and now < self._token_expires_at - 60:
            return self._access_token

        resp = await self._http_client.post(
            TOKEN_URL,
            json={"appId": self._app_id, "clientSecret": self._client_secret},
            timeout=DEFAULT_API_TIMEOUT,
        )
        if resp.status_code != 200:
            raise RuntimeError(
                f"QQ Bot token request failed: {resp.status_code}"
            )
        data = resp.json()
        token = data.get("access_token")
        if not token:
            raise RuntimeError(
                f"QQ Bot token response missing access_token: {data}"
            )
        expires_in = int(data.get("expires_in", 7200))
        self._access_token = token
        self._token_expires_at = now + expires_in
        logger.info(
            "[%s] Access token refreshed, expires in %ds",
            self._log_tag, expires_in,
        )
        return token

    # ── API request ────────────────────────────────────────────────────

    def _auth_headers(self, token: str) -> Dict[str, str]:
        return {
            "Authorization": f"QQBot {token}",
            "Content-Type": "application/json",
            "User-Agent": build_user_agent(),
        }

    async def api_request(
        self,
        method: str,
        path: str,
        body: Optional[Dict[str, Any]] = None,
        timeout: float = DEFAULT_API_TIMEOUT,
    ) -> Dict[str, Any]:
        """Make an authenticated REST API request."""
        token = await self.ensure_token()
        headers = self._auth_headers(token)
        resp = await self._http_client.request(
            method,
            f"{API_BASE}{path}",
            headers=headers,
            json=body,
            timeout=timeout,
        )
        data = resp.json()
        if resp.status_code >= 400:
            raise RuntimeError(
                f"QQ Bot API error [{resp.status_code}] {path}: "
                f"{data.get('message', data)}"
            )
        return data

    async def http_put(
        self,
        url: str,
        data: bytes,
        headers: Optional[Dict[str, str]] = None,
        timeout: float = 120.0,
    ) -> Any:
        """HTTP PUT for chunked-upload parts.  Returns raw response."""
        return await self._http_client.put(
            url, content=data, headers=headers, timeout=timeout
        )

    # ── Upload ─────────────────────────────────────────────────────────

    async def upload_local_file(
        self,
        chat_type: str,
        target_id: str,
        file_path: str,
        file_type: int,
        file_name: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Upload a local file via ChunkedUploader.

        Returns the complete-upload response dict (contains ``file_info``).
        """
        resolved_name = file_name or Path(file_path).name
        uploader = ChunkedUploader(
            api_request=self.api_request,
            http_put=self.http_put,
            log_tag=self._log_tag,
        )
        return await uploader.upload(
            chat_type=chat_type,
            target_id=target_id,
            file_path=file_path,
            file_type=file_type,
            file_name=resolved_name,
        )

    # ── Message sending ────────────────────────────────────────────────

    @staticmethod
    def _endpoint_for(chat_type: str, target_id: str) -> str:
        """Return the REST endpoint path for a given chat type."""
        if chat_type == "c2c":
            return f"/v2/users/{target_id}/messages"
        if chat_type == "group":
            return f"/v2/groups/{target_id}/messages"
        if chat_type == "guild":
            return f"/channels/{target_id}/messages"
        raise ValueError(f"Unknown chat_type: {chat_type}")

    async def send_text(
        self,
        chat_type: str,
        target_id: str,
        content: str,
        msg_seq: int = 1,
    ) -> Dict[str, Any]:
        """Send a text message.  Content is truncated to MAX_MESSAGE_LENGTH."""
        path = self._endpoint_for(chat_type, target_id)
        return await self.api_request(
            "POST",
            path,
            {"content": content[:MAX_MESSAGE_LENGTH], "msg_type": MSG_TYPE_TEXT},
        )

    async def send_media(
        self,
        chat_type: str,
        target_id: str,
        file_info: Dict[str, Any],
        content: Optional[str] = None,
        msg_seq: int = 1,
    ) -> Dict[str, Any]:
        """Send a media message with optional caption."""
        path = self._endpoint_for(chat_type, target_id)
        body: Dict[str, Any] = {
            "msg_type": MSG_TYPE_MEDIA,
            "media": {"file_info": file_info},
            "msg_seq": msg_seq,
        }
        if content and content.strip():
            body["content"] = content[:MAX_MESSAGE_LENGTH]
        return await self.api_request("POST", path, body)


# ── Text chunking ──────────────────────────────────────────────────────

def split_for_qq(text: str, max_len: int) -> List[str]:
    """Split *text* into chunks ≤ *max_len*, preferring newline boundaries."""
    if len(text) <= max_len:
        return [text]
    chunks: List[str] = []
    remaining = text
    while len(remaining) > max_len:
        cut = remaining.rfind("\n", 0, max_len)
        if cut == -1:
            cut = max_len
        chunks.append(remaining[:cut])
        remaining = remaining[cut:].lstrip("\n")
    if remaining.strip():
        chunks.append(remaining)
    return chunks
