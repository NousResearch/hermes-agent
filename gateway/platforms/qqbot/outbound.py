"""Shared QQ Bot REST API client.

Used by both the live ``QQAdapter`` and the standalone sender.  Provides:

* Token acquisition with singleflight + invalidation
* Authenticated API requests (raises :class:`QQApiError` with ``status_code``)
* Target resolution from chat_id prefixes
* File upload via ``ChunkedUploader``
* Text and media message sending
* Text chunking

The adapter and standalone both use the same ``QQApiClient`` class; the
adapter passes its persistent ``httpx.AsyncClient`` while the standalone
creates a temporary one.
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


def classify_media_type(
    ext: str,
    *,
    is_voice: bool = False,
    force_document: bool = False,
) -> int:
    """Classify a file extension into a QQ Bot ``file_type`` constant.

    Args:
        ext: File extension (e.g. ``.jpg``, ``.mp3``).  Case-insensitive.
        is_voice: If ``True`` AND the extension is a supported audio
            format, classify as ``MEDIA_TYPE_VOICE`` (voice message).
            Set by the ``[[audio_as_voice]]`` directive in the response
            text.  Without this flag, audio files are sent as documents.
        force_document: If ``True``, ALL files are classified as
            ``MEDIA_TYPE_FILE``.  Highest priority.

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
    if is_voice and ext in _VOICE_EXTS:
        return MEDIA_TYPE_VOICE
    # Audio extensions without [[audio_as_voice]] → document
    return MEDIA_TYPE_FILE


# ── Structured API exception ───────────────────────────────────────────

class QQApiError(RuntimeError):
    """Raised by :meth:`QQApiClient.api_request` on HTTP errors.

    Carries ``status_code`` so callers can implement targeted fallback
    logic (e.g. C2C → group retry on 404).
    """

    def __init__(
        self,
        message: str,
        *,
        status_code: int = 0,
    ):
        super().__init__(message)
        self.status_code = status_code


# ── Target resolution (standalone — does NOT depend on _chat_type_map) ──

def resolve_target(chat_id: str) -> Tuple[str, str, bool]:
    """Resolve a QQBot chat_id into ``(target_type, target_id, has_prefix)``.

    Handles explicit prefixes::

        c2c:<openid>  → ('c2c', '<openid>', True)
        user:<openid> → ('c2c', '<openid>', True)
        group:<openid> → ('group', '<openid>', True)
        guild:<id>     → ('guild', '<id>', True)

    Raw OpenIDs (no prefix) are returned as ``('c2c', '<openid>', False)``
    — the caller may probe C2C first and fall back to group ONLY on 404.
    Explicit-prefix targets MUST NOT fallback.
    """
    raw = str(chat_id).strip()
    if ":" in raw:
        prefix, rest = raw.split(":", 1)
        prefix = prefix.lower()
        if prefix in {"c2c", "user"}:
            return "c2c", rest, True
        if prefix == "group":
            return "group", rest, True
        if prefix == "guild":
            return "guild", rest, True
    # Raw OpenID — default to C2C; caller may probe group on 404 only
    if raw:
        return "c2c", raw, False
    return "unknown", raw, False


# ── Shared API client ──────────────────────────────────────────────────

class QQApiClient:
    """Authenticated QQ Bot REST client — single source of truth.

    Manages access token lifecycle with singleflight-concurrent caching
    and public invalidation.  Provides authenticated API requests and
    file uploads.  Backed by an ``httpx.AsyncClient`` passed at init
    (adapter-owned persistent or standalone temporary).
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
        self._http_client = http_client  # httpx.AsyncClient
        self._access_token: Optional[str] = None
        self._token_expires_at: float = 0.0
        self._token_lock: asyncio.Lock = asyncio.Lock()
        self._log_tag = log_tag

    # ── Public token API ───────────────────────────────────────────────

    async def ensure_token(self) -> str:
        """Return a valid access token, refreshing if needed.

        Uses ``asyncio.Lock`` for singleflight: concurrent callers
        block on the lock and the second caller re-checks the cache
        after the first finishes the refresh.
        """
        now = time.time()
        if self._access_token and now < self._token_expires_at - 60:
            return self._access_token

        async with self._token_lock:
            # Double-check after acquiring the lock
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

    def invalidate_token(self) -> None:
        """Clear the cached access token.

        Called by the adapter on 4004 (invalid token) so the next
        ``ensure_token()`` or ``api_request()`` call forces a refresh.
        """
        logger.info("[%s] Token invalidated", self._log_tag)
        self._access_token = None
        self._token_expires_at = 0.0

    # Read-only mirrors for backward compat (adapter uses these via
    # ``self._api._access_token`` / ``self._api._token_expires_at``).
    @property
    def access_token(self) -> Optional[str]:
        return self._access_token

    @property
    def token_expires_at(self) -> float:
        return self._token_expires_at

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
        """Make an authenticated REST API request.

        Raises :class:`QQApiError` with ``.status_code`` on HTTP ≥ 400.
        HTTP errors that return non-JSON bodies still produce a
        ``QQApiError`` with the correct ``status_code``; only the
        endpoint path (not token, secret, or full body) appears in the
        message.
        """
        token = await self.ensure_token()
        headers = self._auth_headers(token)
        resp = await self._http_client.request(
            method,
            f"{API_BASE}{path}",
            headers=headers,
            json=body,
            timeout=timeout,
        )
        try:
            data = resp.json()
        except Exception:
            # Non-JSON response body — still raise with correct status_code
            raise QQApiError(
                f"QQ Bot API error [{resp.status_code}] {path}",
                status_code=resp.status_code,
            )
        if resp.status_code >= 400:
            raise QQApiError(
                f"QQ Bot API error [{resp.status_code}] {path}: "
                f"{data.get('message', 'no message')}",
                status_code=resp.status_code,
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
