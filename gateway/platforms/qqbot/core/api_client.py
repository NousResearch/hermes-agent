# -*- coding: utf-8 -*-
"""QQ Bot REST API client — mirrors botgo/openapi.

Encapsulates all HTTP interactions with the QQ Bot REST API:

- OAuth2 client-credentials token lifecycle
- Authenticated request dispatch with trace-id logging
- Message sending (C2C / group / guild)
- Rich-media file upload
- WebSocket gateway URL retrieval

This class has **zero dependencies on hermes**.  It only requires an
``httpx.AsyncClient`` (or any object that satisfies the same interface) to
be injected via :meth:`setup`.

Usage::

    client = QQApiClient(app_id="...", client_secret="...")
    client.setup(http_client)          # inject httpx.AsyncClient
    await client.ensure_token()
    result = await client.post_c2c_message(user_id, body)
"""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from typing import Any, Dict, Optional

from .constants import (
    API_BASE,
    DEFAULT_API_TIMEOUT,
    FILE_UPLOAD_TIMEOUT,
    GATEWAY_URL_PATH,
    MAX_MESSAGE_LENGTH,
    MSG_TYPE_MARKDOWN,
    MSG_TYPE_TEXT,
    TOKEN_URL,
)
from .dto import (
    GuildMessageToCreate,
    InlineKeyboard,
    MarkdownContent,
    MediaInfo,
    MessageReference,
    MessageToCreate,
    QQMessageType,
    RichMediaMessage,
    UploadPrepareRequest,
    UploadPartFinishRequest,
    CompleteUploadRequest,
    UploadPrepareResponse,
    CompleteUploadResponse,
    parse_upload_prepare,
    parse_complete_upload,
)
from .utils import build_user_agent

logger = logging.getLogger(__name__)


# ── URI constants (mirrors botgo/openapi/v1/resource.go) ─────────────

_C2C_MESSAGES_URI = "/v2/users/{user_id}/messages"
_C2C_FILES_URI = "/v2/users/{user_id}/files"
_GROUP_MESSAGES_URI = "/v2/groups/{group_id}/messages"
_GROUP_FILES_URI = "/v2/groups/{group_id}/files"
_GUILD_MESSAGES_URI = "/channels/{channel_id}/messages"


# ── QQApiClient ───────────────────────────────────────────────────────

class QQApiClient:
    """Authenticated HTTP client for the QQ Bot REST API.

    :param app_id: Bot application ID.
    :param client_secret: Bot client secret.
    :param log_tag: Log prefix for disambiguation in multi-instance setups.
    """

    def __init__(
        self,
        app_id: str,
        client_secret: str,
        log_tag: str = "QQBot",
    ) -> None:
        self._app_id = app_id
        self._client_secret = client_secret
        self._log_tag = log_tag

        self._http_client: Any = None

        self._access_token: Optional[str] = None
        self._token_expires_at: float = 0.0
        self._token_lock = asyncio.Lock()

    def setup(self, http_client: Any) -> None:
        """Attach an ``httpx.AsyncClient`` for making requests.

        :param http_client: Any object with ``.request()``, ``.post()``,
            and ``.get()`` async methods compatible with httpx.
        """
        self._http_client = http_client

    @property
    def access_token(self) -> Optional[str]:
        """Current access token (may be expired)."""
        return self._access_token

    # ------------------------------------------------------------------
    # Token management
    # ------------------------------------------------------------------

    async def ensure_token(self) -> str:
        """Return a valid access token, refreshing if needed (singleflight).

        :returns: Valid ``access_token`` string.
        :raises RuntimeError: If the token request fails.
        """
        if self._access_token and time.time() < self._token_expires_at - 60:
            return self._access_token

        async with self._token_lock:
            # Double-check inside lock to avoid stampede.
            if self._access_token and time.time() < self._token_expires_at - 60:
                return self._access_token

            if not self._http_client:
                raise RuntimeError(
                    "HTTP client not initialized — call setup() first"
                )

            try:
                resp = await self._http_client.post(
                    TOKEN_URL,
                    json={
                        "appId": self._app_id,
                        "clientSecret": self._client_secret,
                    },
                    timeout=DEFAULT_API_TIMEOUT,
                )
                resp.raise_for_status()
                data = resp.json()
            except Exception as exc:
                raise RuntimeError(
                    f"Failed to get QQ Bot access token: {exc}"
                ) from exc

            token = data.get("access_token")
            if not token:
                raise RuntimeError(
                    f"QQ Bot token response missing access_token: {data}"
                )

            expires_in = int(data.get("expires_in", 7200))
            self._access_token = token
            self._token_expires_at = time.time() + expires_in
            logger.info(
                "[%s] Access token refreshed, expires in %ds",
                self._log_tag,
                expires_in,
            )
            return self._access_token

    def clear_token(self) -> None:
        """Invalidate the cached token (e.g. after a 4004 close code)."""
        self._access_token = None
        self._token_expires_at = 0.0

    def media_headers(self) -> Dict[str, str]:
        """Return ``Authorization`` headers for QQ CDN media downloads."""
        if self._access_token:
            return {"Authorization": f"QQBot {self._access_token}"}
        return {}

    # ------------------------------------------------------------------
    # Generic authenticated request
    # ------------------------------------------------------------------

    async def request(
        self,
        method: str,
        path: str,
        body: Optional[Dict[str, Any]] = None,
        timeout: float = DEFAULT_API_TIMEOUT,
    ) -> Dict[str, Any]:
        """Make an authenticated REST API request.

        :param method: HTTP method (``GET``, ``POST``, …).
        :param path: API path, e.g. ``/v2/users/{id}/messages``.
        :param body: JSON request body.
        :param timeout: Request timeout in seconds.
        :returns: Parsed JSON response dict.
        :raises RuntimeError: On HTTP errors or missing client.
        """
        if not self._http_client:
            raise RuntimeError("HTTP client not initialized — not connected?")

        import httpx

        token = await self.ensure_token()
        headers = {
            "Authorization": f"QQBot {token}",
            "Content-Type": "application/json",
            "User-Agent": build_user_agent(),
        }

        try:
            resp = await self._http_client.request(
                method,
                f"{API_BASE}{path}",
                headers=headers,
                json=body,
                timeout=timeout,
            )
            trace_id = resp.headers.get("x-tps-trace-id", "-")

            if resp.status_code >= 400:
                logger.error(
                    "[%s] API %s %s failed: status=%d trace_id=%s",
                    self._log_tag, method, path, resp.status_code, trace_id,
                )
                data = resp.json()
                raise RuntimeError(
                    f"QQ Bot API error [{resp.status_code}] {path}: "
                    f"{data.get('message', data)}"
                )

            logger.info(
                "[%s] API %s %s: status=%d trace_id=%s",
                self._log_tag, method, path, resp.status_code, trace_id,
            )
            return resp.json()

        except httpx.TimeoutException as exc:
            logger.error("[%s] API %s %s timeout", self._log_tag, method, path)
            raise RuntimeError(f"QQ Bot API timeout [{path}]: {exc}") from exc

        except Exception:
            logger.error(
                "[%s] API %s %s error",
                self._log_tag, method, path,
                exc_info=True,
            )
            raise

    # ------------------------------------------------------------------
    # Gateway
    # ------------------------------------------------------------------

    async def get_gateway_url(self) -> str:
        """Fetch the WebSocket gateway URL.

        :returns: WebSocket gateway URL string.
        :raises RuntimeError: If the response is missing ``url``.
        """
        data = await self.request("GET", GATEWAY_URL_PATH)
        url = data.get("url")
        if not url:
            raise RuntimeError(
                f"QQ Bot gateway response missing url: {data}"
            )
        return url

    # ------------------------------------------------------------------
    # Message endpoints
    # ------------------------------------------------------------------

    async def post_c2c_message(
        self,
        user_id: str,
        msg: MessageToCreate,
        keyboard: Optional[InlineKeyboard] = None,
    ) -> Dict[str, Any]:
        """Send a message to a C2C user.

        :param user_id: Target user openid.
        :param msg: Message payload.
        :param keyboard: Optional inline keyboard; appended to the body when provided.
        :returns: Raw API response dict.
        """
        path = _C2C_MESSAGES_URI.replace("{user_id}", user_id)
        body = msg.to_dict()
        if keyboard is not None:
            body["keyboard"] = keyboard.to_dict()
        return await self.request("POST", path, body)

    async def post_group_message(
        self,
        group_id: str,
        msg: MessageToCreate,
        keyboard: Optional[InlineKeyboard] = None,
    ) -> Dict[str, Any]:
        """Send a message to a group.

        :param group_id: Target group openid.
        :param msg: Message payload.
        :param keyboard: Optional inline keyboard; appended to the body when provided.
        :returns: Raw API response dict.
        """
        path = _GROUP_MESSAGES_URI.replace("{group_id}", group_id)
        body = msg.to_dict()
        if keyboard is not None:
            body["keyboard"] = keyboard.to_dict()
        return await self.request("POST", path, body)

    async def post_guild_message(
        self,
        channel_id: str,
        msg: GuildMessageToCreate,
    ) -> Dict[str, Any]:
        """Send a message to a guild channel."""
        path = _GUILD_MESSAGES_URI.replace("{channel_id}", channel_id)
        return await self.request("POST", path, msg.to_dict())

    # ------------------------------------------------------------------
    # Media upload endpoints
    # ------------------------------------------------------------------

    async def upload_c2c_file(
        self,
        user_id: str,
        msg: RichMediaMessage,
    ) -> Dict[str, Any]:
        """Upload a rich-media file for a C2C user."""
        path = _C2C_FILES_URI.replace("{user_id}", user_id)
        return await self.request("POST", path, msg.to_dict(), timeout=FILE_UPLOAD_TIMEOUT)

    async def upload_group_file(
        self,
        group_id: str,
        msg: RichMediaMessage,
    ) -> Dict[str, Any]:
        """Upload a rich-media file for a group."""
        path = _GROUP_FILES_URI.replace("{group_id}", group_id)
        return await self.request("POST", path, msg.to_dict(), timeout=FILE_UPLOAD_TIMEOUT)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def next_msg_seq(seed: str = "") -> int:
        """Generate a message sequence number in the range 0–65535.

        :param seed: Unused; kept for API compatibility.
        :returns: Pseudo-random integer in ``[0, 65535]``.
        """
        time_part = int(time.time()) % 100000000
        rand = int(uuid.uuid4().hex[:4], 16)
        return (time_part ^ rand) % 65536

    @staticmethod
    def build_text_body(
        content: str,
        reply_to: Optional[str] = None,
        markdown: bool = True,
        max_length: int = MAX_MESSAGE_LENGTH,
    ) -> MessageToCreate:
        """Build a :class:`MessageToCreate` for C2C/group text sending.

        :param content: Message text.
        :param reply_to: Message ID to reply to (passive message).
        :param markdown: Use msg_type 2 (markdown) when ``True``.
        :param max_length: Maximum content length; truncates if exceeded.
        :returns: Ready-to-send :class:`MessageToCreate` instance.
        """
        msg_seq = QQApiClient.next_msg_seq()
        truncated = content[:max_length]

        if markdown:
            return MessageToCreate(
                msg_type=QQMessageType.MARKDOWN,
                msg_seq=msg_seq,
                msg_id=reply_to or "",
                markdown=MarkdownContent(content=truncated),
            )

        msg = MessageToCreate(
            content=truncated,
            msg_type=QQMessageType.TEXT,
            msg_seq=msg_seq,
            msg_id=reply_to or "",
        )
        if reply_to:
            msg.message_reference = MessageReference(message_id=reply_to)
        return msg

    # ------------------------------------------------------------------
    # Chunked upload (upload_prepare / upload_part_finish / complete_upload)
    # ------------------------------------------------------------------
    #
    # Three-step flow (mirrors TS chunked-upload.ts):
    #   1. upload_prepare → upload_id, block_size, presigned part URLs
    #   2. PUT each part to its presigned URL (COS), then upload_part_finish
    #   3. complete_upload → file_info / file_uuid
    #
    # Error code semantics (from api.ts):
    #   40093001 — part_finish retryable (retry until success or timeout)
    #   40093002 — daily upload limit exceeded (surface to user as error text)

    async def upload_prepare(
        self,
        chat_type: str,
        target_id: str,
        file_type: int,
        file_name: str,
        file_size: int,
        md5: str,
        sha1: str,
        md5_10m: str,
    ) -> UploadPrepareResponse:
        """申请分片上传，返回 upload_id / block_size / parts（含预签名链接）.

        :param chat_type: ``'c2c'`` or ``'group'``.
        :param target_id: User openid or group openid.
        :param file_type: Media type constant (1=image, 2=video, 3=voice, 4=file).
        :param file_name: Original filename.
        :param file_size: File size in bytes.
        :param md5: Full-file MD5 hex string.
        :param sha1: Full-file SHA1 hex string.
        :param md5_10m: MD5 of first 10,002,432 bytes (equals full MD5 for smaller files).
        :returns: Typed :class:`~dto.UploadPrepareResponse`.
        :raises RuntimeError: On API error (biz_code included in message).
        :raises ValueError: If the response is missing required fields.
        """
        if chat_type == "c2c":
            path = f"/v2/users/{target_id}/upload_prepare"
        else:
            path = f"/v2/groups/{target_id}/upload_prepare"
        req = UploadPrepareRequest(
            file_type=file_type,
            file_name=file_name,
            file_size=file_size,
            md5=md5,
            sha1=sha1,
            md5_10m=md5_10m,
        )
        raw = await self.request("POST", path, req.to_dict(), timeout=FILE_UPLOAD_TIMEOUT)
        return parse_upload_prepare(raw)

    async def upload_part_finish(
        self,
        chat_type: str,
        target_id: str,
        upload_id: str,
        part_index: int,
        block_size: int,
        md5: str,
    ) -> None:
        """通知开放平台某个分片已上传完成.

        Returns nothing on success (mirrors proto ``UploadPartFinishRsp`` which is empty).

        :param chat_type: ``'c2c'`` or ``'group'``.
        :param target_id: User or group openid.
        :param upload_id: Upload task ID from ``upload_prepare``.
        :param part_index: 1-based part index.
        :param block_size: Actual byte count of this part.
        :param md5: MD5 hex of this part's data.
        :raises RuntimeError: On API error (biz_code 40093001 = retryable).
        """
        if chat_type == "c2c":
            path = f"/v2/users/{target_id}/upload_part_finish"
        else:
            path = f"/v2/groups/{target_id}/upload_part_finish"
        req = UploadPartFinishRequest(
            upload_id=upload_id,
            part_index=part_index,
            block_size=block_size,
            md5=md5,
        )
        await self.request("POST", path, req.to_dict(), timeout=FILE_UPLOAD_TIMEOUT)

    async def complete_upload(
        self,
        chat_type: str,
        target_id: str,
        upload_id: str,
    ) -> CompleteUploadResponse:
        """完成分片上传，返回 file_info / file_uuid / ttl.

        Reuses the ``/files`` endpoint (same as simple upload) with
        ``upload_id`` to signal the chunked completion path:

        - C2C:   ``POST /v2/users/{user_id}/files``
        - Group: ``POST /v2/groups/{group_id}/files``

        :param chat_type: ``'c2c'`` or ``'group'``.
        :param target_id: User or group openid.
        :param upload_id: Upload task ID from ``upload_prepare``.
        :returns: Typed :class:`~dto.CompleteUploadResponse`.
        """
        if chat_type == "c2c":
            path = f"/v2/users/{target_id}/files"
        else:
            path = f"/v2/groups/{target_id}/files"
        req = CompleteUploadRequest(upload_id=upload_id)
        raw = await self.request(
            "POST", path, req.to_dict(), timeout=FILE_UPLOAD_TIMEOUT
        )
        return parse_complete_upload(raw)

    # ------------------------------------------------------------------
    # Interaction (button callback)
    # ------------------------------------------------------------------

    async def acknowledge_interaction(
        self,
        interaction_id: str,
        code: int = 0,
        data: Optional[Dict[str, Any]] = None,
    ) -> None:
        """ACK a button interaction to stop the client loading indicator.

        Must be called promptly after receiving INTERACTION_CREATE, or the
        button will show an error to the user.

        :param interaction_id: The ``id`` field from the INTERACTION_CREATE event.
        :param code: Response code (0 = success).
        :param data: Optional extra payload (e.g. ``claw_cfg`` for config interactions).
        """
        body: Dict[str, Any] = {"code": code}
        if data:
            body["data"] = data
        await self.request("PUT", f"/interactions/{interaction_id}", body)



