"""QQBot standalone (out-of-process) sender.

Used by ``tools/send_message_tool._send_via_adapter`` when no live QQAdapter
is present in the current process (CLI, cron, standalone scripts).  Shares
token acquisition, API request construction, target resolution, file-type
classification, and chunked-upload with the live adapter — no duplicated
REST protocol.

Does NOT start a WebSocket or gateway listener.  Resources (httpx client,
token) are acquired on first use and released in a ``try/finally`` block.
"""

from __future__ import annotations

import logging
import os
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


def _classify_file(
    ext: str,
    *,
    force_document: bool = False,
) -> int:
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


# ── Target resolution ──────────────────────────────────────────────────

def _resolve_target(chat_id: str) -> Tuple[str, str]:
    """Resolve a QQBot chat_id into ``(target_type, target_id)``.

    Handles explicit prefixes:
      ``c2c:<openid>`` → ``('c2c', '<openid>')``
      ``user:<openid>`` → ``('c2c', '<openid>')``
      ``group:<openid>`` → ``('group', '<openid>')``
      ``guild:<id>`` → ``('guild', '<id>')``

    Raw OpenIDs (no prefix) are returned as ``('c2c', '<openid>')``
    — the caller probes and falls back to group on 404.

    Returns ``('unknown', chat_id)`` if the ID is empty after stripping.
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
    # Raw OpenID — default to C2C, callers may probe group on 404
    if raw.strip():
        return "c2c", raw.strip()
    return "unknown", raw


# ── Standalone sender ──────────────────────────────────────────────────

async def _standalone_send(
    pconfig: Any,
    chat_id: str,
    message: str,
    *,
    thread_id: Optional[str] = None,
    media_files: Optional[List[Tuple[str, bool]]] = None,
    force_document: bool = False,
) -> Dict[str, Any]:
    """Send a message + optional media via QQBot REST API — no gateway needed.

    Parameters match the :class:`PlatformEntry.standalone_sender_fn` signature.

    Returns ``{"success": True, "message_id": "..."}`` on success or
    ``{"error": "..."}`` on failure.  Redacts secrets from error text.
    """
    del thread_id  # QQBot has no thread concept

    try:
        import httpx
    except ImportError:
        return {"error": "QQBot standalone send requires httpx. Run: pip install httpx"}

    extra = getattr(pconfig, "extra", None) or {}
    app_id = str(extra.get("app_id") or os.getenv("QQ_APP_ID", "")).strip()
    secret = str(
        extra.get("client_secret") or getattr(pconfig, "token", None)
        or os.getenv("QQ_CLIENT_SECRET", "")
    ).strip()

    if not app_id or not secret:
        return {"error": "QQBot: QQ_APP_ID / QQ_CLIENT_SECRET not configured."}

    target_type, target_id = _resolve_target(chat_id)
    if not target_id:
        return {"error": f"QQBot: empty target ID in chat_id '{chat_id}'"}

    if target_type == "guild":
        return {
            "error": (
                "QQBot MEDIA delivery is only supported for C2C and group chats, "
                "not guild channels"
            )
        }

    media_files = media_files or []

    # ── Validate + classify media before any HTTP ───────────────────────
    media_items: List[Tuple[str, int]] = []  # [(path, file_type), ...]
    for media_path, _is_voice in media_files:
        mp = Path(media_path)
        if not mp.is_file():
            return {"error": f"Media file not found: {media_path}"}
        ft = _classify_file(mp.suffix, force_document=force_document)
        media_items.append((media_path, ft))

    # ── Group targets cannot receive document-type uploads ─────────────
    if target_type == "group":
        docs = [p for p, ft in media_items if ft == MEDIA_TYPE_FILE]
        if docs:
            names = ", ".join(Path(p).name for p in docs)
            return {
                "error": (
                    f"QQ Bot API does not support document uploads to groups. "
                    f"Rejected: {names}. Use image/video/audio for group targets."
                )
            }

    # ── Chunk text ─────────────────────────────────────────────────────
    text_chunks: List[str] = []
    if message.strip():
        text_chunks = _split_for_qq(message, MAX_MESSAGE_LENGTH)

    timeout = FILE_UPLOAD_TIMEOUT
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            # --- Step 1: access token ---
            token_resp = await client.post(
                TOKEN_URL,
                json={"appId": app_id, "clientSecret": secret},
                timeout=DEFAULT_API_TIMEOUT,
            )
            if token_resp.status_code != 200:
                return {
                    "error": f"QQBot token request failed: {token_resp.status_code}"
                }
            token_data = token_resp.json()
            access_token = token_data.get("access_token")
            if not access_token:
                return {"error": "QQBot: no access_token in response"}

            def _auth_headers() -> Dict[str, str]:
                return {
                    "Authorization": f"QQBot {access_token}",
                    "Content-Type": "application/json",
                    "User-Agent": build_user_agent(),
                }

            # --- Step 2: upload media via shared ChunkedUploader ---
            uploaded: List[Dict[str, Any]] = []  # [{file_info, file_type, name}]

            for media_path, file_type in media_items:
                mp = Path(media_path)
                resolved_name = mp.name

                uploader = ChunkedUploader(
                    api_request=_make_api_request_fn(client, access_token),
                    http_put=_make_http_put_fn(client, access_token),
                    log_tag="QQBot:standalone",
                )
                try:
                    complete = await uploader.upload(
                        chat_type=target_type,
                        target_id=target_id,
                        file_path=str(mp),
                        file_type=file_type,
                        file_name=resolved_name,
                    )
                except UploadDailyLimitExceededError as exc:
                    return {
                        "error": (
                            f"QQ daily upload limit exceeded for {exc.file_name!r} "
                            f"({exc.file_size_human}). Retry tomorrow."
                        )
                    }
                except UploadFileTooLargeError as exc:
                    return {
                        "error": (
                            f"File {exc.file_name!r} ({exc.file_size_human}) "
                            f"exceeds platform limit ({exc.limit_human})"
                        )
                    }
                except (ValueError, RuntimeError, OSError) as exc:
                    return {"error": f"QQBot file upload failed ({resolved_name}): {exc}"}

                fi = complete.get("file_info") or (
                    complete.get("data", {}) or {}
                ).get("file_info")
                if not fi:
                    return {
                        "error": f"QQBot: no file_info for {resolved_name}: {complete}"
                    }
                uploaded.append(
                    {"file_info": fi, "file_type": file_type, "name": resolved_name}
                )

            # --- Step 3: send media messages (caption on first) ---
            ep = "users" if target_type == "c2c" else "groups"
            last_msg_id: Optional[str] = None

            for idx, up in enumerate(uploaded):
                body: Dict[str, Any] = {
                    "msg_type": MSG_TYPE_MEDIA,
                    "media": {"file_info": up["file_info"]},
                    "msg_seq": idx + 1,
                }
                if idx == 0 and text_chunks:
                    body["content"] = text_chunks[0][:MAX_MESSAGE_LENGTH]

                resp = await client.post(
                    f"{API_BASE}/v2/{ep}/{target_id}/messages",
                    json=body,
                    headers=_auth_headers(),
                )
                if resp.status_code not in {200, 201}:
                    return {
                        "error": (
                            f"QQBot media send failed ({up['name']}): "
                            f"{resp.status_code} {_safe_text(resp)}"
                        )
                    }
                last_msg_id = resp.json().get("id")

            # --- Step 4: send remaining text chunks ---
            start = 1 if uploaded else 0
            for chunk in text_chunks[start:]:
                resp = await client.post(
                    f"{API_BASE}/v2/{ep}/{target_id}/messages",
                    json={"content": chunk, "msg_type": MSG_TYPE_TEXT},
                    headers=_auth_headers(),
                )
                if resp.status_code not in {200, 201}:
                    return {
                        "error": (
                            f"QQBot text send failed: "
                            f"{resp.status_code} {_safe_text(resp)}"
                        )
                    }
                last_msg_id = resp.json().get("id")

            return {
                "success": True,
                "platform": "qqbot",
                "chat_id": chat_id,
                "message_id": last_msg_id,
            }

    except Exception as exc:
        return {"error": f"QQBot standalone send failed: {exc}"}


# ── Helpers ────────────────────────────────────────────────────────────


def _split_for_qq(text: str, max_len: int) -> List[str]:
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


def _make_api_request_fn(client: Any, token: str):
    """Build an ``api_request`` callable compatible with :class:`ChunkedUploader`.

    Signature: ``(method, path, body, timeout) -> dict``.
    """

    async def _api_request(
        method: str,
        path: str,
        body: Optional[Dict[str, Any]] = None,
        timeout: float = DEFAULT_API_TIMEOUT,
    ) -> Dict[str, Any]:
        url = f"{API_BASE}{path}"
        headers = {
            "Authorization": f"QQBot {token}",
            "Content-Type": "application/json",
            "User-Agent": build_user_agent(),
        }
        resp = await client.request(
            method, url, headers=headers, json=body, timeout=timeout
        )
        data = resp.json()
        if resp.status_code >= 400:
            msg = data.get("message", data) if isinstance(data, dict) else str(data)
            raise RuntimeError(
                f"QQ Bot API error [{resp.status_code}] {path}: {msg}"
            )
        return data

    return _api_request


def _make_http_put_fn(client: Any, token: str):
    """Build an ``http_put`` callable compatible with :class:`ChunkedUploader`.

    Signature: ``(url, data, headers, timeout) -> response``.
    """

    async def _http_put(
        url: str,
        data: bytes,
        headers: Optional[Dict[str, str]] = None,
        timeout: float = 120.0,
    ) -> Any:
        return await client.put(url, content=data, headers=headers, timeout=timeout)

    return _http_put


def _safe_text(resp: Any) -> str:
    """Extract up to 200 chars of response text without leaking secrets."""
    try:
        raw = resp.text
    except Exception:
        return "(no body)"
    return str(raw)[:200]
