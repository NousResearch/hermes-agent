"""QQBot standalone (out-of-process) sender.

Used by ``tools/send_message_tool._send_via_adapter`` when no live QQAdapter
is present in the current process (CLI, cron, standalone scripts).  Shares
token, auth, API request, upload, and target resolution with the live adapter
via ``QQApiClient`` — no duplicated REST protocol.

Does NOT start a WebSocket or gateway listener.  Resources (httpx client,
token) are acquired on first use and released in a ``try/finally`` block.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from gateway.platforms.qqbot.chunked_upload import (
    UploadDailyLimitExceededError,
    UploadFileTooLargeError,
)
from gateway.platforms.qqbot.constants import (
    DEFAULT_API_TIMEOUT,
    FILE_UPLOAD_TIMEOUT,
    MAX_MESSAGE_LENGTH,
    MEDIA_TYPE_FILE,
)
from gateway.platforms.qqbot.outbound import (
    QQApiClient,
    classify_media_type,
    resolve_target,
    split_for_qq,
)

logger = logging.getLogger(__name__)


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

    target_type, target_id = resolve_target(chat_id)
    if not target_id:
        return {"error": f"QQBot: empty target ID in chat_id '{chat_id}'"}

    if target_type == "guild":
        # Guild channels do not support media upload via the QQ Bot API.
        # Text-only messages are still supported.
        if media_files:
            return {
                "error": (
                    "QQBot MEDIA delivery to guild channels is not supported "
                    "by the QQ Bot API. Use C2C or group targets for media."
                )
            }
        # Fall through — text still works for guilds via /channels/<id>/messages

    media_files = media_files or []

    # ── Validate + classify media before any HTTP ───────────────────────
    media_items: List[Tuple[str, int]] = []  # [(path, file_type), ...]
    for media_path, _is_voice in media_files:
        mp = Path(media_path)
        if not mp.is_file():
            return {"error": f"Media file not found: {media_path}"}
        ft = classify_media_type(mp.suffix, force_document=force_document)
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
        text_chunks = split_for_qq(message, MAX_MESSAGE_LENGTH)

    http_client = None
    try:
        http_client = httpx.AsyncClient(timeout=FILE_UPLOAD_TIMEOUT)
        api = QQApiClient(
            app_id,
            secret,
            http_client,
            log_tag="QQBot:standalone",
        )

        # --- Step 1: upload media via shared ChunkedUploader ---
        uploaded: List[Dict[str, Any]] = []

        for media_path, file_type in media_items:
            try:
                complete = await api.upload_local_file(
                    chat_type=target_type,
                    target_id=target_id,
                    file_path=media_path,
                    file_type=file_type,
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
                return {"error": f"QQBot file upload failed ({Path(media_path).name}): {exc}"}

            fi = complete.get("file_info") or (
                complete.get("data", {}) or {}
            ).get("file_info")
            if not fi:
                return {
                    "error": f"QQBot: no file_info for {Path(media_path).name}: {complete}"
                }
            uploaded.append({"file_info": fi, "file_type": file_type})

        # --- Step 2: send media messages (caption on first) ---
        last_msg_id: Optional[str] = None

        for idx, up in enumerate(uploaded):
            caption = text_chunks[0][:MAX_MESSAGE_LENGTH] if idx == 0 and text_chunks else None
            resp = await api.send_media(
                target_type,
                target_id,
                up["file_info"],
                content=caption,
                msg_seq=idx + 1,
            )
            last_msg_id = resp.get("id")

        # --- Step 3: send remaining text chunks ---
        start = 1 if uploaded else 0
        for i, chunk in enumerate(text_chunks[start:]):
            resp = await api.send_text(
                target_type,
                target_id,
                chunk,
                msg_seq=len(uploaded) + i + 1,
            )
            last_msg_id = resp.get("id")

        return {
            "success": True,
            "platform": "qqbot",
            "chat_id": chat_id,
            "message_id": last_msg_id,
        }

    except Exception as exc:
        return {"error": f"QQBot standalone send failed: {exc}"}
    finally:
        if http_client is not None:
            try:
                await http_client.aclose()
            except Exception:
                pass
