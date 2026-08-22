"""QQBot standalone (out-of-process) sender.

Used when no live ``QQAdapter`` is running in the process (CLI, cron,
isolated workers).  Creates a temporary ``QQApiClient`` backed by a
short-lived ``httpx.AsyncClient``, then sends text and/or media.

For raw OpenIDs (no ``c2c:``/``group:`` prefix), C2C is tried first;
on a **404 only**, the send is retried against the group endpoint.
Explicit-prefix targets, 401/403/429/timeout/5xx/non-JSON errors
never trigger fallback.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import httpx

from gateway.platforms.qqbot.constants import (
    MAX_MESSAGE_LENGTH,
    MEDIA_TYPE_FILE,
    MSG_TYPE_TEXT,
)
from gateway.platforms.qqbot.outbound import (
    QQApiClient,
    QQApiError,
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
    media_files: Optional[List[Any]] = None,
    force_document: bool = False,
) -> Dict[str, Any]:
    """Send text + media through the QQ Bot REST API (no WebSocket).

    Returns ``{"success": True, "message_id": ...}`` or ``{"error": "..."}``.
    """
    from gateway.platforms.qqbot.chunked_upload import (
        UploadDailyLimitExceededError,
        UploadFileTooLargeError,
    )

    extra = getattr(pconfig, "extra", None) or {}
    app_id = str(extra.get("app_id") or "").strip()
    secret = (
        extra.get("client_secret") or getattr(pconfig, "token", None)
        or ""
    ).strip()

    if not app_id or not secret:
        return {"error": "QQBot: QQ_APP_ID / QQ_CLIENT_SECRET not configured."}

    target_type, target_id, has_prefix = resolve_target(chat_id)
    if not target_id:
        return {"error": f"QQBot: empty target ID in chat_id '{chat_id}'"}

    if target_type == "guild":
        if media_files:
            return {
                "error": (
                    "QQBot MEDIA delivery to guild channels is not supported "
                    "by the QQ Bot API. Use C2C or group targets for media."
                )
            }
        # Fall through — text still works for guilds

    media_files = media_files or []

    # ── Classify media ────────────────────────────────────────────────
    media_items: List[Any] = []
    for media_path, _is_voice in media_files:
        mp = Path(media_path)
        if not mp.is_file():
            return {"error": f"Media file not found: {media_path}"}
        ft = classify_media_type(
            mp.suffix, is_voice=_is_voice, force_document=force_document,
        )
        media_items.append((media_path, ft))

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

    # ── Try send (with 404 fallback for raw OpenIDs) ─────────────────
    http_client = None
    last_error: Optional[Dict[str, Any]] = None
    try_types = [target_type]
    if not has_prefix and target_type == "c2c":
        # Raw OpenID — fallback to group on 404
        try_types = ["c2c", "group"]

    for attempt_idx, try_type in enumerate(try_types):
        if attempt_idx > 0:
            logger.info(
                "QQBot:standalone: raw target '%s' 404 on C2C, retrying as group",
                target_id,
            )
        try:
            result = await _do_send(
                app_id=app_id,
                secret=secret,
                chat_type=try_type,
                target_id=target_id,
                message=message,
                media_items=media_items,
                http_client=http_client,
            )
            if isinstance(result, dict) and result.get("success"):
                return result
            last_error = result
            # Don't fallback on non-404 errors
            break
        except QQApiError as e:
            if e.status_code == 404 and attempt_idx == 0 and len(try_types) > 1:
                # 404 on raw OpenID C2C → try group
                last_error = {"error": str(e), "_status": 404}
                continue
            last_error = {"error": str(e)}
            break
        except (httpx.TimeoutException, OSError, RuntimeError) as exc:
            last_error = {"error": f"QQBot send failed: {exc}"}
            break
        except Exception:
            logger.exception("QQBot:standalone: unexpected error")
            last_error = {"error": "QQBot send failed with unexpected error"}
            break

    return last_error or {"error": "QQBot send failed: unknown error"}


async def _do_send(
    *,
    app_id: str,
    secret: str,
    chat_type: str,
    target_id: str,
    message: str,
    media_items: List[Any],
    http_client: Optional[httpx.AsyncClient] = None,
) -> Dict[str, Any]:
    """Core send logic for one target type.  Manages its own ``http_client`` life."""
    from gateway.platforms.qqbot.chunked_upload import (
        UploadDailyLimitExceededError,
        UploadFileTooLargeError,
    )

    # ── Chunk text ─────────────────────────────────────────────────────
    text_chunks: List[str] = []
    if message.strip():
        text_chunks = split_for_qq(message, MAX_MESSAGE_LENGTH)

    own_client = http_client is None
    client = http_client
    api: Optional[QQApiClient] = None
    try:
        if own_client:
            client = httpx.AsyncClient(timeout=60.0)
        api = QQApiClient(
            app_id, secret, client, log_tag="QQBot:standalone",
        )

        # --- Upload media ---
        uploaded: List[Dict[str, Any]] = []
        for media_path, file_type in media_items:
            try:
                complete = await api.upload_local_file(
                    chat_type=chat_type,
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
                return {
                    "error": f"QQBot file upload failed ({Path(media_path).name}): {exc}"
                }

            fi = complete.get("file_info") or (
                complete.get("data", {}) or {}
            ).get("file_info")
            if not fi:
                return {
                    "error": f"QQBot: no file_info for {Path(media_path).name}"
                }
            uploaded.append({"file_info": fi, "file_type": file_type})

        # --- Send media or text ---
        last_msg_id: Optional[str] = None

        if uploaded:
            for idx, up in enumerate(uploaded):
                caption = (
                    text_chunks[0][:MAX_MESSAGE_LENGTH]
                    if idx == 0 and text_chunks
                    else None
                )
                resp = await api.send_media(
                    chat_type,
                    target_id,
                    up["file_info"],
                    content=caption,
                    msg_seq=idx + 1,
                )
                last_msg_id = str(resp.get("id", ""))

            # Send remaining text chunks (if any)
            for chunk in text_chunks if uploaded else text_chunks[:1]:
                if last_msg_id and chunk == text_chunks[0]:
                    continue  # Already sent as caption
                resp = await api.send_text(
                    chat_type, target_id, chunk, msg_seq=1,
                )
                last_msg_id = last_msg_id or str(resp.get("id", ""))
        else:
            # Text only
            for chunk in text_chunks:
                resp = await api.send_text(
                    chat_type, target_id, chunk, msg_seq=1,
                )
                last_msg_id = last_msg_id or str(resp.get("id", ""))

        return {"success": True, "message_id": last_msg_id}

    finally:
        if own_client and client is not None:
            await client.aclose()
        # _api just holds a reference to client — no separate cleanup needed
