"""Out-of-process QQ Bot sender (cron / CLI).

Text, media upload, and rich-media send all go through ``qqbot-agent-sdk``.
Guild channels, C2C chats, and groups are tried in order for text. Media
uploads try C2C then group — guild has no native media path.
"""

from __future__ import annotations

import asyncio
import logging
import os

from tools.send_message_senders import (
    _AUDIO_EXTS, _IMAGE_EXTS, _VIDEO_EXTS, _VOICE_EXTS, _error, _success)

logger = logging.getLogger("tools.send_message_tool")


def _qqbot_media_file_type(media_path: str, is_voice: bool) -> int:
    """Map a local path / voice flag to QQ Bot ``file_type`` constants."""
    from qqbot_agent_sdk import (
        MEDIA_TYPE_FILE, MEDIA_TYPE_IMAGE, MEDIA_TYPE_VIDEO, MEDIA_TYPE_VOICE)

    ext = os.path.splitext(media_path)[1].lower()
    if is_voice or ext in _VOICE_EXTS:
        return MEDIA_TYPE_VOICE
    if ext in _IMAGE_EXTS:
        return MEDIA_TYPE_IMAGE
    if ext in _VIDEO_EXTS:
        return MEDIA_TYPE_VIDEO
    if ext in _AUDIO_EXTS:
        return MEDIA_TYPE_VOICE
    return MEDIA_TYPE_FILE


def _media_uploader(client):
    from qqbot_agent_sdk import MediaUploader

    return MediaUploader(
        client, getattr(client, "_http_client", None), log_tag="send_message.qqbot")


async def _qqbot_send_media_message(client, chat_type, chat_id, file_info, caption=None) -> dict:
    """Send a rich-media message that references an uploaded ``file_info``."""
    from qqbot_agent_sdk import MAX_MESSAGE_LENGTH, MediaInfo, MessageToCreate, QQApiClient, QQMessageType

    text = caption.strip()[:MAX_MESSAGE_LENGTH] if caption and caption.strip() else ""
    msg = MessageToCreate(
        msg_type=int(QQMessageType.RICH_MEDIA),
        msg_seq=QQApiClient.next_msg_seq(chat_id),
        media=MediaInfo(file_info=file_info),
        content=text,
    )
    poster = client.post_group_message if chat_type == "group" else client.post_c2c_message
    data = await poster(chat_id, msg)
    return data if isinstance(data, dict) else {}


async def _qqbot_send_text_message(client, chat_id, message: str) -> dict:
    """Try guild, then C2C, then group via ``QQApiClient.send_text``."""
    statuses = []
    for kind in ("guild", "c2c", "group"):
        try:
            data = await client.send_text(
                kind, chat_id, message or "", markdown=False, retries=1)
            return _success("qqbot", chat_id, message_id=(data or {}).get("id"))
        except Exception as exc:
            logger.debug("QQ text send via %s failed", kind, exc_info=True)
            statuses.append(f"{kind}={exc}")
    return _error(f"QQBot send failed: {' '.join(statuses)}")


async def _qqbot_deliver_one_media(client, chat_id, media_path, is_voice, caption=None) -> dict:
    """Upload one attachment and send it, trying C2C then group (no guild)."""
    from pathlib import Path as _Path
    from urllib.parse import urlparse

    file_type = _qqbot_media_file_type(media_path, is_voice)
    file_name = _Path(urlparse(str(media_path)).path).name or _Path(str(media_path)).name or "media"
    errors: list[str] = []
    uploader = _media_uploader(client)

    for chat_type in ("c2c", "group"):
        try:
            file_info = await uploader.upload(
                chat_type, chat_id, str(media_path), file_type, file_name)
            send_data = await _qqbot_send_media_message(
                client, chat_type, chat_id, file_info, caption=caption)
            return _success(
                "qqbot", chat_id, message_id=send_data.get("id"), chat_type=chat_type)
        except Exception as exc:
            errors.append(f"{chat_type}: {exc}")
            continue

    return _error(
        "QQBot media send failed (guild/channel native media is unsupported; "
        f"tried c2c and group): {'; '.join(errors)}")


async def send_qqbot(pconfig, chat_id, message, media_files=None, caption=None, *,
                     thread_id=None, force_document=False):
    """Send via qqbot-agent-sdk (no WebSocket needed).

    When ``media_files`` is provided, each attachment is uploaded (C2C then
    group) and sent as a rich-media message. Optional ``caption`` rides on the
    first media bubble; non-empty ``message`` is sent as a separate text
    message before the media.
    """
    del thread_id, force_document
    try:
        import httpx
        from qqbot_agent_sdk import QQApiClient

        from .sdk_bridge import apply_sdk_identity
    except ImportError:
        from pm import install_hint
        return _error(f"QQBot direct send requires qqbot-agent-sdk. Run: {install_hint('qqbot')}")

    from gateway.platforms._shared import get_scoped_secret
    extra = pconfig.extra or {}
    appid = extra.get("app_id") or get_scoped_secret("QQ_APP_ID", "")
    secret = pconfig.token or extra.get("client_secret") or get_scoped_secret("QQ_CLIENT_SECRET", "")
    if not appid or not secret:
        return _error("QQBot: QQ_APP_ID / QQ_CLIENT_SECRET not configured.")

    apply_sdk_identity()
    media_files = media_files or []
    client_timeout = 120.0 if media_files else 15.0

    try:
        async with httpx.AsyncClient(timeout=client_timeout) as http:
            client = QQApiClient(str(appid), str(secret), log_tag="send_message.qqbot")
            client.setup(http)

            if media_files:
                last_result = None
                text = (message or "").strip()
                warnings: list[str] = []
                if text and not (caption and caption.strip()):
                    text_result = await _qqbot_send_text_message(client, chat_id, text)
                    if isinstance(text_result, dict) and text_result.get("error"):
                        return text_result
                    last_result = text_result

                for index, (media_path, is_voice) in enumerate(media_files):
                    if not media_path:
                        continue
                    media_caption = caption if index == 0 else None
                    if not await asyncio.to_thread(os.path.exists, media_path):
                        from urllib.parse import urlparse as _urlparse
                        if _urlparse(str(media_path)).scheme not in {"http", "https"}:
                            warnings.append(f"QQBot media file not found, skipping: {media_path}")
                            logger.warning(warnings[-1])
                            continue
                    result = await _qqbot_deliver_one_media(
                        client, chat_id, media_path, bool(is_voice), caption=media_caption)
                    if isinstance(result, dict) and result.get("error"):
                        warnings.append(result["error"])
                        logger.warning(
                            "QQBot media delivery failed, omitting attachment: %s", result["error"])
                        continue
                    last_result = result

                if last_result is None:
                    fallback_text = text or (caption or "").strip()
                    if not fallback_text:
                        return {"error": "QQBot: no deliverable media attachments",
                                **({"warnings": warnings} if warnings else {})}
                    text_result = await _qqbot_send_text_message(client, chat_id, fallback_text)
                    if isinstance(text_result, dict) and text_result.get("error"):
                        return {**text_result, **({"warnings": warnings} if warnings else {})}
                    last_result = text_result
                if warnings and isinstance(last_result, dict):
                    last_result["warnings"] = [*last_result.get("warnings", []), *warnings]
                return last_result

            return await _qqbot_send_text_message(client, chat_id, message or "")
    except Exception as e:
        logger.debug("QQBot standalone send failed", exc_info=True)
        return _error(f"QQBot send failed: {e}")
