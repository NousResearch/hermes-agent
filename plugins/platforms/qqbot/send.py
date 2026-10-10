"""Out-of-process QQ Bot sender (cron / CLI). Text tries guild, then C2C, then group.
Media uploads through C2C then group; guild channels have no native media path.
"""

from __future__ import annotations

import asyncio
import logging
import os
import time

from tools.send_message_senders import (
    _AUDIO_EXTS, _IMAGE_EXTS, _VIDEO_EXTS, _VOICE_EXTS, _error, _success)

logger = logging.getLogger("tools.send_message_tool")


def _qqbot_media_file_type(media_path: str, is_voice: bool) -> int:
    """Map a local path / voice flag to QQ Bot ``file_type`` constants."""
    from .constants import (
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


def _is_sdk_client(client) -> bool:
    return type(client).__name__ == "QQApiClient"


async def _qqbot_api_json(client, headers: dict, method: str, path: str,
                          body: dict | None = None, *, timeout: float = 30.0) -> dict:
    """POST/GET JSON against ``api.sgroup.qq.com``; raise RuntimeError on failure."""
    if _is_sdk_client(client):
        data = await client.request(method, path, body, timeout=timeout)
        return data if isinstance(data, dict) else {}
    from .constants import API_BASE

    resp = await client.request(method, f"{API_BASE}{path}", json=body, headers=headers, timeout=timeout)
    try:
        data = resp.json() if resp.content else {}
    except Exception:
        data = {}
    if resp.status_code >= 400:
        raise RuntimeError(
            f"QQ Bot API error [{resp.status_code}] {path}: "
            f"{data.get('message', data) or resp.text[:200]}")
    return data if isinstance(data, dict) else {}


async def _qqbot_upload_local_file(client, headers, chat_type, chat_id, media_path, file_type) -> dict:
    """Chunked-upload a local file; returns the ``/files`` complete response."""
    from pathlib import Path as _Path
    from .chunked_upload import ChunkedUploader
    from .constants import FILE_UPLOAD_TIMEOUT

    local_path = _Path(media_path).expanduser()
    if not local_path.is_absolute():
        local_path = (_Path.cwd() / local_path).resolve()
    if not local_path.exists() or not local_path.is_file():
        raise FileNotFoundError(f"Media file not found: {local_path}")

    if _is_sdk_client(client):
        from qqbot_agent_sdk import MediaUploader

        file_info = await MediaUploader(
            client, getattr(client, "_http_client", None), log_tag="send_message.qqbot",
        ).upload(chat_type, chat_id, str(local_path), file_type, local_path.name)
        return {"file_info": file_info}

    async def _api_request(method, path, body=None, timeout=FILE_UPLOAD_TIMEOUT):
        return await _qqbot_api_json(client, headers, method, path, body, timeout=timeout)

    uploader = ChunkedUploader(api_request=_api_request, http_put=client.put, log_tag="send_message.qqbot")
    return await uploader.upload(
        chat_type=chat_type, target_id=chat_id, file_path=str(local_path),
        file_type=file_type, file_name=local_path.name)


async def _qqbot_upload_url(client, headers, chat_type, chat_id, url, file_type,
                            file_name=None) -> dict:
    """Ask QQ to fetch ``url`` into the media store; returns upload JSON."""
    from .constants import FILE_UPLOAD_TIMEOUT, MEDIA_TYPE_FILE

    path = f"/v2/users/{chat_id}/files" if chat_type == "c2c" else f"/v2/groups/{chat_id}/files"
    body = {"file_type": file_type, "srv_send_msg": False, "url": url}
    if file_type == MEDIA_TYPE_FILE and file_name:
        body["file_name"] = file_name
    return await _qqbot_api_json(client, headers, "POST", path, body, timeout=FILE_UPLOAD_TIMEOUT)


def _qqbot_extract_file_info(upload: dict):
    if not isinstance(upload, dict):
        return None
    file_info = upload.get("file_info") or (upload.get("data") or {}).get("file_info")
    return file_info if file_info else None


async def _qqbot_send_media_message(client, headers, chat_type, chat_id, file_info,
                                    caption=None) -> dict:
    """POST a RichMedia message referencing an uploaded ``file_info``."""
    from .constants import MAX_MESSAGE_LENGTH, MSG_TYPE_MEDIA

    path = f"/v2/users/{chat_id}/messages" if chat_type == "c2c" else f"/v2/groups/{chat_id}/messages"
    body = {"msg_type": MSG_TYPE_MEDIA, "media": {"file_info": file_info},
            "msg_seq": int(time.time() * 1000) % 1_000_000_000}
    if caption and caption.strip():
        body["content"] = caption.strip()[:MAX_MESSAGE_LENGTH]
    return await _qqbot_api_json(client, headers, "POST", path, body)


async def _qqbot_send_text_message(client, headers, chat_id, message: str) -> dict:
    """Try channel → C2C → group text endpoints (pre-media standalone behavior)."""
    if _is_sdk_client(client):
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
    payload = {"content": (message or "")[:4000], "msg_type": 0}
    endpoints = (("channel", f"https://api.sgroup.qq.com/channels/{chat_id}/messages"),
                 ("c2c", f"https://api.sgroup.qq.com/v2/users/{chat_id}/messages"),
                 ("group", f"https://api.sgroup.qq.com/v2/groups/{chat_id}/messages"))
    statuses = []
    for kind, url in endpoints:
        resp = await client.post(url, json=payload, headers=headers)
        if resp.status_code in {200, 201}:
            return _success("qqbot", chat_id, message_id=resp.json().get("id"))
        statuses.append(f"{kind}={resp.status_code}")
    return _error(f"QQBot send failed: {' '.join(statuses)}")


async def _qqbot_deliver_one_media(client, headers, chat_id, media_path, is_voice,
                                   caption=None) -> dict:
    """Upload one attachment and send it, trying C2C then group (no guild)."""
    from pathlib import Path as _Path
    from urllib.parse import urlparse

    file_type = _qqbot_media_file_type(media_path, is_voice)
    is_url = urlparse(str(media_path)).scheme in {"http", "https"}
    errors: list[str] = []

    for chat_type in ("c2c", "group"):
        try:
            if is_url:
                upload = await _qqbot_upload_url(
                    client, headers, chat_type, chat_id, media_path, file_type,
                    file_name=_Path(urlparse(media_path).path).name or "media")
            else:
                upload = await _qqbot_upload_local_file(
                    client, headers, chat_type, chat_id, media_path, file_type)
            file_info = _qqbot_extract_file_info(upload)
            if not file_info:
                errors.append(f"{chat_type}: upload returned no file_info: {upload}")
                continue
            send_data = await _qqbot_send_media_message(
                client, headers, chat_type, chat_id, file_info, caption=caption)
            return _success("qqbot", chat_id, message_id=send_data.get("id"), chat_type=chat_type)
        except Exception as exc:
            errors.append(f"{chat_type}: {exc}")
            continue

    return _error(
        "QQBot media send failed (guild/channel native media is unsupported; "
        f"tried c2c and group): {'; '.join(errors)}")


async def send_qqbot(pconfig, chat_id, message, media_files=None, caption=None, *,
                     thread_id=None, force_document=False):
    """Send via the QQ Bot Open Platform REST API (no WebSocket needed).

    Guild channels, C2C (private) chats and groups are tried in order for text.
    When ``media_files`` is provided, each attachment is uploaded (C2C then
    group — guild has no native media upload path) and sent as a
    ``msg_type=MEDIA`` RichMedia message (#37315). Optional ``caption`` rides on
    the first media bubble; non-empty ``message`` is sent as a separate text
    message before the media.
    """
    try:
        import httpx
        from qqbot_agent_sdk import QQApiClient
    except ImportError:
        from pm.extras import install_hint
        return _error(f"QQBot direct send requires qqbot-agent-sdk. Run: {install_hint('qqbot')}")

    # Profile-scoped lookup so a multiplex profile never borrows another's QQ credentials.
    from gateway.platforms._shared import get_scoped_secret
    from .constants import FILE_UPLOAD_TIMEOUT
    extra = pconfig.extra or {}
    appid = extra.get("app_id") or get_scoped_secret("QQ_APP_ID", "")
    secret = pconfig.token or extra.get("client_secret") or get_scoped_secret("QQ_CLIENT_SECRET", "")
    if not appid or not secret:
        return _error("QQBot: QQ_APP_ID / QQ_CLIENT_SECRET not configured.")

    media_files = media_files or []
    # Longer timeout when uploading; text-only stays snappy.
    client_timeout = FILE_UPLOAD_TIMEOUT if media_files else 15.0

    try:
        async with httpx.AsyncClient(timeout=client_timeout) as http:
            client = QQApiClient(str(appid), str(secret), log_tag="send_message.qqbot")
            client.setup(http)
            headers = {}

            # --- Media path (#37315) ---
            if media_files:
                last_result = None
                text = (message or "").strip()
                warnings: list[str] = []
                if text and not (caption and caption.strip()):
                    text_result = await _qqbot_send_text_message(client, headers, chat_id, text)
                    if isinstance(text_result, dict) and text_result.get("error"):
                        return text_result
                    last_result = text_result

                for index, (media_path, is_voice) in enumerate(media_files):
                    if not media_path:
                        continue
                    # Caption applies to the first bubble only (single-file
                    # caption split already enforced by the caller).
                    media_caption = caption if index == 0 else None
                    if not await asyncio.to_thread(os.path.exists, media_path):
                        from urllib.parse import urlparse as _urlparse
                        if _urlparse(str(media_path)).scheme not in {"http", "https"}:
                            warnings.append(f"QQBot media file not found, skipping: {media_path}")
                            logger.warning(warnings[-1])
                            continue
                    result = await _qqbot_deliver_one_media(
                        client, headers, chat_id, media_path, bool(is_voice), caption=media_caption)
                    if isinstance(result, dict) and result.get("error"):
                        # Media failure must not discard the send (main degraded to text +
                        # omission warning): collect it and keep delivering.
                        warnings.append(result["error"])
                        logger.warning("QQBot media delivery failed, omitting attachment: %s",
                                       result["error"])
                        continue
                    last_result = result

                if last_result is None:
                    # Nothing was deliverable (caption rides the media, which all failed):
                    # degrade to the text alone, as the pre-media generic path did.
                    fallback_text = text or (caption or "").strip()
                    if not fallback_text:
                        return {"error": "QQBot: no deliverable media attachments",
                                **({"warnings": warnings} if warnings else {})}
                    text_result = await _qqbot_send_text_message(client, headers, chat_id, fallback_text)
                    if isinstance(text_result, dict) and text_result.get("error"):
                        return {**text_result, **({"warnings": warnings} if warnings else {})}
                    last_result = text_result
                if warnings and isinstance(last_result, dict):
                    last_result["warnings"] = [*last_result.get("warnings", []), *warnings]
                return last_result

            # --- Text-only path: first 2xx wins (pre-media behavior) ---
            return await _qqbot_send_text_message(client, headers, chat_id, message or "")
    except Exception as exc:
        logger.warning("QQBot send failed", exc_info=True)
        return _error(f"QQBot send failed: {exc}")




# Tests and older call sites use this name.
_send_qqbot = send_qqbot
