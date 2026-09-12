"""Out-of-process Discord REST delivery for cron jobs."""

from __future__ import annotations

import inspect
import json
import logging
import os
from typing import Any, Dict, Optional, Tuple

from .. import adapter as _adapter

logger = _adapter.logger
_DISCORD_STANDALONE_ERROR_BODY_LIMIT_BYTES = 8 * 1024
_DISCORD_STANDALONE_JSON_BODY_LIMIT_BYTES = 1 * 1024 * 1024


def _derive_forum_thread_name(message: str) -> str:
    return _adapter._derive_forum_thread_name(message)


def _probe_is_forum_cached(chat_id: str):
    return _adapter._probe_is_forum_cached(chat_id)


def _remember_channel_is_forum(chat_id: str, is_forum: bool) -> None:
    _adapter._remember_channel_is_forum(chat_id, is_forum)


def _standalone_sanitize_error(text) -> str:
    s = str(text)
    import re
    return re.sub(r"(Authorization:\s*Bot\s+)\S+", r"\1***", s, flags=re.IGNORECASE)
def _standalone_sanitize_error(text) -> str:
    """Local copy of tools.send_message_tool._sanitize_error_text (strips bot tokens); avoids hard dep."""
    s = str(text)
    import re as _re_san
    return _re_san.sub(r"(Authorization:\s*Bot\s+)\S+", r"\1***", s, flags=_re_san.IGNORECASE)


def _standalone_close_response(resp: Any) -> None:
    close = getattr(resp, "close", None)
    if callable(close):
        close()
        return
    release = getattr(resp, "release", None)
    if callable(release):
        release()


async def _standalone_read_response_bytes_limited(
    resp: Any, limit_bytes: int,
) -> Tuple[Optional[bytes], bool]:
    """Read at most *limit_bytes*; returns ``(body, truncated)``. ``(None, False)`` when the object
    has no streaming ``content.read`` coroutine (proxy/test double) — callers use ``json()``/``text()``."""
    content = getattr(resp, "content", None)
    read = getattr(content, "read", None)
    if content is None or not inspect.iscoroutinefunction(read):
        return None, False
    try:
        chunks: list[bytes] = []
        total = 0
        while total <= limit_bytes:
            chunk = await read(limit_bytes + 1 - total)
            if not chunk:
                break
            if isinstance(chunk, str):
                chunk = chunk.encode("utf-8", "replace")
            total += len(chunk)
            chunks.append(chunk)
            if total > limit_bytes:
                _standalone_close_response(resp)
                return b"".join(chunks)[:limit_bytes], True
        return b"".join(chunks), False
    except (TypeError, AttributeError):
        # Quacked like a stream but wasn't — caller uses native json()/text().
        return None, False


def _standalone_response_encoding(resp: Any) -> str:
    get_encoding = getattr(resp, "get_encoding", None)
    if callable(get_encoding):
        try:
            return get_encoding() or "utf-8"
        except Exception:
            return "utf-8"
    return "utf-8"


async def _standalone_read_text_limited(resp: Any, limit_bytes: int) -> str:
    body, _truncated = await _standalone_read_response_bytes_limited(resp, limit_bytes)
    if body is None:
        return await resp.text()
    return body.decode(_standalone_response_encoding(resp), "replace")


async def _standalone_read_json_limited(resp: Any, limit_bytes: int) -> dict:
    body, truncated = await _standalone_read_response_bytes_limited(resp, limit_bytes)
    if body is None:
        return await resp.json()
    if truncated:
        raise ValueError(f"Discord API JSON response exceeds {limit_bytes} bytes")
    if not body:
        return {}
    data = json.loads(body.decode(_standalone_response_encoding(resp), "replace"))
    return data if isinstance(data, dict) else {}


def _standalone_warn_missing_media(media_path: str) -> str:
    warning = f"Media file not found, skipping: {media_path}"
    logger.warning(warning)
    return warning


async def _standalone_response_json_or_error(resp: Any, error_prefix: str):
    """``(data, None)`` for a 200/201 JSON response, else ``(None, {"error": ...})``
    with the (size-capped) body text appended to ``error_prefix``."""
    if resp.status not in {200, 201}:
        body = await _standalone_read_text_limited(resp, _DISCORD_STANDALONE_ERROR_BODY_LIMIT_BYTES)
        return None, {"error": f"{error_prefix} ({resp.status}): {body}"}
    return await _standalone_read_json_limited(resp, _DISCORD_STANDALONE_JSON_BODY_LIMIT_BYTES), None


async def _standalone_is_forum(aiohttp, chat_id: str, json_headers: dict, sess_kw: dict, req_kw: dict) -> bool:
    """Forum detection: channel directory → process-local probe cache → memoized ``GET /channels/{id}``."""
    _channel_type = None
    try:
        from gateway.channel_directory import lookup_channel_type
        _channel_type = lookup_channel_type("discord", chat_id)
    except Exception:
        pass
    if _channel_type is not None:
        return _channel_type == "forum"
    cached = _probe_is_forum_cached(chat_id)
    if cached is not None:
        return cached
    is_forum = False
    try:
        info_url = f"https://discord.com/api/v10/channels/{chat_id}"
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=15), **sess_kw) as info_sess:
            async with info_sess.get(info_url, headers=json_headers, **req_kw) as info_resp:
                if info_resp.status == 200:
                    info = await _standalone_read_json_limited(info_resp, _DISCORD_STANDALONE_JSON_BODY_LIMIT_BYTES)
                    is_forum = info.get("type") == 15
                    _remember_channel_is_forum(chat_id, is_forum)
    except Exception:
        logger.debug("Failed to probe channel type for %s", chat_id, exc_info=True)
    return is_forum


async def _standalone_send(
    pconfig, chat_id: str, message: str, *, thread_id: Optional[str] = None,
    media_files: Optional[list] = None, force_document: bool = False, caption: Optional[str] = None,
) -> Dict[str, Any]:
    """Send via Discord REST without a live gateway adapter (token: ``pconfig.token`` then env var).
    Forum channels (type 15) reject ``POST /messages``, so a thread post is created via
    ``POST /channels/{id}/threads`` with media as multipart attachments. Channel type: directory
    cache → process-local probe cache → memoized GET. ``force_document`` accepted but unused."""
    try:
        import aiohttp
    except ImportError:
        return {"error": "aiohttp not installed. Run: pip install aiohttp"}
    token = (getattr(pconfig, "token", None) or "").strip()
    if not token:
        # Profile-scoped read: under multiplex the env may hold another profile's token.
        from agent.secret_scope import get_secret
        token = (get_secret("DISCORD_BOT_TOKEN", "") or "").strip()
    if not token:
        return {"error": "Discord standalone send: DISCORD_BOT_TOKEN is not set"}
    try:
        from gateway.platforms.base import resolve_proxy_url, proxy_kwargs_for_aiohttp
        _proxy = resolve_proxy_url(platform_env_var="DISCORD_PROXY")
        _sess_kw, _req_kw = proxy_kwargs_for_aiohttp(_proxy)
        auth_headers = {"Authorization": f"Bot {token}"}
        json_headers = {**auth_headers, "Content-Type": "application/json"}
        media_files = media_files or []
        last_data = None
        warnings = []
        if thread_id:
            url = f"https://discord.com/api/v10/channels/{thread_id}/messages"
        else:
            # Forum channels (type 15) reject POST /messages — create a thread post.
            if await _standalone_is_forum(aiohttp, chat_id, json_headers, _sess_kw, _req_kw):
                thread_name = _derive_forum_thread_name(message)
                thread_url = f"https://discord.com/api/v10/channels/{chat_id}/threads"
                # Filter readable media first to pick JSON vs multipart before opening a session.
                valid_media = []
                for media_path, _is_voice in media_files:
                    if not os.path.exists(media_path):
                        warnings.append(_standalone_warn_missing_media(media_path))
                        continue
                    valid_media.append(media_path)
                async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=60), **_sess_kw) as session:
                    if valid_media:
                        # Multipart payload_json + files[N]: thread + starter + attachments in one call.
                        attachments_meta = [
                            {"id": str(idx), "filename": os.path.basename(path)}
                            for idx, path in enumerate(valid_media)
                        ]
                        starter_message = {"content": (caption or message), "attachments": attachments_meta}
                        payload_json = json.dumps({"name": thread_name, "message": starter_message})
                        form = aiohttp.FormData()
                        form.add_field("payload_json", payload_json, content_type="application/json")
                        try:
                            for idx, media_path in enumerate(valid_media):
                                with open(media_path, "rb") as fh:
                                    form.add_field(
                                        f"files[{idx}]", fh.read(),
                                        filename=os.path.basename(media_path),
                                    )
                            async with session.post(thread_url, headers=auth_headers, data=form, **_req_kw) as resp:
                                data, err = await _standalone_response_json_or_error(resp, "Discord forum thread creation error")
                                if err:
                                    return err
                        except Exception as e:
                            return {"error": _standalone_sanitize_error(f"Discord forum thread upload failed: {e}")}
                    else:
                        # No media: JSON POST creates the thread with the text starter.
                        async with session.post(
                            thread_url, headers=json_headers,
                            json={"name": thread_name, "message": {"content": message}}, **_req_kw,
                        ) as resp:
                            data, err = await _standalone_response_json_or_error(resp, "Discord forum thread creation error")
                            if err:
                                return err
                thread_id_created = data.get("id")
                starter_msg_id = (data.get("message") or {}).get("id", thread_id_created)
                result = {
                    "success": True, "platform": "discord", "chat_id": chat_id,
                    "thread_id": thread_id_created, "message_id": starter_msg_id,
                }
                if warnings:
                    result["warnings"] = warnings
                return result
            url = f"https://discord.com/api/v10/channels/{chat_id}/messages"
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=30), **_sess_kw) as session:
            if message.strip() or not media_files:
                async with session.post(url, headers=json_headers, json={"content": message}, **_req_kw) as resp:
                    last_data, err = await _standalone_response_json_or_error(resp, "Discord API error")
                    if err:
                        return err
            # One multipart upload per file; a MEDIA:<path> caption rides as the attachment message's
            # content, and caption_pending makes a missing file fall back to a plain message.
            caption_pending = bool(caption)
            for media_path, _is_voice in media_files:
                if not os.path.exists(media_path):
                    warnings.append(_standalone_warn_missing_media(media_path))
                    if caption_pending:
                        try:
                            async with session.post(
                                url, headers=json_headers, json={"content": caption}, **_req_kw,
                            ) as resp:
                                if resp.status in {200, 201}:
                                    last_data = await _standalone_read_json_limited(
                                        resp, _DISCORD_STANDALONE_JSON_BODY_LIMIT_BYTES,
                                    )
                                    caption_pending = False
                        except Exception:
                            logger.warning("Discord caption-fallback send failed for missing media")
                    continue
                try:
                    form = aiohttp.FormData()
                    filename = os.path.basename(media_path)
                    if caption_pending:
                        form.add_field(
                            "payload_json", json.dumps({"content": caption}),
                            content_type="application/json",
                        )
                        caption_pending = False
                    with open(media_path, "rb") as f:
                        form.add_field("files[0]", f, filename=filename)
                        async with session.post(url, headers=auth_headers, data=form, **_req_kw) as resp:
                            data, err = await _standalone_response_json_or_error(resp, "Discord API error")
                            if err:
                                warning = _standalone_sanitize_error(f"Failed to send media {media_path}: {err['error']}")
                                logger.error(warning)
                                warnings.append(warning)
                                continue
                            last_data = data
                except Exception as e:
                    warning = _standalone_sanitize_error(f"Failed to send media {media_path}: {e}")
                    logger.error(warning)
                    warnings.append(warning)
        if last_data is None:
            error = "No deliverable text or media remained after processing"
            if warnings:
                return {"error": error, "warnings": warnings}
            return {"error": error}
        result = {"success": True, "platform": "discord", "chat_id": chat_id, "message_id": last_data.get("id")}
        if warnings:
            result["warnings"] = warnings
        return result
    except Exception as e:
        # Include the exception type: str(TimeoutError()) is empty.
        logger.error("Discord standalone send failed", exc_info=True)
        return {"error": _standalone_sanitize_error(f"Discord send failed: {type(e).__name__}: {e}")}
