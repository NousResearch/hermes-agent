"""Discord standalone (out-of-process) REST sender.

Extracted from ``plugins/platforms/discord/adapter.py`` (god-file slice R5-S1,
epic #78647) - the adapter's C4 cluster, moved byte-verbatim.  The adapter
re-exports every name below so ``register()`` wiring and existing importers
keep resolving through ``plugins.platforms.discord.adapter``.
"""

import asyncio
import inspect
import json
import logging
import os
from typing import Any, Dict, Optional, Tuple

logger = logging.getLogger(__name__)

# ── Standalone (out-of-process) sender ────────────────────────────────────────
# Used by ``tools/send_message_tool._send_via_adapter`` when the gateway runner
# is not in this process (e.g. ``hermes cron`` running standalone) and no live
# DiscordAdapter instance is available.  Implements the same forum/thread/
# multipart logic the live adapter would use, via Discord's REST API directly.
#
# This block was previously hosted in ``tools/send_message_tool.py`` as
# ``_send_discord``.  It moved into the plugin so all Discord-specific HTTP
# logic lives next to the adapter — same shape as Teams' ``_standalone_send``.

# Process-local cache for Discord channel-type probes.  Avoids re-probing the
# same channel on every send when the directory cache has no entry (e.g. fresh
# install, or channel created after the last directory build).
_DISCORD_CHANNEL_TYPE_PROBE_CACHE: Dict[str, bool] = {}
_DISCORD_STANDALONE_JSON_BODY_LIMIT_BYTES = 1 * 1024 * 1024
_DISCORD_STANDALONE_ERROR_BODY_LIMIT_BYTES = 8 * 1024


def _remember_channel_is_forum(chat_id: str, is_forum: bool) -> None:
    _DISCORD_CHANNEL_TYPE_PROBE_CACHE[str(chat_id)] = bool(is_forum)


def _probe_is_forum_cached(chat_id: str) -> Optional[bool]:
    return _DISCORD_CHANNEL_TYPE_PROBE_CACHE.get(str(chat_id))


def _derive_forum_thread_name(message: str) -> str:
    """Derive a thread name from the first line of the message, capped at 100 chars."""
    first_line = message.strip().split("\n", 1)[0].strip()
    # Strip common markdown heading prefixes
    first_line = first_line.lstrip("#").strip()
    if not first_line:
        first_line = "New Post"
    return first_line[:100]


def _standalone_sanitize_error(text) -> str:
    """Local copy of tools.send_message_tool._sanitize_error_text — strips bot
    tokens from any error payload before bubbling it up.  Inlined so the
    plugin doesn't introduce a hard dependency on send_message_tool internals.
    """
    s = str(text)
    # Mask anything that looks like a Bot token in an Authorization header.
    import re as _re_san
    return _re_san.sub(
        r"(Authorization:\s*Bot\s+)\S+",
        r"\1***",
        s,
        flags=_re_san.IGNORECASE,
    )


def _standalone_close_response(resp: Any) -> None:
    close = getattr(resp, "close", None)
    if callable(close):
        close()
        return
    release = getattr(resp, "release", None)
    if callable(release):
        release()


async def _standalone_read_response_bytes_limited(
    resp: Any,
    limit_bytes: int,
) -> Tuple[Optional[bytes], bool]:
    """Read at most *limit_bytes* from an aiohttp-style response body.

    Returns ``(body, truncated)``. Returns ``(None, False)`` when the response
    object does not expose a streaming ``content.read`` coroutine (e.g. a
    proxy wrapper or test double) — callers fall back to the object's own
    ``json()`` / ``text()`` in that case.
    """
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
        # Object quacked like a stream but wasn't one — let the caller use
        # its native json()/text() instead of failing the send.
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


async def _standalone_send(
    pconfig,
    chat_id: str,
    message: str,
    *,
    thread_id: Optional[str] = None,
    media_files: Optional[list] = None,
    force_document: bool = False,
    caption: Optional[str] = None,
) -> Dict[str, Any]:
    """Send via Discord REST API without a live gateway adapter.

    Used by ``tools/send_message_tool._send_via_adapter`` when the gateway
    runner is not in this process.  Reads ``DISCORD_BOT_TOKEN`` from
    ``pconfig.token`` (set by the gateway config loader from env) and falls
    back to the ``DISCORD_BOT_TOKEN`` env var.

    Forum channels (type 15) reject ``POST /messages`` — a thread post is
    created automatically via ``POST /channels/{id}/threads``.  Media files
    are uploaded as multipart attachments on the starter message of the new
    thread.  Channel type is resolved from the channel directory first, then
    a process-local probe cache, and only as a last resort with a live
    ``GET /channels/{id}`` probe (whose result is memoized).

    ``force_document`` is accepted for signature parity but unused — Discord
    treats every uploaded file as a generic attachment.
    """
    try:
        import aiohttp
    except ImportError:
        return {"error": "aiohttp not installed. Run: pip install aiohttp"}

    token = (getattr(pconfig, "token", None) or "").strip()
    if not token:
        # Profile-scoped read: under multiplex the process env may hold a
        # different profile's bot token, so honor the secret scope's verdict
        # (scoped miss ⇒ no token; unscoped multiplex ⇒ UnscopedSecretError).
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

        # Thread endpoint: Discord threads are channels; send directly to the thread ID.
        if thread_id:
            url = f"https://discord.com/api/v10/channels/{thread_id}/messages"
        else:
            # Check if the target channel is a forum channel (type 15).
            # Forum channels reject POST /messages — create a thread post instead.
            # Three-layer detection: directory cache → process-local probe
            # cache → GET /channels/{id} probe (with result memoized).
            _channel_type = None
            try:
                from gateway.channel_directory import lookup_channel_type
                _channel_type = lookup_channel_type("discord", chat_id)
            except Exception:
                pass

            if _channel_type == "forum":
                is_forum = True
            elif _channel_type is not None:
                is_forum = False
            else:
                cached = _probe_is_forum_cached(chat_id)
                if cached is not None:
                    is_forum = cached
                else:
                    is_forum = False
                    try:
                        info_url = f"https://discord.com/api/v10/channels/{chat_id}"
                        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=15), **_sess_kw) as info_sess:
                            async with info_sess.get(info_url, headers=json_headers, **_req_kw) as info_resp:
                                if info_resp.status == 200:
                                    info = await _standalone_read_json_limited(
                                        info_resp,
                                        _DISCORD_STANDALONE_JSON_BODY_LIMIT_BYTES,
                                    )
                                    is_forum = info.get("type") == 15
                                    _remember_channel_is_forum(chat_id, is_forum)
                    except Exception:
                        logger.debug("Failed to probe channel type for %s", chat_id, exc_info=True)

            if is_forum:
                thread_name = _derive_forum_thread_name(message)
                thread_url = f"https://discord.com/api/v10/channels/{chat_id}/threads"

                # Filter to readable media files up front so we can pick the
                # right code path (JSON vs multipart) before opening a session.
                valid_media = []
                for media_path, _is_voice in media_files:
                    if not os.path.exists(media_path):
                        warning = f"Media file not found, skipping: {media_path}"
                        logger.warning(warning)
                        warnings.append(warning)
                        continue
                    valid_media.append(media_path)

                async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=60), **_sess_kw) as session:
                    if valid_media:
                        # Multipart: payload_json + files[N] creates a forum
                        # thread with the starter message plus attachments in
                        # a single API call.
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
                                        f"files[{idx}]",
                                        fh.read(),
                                        filename=os.path.basename(media_path),
                                    )
                            async with session.post(thread_url, headers=auth_headers, data=form, **_req_kw) as resp:
                                if resp.status not in {200, 201}:
                                    body = await _standalone_read_text_limited(
                                        resp,
                                        _DISCORD_STANDALONE_ERROR_BODY_LIMIT_BYTES,
                                    )
                                    return {"error": f"Discord forum thread creation error ({resp.status}): {body}"}
                                data = await _standalone_read_json_limited(
                                    resp,
                                    _DISCORD_STANDALONE_JSON_BODY_LIMIT_BYTES,
                                )
                        except Exception as e:
                            return {"error": _standalone_sanitize_error(f"Discord forum thread upload failed: {e}")}
                    else:
                        # No media — simple JSON POST creates the thread with
                        # just the text starter.
                        async with session.post(
                            thread_url,
                            headers=json_headers,
                            json={
                                "name": thread_name,
                                "message": {"content": message},
                            },
                            **_req_kw,
                        ) as resp:
                            if resp.status not in {200, 201}:
                                body = await _standalone_read_text_limited(
                                    resp,
                                    _DISCORD_STANDALONE_ERROR_BODY_LIMIT_BYTES,
                                )
                                return {"error": f"Discord forum thread creation error ({resp.status}): {body}"}
                            data = await _standalone_read_json_limited(
                                resp,
                                _DISCORD_STANDALONE_JSON_BODY_LIMIT_BYTES,
                            )

                thread_id_created = data.get("id")
                starter_msg_id = (data.get("message") or {}).get("id", thread_id_created)
                result = {
                    "success": True,
                    "platform": "discord",
                    "chat_id": chat_id,
                    "thread_id": thread_id_created,
                    "message_id": starter_msg_id,
                }
                if warnings:
                    result["warnings"] = warnings
                return result

            url = f"https://discord.com/api/v10/channels/{chat_id}/messages"

        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=30), **_sess_kw) as session:
            # Send text message (skip if empty and media is present)
            if message.strip() or not media_files:
                async with session.post(url, headers=json_headers, json={"content": message}, **_req_kw) as resp:
                    if resp.status not in {200, 201}:
                        body = await _standalone_read_text_limited(
                            resp,
                            _DISCORD_STANDALONE_ERROR_BODY_LIMIT_BYTES,
                        )
                        return {"error": f"Discord API error ({resp.status}): {body}"}
                    last_data = await _standalone_read_json_limited(
                        resp,
                        _DISCORD_STANDALONE_JSON_BODY_LIMIT_BYTES,
                    )

            # Send each media file as a separate multipart upload. When a
            # MEDIA:<path> caption was supplied, ride it as the message content
            # on the attachment so it appears under the media bubble instead of
            # as a separate message. caption_pending tracks whether the caption
            # still needs delivering, so a missing file falls back to a plain
            # message rather than silently dropping the text.
            caption_pending = bool(caption)
            for media_path, _is_voice in media_files:
                if not os.path.exists(media_path):
                    warning = f"Media file not found, skipping: {media_path}"
                    logger.warning(warning)
                    warnings.append(warning)
                    if caption_pending:
                        try:
                            async with session.post(
                                url, headers=json_headers,
                                json={"content": caption}, **_req_kw,
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
                            "payload_json",
                            json.dumps({"content": caption}),
                            content_type="application/json",
                        )
                        caption_pending = False
                    with open(media_path, "rb") as f:
                        form.add_field("files[0]", f, filename=filename)
                        async with session.post(url, headers=auth_headers, data=form, **_req_kw) as resp:
                            if resp.status not in {200, 201}:
                                body = await _standalone_read_text_limited(
                                    resp,
                                    _DISCORD_STANDALONE_ERROR_BODY_LIMIT_BYTES,
                                )
                                warning = _standalone_sanitize_error(f"Failed to send media {media_path}: Discord API error ({resp.status}): {body}")
                                logger.error(warning)
                                warnings.append(warning)
                                continue
                            last_data = await _standalone_read_json_limited(
                                resp,
                                _DISCORD_STANDALONE_JSON_BODY_LIMIT_BYTES,
                            )
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
        return {"error": _standalone_sanitize_error(f"Discord send failed: {e}")}
