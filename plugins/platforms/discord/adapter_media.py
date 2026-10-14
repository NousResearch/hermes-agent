"""Discord native media uploads and delivery fallbacks."""
from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import os
import tempfile
import uuid
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from agent.i18n import t
from gateway.platforms.base import SendResult

logger = logging.getLogger("plugins.platforms.discord.adapter")

# Default Discord attachment cap for DMs / channels without a guild boost
# context. 20 MiB since the Sep 3 2026 API change (10 MiB before); guild
# channels expose a boost-raised limit via ``guild.filesize_limit``, but
# discord.py's fallback constant can lag the platform default, so the
# effective limit is never taken below this floor. See issue #50846 and
# https://docs.discord.com/developers/change-log (Sep 3, 2026).
_DISCORD_DEFAULT_UPLOAD_LIMIT_BYTES = 20 * 1024 * 1024


class DiscordMediaMixin:
    @staticmethod
    def _discord_upload_limit_bytes(channel: Any) -> int:
        """Return the effective Discord attachment size limit for *channel*.

        Prefer the guild's boost-aware ``filesize_limit`` when present; fall
        back to the platform default for DMs / group DMs without a guild.
        The guild value is floored at the platform default: discord.py's
        unboosted-tier constant can lag a platform-wide raise (10 MiB in
        2.7.1 vs the 20 MiB default since Sep 3 2026), and under-reporting
        makes the preflight reject files Discord would accept.
        """
        guild = getattr(channel, "guild", None)
        if guild is not None:
            limit = getattr(guild, "filesize_limit", None)
            if isinstance(limit, int) and limit > 0:
                return max(limit, _DISCORD_DEFAULT_UPLOAD_LIMIT_BYTES)
        return _DISCORD_DEFAULT_UPLOAD_LIMIT_BYTES

    async def _reject_oversized_upload(
        self, channel: Any, file_path: str, filename: str, *, caption: Optional[str] = None,
    ) -> Optional[SendResult]:
        """Preflight ``file_path`` against the channel's upload cap (#50846): a doomed
        ``413`` round-trip is skipped and the user gets a notice naming the size and the
        limit. Returns the failed result, or ``None`` when the file may be uploaded."""
        try:
            file_size = os.path.getsize(file_path)
        except OSError as exc:
            return SendResult(success=False, error=f"Cannot stat file {filename}: {exc}")
        limit = self._discord_upload_limit_bytes(channel)
        if file_size <= limit:
            return None
        size_mb = file_size / (1024 * 1024)
        limit_mb = limit / (1024 * 1024)
        error = f"File too large for Discord upload: {filename} is {size_mb:.1f} MB (limit {limit_mb:.0f} MB)"
        logger.warning("[%s] %s", self.name, error)
        notice = t(
            "platform.discord.media.upload_too_large",
            filename=filename, size_mb=f"{size_mb:.1f}", limit_mb=f"{limit_mb:.0f}")
        try:
            shown = self.warning_notifications_enabled()
            if shown:
                # Legacy: the notice is posted in-channel only (forum parents get no bare notice).
                if not self._is_forum_parent(channel):
                    await channel.send(content=notice)
            elif caption:
                # Hidden: the requested caption still travels, on the channel's native shape.
                if self._is_forum_parent(channel):
                    await self._send_to_forum(channel, caption)
                else:
                    await channel.send(content=caption)
        except Exception:
            logger.debug("[%s] Failed to send oversized-file notice for %s", self.name, filename, exc_info=True)
        return SendResult(success=False, error=error)

    async def _send_file_attachment(
        self, chat_id: str, file_path: str, caption: Optional[str] = None,
        file_name: Optional[str] = None, metadata: Optional[dict[str, Any]] = None,
    ) -> SendResult:
        """Send a local file as a Discord attachment (forum channels get a new thread). Path-based
        ``discord.File`` only: the open-handle form can race the multipart encoder after an image
        batch and yield zero attachments — a silent drop for video/document MEDIA tags.

        See #66797.
        """
        from plugins.platforms.discord.adapter import _prompt_target_id, discord

        if not self._client:
            return SendResult(success=False, error="Not connected")
        if not os.path.isfile(file_path):
            return SendResult(success=False, error=f"File not found: {file_path}")
        channel = await self._resolve_channel(_prompt_target_id(chat_id, metadata))
        if not channel:
            return SendResult(success=False, error=f"Channel {chat_id} not found")
        filename = file_name or os.path.basename(file_path)
        rejected = await self._reject_oversized_upload(channel, file_path, filename, caption=caption)
        if rejected is not None:
            return rejected
        logger.info(
            "[%s] Sending file attachment %s (%s) to %s", self.name, filename,
            os.path.splitext(filename)[1].lower() or "no-ext", chat_id,
        )
        # Path-based File (discord.py owns open/close); ``files=[...]`` over deprecated ``file=``.
        discord_file = discord.File(file_path, filename=filename)
        if self._is_forum_parent(channel):
            result = await self._forum_post_file(
                channel, content=(caption or "").strip(), files=[discord_file],
            )
            return result
        msg = await channel.send(content=caption if caption else None, files=[discord_file])
        attachments = getattr(msg, "attachments", None) or []
        if not attachments:
            # Discord accepted the message but attached nothing: fail loud instead of a silent drop.
            # Discord accepted the message but attached nothing — the failure mode reported in #66797 (MEDIA
            # video stripped from text, no attachment, no prior log line).
            logger.warning(
                "[%s] Discord returned message %s with no attachments for %s", self.name,
                getattr(msg, "id", "?"), filename,
            )
            return SendResult(
                success=False,
                error=f"Discord accepted the message but attached no files ({filename})",
                message_id=str(getattr(msg, "id", "") or "") or None,
            )
        return SendResult(success=True, message_id=str(msg.id))


    async def send_multiple_images(
        self, chat_id: str, images: list[tuple[str, str]],
        metadata: Optional[dict[str, Any]] = None, human_delay: float = 0.0,
    ) -> SendResult:
        """Send images as one Discord message (<=10 attachments): URLs are downloaded and uploaded
        inline (bare links don't render). Every chunk is verified post-send — Discord can accept a
        multipart POST and return a message with zero attachments (the #66797 silent drop) — and
        missing files are retried / topped up / notified instead of vanishing. Downloaded URLs are
        staged as temp files so every ``File`` in a batch is path-based (open-handle BytesIOs can
        race the multipart encoder); on chunk exception the remainder uses the per-image loop."""
        from plugins.platforms.discord.adapter import _prompt_target_id

        if not self._client:
            return SendResult(success=False, error="Not connected")
        if not images:
            return SendResult(success=False, error="no images to send")
        try:
            import discord  # importability gate; Files are built in _send_image_chunk
        except Exception:  # pragma: no cover
            return await super().send_multiple_images(chat_id, images, metadata, human_delay)
        try:
            channel = await self._resolve_channel(_prompt_target_id(chat_id, metadata))
            if not channel:
                logger.warning("[%s] Channel %s not found for multi-image send", self.name, chat_id)
                return SendResult(success=False, error=f"Channel {chat_id} not found")
        except Exception as e:
            logger.warning("[%s] Failed to resolve channel for multi-image send: %s", self.name, e)
            return await super().send_multiple_images(chat_id, images, metadata, human_delay)
        CHUNK = 10
        chunks = [images[i:i + CHUNK] for i in range(0, len(images), CHUNK)]
        delivered = False
        stage_dir = tempfile.TemporaryDirectory(prefix="hermes_discord_images_")
        try:
            for chunk_idx, chunk in enumerate(chunks):
                if human_delay > 0 and chunk_idx > 0:
                    await asyncio.sleep(human_delay)
                file_specs: list[tuple[str, str]] = []  # (local path, filename) — all path-based
                try:
                    file_specs, captions, skip_notices = await self._stage_chunk_files(
                        chunk, channel, stage_dir.name)
                    if not file_specs:
                        # Everything in this chunk was skipped. Still surface any
                        # oversized-file notices so the drop is not silent.
                        await self._notify_empty_chunk(channel, captions, skip_notices)
                        continue
                    # Use the first caption if any (Discord only has one message body for the group)
                    chunk_content = captions[0] if captions else None
                    if skip_notices:
                        chunk_content = self.warning_text(
                            "\n".join(([chunk_content] if chunk_content else []) + skip_notices), chunk_content)
                    logger.info(
                        "[%s] Sending %d image(s) as single Discord message (chunk %d/%d)",
                        self.name, len(file_specs), chunk_idx + 1, len(chunks),
                    )
                    if await self._send_image_chunk(channel, content=chunk_content, file_specs=file_specs):
                        delivered = True
                except Exception as e:
                    logger.warning(
                        "[%s] Multi-image Discord send failed (chunk %d/%d), retrying once after blip wait: %s",
                        self.name, chunk_idx + 1, len(chunks), e, exc_info=True,
                    )
                    # A transient transport blip (connection reset) would also sink the
                    # per-image fallback, which has no retry of its own — retry the chunk
                    # once after a short wait before degrading to it.
                    delivered = delivered or await self._retry_chunk_then_fallback(
                        channel, chat_id, chunk, file_specs, metadata, human_delay)
        finally:
            try:
                stage_dir.cleanup()
            except Exception:
                logger.debug("[%s] Discord image staging cleanup failed", self.name, exc_info=True)
        return SendResult(success=delivered, error=None if delivered else "all images failed to send")

    @staticmethod
    def _local_image_size(local_path: str) -> Optional[int]:
        """Size of a local image for preflight (runs in a worker thread): None if the file is
        missing, -1 if it exists but is unreadable, else its byte size."""
        if not os.path.exists(local_path):
            return None
        try:
            return os.path.getsize(local_path)
        except OSError:
            return -1

    @staticmethod
    def _stage_image_bytes(stage_dir: str, fname: str, data: bytes) -> tuple[str, str]:
        """Write downloaded image bytes to a temp file under *stage_dir* (sync: invoked via
        ``asyncio.to_thread`` from the async staging path — blocking ``open`` in an async def
        would stall the event loop)."""
        tmp = os.path.join(stage_dir, f"{uuid.uuid4().hex[:10]}_{fname}")
        with open(tmp, "wb") as f:
            f.write(data)
        return tmp, fname

    async def _stage_chunk_files(
        self, chunk: list[tuple[str, str]], channel: Any, stage_dir: str,
    ) -> tuple[list[tuple[str, str]], list[str], list[str]]:
        """Preflight and stage one chunk's images as local paths: ``file://`` entries are
        preflighted for existence/size (#50846: an oversized image would 413 the whole chunk),
        URLs are downloaded and staged as temp files (open-handle BytesIO Files can race the
        multipart encoder and yield zero attachments, #66797; path-based Files re-open cleanly
        and survive a retry). Returns ``(file_specs, captions, skip_notices)``; skipped
        oversized files land in ``skip_notices`` so the drop is never silent."""
        from plugins.platforms.discord.adapter import (
            _image_ext_from_content_type, _read_url_image_with_redirect_guard, is_safe_url,
        )
        from urllib.parse import unquote as _unquote

        file_specs: list[tuple[str, str]] = []
        captions: list[str] = []
        skip_notices: list[str] = []
        aiohttp_session = None
        try:
            for image_url, alt_text in chunk:
                if alt_text:
                    captions.append(alt_text)
                if image_url.startswith("file://"):
                    local_path = _unquote(image_url[7:])
                    _img_size = await asyncio.to_thread(self._local_image_size, local_path)
                    if _img_size is None:
                        logger.warning("[%s] Skipping missing image: %s", self.name, local_path)
                        continue
                    if _img_size < 0:
                        logger.warning("[%s] Skipping unreadable image %s", self.name, local_path)
                        continue
                    _img_limit = self._discord_upload_limit_bytes(channel)
                    if _img_size > _img_limit:
                        logger.warning(
                            "[%s] Skipping oversized image in batch: %s is %.1f MB (limit %.0f MB)",
                            self.name, os.path.basename(local_path),
                            _img_size / (1024 * 1024), _img_limit / (1024 * 1024),
                        )
                        skip_notices.append(t(
                            "platform.discord.media.image_skipped_too_large",
                            filename=os.path.basename(local_path),
                            size_mb=f"{_img_size / (1024 * 1024):.1f}",
                            limit_mb=f"{_img_limit / (1024 * 1024):.0f}"))
                        continue
                    file_specs.append((local_path, os.path.basename(local_path)))
                else:
                    if not is_safe_url(image_url):
                        logger.warning("[%s] Blocked unsafe image URL in batch", self.name)
                        continue
                    try:
                        import aiohttp as _aiohttp
                        from gateway.platforms.base import resolve_proxy_url, proxy_kwargs_for_aiohttp
                        _proxy = resolve_proxy_url(platform_env_var="DISCORD_PROXY")
                        _sess_kw, _req_kw = proxy_kwargs_for_aiohttp(_proxy)
                        if aiohttp_session is None:
                            aiohttp_session = _aiohttp.ClientSession(**_sess_kw)
                        status, data, headers = await _read_url_image_with_redirect_guard(
                            aiohttp_session, image_url,
                            timeout=_aiohttp.ClientTimeout(total=30), request_kwargs=_req_kw,
                        )
                        if status != 200:
                            logger.warning(
                                "[%s] Failed to download image (HTTP %d) in batch: %s",
                                self.name, status, image_url[:80],
                            )
                            continue
                        ext = _image_ext_from_content_type(headers.get("content-type", "image/png"))
                        _fname = f"image_{len(file_specs)}.{ext}"
                        file_specs.append(
                            await asyncio.to_thread(self._stage_image_bytes, stage_dir, _fname, data))
                    except Exception as dl_err:
                        logger.warning(
                            "[%s] Download failed for %s: %s",
                            self.name, image_url[:80], dl_err, exc_info=True,
                        )
                        continue
        finally:
            if aiohttp_session is not None:
                try:
                    await aiohttp_session.close()
                except Exception:
                    logger.debug("[%s] Failed to close image download session", self.name, exc_info=True)
        return file_specs, captions, skip_notices

    async def _notify_empty_chunk(
        self, channel: Any, captions: list[str], skip_notices: list[str],
    ) -> None:
        """Every image in a chunk was skipped: still surface any oversized-file notices so the
        drop is not silent (mirrors the per-image semantics)."""
        shown = self.warning_notifications_enabled()
        if not shown and captions:
            if self._is_forum_parent(channel):
                await self._send_to_forum(channel, captions[0])
            else:
                await channel.send(content=captions[0])
        if skip_notices and shown and not self._is_forum_parent(channel):
            try:
                await channel.send(content="\n".join(skip_notices))
            except Exception:
                logger.debug("[%s] Failed to send oversized-image notices", self.name, exc_info=True)

    async def _retry_chunk_then_fallback(
        self, channel: Any, chat_id: str, chunk: list[tuple[str, str]],
        file_specs: list[tuple[str, str]], metadata: Optional[dict[str, Any]], human_delay: float,
    ) -> bool:
        """A chunk exception is a transport blip: retry once after a short wait (fresh
        path-based Files), then degrade to the per-image fallback, which has no retry of its
        own. Returns True when the retry landed."""
        if file_specs:
            try:
                await asyncio.sleep(2.0)
                if await self._send_image_chunk(channel, content=None, file_specs=file_specs):
                    return True
            except Exception as retry_err:
                logger.warning(
                    "[%s] Multi-image retry failed, falling back to per-image: %s",
                    self.name, retry_err, exc_info=True,
                )
        fallback = await super().send_multiple_images(chat_id, chunk, metadata, human_delay=human_delay)
        return bool(fallback.success)

    async def _send_image_chunk(
        self, channel: Any, *, content: Optional[str], file_specs: list[tuple[str, str]],
    ) -> bool:
        """Send one image chunk (<=10 path-based files) and verify the response carries them.

        Discord can accept a multipart POST and return a message with zero attachments (the
        silent drop reported in #66797): an all-or-nothing result is retried once with fresh
        path-based ``File`` objects (the #66797 race only bit the open-handle form), a partial
        result is topped up with a follow-up message of just the missing files, and whatever is
        still missing after the retry notifies the user instead of vanishing silently (the text
        with stripped tags is already in the channel).

        Returns True when every file in the chunk landed as an attachment.
        """
        from plugins.platforms.discord.adapter import discord
        from gateway.platforms.base import _media_failure_text

        if not file_specs:
            return False
        if self._is_forum_parent(channel):
            # _forum_post_file verifies the starter's attachments and fails loud.
            result = await self._forum_post_file(
                channel, content=(content or "").strip(),
                files=[discord.File(path, filename=fn) for path, fn in file_specs],
            )
            return bool(result.success)

        def _arrived(msg: Any) -> list[str]:
            return [getattr(a, "filename", "") or "" for a in (getattr(msg, "attachments", None) or [])]

        def _missing(msg: Any, specs: list[tuple[str, str]]) -> list[tuple[str, str]]:
            pool = Counter(fn for _, fn in specs)
            for fn in _arrived(msg):
                if pool.get(fn, 0) > 0:
                    pool[fn] -= 1
            out = []
            for path, fn in specs:
                if pool.get(fn, 0) > 0:
                    pool[fn] -= 1
                    out.append((path, fn))
            return out

        body = content
        specs = list(file_specs)
        for attempt in (1, 2):
            msg = await channel.send(
                content=body if attempt == 1 else None,
                files=[discord.File(path, filename=fn) for path, fn in specs],
            )
            raw = getattr(msg, "attachments", None)
            if not isinstance(raw, (list, tuple)):
                # Non-standard return (not a Discord message — e.g. a stubbed
                # client in tests): we can't inspect the attachment list, so
                # don't fire the drop path; assume the POST delivered.
                logger.debug(
                    "[%s] send returned a non-message object for %s; assuming delivered",
                    self.name, [fn for _, fn in specs],
                )
                return True
            missing = _missing(msg, specs)
            if not missing:
                return True
            if attempt == 2:
                break
            if _arrived(msg):
                # Partial delivery: top up just the missing files.
                logger.warning(
                    "[%s] Discord attached %d of %d image(s) on message %s; sending %d missing: %s",
                    self.name, len(_arrived(msg)), len(specs), getattr(msg, "id", "?"),
                    len(missing), [fn for _, fn in missing],
                )
                specs = missing
            else:
                # All-or-nothing drop (the #66797 encoder race): retry the full chunk with
                # fresh Files; the caption rides only on the first attempt.
                logger.warning(
                    "[%s] Discord returned message %s with no attachments for %s; retrying once",
                    self.name, getattr(msg, "id", "?"), [fn for _, fn in specs],
                )
                specs = list(file_specs)
        # Still missing after the retry: fail loud — silence would be a silent drop.
        names = [fn for _, fn in specs]
        logger.error(
            "[%s] %d image attachment(s) never attached to Discord: %s",
            self.name, len(names), names,
        )
        try:
            if self.warning_notifications_enabled():
                await channel.send(
                    content=f"{_media_failure_text('image')} ({', '.join(names[:10])})")
        except Exception:
            logger.debug("[%s] Failed to send image-batch failure notice", self.name, exc_info=True)
        return False


    async def _send_single_file_verified(
        self, channel: Any, *, content: Optional[str], path: str, filename: str,
    ) -> bool:
        """Send one path-based ``file=`` and verify the response carries it.

        A zero-attachment response (the #66797 encoder race) is retried once with a fresh
        ``File``. No user notice is sent here — the caller decides the fallback (a URL link vs.
        a failure). Returns True only when the attachment actually landed.
        """
        from plugins.platforms.discord.adapter import discord

        for attempt in (1, 2):
            msg = await channel.send(
                content=content if attempt == 1 else None,
                file=discord.File(path, filename=filename),
            )
            raw = getattr(msg, "attachments", None)
            if not isinstance(raw, (list, tuple)):
                # Non-standard return: can't inspect, assume delivered.
                return True
            if any(getattr(a, "filename", "") == filename for a in raw):
                return True
            logger.warning(
                "[%s] Discord returned message %s with no attachment for %s; retrying once",
                self.name, getattr(msg, "id", "?"), filename,
            )
        logger.error("[%s] attachment never attached to Discord: %s", self.name, filename)
        return False


    async def send_voice(
        self, chat_id: str, audio_path: str, caption: Optional[str] = None,
        reply_to: Optional[str] = None, metadata: Optional[dict[str, Any]] = None, **kwargs,
    ) -> SendResult:
        """Send audio as a Discord file attachment."""
        from plugins.platforms.discord.adapter import _prompt_target_id, discord

        try:
            import io
            channel = await self._resolve_channel(_prompt_target_id(chat_id, metadata))
            if not channel:
                return SendResult(success=False, error=f"Channel {chat_id} not found")
            if not os.path.exists(audio_path):
                return SendResult(success=False, error=f"Audio file not found: {audio_path}")
            filename = os.path.basename(audio_path)
            rejected = await self._reject_oversized_upload(channel, audio_path, filename, caption=caption)
            if rejected is not None:
                return rejected
            reference = self._reply_reference_for_send(reply_to, channel)
            file_data = await asyncio.to_thread(Path(audio_path).read_bytes)
            # Forum channels reject POST /messages (native voice path too); create a thread post instead.
            if self._is_forum_parent(channel):
                forum_file = discord.File(io.BytesIO(file_data), filename=filename)
                return await self._forum_post_file(
                    channel, content=(caption or "").strip(), file=forum_file,
                )
            # Try sending as a native voice message via raw API (flags=8192).
            try:
                import base64
                try:
                    from mutagen.oggopus import OggOpus
                    duration_secs = OggOpus(audio_path).info.length
                except Exception:
                    duration_secs = max(1.0, len(file_data) / 2000.0)
                payload_data = {
                    "flags": 8192,
                    "attachments": [{
                        "id": "0", "filename": "voice-message.ogg", "duration_secs": round(duration_secs, 2),
                        "waveform": base64.b64encode(bytes([128] * 256)).decode(),
                    }],
                }
                if reference is not None:
                    payload_data["message_reference"] = {"message_id": str(reply_to), "fail_if_not_exists": False}
                form = [
                    {"name": "payload_json", "value": json.dumps(payload_data)},
                    {
                        "name": "files[0]", "value": file_data, "filename": "voice-message.ogg",
                        "content_type": "audio/ogg",
                    },
                ]
                msg_data = await self._client.http.request(
                    discord.http.Route("POST", "/channels/{channel_id}/messages", channel_id=channel.id),
                    form=form,
                )
                return SendResult(success=True, message_id=str(msg_data["id"]))
            except Exception as voice_err:
                logger.debug("Voice message flag failed, falling back to file: %s", voice_err)
                file = discord.File(io.BytesIO(file_data), filename=filename)
                try:
                    msg = await channel.send(file=file, reference=reference)
                except Exception as send_err:
                    if reference is not None and self._is_reply_reference_rejected(send_err):
                        msg = await channel.send(file=file, reference=None)
                    else:
                        raise
                return SendResult(success=True, message_id=str(msg.id))
        except Exception as e:  # pragma: no cover - defensive logging
            logger.error("[%s] Failed to send audio: %s", self.name, e, exc_info=True)
            return SendResult(success=False, error=str(e))


    async def _send_local_file(self, chat_id, path, caption, *, file_name=None, not_found: str, kind: str, metadata=None):
        """A failed attachment is a failed delivery, never a successful text notice."""
        try:
            return await self._send_file_attachment(chat_id, path, caption, file_name=file_name, metadata=metadata)
        except FileNotFoundError:
            return SendResult(success=False, error=f"{not_found}: {path}")
        except Exception as e:  # pragma: no cover - defensive logging
            logger.error("[%s] Failed to send %s: %s", self.name, kind, e, exc_info=True)
            return SendResult(success=False, error=str(e))


    async def send_image_file(
        self, chat_id: str, image_path: str, caption: Optional[str] = None,
        reply_to: Optional[str] = None, metadata: Optional[dict[str, Any]] = None,
    ) -> SendResult:
        """Send a local image file natively as a Discord file attachment."""
        return await self._send_local_file(
            chat_id, image_path, caption, not_found="Image file not found", kind="local image",
            metadata=metadata,
        )


    async def _send_url_media(
        self, chat_id: str, url: str, caption: Optional[str], *, kind: str,
        filename_for, fallback, metadata: Optional[dict], error_metadata: Optional[dict],
    ) -> SendResult:
        """Download ``url`` and post it as a native attachment (Discord renders those inline).
        ``fallback(metadata)`` is the base-adapter URL send (``error_metadata`` after download failure)."""
        from plugins.platforms.discord.adapter import _prompt_target_id, _read_url_image_with_redirect_guard, discord, is_safe_url

        if not self._client:
            return SendResult(success=False, error="Not connected")
        if not is_safe_url(url):
            logger.warning("[%s] Blocked unsafe %s URL during Discord send_%s", self.name, kind, kind)
            return await fallback(metadata)
        try:
            import aiohttp
            channel = await self._resolve_channel(_prompt_target_id(chat_id, metadata))
            if not channel:
                return SendResult(success=False, error=f"Channel {chat_id} not found")
            from gateway.platforms.base import resolve_proxy_url, proxy_kwargs_for_aiohttp
            _sess_kw, _req_kw = proxy_kwargs_for_aiohttp(resolve_proxy_url(platform_env_var="DISCORD_PROXY"))
            async with aiohttp.ClientSession(**_sess_kw) as session:
                status, data, headers = await _read_url_image_with_redirect_guard(
                    session, url, timeout=aiohttp.ClientTimeout(total=30), request_kwargs=_req_kw,
                )
                if status != 200:
                    raise Exception(f"Failed to download {kind}: HTTP {status}")
                fname = filename_for(headers)
                # Stage to a temp file: an open-handle (BytesIO) File can race the
                # multipart encoder and yield a zero-attachment message (#66797);
                # path-based Files re-open cleanly and survive a retry.
                _ext = fname.rsplit(".", 1)[-1] if "." in fname else "bin"
                with tempfile.NamedTemporaryFile(
                    prefix=f"hermes_{kind}_", suffix=f".{_ext}", delete=False,
                ) as _tf:
                    _tf.write(data)
                    staged = _tf.name
                try:
                    if self._is_forum_parent(channel):
                        # _forum_post_file verifies the starter's attachments.
                        return await self._forum_post_file(
                            channel, content=(caption or "").strip(),
                            file=discord.File(staged, filename=fname))
                    if await self._send_single_file_verified(
                        channel, content=caption, path=staged, filename=fname):
                        return SendResult(success=True)
                    # Confirmed drop: fall back to the original URL (renders as a link).
                    return await fallback(error_metadata)
                finally:
                    with contextlib.suppress(OSError):
                        os.unlink(staged)
        except ImportError:
            logger.warning("[%s] aiohttp not installed, falling back to URL. Run: pip install aiohttp", self.name, exc_info=True)
            return await fallback(error_metadata)
        except Exception as e:  # pragma: no cover - defensive logging
            logger.error("[%s] Failed to send %s attachment, falling back to URL: %s", self.name, kind, e, exc_info=True)
            return await fallback(error_metadata)


    async def send_image(
        self, chat_id: str, image_url: str, caption: Optional[str] = None,
        reply_to: Optional[str] = None, metadata: Optional[dict[str, Any]] = None,
    ) -> SendResult:
        """Send an image natively as a Discord file attachment."""
        from plugins.platforms.discord.adapter import _prompt_target_id, _image_ext_from_content_type

        return await self._send_url_media(
            chat_id, image_url, caption, kind="image",
            filename_for=lambda h: f"image.{_image_ext_from_content_type(h.get('content-type', 'image/png'))}",
            fallback=lambda md: super(DiscordMediaMixin, self).send_image(chat_id, image_url, caption, reply_to, metadata=md),
            metadata=metadata, error_metadata=metadata,
        )


    async def send_animation(
        self, chat_id: str, animation_url: str, caption: Optional[str] = None,
        reply_to: Optional[str] = None, metadata: Optional[dict[str, Any]] = None,
    ) -> SendResult:
        """Send an animated GIF natively as a Discord file attachment."""
        return await self._send_url_media(
            chat_id, animation_url, caption, kind="animation", filename_for=lambda _h: "animation.gif",
            fallback=lambda md: super(DiscordMediaMixin, self).send_animation(chat_id, animation_url, caption, reply_to, metadata=md),
            metadata=metadata, error_metadata=metadata,
        )


    async def send_video(
        self, chat_id: str, video_path: str, caption: Optional[str] = None,
        reply_to: Optional[str] = None, metadata: Optional[dict[str, Any]] = None,
    ) -> SendResult:
        """Send a local video file natively as a Discord attachment."""
        return await self._send_local_file(
            chat_id, video_path, caption, not_found="Video file not found", kind="local video",
            metadata=metadata,
        )


    async def send_document(
        self, chat_id: str, file_path: str, caption: Optional[str] = None,
        file_name: Optional[str] = None, reply_to: Optional[str] = None,
        metadata: Optional[dict[str, Any]] = None,
    ) -> SendResult:
        """Send an arbitrary file natively as a Discord attachment."""
        return await self._send_local_file(
            chat_id, file_path, caption, file_name=file_name, not_found="File not found", kind="document",
            metadata=metadata,
        )

