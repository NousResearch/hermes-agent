"""Discord native media uploads and delivery fallbacks."""
from __future__ import annotations

import asyncio
import json
import logging
import os
from typing import Any, Dict, List, Optional, Tuple

from gateway.platforms.base import SendResult

logger = logging.getLogger("plugins.platforms.discord.adapter")


class DiscordMediaMixin:
    async def _send_file_attachment(
        self, chat_id: str, file_path: str, caption: Optional[str] = None,
        file_name: Optional[str] = None, metadata: Optional[Dict[str, Any]] = None,
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
        self, chat_id: str, images: List[Tuple[str, str]],
        metadata: Optional[Dict[str, Any]] = None, human_delay: float = 0.0,
    ) -> None:
        """Send images as one Discord message (<=10 attachments): URLs are downloaded and uploaded
        inline (bare links don't render); on chunk failure the remainder uses the per-image loop."""
        from plugins.platforms.discord.adapter import _prompt_target_id, _image_ext_from_content_type, _read_url_image_with_redirect_guard, is_safe_url

        if not self._client:
            return
        if not images:
            return
        try:
            import discord as _discord_mod
            import io as _io
            from urllib.parse import unquote as _unquote
        except Exception:  # pragma: no cover
            await super().send_multiple_images(chat_id, images, metadata, human_delay)
            return
        try:
            channel = await self._resolve_channel(_prompt_target_id(chat_id, metadata))
            if not channel:
                logger.warning("[%s] Channel %s not found for multi-image send", self.name, chat_id)
                return
        except Exception as e:
            logger.warning("[%s] Failed to resolve channel for multi-image send: %s", self.name, e)
            await super().send_multiple_images(chat_id, images, metadata, human_delay)
            return
        CHUNK = 10
        chunks = [images[i:i + CHUNK] for i in range(0, len(images), CHUNK)]
        for chunk_idx, chunk in enumerate(chunks):
            if human_delay > 0 and chunk_idx > 0:
                await asyncio.sleep(human_delay)
            files: List[Any] = []
            captions: List[str] = []
            aiohttp_session = None
            try:
                for image_url, alt_text in chunk:
                    if alt_text:
                        captions.append(alt_text)
                    if image_url.startswith("file://"):
                        local_path = _unquote(image_url[7:])
                        if not os.path.exists(local_path):
                            logger.warning("[%s] Skipping missing image: %s", self.name, local_path)
                            continue
                        files.append(_discord_mod.File(local_path, filename=os.path.basename(local_path)))
                    else:
                        if not is_safe_url(image_url):
                            logger.warning("[%s] Blocked unsafe image URL in batch", self.name)
                            continue
                        # Download to BytesIO so it renders inline
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
                            files.append(_discord_mod.File(_io.BytesIO(data), filename=f"image_{len(files)}.{ext}"))
                        except Exception as dl_err:
                            logger.warning("[%s] Download failed for %s: %s", self.name, image_url[:80], dl_err)
                            continue
                if not files:
                    continue
                # Use the first caption if any (Discord only has one message body for the group)
                content = captions[0] if captions else None
                logger.info(
                    "[%s] Sending %d image(s) as single Discord message (chunk %d/%d)",
                    self.name, len(files), chunk_idx + 1, len(chunks),
                )
                if self._is_forum_parent(channel):
                    await self._forum_post_file(
                        channel, content=(content or "").strip(), files=files,
                    )
                else:
                    await channel.send(content=content, files=files)
            except Exception as e:
                logger.warning(
                    "[%s] Multi-image Discord send failed (chunk %d/%d), falling back to per-image: %s",
                    self.name, chunk_idx + 1, len(chunks), e, exc_info=True,
                )
                await super().send_multiple_images(chat_id, chunk, metadata, human_delay=human_delay)
            finally:
                if aiohttp_session is not None:
                    try:
                        await aiohttp_session.close()
                    except Exception:
                        pass


    async def send_voice(
        self, chat_id: str, audio_path: str, caption: Optional[str] = None,
        reply_to: Optional[str] = None, metadata: Optional[Dict[str, Any]] = None, **kwargs,
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
            reference = self._reply_reference_for_send(reply_to, channel)
            with open(audio_path, "rb") as f:
                file_data = f.read()
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
        reply_to: Optional[str] = None, metadata: Optional[Dict[str, Any]] = None,
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
                import io
                file = discord.File(io.BytesIO(data), filename=filename_for(headers))
                if self._is_forum_parent(channel):
                    return await self._forum_post_file(channel, content=(caption or "").strip(), file=file)
                msg = await channel.send(content=caption if caption else None, file=file)
                return SendResult(success=True, message_id=str(msg.id))
        except ImportError:
            logger.warning("[%s] aiohttp not installed, falling back to URL. Run: pip install aiohttp", self.name, exc_info=True)
            return await fallback(error_metadata)
        except Exception as e:  # pragma: no cover - defensive logging
            logger.error("[%s] Failed to send %s attachment, falling back to URL: %s", self.name, kind, e, exc_info=True)
            return await fallback(error_metadata)


    async def send_image(
        self, chat_id: str, image_url: str, caption: Optional[str] = None,
        reply_to: Optional[str] = None, metadata: Optional[Dict[str, Any]] = None,
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
        reply_to: Optional[str] = None, metadata: Optional[Dict[str, Any]] = None,
    ) -> SendResult:
        """Send an animated GIF natively as a Discord file attachment."""
        return await self._send_url_media(
            chat_id, animation_url, caption, kind="animation", filename_for=lambda _h: "animation.gif",
            fallback=lambda md: super(DiscordMediaMixin, self).send_animation(chat_id, animation_url, caption, reply_to, metadata=md),
            metadata=metadata, error_metadata=metadata,
        )


    async def send_video(
        self, chat_id: str, video_path: str, caption: Optional[str] = None,
        reply_to: Optional[str] = None, metadata: Optional[Dict[str, Any]] = None,
    ) -> SendResult:
        """Send a local video file natively as a Discord attachment."""
        return await self._send_local_file(
            chat_id, video_path, caption, not_found="Video file not found", kind="local video",
            metadata=metadata,
        )


    async def send_document(
        self, chat_id: str, file_path: str, caption: Optional[str] = None,
        file_name: Optional[str] = None, reply_to: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> SendResult:
        """Send an arbitrary file natively as a Discord attachment."""
        return await self._send_local_file(
            chat_id, file_path, caption, file_name=file_name, not_found="File not found", kind="document",
            metadata=metadata,
        )

