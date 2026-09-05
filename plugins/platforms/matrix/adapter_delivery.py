"""Matrix delivery methods; runtime dependencies remain on the adapter facade."""

from __future__ import annotations
from gateway.native_document_guard import check_document_fallback, mark_native_document_guard

from typing import Any, Dict, Optional
from gateway.platforms.base import SendResult


class MatrixDeliveryMixin:
    async def send(
        self, chat_id: str, content: str, reply_to: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None) -> SendResult:
        from . import adapter as _adapter

        if not content:
            return _adapter.SendResult(success=True)
        last_event_id = None
        for chunk in self.truncate_message(self.format_message(content), self.max_message_length):
            msg_content = self._build_text_message_content(chunk)
            self._apply_relation_metadata(msg_content, reply_to=reply_to, metadata=metadata)
            try:
                last_event_id = await self._send_room_message(chat_id, msg_content)
                _adapter.logger.info("Matrix: sent event %s to %s", last_event_id, chat_id)
            except Exception as exc:
                if not (self._encryption and getattr(self._client, "crypto", None)):
                    _adapter.logger.error("Matrix: failed to send to %s: %s", chat_id, exc)
                    return _adapter.SendResult(success=False, error=str(exc))
                try:  # E2EE error: retry once after sharing keys
                    await self._client.crypto.share_keys()
                    last_event_id = await self._send_room_message(chat_id, msg_content)
                    _adapter.logger.info("Matrix: sent event %s to %s (after key share)", last_event_id, chat_id)
                except Exception as retry_exc:
                    _adapter.logger.error("Matrix: failed to send to %s after retry: %s", chat_id, retry_exc)
                    return _adapter.SendResult(success=False, error=str(retry_exc))
        return _adapter.SendResult(success=True, message_id=last_event_id)

    async def _send_room_message(self, chat_id: str, msg_content: Dict[str, Any]) -> str:
        """Send one m.room.message event (45s cap) and return its event ID as str."""
        from . import adapter as _adapter

        event_id = await _adapter.asyncio.wait_for(
            self._client.send_message_event(_adapter.RoomID(chat_id), _adapter.EventType.ROOM_MESSAGE, msg_content), timeout=45)
        return str(event_id)

    async def get_chat_info(self, chat_id: str) -> Dict[str, Any]:
        identity = await self._resolve_room_identity(chat_id)
        return {"name": identity.display_name, "type": "dm" if identity.chat_type == "dm" else "group"}

    def get_diagnostics(self) -> Dict[str, Any]:
        from . import adapter as _adapter

        now = _adapter.time.time()
        token_present = bool(self._access_token)
        user_id = self._user_id or getattr(self._client, "mxid", "") or ""
        device_id = self._device_id or getattr(self._client, "device_id", "") or ""
        return {
            "platform": "matrix", "homeserver": self._homeserver,
            "auth": {
                "access_token_present": token_present, "password_present": bool(self._password),
                "token_preview": "***" if token_present else "", "user_id": user_id,
                "device_id_present": bool(device_id), "device_id_preview": "***" if str(device_id or "").strip() else ""},
            "sync": {
                "connected": self._client is not None, "joined_room_count": len(self._joined_rooms),
                "last_sync_age_seconds": max(0.0, now - self._last_sync_ts) if self._last_sync_ts else None},
            "e2ee": {
                "mode": self._e2ee_mode, "enabled": bool(self._encryption), "deps_available": _adapter._check_e2ee_deps(),
                "crypto_store_path": str(self._crypto_db_path),
                "recovery_key_configured": bool(_adapter._scoped_recovery_key().strip())},
            "policy": {
                "allowed_user_count": len(self._allowed_user_ids), "allowed_room_count": len(self._allowed_room_ids),
                "ignored_user_pattern_count": len(self._ignored_user_patterns),
                "require_mention": self._require_mention, "free_response_room_count": len(self._free_rooms),
                "allow_room_mentions": self._allow_room_mentions, "process_notices": self._process_notices,
                "allow_public_rooms": _adapter._env_truthy("MATRIX_ALLOW_PUBLIC_ROOMS")},
            "media": {"max_media_bytes": self._max_media_bytes}}

    async def _set_typing(self, chat_id: str, timeout: int) -> None:
        from . import adapter as _adapter

        if self._client:
            with _adapter.suppress(Exception):
                await self._client.set_typing(_adapter.RoomID(chat_id), timeout=timeout)

    async def send_typing(self, chat_id: str, metadata: Optional[Dict[str, Any]] = None) -> None:
        await self._set_typing(chat_id, 30000)

    async def stop_typing(self, chat_id: str) -> None:
        await self._set_typing(chat_id, 0)

    async def edit_message(self, chat_id: str, message_id: str, content: str, *, finalize: bool = False) -> SendResult:
        from . import adapter as _adapter

        formatted = self.format_message(content)
        new_content = self._build_text_message_content(formatted)
        msg_content: _adapter.Dict[str, _adapter.Any] = {"msgtype": "m.text", "body": f"* {formatted}", "m.new_content": new_content}
        if "m.mentions" in new_content:
            msg_content["m.mentions"] = new_content["m.mentions"]
        if "formatted_body" in new_content:
            msg_content["format"] = "org.matrix.custom.html"
            msg_content["formatted_body"] = f'* {new_content["formatted_body"]}'
        msg_content["m.relates_to"] = {"rel_type": "m.replace", "event_id": message_id}
        return await self._send_content_event(chat_id, msg_content)

    async def send_image(
        self, chat_id: str, image_url: str, caption: Optional[str] = None, reply_to: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None) -> SendResult:
        from . import adapter as _adapter

        from tools.url_safety import is_safe_url
        if not is_safe_url(image_url):
            _adapter.logger.warning("Matrix: blocked unsafe image URL (SSRF protection)")
            return await super().send_image(chat_id, image_url, caption, reply_to, metadata=metadata)
        try:
            data, ct, fname = await self._download_external_media_with_cap(image_url)
        except Exception as exc:
            _adapter.logger.warning("Matrix: failed to download image %s: %s", _adapter._redact_url_for_log(image_url), exc)
            fallback = ("I couldn't download and upload the image to Matrix. "
                        "The source URL was not shown because it may contain private tokens.")
            return await self.send(chat_id, f"{caption}\n{fallback}" if caption else fallback, reply_to)
        return await self._upload_and_send(chat_id, data, fname, ct, "m.image", caption, reply_to, metadata)

    async def _download_external_media_with_cap(self, url: str) -> tuple[bytes, str, str]:
        """Download external media while enforcing redirect safety and size caps."""
        from . import adapter as _adapter

        from tools.url_safety import is_safe_url
        if not is_safe_url(url):
            raise ValueError("blocked unsafe media URL")

        async def _read_capped(resp, chunks, content_type) -> tuple[bytes, str]:
            """Enforce Content-Length + streamed size caps, then require an image/* type."""
            try:
                size = int(resp.headers.get("Content-Length") or resp.headers.get("content-length"))
            except Exception:
                size = None
            if size is not None and size > self._max_media_bytes:
                raise ValueError(f"media exceeds Matrix limit ({size} > {self._max_media_bytes} bytes)")
            parts: list[bytes] = []
            total = 0
            async for chunk in chunks:
                total += len(chunk)
                if total > self._max_media_bytes:
                    raise ValueError(f"media exceeds Matrix limit (> {self._max_media_bytes} bytes)")
                parts.append(bytes(chunk))
            content_type = str(content_type or "").split(";", 1)[0].strip().lower()
            if not content_type.startswith("image/"):
                raise ValueError("external media is not an image")
            return b"".join(parts), content_type
        fname = url.rsplit("/", 1)[-1].split("?")[0] or "image.png"
        try:
            import aiohttp as _aiohttp
            _sess_kw, _req_kw = _adapter.proxy_kwargs_for_aiohttp(self._proxy_url)
            async with _aiohttp.ClientSession(**_sess_kw) as http:
                fetch_url = url
                for _ in range(20):
                    async with http.get(
                        fetch_url, timeout=_aiohttp.ClientTimeout(total=30), allow_redirects=False, **_req_kw) as resp:
                        if resp.status in {301, 302, 303, 307, 308}:
                            location = resp.headers.get("Location")
                            if not location:
                                raise ValueError("redirect missing Location")
                            # Re-validate EVERY hop: a public URL can 302 toward loopback/metadata endpoints,
                            # and checking only the final URL is too late (the hop already connected).
                            fetch_url = _adapter.urljoin(fetch_url, location)
                            if not is_safe_url(fetch_url):
                                raise ValueError("blocked unsafe redirect URL")
                            continue
                        resp.raise_for_status()
                        data, ct = await _read_capped(
                            resp, resp.content.iter_chunked(65536),
                            getattr(resp, "content_type", None)
                            or resp.headers.get("content-type", "application/octet-stream"))
                        return data, ct, fname
                raise ValueError("too many redirects")
        except ImportError:
            from tools.url_safety import create_ssrf_safe_async_client
            _httpx_kw: dict = {"proxy": self._proxy_url} if self._proxy_url else {}
            _httpx_kw["event_hooks"] = {"response": [_adapter._ssrf_redirect_guard]}
            async with create_ssrf_safe_async_client(**_httpx_kw) as http:
                async with http.stream("GET", url, follow_redirects=True, timeout=30) as resp:
                    resp.raise_for_status()
                    data, ct = await _read_capped(
                        resp, resp.aiter_bytes(), resp.headers.get("content-type", "application/octet-stream"))
                    return data, ct, fname

    async def send_image_file(
        self, chat_id: str, image_path: str, caption: Optional[str] = None, reply_to: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None) -> SendResult:
        return await self._send_local_file(chat_id, image_path, "m.image", caption, reply_to, metadata=metadata)

    async def send_multiple_images(
        self, chat_id: str, images: list[tuple[str, str]], metadata: Optional[Dict[str, Any]] = None,
        human_delay: float = 0.0) -> None:
        from . import adapter as _adapter

        if not images:
            return
        from urllib.parse import unquote as _unquote
        total = len(images)
        for idx, (image_url, alt_text) in enumerate(images, start=1):
            if human_delay > 0 and idx > 1:
                await _adapter.asyncio.sleep(human_delay)
            caption = f"{alt_text} ({idx}/{total})" if alt_text and total > 1 else (alt_text or None)
            if image_url.startswith("file://"):
                result = await self.send_image_file(
                    chat_id=chat_id, image_path=_unquote(image_url[7:]), caption=caption, metadata=metadata)
            else:
                result = await self.send_image(chat_id=chat_id, image_url=image_url, caption=caption, metadata=metadata)
            if not result.success:
                _adapter.logger.warning("Matrix: failed to send image %d/%d: %s", idx, total, result.error)

    @mark_native_document_guard
    async def send_document(
        self, chat_id: str, file_path: str, caption: Optional[str] = None, file_name: Optional[str] = None,
        reply_to: Optional[str] = None, metadata: Optional[Dict[str, Any]] = None) -> SendResult:
        return await self._send_local_file(chat_id, file_path, "m.file", caption, reply_to, file_name, metadata)

    async def send_voice(
        self, chat_id: str, audio_path: str, caption: Optional[str] = None, reply_to: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None) -> SendResult:
        """Upload audio as an MSC3245 voice message. Voice bubbles need Ogg/Opus but callers pass any
        format (e.g. TTS output), so transcode here — best-effort: without ffmpeg the original is sent."""
        from . import adapter as _adapter

        converted_path: _adapter.Optional[str] = None
        if not str(audio_path).lower().endswith((".ogg", ".oga", ".opus")):
            converted_path = await _adapter.asyncio.to_thread(_adapter._matrix_transcode_voice_to_ogg, audio_path)
        try:
            return await self._send_local_file(
                chat_id, converted_path or audio_path, "m.audio", caption, reply_to,
                # keep the caller's basename (the temp transcode file has a generated name)
                file_name=(_adapter.Path(audio_path).with_suffix(".ogg").name if converted_path else None),
                metadata=metadata, is_voice=True)
        finally:
            if converted_path:
                with _adapter.suppress(OSError):
                    _adapter.os.unlink(converted_path)

    async def send_video(
        self, chat_id: str, video_path: str, caption: Optional[str] = None, reply_to: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None) -> SendResult:
        return await self._send_local_file(chat_id, video_path, "m.video", caption, reply_to, metadata=metadata)

    def format_message(self, content: str) -> str:
        """Markdown passes through; strip image markdown (media is uploaded separately)."""
        from . import adapter as _adapter

        return _adapter.re.sub(r"!\[([^\]]*)\]\(([^)]+)\)", r"\2", content)

    async def _upload_and_send(
        self, room_id: str, data: bytes, filename: str, content_type: str, msgtype: str,
        caption: Optional[str] = None, reply_to: Optional[str] = None, metadata: Optional[Dict[str, Any]] = None,
        is_voice: bool = False, voice_metadata: Optional[Dict[str, Any]] = None) -> SendResult:
        from . import adapter as _adapter

        if len(data) > self._max_media_bytes:
            return self._media_too_large(len(data))
        upload_data = data
        encrypted_file = None
        if await self._room_needs_encrypted_upload(room_id):
            try:
                from mautrix.crypto.attachments import encrypt_attachment
                upload_data, encrypted_file = encrypt_attachment(data)
            except Exception as exc:
                _adapter.logger.error("Matrix: attachment encryption failed: %s", exc)
                return _adapter.SendResult(success=False, error=str(exc))
        try:
            mxc_url = await self._client.upload_media(
                upload_data, mime_type=content_type, filename=filename, size=len(upload_data))
        except Exception as exc:
            _adapter.logger.error("Matrix: upload failed: %s", exc)
            return _adapter.SendResult(success=False, error=str(exc))
        msg_content: _adapter.Dict[str, _adapter.Any] = {
            "msgtype": msgtype, "body": caption or filename, "info": {"mimetype": content_type, "size": len(data)}}
        if encrypted_file is not None:
            msg_content["file"] = {**encrypted_file.serialize(), "url": str(mxc_url)}
        else:
            msg_content["url"] = str(mxc_url)
        if is_voice:  # MSC3245 native voice flag + MSC1767 audio metadata
            msg_content["org.matrix.msc3245.voice"] = {}
            audio_metadata = {
                k: v for k in ("duration", "waveform") if (v := (voice_metadata or {}).get(k)) is not None}
            if "duration" in audio_metadata:
                msg_content["info"]["duration"] = audio_metadata["duration"]
            if audio_metadata:
                msg_content["org.matrix.msc1767.audio"] = audio_metadata
        self._apply_relation_metadata(msg_content, reply_to=reply_to, metadata=metadata)
        return await self._send_content_event(room_id, msg_content)

    async def _room_needs_encrypted_upload(self, room_id: str) -> bool:
        """E2EE on, Olm machine loaded, and the state store says the room is encrypted."""
        from . import adapter as _adapter

        if not (self._encryption and getattr(self._client, "crypto", None)):
            return False
        state_store = getattr(self._client, "state_store", None)
        if not state_store:
            return False
        try:
            return bool(await state_store.is_encrypted(_adapter.RoomID(room_id)))
        except Exception:
            return False

    def _media_too_large(self, size: int) -> SendResult:
        from . import adapter as _adapter

        return _adapter.SendResult(
            success=False, error=f"Media file exceeds Matrix limit ({size} > {self._max_media_bytes} bytes)")

    async def _send_content_event(self, room_id: str, msg_content: Dict[str, Any]) -> SendResult:
        """Send a prebuilt m.room.message payload, mapping exceptions to SendResult."""
        from . import adapter as _adapter

        try:
            event_id = await self._client.send_message_event(_adapter.RoomID(room_id), _adapter.EventType.ROOM_MESSAGE, msg_content)
            return _adapter.SendResult(success=True, message_id=str(event_id))
        except Exception as exc:
            return _adapter.SendResult(success=False, error=str(exc))

    async def _send_local_file(
        self, room_id: str, file_path: str, msgtype: str, caption: Optional[str] = None,
        reply_to: Optional[str] = None, file_name: Optional[str] = None, metadata: Optional[Dict[str, Any]] = None,
        is_voice: bool = False) -> SendResult:
        from . import adapter as _adapter

        p = _adapter.Path(file_path).expanduser()
        if not p.exists():
            # file_path is host-local; never echo it into chat.
            _adapter.logger.warning("[%s] upload fallback: media file not found for %s", self.name, file_path)
            check_document_fallback()
            text = "⚠️ Couldn't deliver the attachment."
            return await self.send(room_id, f"{caption}\n{text}" if caption else text, reply_to)
        try:
            file_size = p.stat().st_size
        except OSError:
            file_size = 0
        if file_size > self._max_media_bytes:
            return self._media_too_large(file_size)
        fname = file_name or p.name
        # ffprobe/ffmpeg probing is blocking (subprocess timeouts up to 15s) —
        # run it off the event loop so voice uploads never stall the adapter.
        voice_metadata = await _adapter.asyncio.to_thread(_adapter._matrix_voice_metadata_for_file, p) if is_voice else None
        return await self._upload_and_send(
            room_id, p.read_bytes(), fname, _adapter.mimetypes.guess_type(fname)[0] or "application/octet-stream", msgtype,
            caption, reply_to, metadata, is_voice, voice_metadata)
