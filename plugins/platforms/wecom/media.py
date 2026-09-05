"""WeCom media mixin: inbound attachment caching; outbound chunked ``aibot_upload_media_*`` upload then
native send (image/video/voice/file)."""

from __future__ import annotations

import asyncio
import base64
import hashlib
import logging
import mimetypes
import re
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import unquote, urlparse

from gateway.platforms.base import SendResult, cache_document_from_bytes_async, cache_image_from_bytes_async

logger = logging.getLogger("plugins.platforms.wecom.adapter")

APP_CMD_SEND = "aibot_send_msg"
APP_CMD_UPLOAD_MEDIA_INIT = "aibot_upload_media_init"
APP_CMD_UPLOAD_MEDIA_CHUNK = "aibot_upload_media_chunk"
APP_CMD_UPLOAD_MEDIA_FINISH = "aibot_upload_media_finish"

IMAGE_MAX_BYTES = 10 * 1024 * 1024
VIDEO_MAX_BYTES = 10 * 1024 * 1024
VOICE_MAX_BYTES = 2 * 1024 * 1024
FILE_MAX_BYTES = 20 * 1024 * 1024
ABSOLUTE_MAX_BYTES = FILE_MAX_BYTES
UPLOAD_CHUNK_SIZE = 512 * 1024
MAX_UPLOAD_CHUNKS = 100
VOICE_SUPPORTED_MIMES = {"audio/amr"}

_IMAGE_MAGIC = ((b"\x89PNG\r\n\x1a\n", ".png"), (b"\xff\xd8\xff", ".jpg"), ((b"GIF87a", b"GIF89a"), ".gif"))
_MIME_PREFIX_KINDS = (("image/", "image"), ("video/", "video"), ("audio/", "voice"))
# type -> (max bytes, Chinese label, human cap) for the "downgrade to file" notice
_TYPE_LIMITS = {"image": (IMAGE_MAX_BYTES, "图片", "10MB"), "video": (VIDEO_MAX_BYTES, "视频", "10MB"), "voice": (VOICE_MAX_BYTES, "语音", "2MB")}


def _size_verdict(final_type: str, *, reject: Optional[str] = None, downgrade: Optional[str] = None) -> Dict[str, Any]:
    return {"final_type": final_type, "rejected": reject is not None, "reject_reason": reject, "downgraded": downgrade is not None, "downgrade_note": downgrade}


def _dict_at(container: Dict[str, Any], key: str) -> Dict[str, Any]:
    return container.get(key) if isinstance(container.get(key), dict) else {}


def _media_body(media_type: str, media_id: str) -> Dict[str, Any]:
    return {"msgtype": media_type, media_type: {"media_id": media_id}}


class WeComMediaMixin:
    """Media helpers mixed into WeComAdapter (uses its transport, req_id cache and stream registry)."""

    async def _extract_media(self, body: Dict[str, Any]) -> Tuple[List[str], List[str]]:
        refs: List[Tuple[str, Dict[str, Any]]] = []
        msgtype = str(body.get("msgtype") or "").lower()

        def _ref(kind: str, container: Dict[str, Any]) -> bool:
            found = isinstance(container.get(kind), dict)
            if found:
                refs.append((kind, container[kind]))
            return found

        if msgtype == "mixed":
            mixed = _dict_at(body, "mixed")
            for item in mixed.get("msg_item") if isinstance(mixed.get("msg_item"), list) else []:
                if isinstance(item, dict) and str(item.get("msgtype") or "").lower() == "image":
                    _ref("image", item)
        else:
            _ref("image", body)
            if msgtype == "file":
                _ref("file", body)
            if msgtype == "appmsg" and isinstance(body.get("appmsg"), dict):  # AI Bot attachments (PDF/Word/Excel)
                _ref("file", body["appmsg"]) or _ref("image", body["appmsg"])
        quote = _dict_at(body, "quote")
        quote_type = str(quote.get("msgtype") or "").lower()
        if quote_type in ("image", "file"):
            _ref(quote_type, quote)
        cached = [c for c in [await self._cache_media(kind, ref) for kind, ref in refs] if c]
        return [c[0] for c in cached], [c[1] for c in cached]

    async def _cache_media(self, kind: str, media: Dict[str, Any]) -> Optional[Tuple[str, str]]:
        """Cache an inbound image/file reference (inline base64 or URL) to local storage."""
        if media.get("base64"):
            try:
                raw = self._decode_base64(media["base64"])
            except Exception as exc:
                logger.debug("[%s] Failed to decode %s base64 media: %s", self.name, kind, exc)
                return None
            filename = str(media.get("filename") or media.get("name") or "wecom_file")
            return await self._store_media(kind, raw, self._detect_image_ext(raw), "", filename, mimetypes.guess_type(filename)[0] or "application/octet-stream", "")
        url = str(media.get("url") or "").strip()
        if not url:
            return None
        aes_key = str(media.get("aeskey") or "").strip()
        try:
            step = "download"
            raw, headers = await self._download_remote_bytes(url, max_bytes=ABSOLUTE_MAX_BYTES)
            step = "decrypt"
            raw = self._decrypt_file_bytes(raw, aes_key) if aes_key else raw
        except Exception as exc:
            logger.debug("[%s] Failed to %s %s from %s: %s", self.name, step, kind, url, exc)
            return None
        content_type = str(headers.get("content-type") or "").split(";", 1)[0].strip() or "application/octet-stream"
        ext = self._guess_extension(url, content_type, fallback=self._detect_image_ext(raw))
        return await self._store_media(kind, raw, ext, content_type, self._guess_filename(url, headers.get("content-disposition"), content_type), content_type, f" from {url}")

    async def _store_media(self, kind, raw, ext, image_mime, filename, doc_mime, origin) -> Optional[Tuple[str, str]]:
        """Cache bytes as an image (``kind == "image"``) or a document; returns (path, mime)."""
        if kind != "image":
            return await cache_document_from_bytes_async(raw, filename), doc_mime
        try:
            return await cache_image_from_bytes_async(raw, ext), image_mime or self._mime_for_ext(ext, fallback="image/jpeg")
        except ValueError as exc:
            logger.warning("[%s] Rejected non-image bytes%s: %s", self.name, origin, exc)
            return None

    @staticmethod
    def _decode_base64(data: str) -> bytes:
        return base64.b64decode(data.split(",", 1)[-1].strip())

    @staticmethod
    def _detect_image_ext(data: bytes) -> str:
        webp = ".webp" if data.startswith(b"RIFF") and data[8:12] == b"WEBP" else ".jpg"
        return next((ext for magic, ext in _IMAGE_MAGIC if data.startswith(magic)), webp)

    @staticmethod
    def _mime_for_ext(ext: str, fallback: str = "application/octet-stream") -> str:
        return mimetypes.types_map.get(ext.lower(), fallback)

    @staticmethod
    def _guess_extension(url: str, content_type: str, fallback: str) -> str:
        ext = mimetypes.guess_extension(content_type) if content_type else None
        return ext or Path(urlparse(url).path).suffix or fallback

    @staticmethod
    def _guess_filename(url: str, content_disposition: Optional[str], content_type: str) -> str:
        match = re.search(r'filename="?([^";]+)"?', content_disposition or "")
        if match:
            return match.group(1)
        name = Path(urlparse(url).path).name or "document"
        return name if "." in name else f"{name}{mimetypes.guess_extension(content_type) or '.bin'}"

    @staticmethod
    def _decrypt_file_bytes(encrypted_data: bytes, aes_key: str) -> bytes:
        if not encrypted_data:
            raise ValueError("encrypted_data is empty")
        if not aes_key:
            raise ValueError("aes_key is required")
        key = base64.b64decode(aes_key + '=' * ((4 - len(aes_key) % 4) % 4))  # WeCom doesn't pad base64 keys
        if len(key) != 32:
            raise ValueError(f"Invalid WeCom AES key length: expected 32 bytes, got {len(key)}")
        try:
            from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
        except ImportError as exc:  # pragma: no cover - dependency is environment-specific
            raise RuntimeError("cryptography is required for WeCom media decryption") from exc
        decryptor = Cipher(algorithms.AES(key), modes.CBC(key[:16])).decryptor()
        decrypted = decryptor.update(encrypted_data) + decryptor.finalize()
        pad_len = decrypted[-1]
        if pad_len < 1 or pad_len > 32 or pad_len > len(decrypted):
            raise ValueError(f"Invalid PKCS#7 padding value: {pad_len}")
        if any(byte != pad_len for byte in decrypted[-pad_len:]):
            raise ValueError("Invalid PKCS#7 padding: padding bytes mismatch")
        return decrypted[:-pad_len]

    async def _download_remote_bytes(self, url: str, max_bytes: int) -> Tuple[bytes, Dict[str, str]]:
        from gateway.platforms.base import _ssrf_redirect_guard
        from tools.url_safety import create_ssrf_safe_async_client, is_safe_url
        from plugins.platforms.wecom import adapter as _adapter_mod
        if not is_safe_url(url):
            raise ValueError(f"Blocked unsafe URL (SSRF protection): {url[:80]}")
        if not _adapter_mod.HTTPX_AVAILABLE:
            raise RuntimeError("httpx is required for WeCom media download")
        client = self._http_client or create_ssrf_safe_async_client(timeout=30.0, follow_redirects=True, event_hooks={"response": [_ssrf_redirect_guard]})
        try:
            async with client.stream("GET", url, headers={"User-Agent": "HermesAgent/1.0", "Accept": "*/*"}) as response:
                response.raise_for_status()
                headers = {key.lower(): value for key, value in response.headers.items()}
                content_length = headers.get("content-length")
                if content_length and content_length.isdigit() and int(content_length) > max_bytes:
                    raise ValueError(f"Remote media exceeds WeCom limit: {int(content_length)} bytes > {max_bytes} bytes")
                data = bytearray()
                async for chunk in response.aiter_bytes():
                    data.extend(chunk)
                    if len(data) > max_bytes:
                        raise ValueError(f"Remote media exceeds WeCom limit while downloading: {len(data)} bytes > {max_bytes} bytes")
                return bytes(data), headers
        finally:
            if client is not self._http_client:
                await client.aclose()

    @staticmethod
    def _guess_mime_type(filename: str) -> str:
        return mimetypes.guess_type(filename)[0] or ("audio/amr" if Path(filename).suffix.lower() == ".amr" else "application/octet-stream")

    @staticmethod
    def _normalize_content_type(content_type: str, filename: str) -> str:
        normalized = str(content_type or "").split(";", 1)[0].strip().lower()
        return normalized if normalized and normalized not in {"application/octet-stream", "text/plain"} else WeComMediaMixin._guess_mime_type(filename)

    @staticmethod
    def _detect_wecom_media_type(content_type: str) -> str:
        mime_type = str(content_type or "").strip().lower()
        return "voice" if mime_type == "application/ogg" else next((kind for prefix, kind in _MIME_PREFIX_KINDS if mime_type.startswith(prefix)), "file")

    @staticmethod
    def _apply_file_size_limits(file_size: int, detected_type: str, content_type: Optional[str] = None) -> Dict[str, Any]:
        file_size_mb = file_size / (1024 * 1024)
        normalized_type = str(detected_type or "file").lower()
        normalized_content_type = str(content_type or "").strip().lower()
        if file_size > ABSOLUTE_MAX_BYTES:
            return _size_verdict(normalized_type, reject=(f"文件大小 {file_size_mb:.2f}MB 超过了企业微信允许的最大限制 20MB，无法发送。" "请尝试压缩文件或减小文件大小。"))
        if normalized_type == "voice" and normalized_content_type and normalized_content_type not in VOICE_SUPPORTED_MIMES:
            return _size_verdict("file", downgrade=f"语音格式 {normalized_content_type} 不支持，企微仅支持 AMR 格式，已转为文件格式发送")
        max_bytes, label, cap = _TYPE_LIMITS.get(normalized_type, (None, "", ""))
        if max_bytes is not None and file_size > max_bytes:
            return _size_verdict("file", downgrade=f"{label}大小 {file_size_mb:.2f}MB 超过 {cap} 限制，已转为文件格式发送")
        return _size_verdict(normalized_type)

    @staticmethod
    def _looks_like_url(media_source: str) -> bool:
        return urlparse(str(media_source or "")).scheme in {"http", "https"}

    async def _load_outbound_media(self, media_source: str, file_name: Optional[str] = None) -> Tuple[bytes, str, str]:
        source = str(media_source or "").strip()
        if not source:
            raise ValueError("media source is required")
        if re.fullmatch(r"<[^>\n]+>", source):
            raise ValueError(f"Media placeholder was not replaced with a real file path: {source}")
        parsed = urlparse(source)
        if parsed.scheme in {"http", "https"}:
            data, headers = await self._download_remote_bytes(source, max_bytes=ABSOLUTE_MAX_BYTES)
            resolved_name = file_name or self._guess_filename(source, headers.get("content-disposition"), headers.get("content-type", ""))
            return data, self._normalize_content_type(headers.get("content-type", ""), resolved_name), resolved_name
        local_path = Path(unquote(parsed.path) if parsed.scheme == "file" else source).expanduser()
        local_path = local_path if local_path.is_absolute() else (Path.cwd() / local_path).resolve()
        if not local_path.is_file():
            raise FileNotFoundError(f"Media file not found: {local_path}")
        resolved_name = file_name or local_path.name
        return local_path.read_bytes(), self._normalize_content_type("", resolved_name), resolved_name

    async def _prepare_outbound_media(self, media_source: str, file_name: Optional[str] = None) -> Dict[str, Any]:
        data, content_type, resolved_name = await self._load_outbound_media(media_source, file_name=file_name)
        detected_type = self._detect_wecom_media_type(content_type)
        return {"data": data, "content_type": content_type, "file_name": resolved_name, "detected_type": detected_type, **self._apply_file_size_limits(len(data), detected_type, content_type)}

    async def _checked_request(self, cmd: str, body: Dict[str, Any], operation: str) -> Dict[str, Any]:
        self._raise_for_wecom_error(response := await self._send_request(cmd, body), operation)
        return response

    async def _upload_media_bytes(self, data: bytes, media_type: str, filename: str) -> Dict[str, Any]:
        if not data:
            raise ValueError("Cannot upload empty media")
        total_size, total_chunks = len(data), (len(data) + UPLOAD_CHUNK_SIZE - 1) // UPLOAD_CHUNK_SIZE
        if total_chunks > MAX_UPLOAD_CHUNKS:
            raise ValueError(f"File too large: {total_chunks} chunks exceeds maximum of {MAX_UPLOAD_CHUNKS} chunks")
        init_payload = {"type": media_type, "filename": filename, "total_size": total_size, "total_chunks": total_chunks, "md5": hashlib.md5(data).hexdigest()}
        init_response = await self._checked_request(APP_CMD_UPLOAD_MEDIA_INIT, init_payload, "media upload init")
        upload_id = str(_dict_at(init_response, "body").get("upload_id") or "").strip()
        if not upload_id:
            raise RuntimeError(f"media upload init failed: missing upload_id in response {init_response}")
        for chunk_index, start in enumerate(range(0, total_size, UPLOAD_CHUNK_SIZE)):  # official SDK uses 0-based chunk indexes
            chunk_b64 = base64.b64encode(data[start : start + UPLOAD_CHUNK_SIZE]).decode("ascii")
            await self._checked_request(APP_CMD_UPLOAD_MEDIA_CHUNK, {"upload_id": upload_id, "chunk_index": chunk_index, "base64_data": chunk_b64}, f"media upload chunk {chunk_index}")
        finish_response = await self._checked_request(APP_CMD_UPLOAD_MEDIA_FINISH, {"upload_id": upload_id}, "media upload finish")
        finish_body = _dict_at(finish_response, "body")
        media_id = str(finish_body.get("media_id") or "").strip()
        if not media_id:
            raise RuntimeError(f"media upload finish failed: missing media_id in response {finish_response}")
        return {"type": str(finish_body.get("type") or media_type), "media_id": media_id, "created_at": finish_body.get("created_at")}

    async def _send_media_message(self, chat_id: str, media_type: str, media_id: str) -> Dict[str, Any]:
        return await self._checked_request(APP_CMD_SEND, {"chatid": chat_id, **_media_body(media_type, media_id)}, "send media message")

    async def _send_followup_markdown(self, chat_id: str, content: str, reply_to: Optional[str] = None) -> Optional[SendResult]:
        if not content:
            return None
        result = await self.send(chat_id=chat_id, content=content, reply_to=reply_to)
        if not result.success:
            logger.warning("[%s] Follow-up markdown send failed: %s", self.name, result.error)
        return result

    async def _send_media_source(self, chat_id: str, media_source: str, caption: Optional[str] = None, file_name: Optional[str] = None, reply_to: Optional[str] = None) -> SendResult:
        if not chat_id:
            return SendResult(success=False, error="chat_id is required")
        try:
            prepared = await self._prepare_outbound_media(media_source, file_name=file_name)
        except Exception as exc:
            if not isinstance(exc, FileNotFoundError):
                logger.error("[%s] Failed to prepare outbound media %s: %s", self.name, media_source, exc)
            return SendResult(success=False, error=str(exc))
        if prepared["rejected"]:
            await self._send_followup_markdown(chat_id, f"⚠️ {prepared['reject_reason']}", reply_to=reply_to)
            return SendResult(success=False, error=prepared["reject_reason"])
        reply_req_id = self._cached_reply_req_id(chat_id, reply_to)
        # Active/expired stream owns the req_id (passive replyMedia is never acked): go proactive.
        if self._find_active_turn_for_chat(chat_id) or chat_id in self._stream_expired_chats:
            reply_req_id = None
        try:
            upload_result = await self._upload_media_bytes(prepared["data"], prepared["final_type"], prepared["file_name"])
            logger.info("[%s] upload_media_bytes OK: media_id=%s type=%s", self.name, upload_result.get("media_id"), prepared["final_type"])
            if reply_req_id:  # passive reply when a req_id is usable, else proactive APP_CMD_SEND
                media_response = await self._send_reply_request(reply_req_id, _media_body(prepared["final_type"], upload_result["media_id"]))
                self._raise_for_wecom_error(media_response, "send reply media message")
            else:
                media_response = await self._send_media_message(chat_id, prepared["final_type"], upload_result["media_id"])
            logger.info("[%s] %s OK: %s", self.name, "send_reply_media" if reply_req_id else "send_media_message", media_response)
        except asyncio.TimeoutError:
            logger.error("[%s] TIMEOUT in _send_media_source for %s", self.name, media_source)
            return SendResult(success=False, error="Timeout sending media to WeCom")
        except Exception as exc:
            logger.error("[%s] Failed to send media %s: %s", self.name, media_source, exc)
            return SendResult(success=False, error=str(exc))
        raw: Dict[str, Any] = {"upload": upload_result, "media": media_response}
        for key, text in (("caption", caption), ("downgrade", f"ℹ️ {prepared['downgrade_note']}" if prepared["downgraded"] and prepared["downgrade_note"] else None)):
            followup = await self._send_followup_markdown(chat_id, text, reply_to=reply_to) if text else None
            raw[key] = followup.raw_response if followup else None
            raw[f"{key}_error"] = followup.error if followup and not followup.success else None
        return SendResult(success=True, message_id=self._payload_req_id(media_response) or uuid.uuid4().hex[:12], raw_response=raw)

    async def send_image(self, chat_id: str, image_url: str, caption: Optional[str] = None, reply_to: Optional[str] = None, metadata: Optional[Dict[str, Any]] = None) -> SendResult:
        result = await self._send_media_source(chat_id=chat_id, media_source=image_url, caption=caption, reply_to=reply_to)
        if result.success or not self._looks_like_url(image_url):
            return result
        logger.warning("[%s] Falling back to text send for image URL %s: %s", self.name, image_url, result.error)
        return await self.send(chat_id=chat_id, content=f"{caption}\n{image_url}" if caption else image_url, reply_to=reply_to)

    async def send_image_file(self, chat_id: str, image_path: str, caption: Optional[str] = None, reply_to: Optional[str] = None, **kwargs) -> SendResult:
        return await self._send_media_source(chat_id=chat_id, media_source=image_path, caption=caption, reply_to=reply_to)

    async def send_document(self, chat_id: str, file_path: str, caption: Optional[str] = None, file_name: Optional[str] = None, reply_to: Optional[str] = None, **kwargs) -> SendResult:
        logger.info("[%s] send_document called: chat=%s file=%s", self.name, chat_id, file_path)
        return await self._send_media_source(chat_id=chat_id, media_source=file_path, caption=caption, file_name=file_name, reply_to=reply_to)

    async def send_voice(self, chat_id: str, audio_path: str, caption: Optional[str] = None, reply_to: Optional[str] = None, **kwargs) -> SendResult:
        return await self._send_media_source(chat_id=chat_id, media_source=audio_path, caption=caption, reply_to=reply_to)

    async def send_video(self, chat_id: str, video_path: str, caption: Optional[str] = None, reply_to: Optional[str] = None, **kwargs) -> SendResult:
        return await self._send_media_source(chat_id=chat_id, media_source=video_path, caption=caption, reply_to=reply_to)
