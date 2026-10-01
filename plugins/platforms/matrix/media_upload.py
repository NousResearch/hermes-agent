"""Outbound Matrix media upload, encryption and message payloads."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Optional

from gateway.platforms.base import SendResult

if TYPE_CHECKING:
    from plugins.platforms.matrix.adapter import MatrixAdapter

logger = logging.getLogger("plugins.platforms.matrix.adapter")


class MatrixMediaUploadMixin:
    async def _upload_and_send(
        self: MatrixAdapter, room_id: str, data: bytes, filename: str, content_type: str, msgtype: str,
        caption: Optional[str] = None, reply_to: Optional[str] = None, metadata: Optional[dict[str, Any]] = None,
        is_voice: bool = False, voice_metadata: Optional[dict[str, Any]] = None) -> SendResult:
        if len(data) > self._max_media_bytes:
            return self._media_too_large(len(data))
        upload_data = data
        encrypted_file = None
        if await self._room_needs_encrypted_upload(room_id):
            try:
                from mautrix.crypto.attachments import encrypt_attachment
                upload_data, encrypted_file = encrypt_attachment(data)
            except Exception as exc:
                logger.error("Matrix: attachment encryption failed: %s", exc)
                return SendResult(success=False, error=str(exc))
        try:
            mxc_url = await self._client.upload_media(
                upload_data, mime_type=content_type, filename=filename, size=len(upload_data))
        except Exception as exc:
            logger.error("Matrix: upload failed: %s", exc)
            return SendResult(success=False, error=str(exc))
        msg_content: dict[str, Any] = {
            "msgtype": msgtype, "body": caption or filename, "info": {"mimetype": content_type, "size": len(data)}}
        if encrypted_file is not None:
            from mautrix.types import ContentURI

            encrypted_file.url = ContentURI(str(mxc_url))
            msg_content["file"] = encrypted_file.serialize()
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
