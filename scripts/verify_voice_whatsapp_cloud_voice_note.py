#!/usr/bin/env python3
"""Verify WhatsApp Cloud voice-note upload behavior without live Graph calls."""

from __future__ import annotations

import asyncio
from collections import OrderedDict
import json
import os
from pathlib import Path
import sys
import tempfile
from typing import Any
from unittest.mock import AsyncMock, MagicMock

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from gateway.config import Platform
from gateway.platforms.whatsapp_cloud import WhatsAppCloudAdapter


WHATSAPP_OPUS_MIME = "audio/ogg; codecs=opus"


class VerificationError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise VerificationError(message)


def make_response(body: dict[str, Any], *, status_code: int = 200) -> MagicMock:
    response = MagicMock()
    response.status_code = status_code
    response.json = MagicMock(return_value=body)
    response.text = json.dumps(body)
    return response


def upload_response(media_id: str) -> MagicMock:
    return make_response({"id": media_id})


def message_response(wamid: str = "wamid.voice-note") -> MagicMock:
    return make_response({"messages": [{"id": wamid}]})


def temp_file(suffix: str, content: bytes) -> str:
    fd, path = tempfile.mkstemp(suffix=suffix)
    with os.fdopen(fd, "wb") as handle:
        handle.write(content)
    return path


def make_adapter() -> WhatsAppCloudAdapter:
    adapter = WhatsAppCloudAdapter.__new__(WhatsAppCloudAdapter)
    adapter.platform = Platform.WHATSAPP_CLOUD
    adapter.config = MagicMock()
    adapter.config.extra = {}
    adapter._phone_number_id = "1234567890"
    adapter._access_token = "test-token"
    adapter._api_version = "v20.0"
    adapter._http_client = None
    adapter._calling_sidecar_call_ids = set()
    adapter._seen_wamids = OrderedDict()
    adapter._warned_no_ffmpeg = False
    return adapter


def upload_files(adapter: WhatsAppCloudAdapter) -> dict[str, Any]:
    return adapter._http_client.post.call_args_list[0].kwargs["files"]


def send_payload(adapter: WhatsAppCloudAdapter) -> dict[str, Any]:
    return adapter._http_client.post.call_args_list[1].kwargs["json"]


async def verify_direct_ogg_upload() -> dict[str, Any]:
    adapter = make_adapter()
    adapter._http_client = MagicMock()
    adapter._http_client.post = AsyncMock(
        side_effect=[upload_response("voice_id"), message_response()]
    )
    adapter._convert_to_opus = AsyncMock()

    path = temp_file(".ogg", b"OggS")
    try:
        result = await adapter.send_voice("15551234567", path)
        require(result.success is True, f"direct Ogg send failed: {result.error}")
        adapter._convert_to_opus.assert_not_awaited()
        files = upload_files(adapter)
        payload = send_payload(adapter)
        require(
            files["file"][2] == WHATSAPP_OPUS_MIME,
            f"direct Ogg file MIME was {files['file'][2]!r}",
        )
        require(
            files["type"][1] == WHATSAPP_OPUS_MIME,
            f"direct Ogg type MIME was {files['type'][1]!r}",
        )
        require(payload["type"] == "audio", "direct Ogg did not send audio type")
        require(payload["audio"]["id"] == "voice_id", "direct Ogg media id mismatch")
        return {
            "success": True,
            "converted": False,
            "upload_mime": files["file"][2],
            "send_type": payload["type"],
        }
    finally:
        os.unlink(path)


async def verify_converted_upload() -> dict[str, Any]:
    adapter = make_adapter()
    adapter._http_client = MagicMock()
    adapter._http_client.post = AsyncMock(
        side_effect=[upload_response("converted_id"), message_response()]
    )

    source_path = temp_file(".mp3", b"ID3")
    opus_path = temp_file(".ogg", b"OggS")
    adapter._convert_to_opus = AsyncMock(return_value=opus_path)
    try:
        result = await adapter.send_voice("15551234567", source_path)
        require(result.success is True, f"converted send failed: {result.error}")
        adapter._convert_to_opus.assert_awaited_once_with(source_path)
        files = upload_files(adapter)
        payload = send_payload(adapter)
        require(
            files["file"][2] == WHATSAPP_OPUS_MIME,
            f"converted file MIME was {files['file'][2]!r}",
        )
        require(
            files["type"][1] == WHATSAPP_OPUS_MIME,
            f"converted type MIME was {files['type'][1]!r}",
        )
        require(payload["type"] == "audio", "converted send did not use audio type")
        require(not os.path.exists(opus_path), "temporary converted Ogg was not removed")
        return {
            "success": True,
            "converted": True,
            "upload_mime": files["file"][2],
            "send_type": payload["type"],
            "temporary_removed": True,
        }
    finally:
        os.unlink(source_path)
        if os.path.exists(opus_path):
            os.unlink(opus_path)


async def verify_mp3_fallback() -> dict[str, Any]:
    adapter = make_adapter()
    adapter._http_client = MagicMock()
    adapter._http_client.post = AsyncMock(
        side_effect=[upload_response("audio_id"), message_response()]
    )
    adapter._convert_to_opus = AsyncMock(return_value=None)

    path = temp_file(".mp3", b"ID3")
    try:
        result = await adapter.send_voice("15551234567", path)
        require(result.success is True, f"fallback send failed: {result.error}")
        adapter._convert_to_opus.assert_awaited_once_with(path)
        files = upload_files(adapter)
        payload = send_payload(adapter)
        require(
            files["file"][2] == "audio/mpeg",
            f"fallback file MIME was {files['file'][2]!r}",
        )
        require(
            files["type"][1] == "audio/mpeg",
            f"fallback type MIME was {files['type'][1]!r}",
        )
        require(payload["type"] == "audio", "fallback send did not use audio type")
        return {
            "success": True,
            "converted": False,
            "upload_mime": files["file"][2],
            "send_type": payload["type"],
        }
    finally:
        os.unlink(path)


async def verify() -> dict[str, Any]:
    return {
        "success": True,
        "checks": {
            "direct_ogg": await verify_direct_ogg_upload(),
            "converted_audio": await verify_converted_upload(),
            "fallback_audio": await verify_mp3_fallback(),
        },
    }


def main() -> int:
    try:
        result = asyncio.run(verify())
    except VerificationError as exc:
        print(json.dumps({"success": False, "error": str(exc)}, indent=2))
        return 1
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
