"""Contract tests for the media + caption fallback paths.

Background: the SDK natively supports caption only for image/video. For
audio and generic files, Feishu does not render a "media + caption" atomic
message, so Hermes sends two messages: caption first, then the native
attachment.

Coverage:
  - send_voice (audio) with markdown caption
  - send_voice without caption (regular SDK path — sanity check)
  - send_document (file) with caption + file_name
  - send_document without caption (regular SDK path — sanity check)
  - send_voice when native audio send fails
  - send_voice with empty caption (falsy → no-caption path)
"""

from __future__ import annotations

import asyncio
import builtins
import json
import struct
import threading
from unittest.mock import AsyncMock

import pytest

pytest.importorskip("lark_oapi.channel")

from lark_oapi.channel.errors import FeishuChannelError, FeishuChannelErrorCode


@pytest.fixture
def fake_audio_path(tmp_path):
    p = tmp_path / "voice.ogg"
    p.write_bytes(b"\x00\x00fake_ogg_payload")
    return str(p)


@pytest.fixture
def fake_pdf_path(tmp_path):
    p = tmp_path / "report.pdf"
    p.write_bytes(b"%PDF-1.4 fake")
    return str(p)


def _project_post(captured) -> dict:
    """Parse a captured CapturedSend into a comparable dict shape."""
    body = captured.body
    content_str = body.get("content", "{}")
    try:
        content_parsed = json.loads(content_str) if isinstance(content_str, str) else content_str
    except json.JSONDecodeError:
        content_parsed = content_str
    return {
        "endpoint": captured.endpoint,
        "msg_type": body.get("msg_type"),
        "content": content_parsed,
        "extra": {k: v for k, v in captured.extra.items() if k == "receive_id_type"},
    }


# ---------------------------------------------------------------------------
# send_voice path
# ---------------------------------------------------------------------------


def test_send_voice_with_caption_sends_caption_then_audio(
    adapter_harness, fake_audio_path
):
    """Audio + caption: send caption first, then native audio attachment."""
    upload_mock = AsyncMock(return_value="file_unused_old_caption_path")
    adapter_harness.adapter._channel.upload_media = upload_mock

    result = asyncio.run(adapter_harness.adapter.send_voice(
        chat_id="oc_testchat",
        audio_path=fake_audio_path,
        caption="**Voice memo** for review",
    ))

    assert result.success
    upload_mock.assert_not_called()
    assert len(adapter_harness.captured_sends) == 2
    caption_send = _project_post(adapter_harness.captured_sends[0])
    audio_send = _project_post(adapter_harness.captured_sends[1])

    assert caption_send["msg_type"] == "post"
    assert caption_send["extra"] == {"receive_id_type": "chat_id"}

    md_nodes_text = [
        n["text"]
        for row in caption_send["content"]["zh_cn"]["content"]
        for n in row
        if n.get("tag") == "md"
    ]
    assert any("**Voice memo**" in t for t in md_nodes_text), (
        f"Expected caption markdown in tag:md nodes; got md texts: {md_nodes_text}"
    )
    assert audio_send["msg_type"] in ("audio", "post"), (
        f"Expected audio/post msg_type after caption; got {audio_send['msg_type']}"
    )


def test_send_voice_without_caption_uses_sdk_audio_direct(
    adapter_harness, fake_audio_path
):
    """No-caption path: channel.send({"audio": {"source": path}}) — upload_media not called."""
    upload_mock = AsyncMock()
    adapter_harness.adapter._channel.upload_media = upload_mock
    sent_messages = []
    original_send = adapter_harness.adapter._channel.send

    async def capture_send(chat_id, message, opts=None):
        sent_messages.append(message)
        return await original_send(chat_id, message, opts=opts)

    adapter_harness.adapter._channel.send = capture_send

    result = asyncio.run(adapter_harness.adapter.send_voice(
        chat_id="oc_testchat",
        audio_path=fake_audio_path,
    ))

    assert result.success
    upload_mock.assert_not_called()  # caption-less path bypasses upload helper
    assert sent_messages == [{"audio": {"source": fake_audio_path}}]
    assert len(adapter_harness.captured_sends) == 1
    actual = _project_post(adapter_harness.captured_sends[0])

    # No-caption path goes through conftest mock's audio branch.
    assert actual["msg_type"] in ("audio", "post"), (
        f"Expected audio/post msg_type without caption; got {actual['msg_type']}"
    )


def test_send_voice_upload_failure_returns_error_result(
    adapter_harness, fake_audio_path
):
    """When native audio send raises FeishuChannelError, helper returns SendResult.fail."""
    original_send = adapter_harness.adapter._channel.send

    async def fail_audio_send(chat_id, message, opts=None):
        if isinstance(message, dict) and "audio" in message:
            raise FeishuChannelError(
                FeishuChannelErrorCode.UPLOAD_FAILED,
                "simulated upload failure",
            )
        return await original_send(chat_id, message, opts=opts)

    adapter_harness.adapter._channel.send = fail_audio_send
    adapter_harness.adapter._channel.upload_media = AsyncMock(
        return_value="file_unused_old_caption_path"
    )

    result = asyncio.run(adapter_harness.adapter.send_voice(
        chat_id="oc_testchat",
        audio_path=fake_audio_path,
        caption="caption that wont be sent",
    ))

    assert not result.success
    assert "simulated upload failure" in (result.error or "").lower()
    assert len(adapter_harness.captured_sends) == 1
    caption_send = _project_post(adapter_harness.captured_sends[0])
    assert caption_send["msg_type"] == "post"


# ---------------------------------------------------------------------------
# send_document path
# ---------------------------------------------------------------------------


def test_send_document_with_caption_sends_caption_then_file(
    adapter_harness, fake_pdf_path
):
    """File + caption: send caption first, then send native file attachment."""
    upload_mock = AsyncMock()
    adapter_harness.adapter._channel.upload_media = upload_mock

    result = asyncio.run(adapter_harness.adapter.send_document(
        chat_id="oc_testchat",
        file_path=fake_pdf_path,
        caption="## Scan Result\n\nSee attachment for details",
        file_name="scan.pdf",
    ))

    assert result.success
    upload_mock.assert_not_called()
    assert len(adapter_harness.captured_sends) == 2
    caption_send = _project_post(adapter_harness.captured_sends[0])
    file_send = _project_post(adapter_harness.captured_sends[1])

    assert caption_send["msg_type"] == "post"
    md_nodes_text = [
        n["text"]
        for row in caption_send["content"]["zh_cn"]["content"]
        for n in row
        if n.get("tag") == "md"
    ]
    assert any("## Scan Result" in t for t in md_nodes_text), (
        f"Expected H2 markdown in tag:md nodes; got md texts: {md_nodes_text}"
    )
    assert file_send["msg_type"] == "file"


def test_send_document_without_caption_uses_sdk_file_direct(
    adapter_harness, fake_pdf_path
):
    """No-caption path: channel.send({"file": {...}}) — upload_media not called."""
    upload_mock = AsyncMock()
    adapter_harness.adapter._channel.upload_media = upload_mock

    result = asyncio.run(adapter_harness.adapter.send_document(
        chat_id="oc_testchat",
        file_path=fake_pdf_path,
        file_name="report.pdf",
    ))

    assert result.success
    upload_mock.assert_not_called()
    actual = _project_post(adapter_harness.captured_sends[0])
    assert actual["msg_type"] in ("file", "post"), (
        f"Expected file/post msg_type without caption; got {actual['msg_type']}"
    )


# ---------------------------------------------------------------------------
# Edge case: empty caption falls through
# ---------------------------------------------------------------------------


def test_send_voice_with_empty_caption_falls_through_to_no_caption_path(
    adapter_harness, fake_audio_path
):
    """Empty string caption is falsy, so only the native attachment is sent."""
    upload_mock = AsyncMock()
    adapter_harness.adapter._channel.upload_media = upload_mock

    result = asyncio.run(adapter_harness.adapter.send_voice(
        chat_id="oc_testchat",
        audio_path=fake_audio_path,
        caption="",
    ))

    assert result.success
    upload_mock.assert_not_called()
    actual = _project_post(adapter_harness.captured_sends[0])
    assert actual["msg_type"] in ("audio", "post"), (
        f"Empty caption should not route through caption path; got {actual['msg_type']}"
    )


def test_send_voice_uploads_opus_duration_before_sending(adapter_harness, tmp_path):
    audio_path = tmp_path / "voice.opus"
    ogg_page = bytearray(27)
    ogg_page[0:4] = b"OggS"
    struct.pack_into("<q", ogg_page, 6, 96000)
    audio_path.write_bytes(bytes(ogg_page))

    result = asyncio.run(adapter_harness.adapter.send_voice(
        chat_id="oc_testchat",
        audio_path=str(audio_path),
    ))

    assert result.success
    upload = adapter_harness.captured_sends[0]
    assert upload.endpoint == "im.v1.file.create"
    assert upload.body["file_type"] == "opus"
    assert upload.body["duration"] == 2000
    actual = _project_post(adapter_harness.captured_sends[1])
    assert actual["content"]["file_key"] == "file_test_1"


def test_send_voice_probes_duration_off_event_loop(adapter_harness, tmp_path, monkeypatch):
    audio_path = tmp_path / "voice.opus"
    audio_path.write_bytes(b"OggS" + b"\x00" * 32)
    event_loop_thread = threading.get_ident()
    probe_thread_ids = []

    def probe_duration(_file_path, _file_type):
        probe_thread_ids.append(threading.get_ident())
        return 2000

    monkeypatch.setattr(
        adapter_harness.adapter,
        "_probe_upload_duration_ms",
        probe_duration,
    )

    result = asyncio.run(adapter_harness.adapter.send_voice(
        chat_id="oc_testchat",
        audio_path=str(audio_path),
    ))

    assert result.success
    assert probe_thread_ids
    assert probe_thread_ids[0] != event_loop_thread


def test_duration_probe_does_not_depend_on_sdk_media_parsers(
    adapter_harness, tmp_path, monkeypatch
):
    audio_path = tmp_path / "voice.opus"
    ogg_page = bytearray(27)
    ogg_page[0:4] = b"OggS"
    struct.pack_into("<q", ogg_page, 6, 96000)
    audio_path.write_bytes(bytes(ogg_page))

    real_import = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        if name.startswith("lark_oapi.channel.outbound.media"):
            raise AssertionError("duration probing must not depend on SDK media parsers")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)

    assert adapter_harness.adapter._probe_upload_duration_ms(str(audio_path), "opus") == 2000


def test_send_video_uploads_mp4_duration_before_sending(adapter_harness, tmp_path):
    video_path = tmp_path / "clip.mp4"
    mvhd_payload = bytearray(32)
    mvhd_payload[0] = 0
    struct.pack_into(">I", mvhd_payload, 12, 1000)
    struct.pack_into(">I", mvhd_payload, 16, 5000)
    mvhd_box = struct.pack(">I4s", 8 + len(mvhd_payload), b"mvhd") + bytes(mvhd_payload)
    moov_box = struct.pack(">I4s", 8 + len(mvhd_box), b"moov") + mvhd_box
    video_path.write_bytes(moov_box)

    result = asyncio.run(adapter_harness.adapter.send_video(
        chat_id="oc_testchat",
        video_path=str(video_path),
    ))

    assert result.success
    upload = adapter_harness.captured_sends[0]
    assert upload.endpoint == "im.v1.file.create"
    assert upload.body["file_type"] == "mp4"
    assert upload.body["duration"] == 5000
    actual = _project_post(adapter_harness.captured_sends[1])
    assert actual["content"]["file_key"] == "file_test_1"


def test_send_video_without_caption_uses_sdk_video_direct(adapter_harness, tmp_path):
    video_path = tmp_path / "clip.mp4"
    video_path.write_bytes(b"\x00\x00fake_mp4_payload")
    sent_messages = []
    original_send = adapter_harness.adapter._channel.send

    async def capture_send(chat_id, message, opts=None):
        sent_messages.append(message)
        return await original_send(chat_id, message, opts=opts)

    adapter_harness.adapter._channel.send = capture_send

    result = asyncio.run(adapter_harness.adapter.send_video(
        chat_id="oc_testchat",
        video_path=str(video_path),
    ))

    assert result.success
    assert sent_messages == [{"video": {"source": str(video_path)}}]
    assert len(adapter_harness.captured_sends) == 1
    actual = _project_post(adapter_harness.captured_sends[0])
    assert actual["msg_type"] in ("video", "media", "post"), (
        f"Expected video/media/post msg_type without caption; got {actual['msg_type']}"
    )


def test_send_voice_rejects_empty_file(adapter_harness, tmp_path):
    empty_audio = tmp_path / "empty.opus"
    empty_audio.write_bytes(b"")

    result = asyncio.run(adapter_harness.adapter.send_voice(
        chat_id="oc_testchat",
        audio_path=str(empty_audio),
    ))

    assert not result.success
    assert "empty" in (result.error or "").lower()
    assert not adapter_harness.captured_sends


def test_send_video_rejects_empty_file(adapter_harness, tmp_path):
    empty_video = tmp_path / "empty.mp4"
    empty_video.write_bytes(b"")

    result = asyncio.run(adapter_harness.adapter.send_video(
        chat_id="oc_testchat",
        video_path=str(empty_video),
    ))

    assert not result.success
    assert "empty" in (result.error or "").lower()
    assert not adapter_harness.captured_sends
