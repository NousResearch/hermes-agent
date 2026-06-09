"""Inbound dispatch + dedup tests for PhotonAdapter.

These tests bypass the aiohttp server — they call ``_dispatch_inbound``
and ``_is_duplicate`` directly. That keeps them fast and means we can
exercise the message-shape parsing logic without binding ports.
"""
from __future__ import annotations

from typing import List

import pytest

from gateway.config import Platform, PlatformConfig
from gateway.platforms.base import MessageEvent, MessageType
from plugins.platforms.photon.adapter import PhotonAdapter


def _make_adapter(monkeypatch: pytest.MonkeyPatch) -> PhotonAdapter:
    # Avoid touching real auth.json / env.
    monkeypatch.setenv("PHOTON_PROJECT_ID", "test-project-id")
    monkeypatch.setenv("PHOTON_PROJECT_SECRET", "test-project-secret")
    monkeypatch.delenv("PHOTON_WEBHOOK_SECRET", raising=False)
    cfg = PlatformConfig(enabled=True, token="", extra={})
    return PhotonAdapter(cfg)


@pytest.mark.asyncio
async def test_dispatch_text_dm(monkeypatch: pytest.MonkeyPatch) -> None:
    adapter = _make_adapter(monkeypatch)
    captured: List[MessageEvent] = []

    async def fake_handle(event: MessageEvent) -> None:
        captured.append(event)

    monkeypatch.setattr(adapter, "handle_message", fake_handle)

    payload = {
        "event": "messages",
        "space": {"id": "any;-;+15551234567", "platform": "iMessage"},
        "message": {
            "id": "spc-msg-abc",
            "platform": "iMessage",
            "direction": "inbound",
            "timestamp": "2026-05-14T19:06:32.000Z",
            "sender": {"id": "+15551234567", "platform": "iMessage"},
            "space": {"id": "any;-;+15551234567", "platform": "iMessage"},
            "content": {"type": "text", "text": "hello world"},
        },
    }
    await adapter._dispatch_inbound(payload)

    assert len(captured) == 1
    event = captured[0]
    assert event.text == "hello world"
    assert event.message_type == MessageType.TEXT
    assert event.message_id == "spc-msg-abc"
    src = event.source
    assert src is not None
    assert src.platform == Platform("photon")
    assert src.chat_id == "any;-;+15551234567"
    assert src.chat_type == "dm"
    assert src.user_id == "+15551234567"


@pytest.mark.asyncio
async def test_dispatch_group_id_detected(monkeypatch: pytest.MonkeyPatch) -> None:
    adapter = _make_adapter(monkeypatch)
    captured: List[MessageEvent] = []

    async def fake_handle(event: MessageEvent) -> None:
        captured.append(event)

    monkeypatch.setattr(adapter, "handle_message", fake_handle)

    payload = {
        "event": "messages",
        "space": {"id": "any;+;group-guid-xyz", "platform": "iMessage"},
        "message": {
            "id": "spc-msg-grp",
            "timestamp": "2026-05-14T19:06:32.000Z",
            "sender": {"id": "+15551234567"},
            "space": {"id": "any;+;group-guid-xyz"},
            "content": {"type": "text", "text": "hi group"},
        },
    }
    await adapter._dispatch_inbound(payload)
    assert captured[0].source.chat_type == "group"


@pytest.mark.asyncio
async def test_dispatch_attachment_surfaces_marker(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter = _make_adapter(monkeypatch)
    captured: List[MessageEvent] = []

    async def fake_handle(event: MessageEvent) -> None:
        captured.append(event)

    monkeypatch.setattr(adapter, "handle_message", fake_handle)

    payload = {
        "event": "messages",
        "message": {
            "id": "spc-msg-att",
            "timestamp": "2026-05-14T19:06:32.000Z",
            "sender": {"id": "+15551234567"},
            "space": {"id": "any;-;+15551234567"},
            "content": {
                "type": "attachment",
                "name": "IMG_4127.HEIC",
                "mimeType": "image/heic",
                "size": 12345,
            },
        },
    }
    await adapter._dispatch_inbound(payload)
    assert len(captured) == 1
    event = captured[0]
    # Attachment carries metadata marker; mime → MessageType.PHOTO.
    assert "Photon attachment received" in event.text
    assert "IMG_4127.HEIC" in event.text
    assert event.message_type == MessageType.PHOTO


@pytest.mark.asyncio
async def test_dispatch_attachment_with_local_path(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    """When the sidecar has downloaded the bytes and forwarded ``localPath``,
    the adapter surfaces the real file via ``media_urls`` (vision-tool access)
    instead of the metadata-only marker."""
    monkeypatch.setenv("PHOTON_DOWNLOAD_DIR", str(tmp_path))
    adapter = _make_adapter(monkeypatch)
    captured: List[MessageEvent] = []

    async def fake_handle(event: MessageEvent) -> None:
        captured.append(event)

    monkeypatch.setattr(adapter, "handle_message", fake_handle)

    img = tmp_path / "IMG_4127.jpg"
    img.write_bytes(b"\xff\xd8\xff\xe0 jpeg bytes")

    payload = {
        "event": "messages",
        "message": {
            "id": "spc-msg-att-dl",
            "timestamp": "2026-05-14T19:06:32.000Z",
            "sender": {"id": "+15551234567"},
            "space": {"id": "any;-;+15551234567"},
            "content": {
                "type": "attachment",
                "name": "IMG_4127.jpg",
                "mimeType": "image/jpeg",
                "size": 12,
                "localPath": str(img),
            },
        },
    }
    await adapter._dispatch_inbound(payload)
    assert len(captured) == 1
    event = captured[0]
    assert event.media_urls == [str(img)]
    assert event.media_types == ["image/jpeg"]
    assert event.message_type == MessageType.PHOTO
    # No metadata marker once the real file is available.
    assert "Photon attachment received" not in event.text


@pytest.mark.asyncio
async def test_dispatch_attachment_missing_local_path_falls_back(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A non-existent ``localPath`` (download failed in the sidecar) falls back
    to the metadata marker rather than handing the agent a dead path."""
    adapter = _make_adapter(monkeypatch)
    captured: List[MessageEvent] = []

    async def fake_handle(event: MessageEvent) -> None:
        captured.append(event)

    monkeypatch.setattr(adapter, "handle_message", fake_handle)

    payload = {
        "event": "messages",
        "message": {
            "id": "spc-msg-att-fail",
            "timestamp": "2026-05-14T19:06:32.000Z",
            "sender": {"id": "+15551234567"},
            "space": {"id": "any;-;+15551234567"},
            "content": {
                "type": "attachment",
                "name": "IMG_4127.jpg",
                "mimeType": "image/jpeg",
                "size": 12,
                "localPath": "/tmp/does-not-exist-xyz.jpg",
            },
        },
    }
    await adapter._dispatch_inbound(payload)
    event = captured[0]
    assert event.media_urls == []
    assert "Photon attachment received" in event.text


@pytest.mark.asyncio
async def test_dispatch_reaction(monkeypatch: pytest.MonkeyPatch) -> None:
    """Tapback / emoji reactions arrive on the same stream and are surfaced."""
    adapter = _make_adapter(monkeypatch)
    captured: List[MessageEvent] = []

    async def fake_handle(event: MessageEvent) -> None:
        captured.append(event)

    monkeypatch.setattr(adapter, "handle_message", fake_handle)

    payload = {
        "event": "messages",
        "message": {
            "id": "spc-msg-react",
            "timestamp": "2026-05-14T19:06:32.000Z",
            "sender": {"id": "+15551234567"},
            "space": {"id": "any;-;+15551234567"},
            "content": {
                "type": "reaction",
                "emoji": "❤️",
                "target": "spc-msg-abc",
            },
        },
    }
    await adapter._dispatch_inbound(payload)
    event = captured[0]
    assert "❤️" in event.text
    assert event.message_type == MessageType.TEXT


@pytest.mark.asyncio
async def test_inbound_audio_maps_to_voice(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    """A voice memo (audio/*) must map to VOICE so the gateway transcribes it
    (AUDIO is the never-STT bucket)."""
    monkeypatch.setenv("PHOTON_DOWNLOAD_DIR", str(tmp_path))
    adapter = _make_adapter(monkeypatch)
    captured: List[MessageEvent] = []

    async def fake_handle(event: MessageEvent) -> None:
        captured.append(event)

    monkeypatch.setattr(adapter, "handle_message", fake_handle)

    memo = tmp_path / "memo.caf"
    memo.write_bytes(b"caf fake audio")
    payload = {
        "event": "messages",
        "message": {
            "id": "spc-msg-voice",
            "timestamp": "2026-05-14T19:06:32.000Z",
            "sender": {"id": "+15551234567"},
            "space": {"id": "any;-;+15551234567"},
            "content": {
                "type": "attachment",
                "name": "memo.caf",
                "mimeType": "audio/x-caf",
                "size": 14,
                "localPath": str(memo),
            },
        },
    }
    await adapter._dispatch_inbound(payload)
    event = captured[0]
    assert event.message_type == MessageType.VOICE
    assert event.media_urls == [str(memo)]


@pytest.mark.asyncio
async def test_inbound_caf_voice_transcoded_to_wav(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    """An iMessage .caf voice memo (which arrives as application/octet-stream
    and isn't an STT-accepted format) is typed VOICE and transcoded to .wav so
    the gateway's transcription pipeline accepts it."""
    import shutil
    import subprocess

    if not shutil.which("ffmpeg"):
        pytest.skip("ffmpeg not available")

    caf = tmp_path / "Audio Message.caf"
    subprocess.run(
        ["ffmpeg", "-y", "-f", "lavfi", "-i", "sine=frequency=440:duration=1",
         str(caf)],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True,
    )

    monkeypatch.setenv("PHOTON_DOWNLOAD_DIR", str(tmp_path))
    adapter = _make_adapter(monkeypatch)
    captured: List[MessageEvent] = []

    async def fake_handle(event: MessageEvent) -> None:
        captured.append(event)

    monkeypatch.setattr(adapter, "handle_message", fake_handle)

    payload = {
        "event": "messages",
        "message": {
            "id": "spc-msg-caf",
            "timestamp": "2026-05-14T19:06:32.000Z",
            "sender": {"id": "+15551234567"},
            "space": {"id": "any;-;+15551234567"},
            "content": {
                "type": "attachment",
                "name": "Audio Message.caf",
                "mimeType": "application/octet-stream",  # how iMessage delivers it
                "size": caf.stat().st_size,
                "localPath": str(caf),
            },
        },
    }
    await adapter._dispatch_inbound(payload)
    event = captured[0]
    assert event.message_type == MessageType.VOICE
    assert len(event.media_urls) == 1
    assert event.media_urls[0].endswith(".wav")
    assert event.media_types == ["audio/wav"]


@pytest.mark.asyncio
async def test_inbound_video_extracts_frames_and_audio(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    """An inbound video is normalized into keyframes (image/*) + audio track
    (audio/*) so the existing vision + STT pipelines can consume it."""
    import shutil
    import subprocess

    if not shutil.which("ffmpeg"):
        pytest.skip("ffmpeg not available")

    clip = tmp_path / "clip.mp4"
    subprocess.run(
        [
            "ffmpeg", "-y", "-f", "lavfi",
            "-i", "testsrc=duration=2:size=320x240:rate=10",
            "-f", "lavfi", "-i", "sine=frequency=440:duration=2",
            "-c:v", "libx264", "-c:a", "aac", "-shortest", str(clip),
        ],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True,
    )

    monkeypatch.setenv("PHOTON_DOWNLOAD_DIR", str(tmp_path))
    adapter = _make_adapter(monkeypatch)
    captured: List[MessageEvent] = []

    async def fake_handle(event: MessageEvent) -> None:
        captured.append(event)

    monkeypatch.setattr(adapter, "handle_message", fake_handle)

    payload = {
        "event": "messages",
        "message": {
            "id": "spc-msg-video",
            "timestamp": "2026-05-14T19:06:32.000Z",
            "sender": {"id": "+15551234567"},
            "space": {"id": "any;-;+15551234567"},
            "content": {
                "type": "attachment",
                "name": "clip.mp4",
                "mimeType": "video/mp4",
                "size": clip.stat().st_size,
                "localPath": str(clip),
            },
        },
    }
    await adapter._dispatch_inbound(payload)
    event = captured[0]
    # Frames (image/jpeg) for vision + audio track (audio/wav) for STT.
    assert any(t == "image/jpeg" for t in event.media_types)
    assert any(t.startswith("audio/") for t in event.media_types)
    # No raw .mp4 left in the media set — it was normalized.
    assert not any(u.endswith(".mp4") for u in event.media_urls)


@pytest.mark.asyncio
async def test_inbound_too_large_attachment_note(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An oversized attachment (sidecar skipped the download, tooLarge=True)
    surfaces an actionable note so the agent replies promptly instead of the
    message blocking on a huge download."""
    adapter = _make_adapter(monkeypatch)
    captured: List[MessageEvent] = []

    async def fake_handle(event: MessageEvent) -> None:
        captured.append(event)

    monkeypatch.setattr(adapter, "handle_message", fake_handle)

    payload = {
        "event": "messages",
        "message": {
            "id": "spc-msg-big",
            "timestamp": "2026-05-14T19:06:32.000Z",
            "sender": {"id": "+15551234567"},
            "space": {"id": "any;-;+15551234567"},
            "content": {
                "type": "attachment",
                "name": "ScreenRecording.mov",
                "mimeType": "video/quicktime",
                "size": 60 * 1024 * 1024,
                "localPath": None,
                "tooLarge": True,
            },
        },
    }
    await adapter._dispatch_inbound(payload)
    event = captured[0]
    assert event.media_urls == []
    assert "too large" in event.text.lower()
    assert "60 MB" in event.text


@pytest.mark.asyncio
async def test_inbound_rejects_out_of_scope_localpath(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A forged localPath outside the sidecar download dir must NOT be handed to
    the agent (no arbitrary-file read via media_urls)."""
    adapter = _make_adapter(monkeypatch)
    captured: List[MessageEvent] = []

    async def fake_handle(event: MessageEvent) -> None:
        captured.append(event)

    monkeypatch.setattr(adapter, "handle_message", fake_handle)

    payload = {
        "event": "messages",
        "message": {
            "id": "spc-msg-evil",
            "timestamp": "2026-05-14T19:06:32.000Z",
            "sender": {"id": "+15551234567"},
            "space": {"id": "any;-;+15551234567"},
            "content": {
                "type": "attachment",
                "name": "passwd",
                "mimeType": "text/plain",
                "size": 100,
                "localPath": "/etc/passwd",
            },
        },
    }
    await adapter._dispatch_inbound(payload)
    event = captured[0]
    assert event.media_urls == []
    assert "/etc/passwd" not in (event.text or "")


def test_is_duplicate_window(monkeypatch: pytest.MonkeyPatch) -> None:
    adapter = _make_adapter(monkeypatch)
    assert adapter._is_duplicate("id-1") is False
    assert adapter._is_duplicate("id-1") is True
    assert adapter._is_duplicate("id-2") is False
    assert adapter._is_duplicate("id-1") is True  # still dup


def test_check_requirements_without_node(monkeypatch: pytest.MonkeyPatch) -> None:
    # If no node binary on PATH the adapter should refuse to start.
    from plugins.platforms.photon import adapter as adapter_mod

    monkeypatch.setattr(adapter_mod.shutil, "which", lambda _name: None)
    assert adapter_mod.check_requirements() is False


def test_webhook_binds_loopback_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    # Secure-by-default: the inbound receiver must not be exposed to the network
    # unless the operator explicitly opts in (the sidecar bridges over loopback).
    monkeypatch.delenv("PHOTON_WEBHOOK_BIND", raising=False)
    adapter = _make_adapter(monkeypatch)
    assert adapter._webhook_bind == "127.0.0.1"
