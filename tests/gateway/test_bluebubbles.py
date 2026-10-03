"""Tests for the BlueBubbles iMessage gateway adapter."""
import asyncio
import json
import threading
from pathlib import Path
from unittest.mock import AsyncMock

import httpx
import pytest

from gateway.config import Platform, PlatformConfig
from gateway.platforms.base import BasePlatformAdapter


def _make_adapter(monkeypatch, **extra):
    monkeypatch.setenv("BLUEBUBBLES_SERVER_URL", "http://localhost:1234")
    monkeypatch.setenv("BLUEBUBBLES_PASSWORD", "secret")
    from gateway.platforms.bluebubbles import BlueBubblesAdapter

    cfg = PlatformConfig(
        enabled=True,
        extra={
            "server_url": "http://localhost:1234",
            "password": "secret",
            **extra,
        },
    )
    return BlueBubblesAdapter(cfg)


class TestBlueBubblesConfigLoading:
    def test_apply_env_overrides_bluebubbles(self, monkeypatch):
        monkeypatch.setenv("BLUEBUBBLES_SERVER_URL", "http://localhost:1234")
        monkeypatch.setenv("BLUEBUBBLES_PASSWORD", "secret")
        monkeypatch.setenv("BLUEBUBBLES_WEBHOOK_PORT", "9999")
        monkeypatch.setenv("BLUEBUBBLES_REQUIRE_MENTION", "true")
        monkeypatch.setenv("BLUEBUBBLES_MENTION_PATTERNS", r'["(?i)^amos\\b"]')
        from gateway.config import GatewayConfig, _apply_env_overrides

        config = GatewayConfig()
        _apply_env_overrides(config)
        assert Platform.BLUEBUBBLES in config.platforms
        bc = config.platforms[Platform.BLUEBUBBLES]
        assert bc.enabled is True
        assert bc.extra["server_url"] == "http://localhost:1234"
        assert bc.extra["password"] == "secret"
        assert bc.extra["webhook_port"] == 9999
        assert bc.extra["require_mention"] is True
        assert bc.extra["mention_patterns"] == ["(?i)^amos\\b"]


class TestBlueBubblesHelpers:


    def test_format_message_preserves_underscores_in_identifiers(self, monkeypatch):
        adapter = _make_adapter(monkeypatch)
        text = "Use /api_v2 with FEATURE_FLAG_NAME and config_file.json"
        assert adapter.format_message(text) == text

    def test_strip_markdown_headers(self, monkeypatch):
        adapter = _make_adapter(monkeypatch)
        assert adapter.format_message("## Heading\ntext") == "Heading\ntext"


    def test_init_normalizes_webhook_path(self, monkeypatch):
        adapter = _make_adapter(monkeypatch, webhook_path="bluebubbles-webhook")
        assert adapter.webhook_path == "/bluebubbles-webhook"


    def test_server_url_normalized(self, monkeypatch):
        adapter = _make_adapter(monkeypatch, server_url="http://localhost:1234/")
        assert adapter.server_url == "http://localhost:1234"


class _FakeBlueBubblesRequest:
    def __init__(self, payload, password="secret"):
        self.query = {"password": password}
        self.headers = {}
        self._body = json.dumps(payload).encode("utf-8")

    async def read(self):
        return self._body


class TestBlueBubblesMentionGating:
    @pytest.mark.asyncio
    async def test_group_message_without_mention_is_acknowledged_and_skipped(self, monkeypatch):
        adapter = _make_adapter(
            monkeypatch,
            require_mention=True,
            send_read_receipts=False,
        )
        handled = []

        async def fake_handle_message(event):
            handled.append(event)

        monkeypatch.setattr(adapter, "handle_message", fake_handle_message)
        response = await adapter._handle_webhook(_FakeBlueBubblesRequest({
            "type": "new-message",
            "data": {
                "guid": "msg-1",
                "text": "casual family chatter",
                "handle": {"address": "+15555550100"},
                "isFromMe": False,
                "isGroup": True,
                "chats": [{"guid": "iMessage;+;group-chat"}],
            },
        }))
        await asyncio.sleep(0)

        assert response.status == 200
        assert handled == []


class TestBlueBubblesWebhookParsing:

    def test_webhook_can_fall_back_to_sender_when_chat_fields_missing(self, monkeypatch):
        adapter = _make_adapter(monkeypatch)
        payload = {
            "data": {
                "guid": "MESSAGE-GUID",
                "text": "hello",
                "handle": {"address": "user@example.com"},
                "isFromMe": False,
            }
        }
        record = adapter._extract_payload_record(payload) or {}
        chat_guid = adapter._value(
            record.get("chatGuid"),
            payload.get("chatGuid"),
            record.get("chat_guid"),
            payload.get("chat_guid"),
            payload.get("guid"),
        )
        chat_identifier = adapter._value(
            record.get("chatIdentifier"),
            record.get("identifier"),
            payload.get("chatIdentifier"),
            payload.get("identifier"),
        )
        sender = (
            adapter._value(
                record.get("handle", {}).get("address")
                if isinstance(record.get("handle"), dict)
                else None,
                record.get("sender"),
                record.get("from"),
                record.get("address"),
            )
            or chat_identifier
            or chat_guid
        )
        if not (chat_guid or chat_identifier) and sender:
            chat_identifier = sender
        assert chat_identifier == "user@example.com"


    def test_extract_payload_record_accepts_list_data(self, monkeypatch):
        adapter = _make_adapter(monkeypatch)
        payload = {
            "type": "new-message",
            "data": [
                {
                    "text": "hello",
                    "chatGuid": "iMessage;-;user@example.com",
                    "chatIdentifier": "user@example.com",
                }
            ],
        }
        record = adapter._extract_payload_record(payload)
        assert record == payload["data"][0]


class TestBlueBubblesGuidResolution:


    @pytest.mark.asyncio
    async def test_participant_only_match_does_not_resolve_to_group(self, monkeypatch):
        """Regression for #24157: contact appearing as a participant in a group
        chat must NOT be selected when no DM with that exact chatIdentifier exists.

        Otherwise an outbound DM reply leaks into the group thread.
        """
        adapter = _make_adapter(monkeypatch)

        async def fake_api_post(path, payload):
            return {
                "data": [
                    {
                        "guid": "iMessage;+;chat0000000000-family-group",
                        "chatIdentifier": "chat0000000000",
                        "participants": [
                            {"address": "user@example.com"},
                            {"address": "+15555550100"},
                        ],
                    }
                ]
            }

        monkeypatch.setattr(adapter, "_api_post", fake_api_post)
        result = await adapter._resolve_chat_guid("user@example.com")
        assert result is None, (
            "participant-only match must not resolve to a group GUID — DM "
            "replies would leak into the group thread"
        )


    @pytest.mark.asyncio
    async def test_unresolved_target_is_not_cached(self, monkeypatch):
        """When no exact match is found, the resolver must NOT cache anything.

        Otherwise a later attempt — after the DM has been created — would
        keep returning the stale ``None`` from cache. Also guards against a
        latent variant of #24157 where a group GUID could be cached under a
        bare address key and persist across calls.
        """
        adapter = _make_adapter(monkeypatch)

        async def fake_api_post(path, payload):
            return {
                "data": [
                    {
                        "guid": "iMessage;+;chat0000000000-family-group",
                        "chatIdentifier": "chat0000000000",
                        "participants": [{"address": "user@example.com"}],
                    }
                ]
            }

        monkeypatch.setattr(adapter, "_api_post", fake_api_post)
        await adapter._resolve_chat_guid("user@example.com")
        assert "user@example.com" not in adapter._guid_cache


class TestBlueBubblesAttachmentDownload:
    """Verify _download_attachment routes to the correct cache helper."""

    def test_download_image_uses_image_cache(self, monkeypatch):
        """Image MIME routes to cache_image_from_bytes."""
        adapter = _make_adapter(monkeypatch)
        import asyncio

        # Mock the HTTP client response
        class MockResponse:
            status_code = 200
            content = b"\x89PNG\r\n\x1a\n"

            def raise_for_status(self):
                pass

        async def mock_get(*args, **kwargs):
            return MockResponse()

        adapter.client = type("MockClient", (), {"get": mock_get})()

        cached_path = None

        async def mock_cache_image(data, ext):
            nonlocal cached_path
            cached_path = f"test_image{ext}"
            return cached_path

        monkeypatch.setattr(
            "gateway.platforms.bluebubbles.cache_image_from_bytes_async",
            mock_cache_image,
        )

        att_meta = {"mimeType": "image/png", "transferName": "photo.png"}
        result = asyncio.get_event_loop().run_until_complete(
            adapter._download_attachment("att-guid-123", att_meta)
        )
        assert result == "test_image.png"

    @pytest.mark.asyncio
    async def test_download_caf_audio_preserves_caf_extension(self, monkeypatch):
        adapter = _make_adapter(monkeypatch)

        class MockResponse:
            content = b"caffake"

            def raise_for_status(self):
                pass

        async def mock_get(*args, **kwargs):
            return MockResponse()

        adapter.client = type("MockClient", (), {"get": mock_get})()  # type: ignore[reportAttributeAccessIssue]
        captured = {}

        async def mock_cache_audio(data, ext):
            captured["data"] = data
            captured["ext"] = ext
            return f"test_audio{ext}"

        monkeypatch.setattr(
            "gateway.platforms.bluebubbles.cache_audio_from_bytes_async",
            mock_cache_audio,
        )
        result = await adapter._download_attachment(
            "att-guid-caf",
            {"mimeType": "audio/x-caf", "transferName": "Audio Message.caf"},
        )
        assert result == "test_audio.caf"
        assert captured == {"data": b"caffake", "ext": ".caf"}


# ---------------------------------------------------------------------------
# Native voice sending
# ---------------------------------------------------------------------------


class TestBlueBubblesVoiceSend:
    @pytest.mark.asyncio
    async def test_existing_caf_uploads_as_private_api_audio_message(
        self, monkeypatch, tmp_path
    ):
        adapter = _make_adapter(monkeypatch)
        adapter._private_api_enabled = True
        adapter._helper_connected = True
        audio_path = tmp_path / "Audio Message.caf"
        audio_path.write_bytes(b"caffake")

        async def fake_resolve_chat_guid(chat_id):
            return "iMessage;-;user@example.com"

        captured = {}

        async def fake_post(self, url, *, files, data, timeout):
            captured["files"] = files
            captured["data"] = data

            class Response:
                def raise_for_status(self):
                    pass

                def json(self):
                    return {"status": 200, "data": {"guid": "out-guid"}}

            return Response()

        adapter.client = type("MockClient", (), {"post": fake_post})()  # type: ignore[reportAttributeAccessIssue]
        monkeypatch.setattr(adapter, "_resolve_chat_guid", fake_resolve_chat_guid)

        result = await adapter.send_voice("user@example.com", str(audio_path))

        assert result.success is True
        assert captured["data"]["isAudioMessage"] == "true"
        assert captured["data"]["method"] == "private-api"
        assert captured["files"]["attachment"][0] == "Audio Message.caf"
        assert captured["files"]["attachment"][2] == "audio/x-caf"
        assert audio_path.exists()

    @pytest.mark.asyncio
    async def test_native_caf_without_helper_omits_private_api_method(
        self, monkeypatch, tmp_path
    ):
        adapter = _make_adapter(monkeypatch)
        adapter._private_api_enabled = True
        adapter._helper_connected = False
        audio_path = tmp_path / "Audio Message.caf"
        audio_path.write_bytes(b"caffake")

        async def fake_resolve_chat_guid(chat_id):
            return "iMessage;-;user@example.com"

        captured = {}

        async def fake_post(self, url, *, files, data, timeout):
            captured["data"] = dict(data)

            class Response:
                def raise_for_status(self):
                    pass

                def json(self):
                    return {"status": 200, "data": {"guid": "out-guid"}}

            return Response()

        adapter.client = type("MockClient", (), {"post": fake_post})()  # type: ignore[reportAttributeAccessIssue]
        monkeypatch.setattr(adapter, "_resolve_chat_guid", fake_resolve_chat_guid)

        result = await adapter.send_voice("user@example.com", str(audio_path))

        assert result.success is True
        assert captured["data"]["isAudioMessage"] == "true"
        assert "method" not in captured["data"]
        assert audio_path.exists()

    def test_prepare_mp3_normalizes_to_24khz_mono_opus_caf(
        self, monkeypatch, tmp_path
    ):
        adapter = _make_adapter(monkeypatch)
        source = tmp_path / "voice.mp3"
        source.write_bytes(b"mp3fake")
        calls = []
        monkeypatch.setattr(
            "gateway.platforms.bluebubbles.shutil.which",
            lambda name: f"/usr/bin/{name}"
            if name in {"ffmpeg", "afconvert"}
            else None,
        )

        def fake_run(command, **kwargs):
            calls.append((command, kwargs))
            Path(command[-1]).write_bytes(b"audio")

        monkeypatch.setattr(
            "gateway.platforms.bluebubbles.subprocess.run", fake_run
        )
        prepared = adapter._prepare_voice_attachment(str(source))
        try:
            assert prepared.filename == "Audio Message.caf"
            assert prepared.content_type == "audio/x-caf"
            assert prepared.cleanup is True
            assert len(calls) == 2
            assert calls[0][0][0] == "/usr/bin/ffmpeg"
            assert calls[0][0][calls[0][0].index("-ac") + 1] == "1"
            assert calls[0][0][calls[0][0].index("-ar") + 1] == "24000"
            assert calls[1][0][0] == "/usr/bin/afconvert"
            assert "opus@24000" in calls[1][0]
            assert all(call[1]["stdin"] is not None for call in calls)
            from hermes_cli._subprocess_compat import windows_hide_flags
            assert all(call[1]["creationflags"] == windows_hide_flags() for call in calls)
        finally:
            Path(prepared.path).unlink(missing_ok=True)

    def test_conversion_failure_removes_temporary_caf_and_normalized_wav(
        self, monkeypatch, tmp_path
    ):
        from gateway.platforms import bluebubbles as bluebubbles_module

        adapter = _make_adapter(monkeypatch)
        source = tmp_path / "voice.mp3"
        source.write_bytes(b"mp3fake")
        created = []
        original_tempfile = bluebubbles_module.tempfile.NamedTemporaryFile

        def track_tempfile(*args, **kwargs):
            handle = original_tempfile(*args, dir=tmp_path, **kwargs)
            created.append(Path(handle.name))
            return handle

        monkeypatch.setattr(bluebubbles_module.tempfile, "NamedTemporaryFile", track_tempfile)
        monkeypatch.setattr(
            bluebubbles_module.shutil,
            "which",
            lambda name: f"/usr/bin/{name}" if name in {"ffmpeg", "afconvert"} else None,
        )

        def fake_run(command, **kwargs):
            if command[0] == "/usr/bin/ffmpeg":
                Path(command[-1]).write_bytes(b"normalized wav")
            # Simulate afconvert exiting successfully without writing CAF output.

        monkeypatch.setattr(bluebubbles_module.subprocess, "run", fake_run)

        assert adapter._convert_audio_to_caf(str(source)) is None
        assert len(created) == 2
        assert all(not path.exists() for path in created)

    def test_missing_ffmpeg_removes_created_caf_temp(self, monkeypatch, tmp_path):
        from gateway.platforms import bluebubbles as bluebubbles_module

        adapter = _make_adapter(monkeypatch)
        source = tmp_path / "voice.mp3"
        source.write_bytes(b"mp3fake")
        created = []
        original_tempfile = bluebubbles_module.tempfile.NamedTemporaryFile

        def track_tempfile(*args, **kwargs):
            handle = original_tempfile(*args, dir=tmp_path, **kwargs)
            created.append(Path(handle.name))
            return handle

        monkeypatch.setattr(bluebubbles_module.tempfile, "NamedTemporaryFile", track_tempfile)
        monkeypatch.setattr(
            bluebubbles_module.shutil,
            "which",
            lambda name: "/usr/bin/afconvert" if name == "afconvert" else None,
        )

        assert adapter._convert_audio_to_caf(str(source)) is None
        assert len(created) == 1
        assert not created[0].exists()

    @pytest.mark.asyncio
    async def test_default_mp3_voice_uploads_as_native_caf_and_cleans_temp(
        self, monkeypatch, tmp_path
    ):
        from gateway.platforms import bluebubbles as bluebubbles_module

        adapter = _make_adapter(monkeypatch)
        adapter._private_api_enabled = True
        adapter._helper_connected = True
        source = tmp_path / "generated-voice.mp3"
        source.write_bytes(b"mp3fake")
        created = []
        original_tempfile = bluebubbles_module.tempfile.NamedTemporaryFile
        captured = {}

        def track_tempfile(*args, **kwargs):
            handle = original_tempfile(*args, dir=tmp_path, **kwargs)
            created.append(Path(handle.name))
            return handle

        async def fake_resolve_chat_guid(chat_id):
            return "iMessage;-;user@example.com"

        def fake_run(command, **kwargs):
            Path(command[-1]).write_bytes(b"normalized wav" if command[0] == "/usr/bin/ffmpeg" else b"opus caf")

        async def fake_post(self, url, *, files, data, timeout):
            captured["filename"] = files["attachment"][0]
            captured["payload"] = files["attachment"][1]
            captured["content_type"] = files["attachment"][2]
            captured["data"] = dict(data)

            class Response:
                def raise_for_status(self):
                    pass

                def json(self):
                    return {"status": 200, "data": {"guid": "native-voice"}}

            return Response()

        monkeypatch.setattr(bluebubbles_module.tempfile, "NamedTemporaryFile", track_tempfile)
        monkeypatch.setattr(
            bluebubbles_module.shutil,
            "which",
            lambda name: f"/usr/bin/{name}" if name in {"ffmpeg", "afconvert"} else None,
        )
        monkeypatch.setattr(bluebubbles_module.subprocess, "run", fake_run)
        monkeypatch.setattr(adapter, "_resolve_chat_guid", fake_resolve_chat_guid)
        adapter.client = type("MockClient", (), {"post": fake_post})()  # type: ignore[reportAttributeAccessIssue]

        result = await adapter.send_voice("user@example.com", str(source))

        assert result.success is True
        assert captured["filename"] == "Audio Message.caf"
        assert captured["payload"] == b"opus caf"
        assert captured["content_type"] == "audio/x-caf"
        assert captured["data"]["isAudioMessage"] == "true"
        assert captured["data"]["method"] == "private-api"
        assert len(created) == 2
        assert all(not path.exists() for path in created)

    @pytest.mark.parametrize(
        ("suffix", "expected_content_type"),
        [(".mp3", "audio/mpeg"), (".m2a", "audio/mpeg")],
    )
    @pytest.mark.asyncio
    async def test_send_voice_conversion_unavailable_is_ordinary_audio(
        self, monkeypatch, tmp_path, caplog, suffix, expected_content_type
    ):
        adapter = _make_adapter(monkeypatch)
        adapter._private_api_enabled = True
        adapter._helper_connected = True
        source = tmp_path / f"voice{suffix}"
        source.write_bytes(b"mp3fake")

        async def fake_resolve_chat_guid(chat_id):
            return "iMessage;-;user@example.com"

        captured = {}

        async def fake_post(self, url, *, files, data, timeout):
            captured["filename"] = files["attachment"][0]
            captured["content_type"] = files["attachment"][2]
            captured["data"] = dict(data)

            class Response:
                def raise_for_status(self):
                    pass

                def json(self):
                    return {"status": 200, "data": {"guid": "out-guid"}}

            return Response()

        adapter.client = type("MockClient", (), {"post": fake_post})()  # type: ignore[reportAttributeAccessIssue]
        monkeypatch.setattr(adapter, "_resolve_chat_guid", fake_resolve_chat_guid)
        monkeypatch.setattr(adapter, "_convert_audio_to_caf", lambda path: None)

        result = await adapter.send_voice("user@example.com", str(source))

        assert result.success is True
        assert captured["filename"] == f"voice{suffix}"
        assert captured["content_type"] == expected_content_type
        assert "isAudioMessage" not in captured["data"]
        assert "method" not in captured["data"]
        assert "ordinary audio attachment" in caplog.text

    @pytest.mark.asyncio
    async def test_tempfile_setup_failure_falls_back_to_typed_audio(self, monkeypatch, tmp_path):
        from gateway.platforms import bluebubbles as bluebubbles_module

        adapter = _make_adapter(monkeypatch)
        adapter._private_api_enabled = True
        adapter._helper_connected = True
        source = tmp_path / "voice.mp3"
        source.write_bytes(b"mp3fake")
        captured = {}

        async def fake_resolve_chat_guid(chat_id):
            return "iMessage;-;user@example.com"

        async def fake_post(self, url, *, files, data, timeout):
            captured["filename"] = files["attachment"][0]
            captured["content_type"] = files["attachment"][2]
            captured["data"] = dict(data)

            class Response:
                def raise_for_status(self):
                    pass

                def json(self):
                    return {"status": 200, "data": {"guid": "audio-fallback"}}

            return Response()

        def fail_tempfile(*args, **kwargs):
            raise OSError("temporary directory unavailable")

        monkeypatch.setattr(adapter, "_resolve_chat_guid", fake_resolve_chat_guid)
        monkeypatch.setattr(bluebubbles_module.shutil, "which", lambda name: "/usr/bin/afconvert")
        monkeypatch.setattr(bluebubbles_module.tempfile, "NamedTemporaryFile", fail_tempfile)
        adapter.client = type("MockClient", (), {"post": fake_post})()  # type: ignore[reportAttributeAccessIssue]

        result = await adapter.send_voice("user@example.com", str(source))

        assert result.success is True
        assert captured["filename"] == "voice.mp3"
        assert captured["content_type"] == "audio/mpeg"
        assert "isAudioMessage" not in captured["data"]
        assert "method" not in captured["data"]

    @pytest.mark.asyncio
    async def test_send_voice_prepares_off_loop_and_cleans_generated_caf(
        self, monkeypatch, tmp_path
    ):
        from gateway.platforms.bluebubbles import _PreparedAttachment

        adapter = _make_adapter(monkeypatch)
        adapter._private_api_enabled = True
        adapter._helper_connected = True
        source = tmp_path / "voice.wav"
        source.write_bytes(b"wavfake")
        generated = tmp_path / "generated.caf"
        generated.write_bytes(b"caffake")
        offloaded = []
        captured = {}

        async def fake_resolve_chat_guid(chat_id):
            return "iMessage;-;user@example.com"

        def fake_prepare(path, filename):
            return _PreparedAttachment(
                path=str(generated),
                filename="Audio Message.caf",
                content_type="audio/x-caf",
                cleanup=True,
                native_voice=True,
            )

        async def fake_to_thread(func, *args):
            offloaded.append((func, args))
            return func(*args)

        async def fake_post(self, url, *, files, data, timeout):
            captured["exists_during_upload"] = generated.exists()
            captured["filename"] = files["attachment"][0]
            captured["payload"] = files["attachment"][1]
            captured["content_type"] = files["attachment"][2]
            captured["data"] = dict(data)

            class Response:
                def raise_for_status(self):
                    pass

                def json(self):
                    return {"status": 200, "data": {"guid": "out-guid"}}

            return Response()

        adapter.client = type("MockClient", (), {"post": fake_post})()  # type: ignore[reportAttributeAccessIssue]
        monkeypatch.setattr(adapter, "_resolve_chat_guid", fake_resolve_chat_guid)
        monkeypatch.setattr(adapter, "_prepare_voice_attachment", fake_prepare)
        monkeypatch.setattr(asyncio, "to_thread", fake_to_thread)

        result = await adapter.send_voice("user@example.com", str(source))

        assert result.success is True
        assert any(func is fake_prepare for func, _ in offloaded)
        assert any(
            getattr(func, "__name__", "") == "read_bytes" for func, _ in offloaded
        )
        assert captured["exists_during_upload"] is True
        assert captured["filename"] == "Audio Message.caf"
        assert captured["payload"] == b"caffake"
        assert captured["content_type"] == "audio/x-caf"
        assert captured["data"]["isAudioMessage"] == "true"
        assert captured["data"]["method"] == "private-api"
        assert not generated.exists()

    @pytest.mark.asyncio
    async def test_cancelled_preparation_cleans_completed_thread_result(
        self, monkeypatch, tmp_path
    ):
        from gateway.platforms.bluebubbles import _PreparedAttachment

        adapter = _make_adapter(monkeypatch)
        adapter.client = object()  # type: ignore[reportAttributeAccessIssue]
        source = tmp_path / "voice.mp3"
        source.write_bytes(b"mp3fake")
        generated = tmp_path / "cancelled.caf"
        generated.write_bytes(b"caffake")
        started = threading.Event()
        release = threading.Event()
        cleanup_complete = threading.Event()

        async def fake_resolve_chat_guid(chat_id):
            return "iMessage;-;user@example.com"

        def blocking_prepare(path, filename):
            started.set()
            release.wait(timeout=5)
            return _PreparedAttachment(
                path=str(generated),
                filename="Audio Message.caf",
                content_type="audio/x-caf",
                cleanup=True,
            )

        original_cleanup = adapter._cleanup_prepared_attachment

        def signal_cleanup(prepared):
            original_cleanup(prepared)
            cleanup_complete.set()

        monkeypatch.setattr(adapter, "_resolve_chat_guid", fake_resolve_chat_guid)
        monkeypatch.setattr(adapter, "_prepare_voice_attachment", blocking_prepare)
        monkeypatch.setattr(adapter, "_cleanup_prepared_attachment", signal_cleanup)
        send_task = asyncio.create_task(
            adapter.send_voice("user@example.com", str(source))
        )
        assert await asyncio.to_thread(started.wait, 5)
        send_task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await send_task
        release.set()
        assert await asyncio.to_thread(cleanup_complete.wait, 5)
        assert not generated.exists()

    @pytest.mark.asyncio
    async def test_cancelled_attachment_read_consumes_error_and_cleans_payload(
        self, monkeypatch, tmp_path
    ):
        import os
        from gateway.platforms.bluebubbles import _PreparedAttachment

        adapter = _make_adapter(monkeypatch)
        adapter.client = object()  # type: ignore[reportAttributeAccessIssue]
        source = tmp_path / "voice.mp3"
        source.write_bytes(b"mp3fake")
        generated = tmp_path / "reading.caf"
        generated.write_bytes(b"caffake")
        read_started = asyncio.Event()
        release_read = asyncio.Event()
        cleanup_complete = asyncio.Event()
        prepared = _PreparedAttachment(
            path=str(generated),
            filename="Audio Message.caf",
            content_type="audio/x-caf",
            cleanup=True,
            native_voice=True,
        )

        async def fake_resolve_chat_guid(chat_id):
            return "iMessage;-;user@example.com"

        def fake_prepare(path, filename):
            return prepared

        async def fake_to_thread(func, *args):
            if func is os.path.isfile:
                return func(*args)
            if getattr(func, "__name__", "") == "fake_prepare":
                return prepared
            if getattr(func, "__name__", "") == "read_bytes":
                read_started.set()
                await release_read.wait()
                raise OSError("synthetic read failure after cancellation")
            raise AssertionError(f"unexpected worker: {func!r}")

        original_cleanup = adapter._cleanup_prepared_attachment

        def signal_cleanup(attachment):
            original_cleanup(attachment)
            cleanup_complete.set()

        monkeypatch.setattr(adapter, "_resolve_chat_guid", fake_resolve_chat_guid)
        monkeypatch.setattr(adapter, "_prepare_voice_attachment", fake_prepare)
        monkeypatch.setattr(adapter, "_cleanup_prepared_attachment", signal_cleanup)
        monkeypatch.setattr(asyncio, "to_thread", fake_to_thread)
        send_task = asyncio.create_task(adapter.send_voice("user@example.com", str(source)))
        await asyncio.wait_for(read_started.wait(), timeout=5)
        send_task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await send_task
        assert generated.exists()

        release_read.set()
        await asyncio.wait_for(cleanup_complete.wait(), timeout=5)
        assert not generated.exists()

    @pytest.mark.asyncio
    async def test_preparation_failure_is_contained(self, monkeypatch, tmp_path):
        adapter = _make_adapter(monkeypatch)
        adapter.client = object()  # type: ignore[reportAttributeAccessIssue]
        source = tmp_path / "voice.mp3"
        source.write_bytes(b"mp3fake")

        async def fake_resolve_chat_guid(chat_id):
            return "iMessage;-;user@example.com"

        def fail_prepare(path, filename):
            raise OSError("temporary file unavailable")

        monkeypatch.setattr(adapter, "_resolve_chat_guid", fake_resolve_chat_guid)
        monkeypatch.setattr(adapter, "_prepare_voice_attachment", fail_prepare)
        result = await adapter.send_voice("user@example.com", str(source))
        assert result.success is False
        assert result.error == "temporary file unavailable"


class TestBlueBubblesAttachmentSend:
    @pytest.mark.asyncio
    async def test_attachment_payload_is_read_before_async_upload(self, monkeypatch, tmp_path):
        adapter = _make_adapter(monkeypatch)
        file_path = tmp_path / "payload.bin"
        payload = b"attachment-payload"
        file_path.write_bytes(payload)

        captured = {}

        async def fake_resolve_chat_guid(chat_id):
            return "iMessage;+;chat-guid"

        class MockResponse:
            def raise_for_status(self):
                pass

            def json(self):
                return {"status": 200, "data": {"guid": "message-guid"}}

        class MockClient:
            async def post(self, url, *, files, data, timeout):
                captured.update(url=url, files=files, data=data, timeout=timeout)
                return MockResponse()

        monkeypatch.setattr(adapter, "_resolve_chat_guid", fake_resolve_chat_guid)
        adapter.client = MockClient()

        result = await adapter._send_attachment(
            "target", str(file_path), filename="payload.bin"
        )

        assert result.success is True
        assert captured["files"]["attachment"] == (
            "payload.bin",
            payload,
            "application/octet-stream",
        )
        assert captured["data"]["chatGuid"] == "iMessage;+;chat-guid"


# ---------------------------------------------------------------------------
# Webhook registration
# ---------------------------------------------------------------------------


class TestBlueBubblesWebhookUrl:
    """_webhook_url property normalises local hosts to 'localhost'."""

    def test_default_host(self, monkeypatch):
        monkeypatch.delenv("BLUEBUBBLES_WEBHOOK_HOST", raising=False)
        adapter = _make_adapter(monkeypatch)
        # Default local bind host is normalized to localhost for callbacks.
        assert "localhost" in adapter._webhook_url
        assert str(adapter.webhook_port) in adapter._webhook_url
        assert adapter.webhook_path in adapter._webhook_url


    def test_register_url_omits_query_when_no_password(self, monkeypatch):
        """If no password is configured, the register URL should be the bare URL."""
        monkeypatch.delenv("BLUEBUBBLES_PASSWORD", raising=False)
        from gateway.platforms.bluebubbles import BlueBubblesAdapter
        cfg = PlatformConfig(
            enabled=True,
            extra={"server_url": "http://localhost:1234", "password": ""},
        )
        adapter = BlueBubblesAdapter(cfg)
        assert adapter._webhook_register_url == adapter._webhook_url


class TestBlueBubblesWebhookRegistration:
    """Tests for _register_webhook, _unregister_webhook, _find_registered_webhooks."""

    @staticmethod
    def _mock_client(get_response=None, post_response=None, delete_ok=True):
        """Build a tiny mock httpx.AsyncClient."""

        async def mock_get(*args, **kwargs):
            class R:
                status_code = 200
                def raise_for_status(self):
                    pass
                def json(self):
                    return get_response or {"status": 200, "data": []}
            return R()

        async def mock_post(*args, **kwargs):
            class R:
                status_code = 200
                def raise_for_status(self):
                    pass
                def json(self):
                    return post_response or {"status": 200, "data": {}}
            return R()

        async def mock_delete(*args, **kwargs):
            class R:
                status_code = 200 if delete_ok else 500
                def raise_for_status(self_inner):
                    if not delete_ok:
                        raise Exception("delete failed")
            return R()

        return type(
            "MockClient", (),
            {"get": mock_get, "post": mock_post, "delete": mock_delete},
        )()

    # -- _find_registered_webhooks --

    def test_find_registered_webhooks_returns_matches(self, monkeypatch):
        import asyncio
        adapter = _make_adapter(monkeypatch)
        url = adapter._webhook_url
        adapter.client = self._mock_client(
            get_response={"status": 200, "data": [
                {"id": 1, "url": url, "events": ["new-message"]},
                {"id": 2, "url": "http://other:9999/hook", "events": ["message"]},
            ]}
        )
        result = asyncio.get_event_loop().run_until_complete(
            adapter._find_registered_webhooks(url)
        )
        assert len(result) == 1
        assert result[0]["id"] == 1


    # -- _register_webhook --

    def test_register_fresh(self, monkeypatch):
        """No existing webhook → POST creates one."""
        import asyncio
        adapter = _make_adapter(monkeypatch)
        adapter.client = self._mock_client(
            get_response={"status": 200, "data": []},
            post_response={"status": 200, "data": {"id": 42}},
        )
        ok = asyncio.get_event_loop().run_until_complete(
            adapter._register_webhook()
        )
        assert ok is True


    def test_register_reuses_existing(self, monkeypatch):
        """Crash resilience — existing registration is reused, no POST needed."""
        import asyncio
        adapter = _make_adapter(monkeypatch)
        url = adapter._webhook_register_url
        adapter.client = self._mock_client(
            get_response={"status": 200, "data": [
                {"id": 7, "url": url, "events": ["new-message"]},
            ]},
        )

        # Track whether POST was called
        post_called = False
        orig_api_post = adapter._api_post
        async def tracking_post(path, payload):
            nonlocal post_called
            post_called = True
            return await orig_api_post(path, payload)
        adapter._api_post = tracking_post

        ok = asyncio.get_event_loop().run_until_complete(
            adapter._register_webhook()
        )
        assert ok is True
        assert not post_called, "Should reuse existing, not POST again"


    # -- _unregister_webhook --


    def test_unregister_removes_all_duplicates(self, monkeypatch):
        """Multiple orphaned registrations for same URL — all get removed."""
        import asyncio
        adapter = _make_adapter(monkeypatch)
        url = adapter._webhook_register_url
        deleted_ids = []

        async def mock_delete(*args, **kwargs):
            # Extract ID from URL
            url_str = args[0] if args else ""
            deleted_ids.append(url_str)
            class R:
                status_code = 200
                def raise_for_status(self):
                    pass
            return R()

        adapter.client = self._mock_client(
            get_response={"status": 200, "data": [
                {"id": 1, "url": url},
                {"id": 2, "url": url},
                {"id": 3, "url": "http://other/hook"},
            ]},
        )
        adapter.client.delete = mock_delete

        ok = asyncio.get_event_loop().run_until_complete(
            adapter._unregister_webhook()
        )
        assert ok is True
        assert len(deleted_ids) == 2


# ---------------------------------------------------------------------------
# Regression for #78183: httpx timeout exceptions stringify to "" which
# defeats _is_timeout_error, causing the plain-text fallback to re-send an
# already-delivered message (duplicate delivery).
# ---------------------------------------------------------------------------

class TestBlueBubblesTimeoutErrorNormalization:
    """When an httpx timeout has an empty string representation, the adapter
    must fall back to the exception type name so the base-layer timeout guard
    can still recognise it."""

    @pytest.mark.asyncio
    async def test_send_read_timeout_produces_matchable_error(self, monkeypatch):
        adapter = _make_adapter(monkeypatch)

        async def fake_resolve(chat_id):
            return "iMessage;+;chat-123"
        monkeypatch.setattr(adapter, "_resolve_chat_guid", fake_resolve)

        async def fake_api_post(path, payload):
            raise httpx.ReadTimeout("")
        monkeypatch.setattr(adapter, "_api_post", fake_api_post)

        result = await adapter.send("chat-1", "hello world")

        assert not result.success
        assert result.error, "error must not be empty"
        assert BasePlatformAdapter._is_timeout_error(result.error), (
            f"_is_timeout_error must recognise {result.error!r}"
        )


    @pytest.mark.asyncio
    async def test_create_chat_for_handle_timeout_produces_matchable_error(
        self, monkeypatch,
    ):
        """Sibling call path — _create_chat_for_handle has the same
        error=str(exc) pattern and must also preserve the exception type."""
        adapter = _make_adapter(monkeypatch)

        async def fake_api_post(path, payload):
            raise httpx.ReadTimeout("")
        monkeypatch.setattr(adapter, "_api_post", fake_api_post)

        result = await adapter._create_chat_for_handle("test@example.com", "hi")

        assert not result.success
        assert result.error
        assert BasePlatformAdapter._is_timeout_error(result.error)

    @pytest.mark.asyncio
    async def test_non_empty_error_string_is_unchanged(self, monkeypatch):
        """A normal exception with a message must keep its original text."""
        adapter = _make_adapter(monkeypatch)

        async def fake_resolve(chat_id):
            return "iMessage;+;chat-123"
        monkeypatch.setattr(adapter, "_resolve_chat_guid", fake_resolve)

        async def fake_api_post(path, payload):
            raise RuntimeError("Server error '500 Internal Server Error'")
        monkeypatch.setattr(adapter, "_api_post", fake_api_post)

        result = await adapter.send("chat-1", "hello world")

        assert not result.success
        assert "500 Internal Server Error" in (result.error or "")




class TestBlueBubblesGateBeforeDownload:
    """The require_mention gate must run BEFORE attachments are downloaded (review follow-up)."""

    @pytest.mark.asyncio
    @pytest.mark.parametrize("text, downloads, handled_count", [
        ("look at this", 0, 0),          # unmentioned group attachment: never fetched
        ("hermes look at this", 1, 1),   # mentioned: fetched and dispatched
    ])
    async def test_unmentioned_group_attachment_is_not_downloaded(
            self, monkeypatch, text, downloads, handled_count):
        adapter = _make_adapter(monkeypatch, require_mention=True, send_read_receipts=False)
        handled = []

        async def fake_handle_message(event):
            handled.append(event)

        download = AsyncMock(return_value="cached.jpg")
        monkeypatch.setattr(adapter, "handle_message", fake_handle_message)
        monkeypatch.setattr(adapter, "_download_attachment", download)
        response = await adapter._handle_webhook(_FakeBlueBubblesRequest({
            "type": "new-message",
            "data": {
                "guid": "msg-att-1",
                "text": text,
                "handle": {"address": "+15555550100"},
                "isFromMe": False,
                "isGroup": True,
                "chats": [{"guid": "iMessage;+;group-chat"}],
                "attachments": [{"guid": "att-1", "mimeType": "image/jpeg"}],
            },
        }))
        await asyncio.sleep(0)

        assert response.status == 200
        assert download.await_count == downloads
        assert len(handled) == handled_count
