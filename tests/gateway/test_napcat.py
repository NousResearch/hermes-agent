"""Tests for the NapCat platform adapter (OneBot 11 reverse WebSocket)."""

from __future__ import annotations

import asyncio
import hashlib
import os
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from gateway.config import Platform, PlatformConfig
from gateway.platforms.base import SendResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_config(**extra) -> PlatformConfig:
    return PlatformConfig(enabled=True, extra=extra)


def _make_adapter(**extra):
    """Construct a NapCatAdapter with minimal valid config."""
    from gateway.platforms.napcat import NapCatAdapter

    extra.setdefault("token", "test-token")
    return NapCatAdapter(_make_config(**extra))


def _run(coro):
    return asyncio.run(coro)


def _ok(data=None, message_id="m1"):
    """Build an OneBot ok response with optional message_id payload."""
    payload = {"status": "ok", "retcode": 0, "data": data or {}}
    if message_id is not None and "message_id" not in payload["data"]:
        payload["data"]["message_id"] = message_id
    return payload


# ---------------------------------------------------------------------------
# call_action / _call_action alias
# ---------------------------------------------------------------------------


class TestCallActionAlias:
    def test_public_and_private_are_same_callable(self):
        from gateway.platforms.napcat import NapCatAdapter

        # Both names must resolve to the same coroutine function so the
        # internal _send_chunk path keeps working alongside napcat_tool's
        # use of the public `call_action`.
        assert NapCatAdapter.call_action is NapCatAdapter._call_action

    def test_direct_upload_private_file_stream_uploads_local_file_first(self, tmp_path):
        adapter = _make_adapter()
        adapter._mark_connected()
        adapter._stream_upload_chunk_size = 4
        local_file = tmp_path / "hermes_napcat_file_test.txt"
        file_bytes = b"hello from hermes"
        local_file.write_bytes(file_bytes)
        uploaded_path = r"C:\NapCat\Temp\hermes_napcat_file_test.txt"
        payloads = []

        class FakeWebSocket:
            closed = False

            async def send_json(self, payload):
                payloads.append(payload)
                action = payload["action"]
                params = payload["params"]
                if action == "upload_file_stream" and params.get("is_complete"):
                    response = {
                        "status": "ok",
                        "retcode": 0,
                        "data": {"status": "file_complete", "file_path": uploaded_path},
                        "echo": payload["echo"],
                    }
                elif action == "upload_file_stream":
                    response = {
                        "status": "ok",
                        "retcode": 0,
                        "data": {"received_chunks": params["chunk_index"] + 1},
                        "echo": payload["echo"],
                    }
                else:
                    response = _ok(message_id="uploaded")
                    response["echo"] = payload["echo"]
                adapter._pending_responses[payload["echo"]].set_result(response)

        adapter._ws = FakeWebSocket()

        result = _run(
            adapter.call_action(
                "upload_private_file",
                {"user_id": 10001, "file": str(local_file), "name": local_file.name},
            )
        )

        assert result["status"] == "ok"
        actions = [payload["action"] for payload in payloads]
        assert "upload_file_stream" in actions
        assert actions[-1] == "upload_private_file"
        assert payloads[-1]["params"]["file"] == uploaded_path

    def test_direct_upload_private_file_combines_stream_and_upload_errors(self, tmp_path):
        adapter = _make_adapter()
        adapter._mark_connected()
        local_file = tmp_path / "hermes_napcat_file_test.txt"
        local_file.write_text("hello")

        class FakeWebSocket:
            closed = False

            async def send_json(self, payload):
                if payload["action"] == "upload_file_stream":
                    response = {
                        "status": "failed",
                        "retcode": 404,
                        "message": "STREAM_UNSUPPORTED",
                        "echo": payload["echo"],
                    }
                else:
                    response = {
                        "status": "failed",
                        "retcode": 1,
                        "message": "识别URL失败",
                        "echo": payload["echo"],
                    }
                adapter._pending_responses[payload["echo"]].set_result(response)

        adapter._ws = FakeWebSocket()

        result = _run(
            adapter.call_action(
                "upload_private_file",
                {"user_id": 10001, "file": str(local_file), "name": local_file.name},
            )
        )

        assert result["status"] == "failed"
        assert "STREAM_UNSUPPORTED" in result["message"]
        assert "识别URL失败" in result["message"]


# ---------------------------------------------------------------------------
# _segment_marker — inbound rich-media markers
# ---------------------------------------------------------------------------


class TestSegmentMarker:
    def test_image_with_file_id(self):
        from gateway.platforms.napcat import NapCatAdapter

        marker = NapCatAdapter._segment_marker("image", {"file": "abcd1234"})
        assert marker == "[图片:abcd1234]"

    def test_image_falls_back_to_url(self):
        from gateway.platforms.napcat import NapCatAdapter

        marker = NapCatAdapter._segment_marker("image", {"url": "https://x.example/y.png"})
        assert marker == "[图片:https://x.example/y.png]"

    def test_record_segment(self):
        from gateway.platforms.napcat import NapCatAdapter

        marker = NapCatAdapter._segment_marker("record", {"file": "voice123"})
        assert marker == "[语音:voice123]"

    def test_file_segment_with_name_and_id(self):
        from gateway.platforms.napcat import NapCatAdapter

        marker = NapCatAdapter._segment_marker(
            "file", {"name": "report.pdf", "file_id": "/biz/abc"}
        )
        assert marker == "[文件:report.pdf:/biz/abc]"

    def test_face_segment(self):
        from gateway.platforms.napcat import NapCatAdapter

        marker = NapCatAdapter._segment_marker("face", {"id": "76"})
        assert marker == "[表情:76]"

    def test_unknown_segment_returns_none(self):
        from gateway.platforms.napcat import NapCatAdapter

        assert NapCatAdapter._segment_marker("xml", {"data": "..."}) is None


# ---------------------------------------------------------------------------
# _extract_reply_and_text — private chats & at segments
# ---------------------------------------------------------------------------


class TestExtractReplyAndText:
    def test_text_only(self):
        from gateway.platforms.napcat import NapCatAdapter

        reply, text = NapCatAdapter._extract_reply_and_text(
            [{"type": "text", "data": {"text": "hello"}}]
        )
        assert reply is None
        assert text == "hello"

    def test_reply_segment_extracted(self):
        from gateway.platforms.napcat import NapCatAdapter

        reply, text = NapCatAdapter._extract_reply_and_text(
            [
                {"type": "reply", "data": {"id": "999"}},
                {"type": "text", "data": {"text": "hi"}},
            ]
        )
        assert reply == "999"
        assert text == "hi"

    def test_image_segment_becomes_marker(self):
        from gateway.platforms.napcat import NapCatAdapter

        _, text = NapCatAdapter._extract_reply_and_text(
            [
                {"type": "text", "data": {"text": "看这个"}},
                {"type": "image", "data": {"file": "fid_xyz"}},
            ]
        )
        assert "看这个" in text
        assert "[图片:fid_xyz]" in text

    def test_at_segment_includes_qq_marker(self):
        from gateway.platforms.napcat import NapCatAdapter

        _, text = NapCatAdapter._extract_reply_and_text(
            [
                {"type": "at", "data": {"qq": "12345"}},
                {"type": "text", "data": {"text": "ping"}},
            ]
        )
        # In the private/text-extraction path (no self_id stripping), an at
        # segment is surfaced as @<qq> for the agent.
        assert "@12345" in text
        assert "ping" in text


# ---------------------------------------------------------------------------
# _strip_self_mention — group chat gating
# ---------------------------------------------------------------------------


class TestStripSelfMention:
    def test_no_self_id_returns_not_mentioned(self):
        adapter = _make_adapter()
        adapter._self_id = None
        mentioned, text = adapter._strip_self_mention(
            [{"type": "text", "data": {"text": "hi"}}]
        )
        assert mentioned is False
        assert text == ""

    def test_at_self_marks_mentioned_and_strips(self):
        adapter = _make_adapter()
        adapter._self_id = "1000"
        mentioned, text = adapter._strip_self_mention(
            [
                {"type": "at", "data": {"qq": "1000"}},
                {"type": "text", "data": {"text": " 帮我看看"}},
            ]
        )
        assert mentioned is True
        assert text == "帮我看看"

    def test_at_other_kept_as_marker_no_mention(self):
        adapter = _make_adapter()
        adapter._self_id = "1000"
        mentioned, text = adapter._strip_self_mention(
            [
                {"type": "at", "data": {"qq": "2000"}},
                {"type": "text", "data": {"text": "看这"}},
            ]
        )
        assert mentioned is False
        assert "@2000" in text
        assert "看这" in text

    def test_image_and_file_become_markers_when_mentioned(self):
        adapter = _make_adapter()
        adapter._self_id = "1000"
        segments = [
            {"type": "at", "data": {"qq": "1000"}},
            {"type": "text", "data": {"text": "这是啥"}},
            {"type": "image", "data": {"file": "img_1"}},
            {"type": "file", "data": {"name": "a.txt", "file_id": "fid_a"}},
        ]
        mentioned, text = adapter._strip_self_mention(segments)
        assert mentioned is True
        assert "[图片:img_1]" in text
        assert "[文件:a.txt:fid_a]" in text
        assert "这是啥" in text


# ---------------------------------------------------------------------------
# _resolve_chat_target
# ---------------------------------------------------------------------------


class TestResolveChatTarget:
    def test_cached_chat_type_wins(self):
        adapter = _make_adapter()
        adapter._chat_type_map["12345"] = "group"
        chat_type, normalized = adapter._resolve_chat_target("12345")
        assert chat_type == "group"
        assert normalized == "12345"

    def test_group_prefix_explicit(self):
        adapter = _make_adapter()
        chat_type, normalized = adapter._resolve_chat_target("group:99999")
        assert chat_type == "group"
        assert normalized == "99999"

    def test_private_prefix_explicit(self):
        adapter = _make_adapter()
        chat_type, normalized = adapter._resolve_chat_target("private:10001")
        assert chat_type == "private"
        assert normalized == "10001"

    def test_bare_numeric_falls_back_to_private(self):
        adapter = _make_adapter()
        chat_type, normalized = adapter._resolve_chat_target("42")
        assert chat_type == "private"
        assert normalized == "42"


# ---------------------------------------------------------------------------
# _media_file_uri — local path / URL handling
# ---------------------------------------------------------------------------


class TestMediaFileUri:
    def test_http_url_unchanged(self):
        from gateway.platforms.napcat import NapCatAdapter

        assert NapCatAdapter._media_file_uri("https://x.example/img.png") == \
            "https://x.example/img.png"

    def test_file_uri_unchanged(self):
        from gateway.platforms.napcat import NapCatAdapter

        assert NapCatAdapter._media_file_uri("file:///tmp/x.png") == "file:///tmp/x.png"

    def test_base64_unchanged(self):
        from gateway.platforms.napcat import NapCatAdapter

        assert NapCatAdapter._media_file_uri("base64://aGVsbG8=") == "base64://aGVsbG8="

    def test_unix_absolute_path_becomes_file_uri(self):
        from gateway.platforms.napcat import NapCatAdapter

        result = NapCatAdapter._media_file_uri("/tmp/img.png")
        assert result.startswith("file://")
        assert result.endswith("/tmp/img.png")

    def test_empty_returns_empty(self):
        from gateway.platforms.napcat import NapCatAdapter

        assert NapCatAdapter._media_file_uri("") == ""


# ---------------------------------------------------------------------------
# Outbound media — send_image_file / send_voice / send_video / send_document
# ---------------------------------------------------------------------------


class TestSendImageFile:
    def _setup_adapter(self):
        adapter = _make_adapter()
        adapter._mark_connected()
        # Build a fake connected websocket so the media short-circuit doesn't fire.
        ws = MagicMock()
        ws.closed = False
        adapter._ws = ws
        adapter._chat_type_map["555"] = "group"
        return adapter

    def test_sends_image_segment_via_send_group_msg(self):
        adapter = self._setup_adapter()
        captured = {}

        async def fake_call_action(action, params, *, timeout=None):
            captured["action"] = action
            captured["params"] = params
            return _ok(message_id="img_msg_1")

        adapter.call_action = fake_call_action  # public
        adapter._call_action = fake_call_action  # private alias
        result = _run(adapter.send_image_file("555", "/tmp/cat.png", caption="meow"))

        assert result.success is True
        assert result.message_id == "img_msg_1"
        assert captured["action"] == "send_group_msg"
        assert captured["params"]["group_id"] == 555

        segments = captured["params"]["message"]
        types = [seg["type"] for seg in segments]
        assert "image" in types
        # Caption appears as a trailing text segment.
        assert "text" in types
        image_seg = next(s for s in segments if s["type"] == "image")
        assert image_seg["data"]["file"].startswith("file://")

    def test_falls_back_to_text_on_failure(self):
        adapter = self._setup_adapter()
        attempts = []

        async def fake_call_action(action, params, *, timeout=None):
            attempts.append(action)
            if action == "send_group_msg" and any(
                seg["type"] == "image" for seg in params.get("message", [])
            ):
                return {"status": "failed", "retcode": 100, "message": "BAD_FILE"}
            # Plain text fallback succeeds.
            return _ok(message_id="text_msg_1")

        adapter.call_action = fake_call_action
        adapter._call_action = fake_call_action
        result = _run(adapter.send_image_file("555", "/tmp/cat.png", caption="cap"))

        # Two send_group_msg calls: image attempt, then text fallback.
        assert attempts == ["send_group_msg", "send_group_msg"]
        assert result.success is False
        assert result.message_id == "text_msg_1"
        assert result.error == "BAD_FILE"

    def test_existing_local_image_is_stream_uploaded_before_send(self, tmp_path):
        adapter = self._setup_adapter()
        adapter._stream_upload_chunk_size = 4
        image_path = tmp_path / "cat.png"
        image_bytes = b"napcat-image-bytes"
        image_path.write_bytes(image_bytes)
        uploaded_path = r"C:\NapCat\Temp\stream-cat.png"
        calls = []

        async def fake_call_action(action, params, *, timeout=None):
            calls.append((action, params))
            if action == "upload_file_stream" and params.get("is_complete"):
                return {
                    "status": "ok",
                    "retcode": 0,
                    "data": {
                        "status": "file_complete",
                        "file_path": uploaded_path,
                        "file_size": len(image_bytes),
                        "sha256": hashlib.sha256(image_bytes).hexdigest(),
                    },
                }
            if action == "upload_file_stream":
                return {
                    "status": "ok",
                    "retcode": 0,
                    "data": {"received_chunks": params["chunk_index"] + 1},
                }
            return _ok(message_id="img_msg_streamed")

        adapter.call_action = fake_call_action
        adapter._call_action = fake_call_action
        adapter._call_action_raw = fake_call_action
        result = _run(adapter.send_image_file("555", str(image_path), caption="meow"))

        assert result.success is True
        upload_calls = [params for action, params in calls if action == "upload_file_stream"]
        assert len(upload_calls) > 1
        first_chunk = upload_calls[0]
        assert first_chunk["stream_id"]
        assert first_chunk["chunk_index"] == 0
        assert first_chunk["total_chunks"] == 5
        assert first_chunk["filename"] == "cat.png"
        assert first_chunk["file_size"] == len(image_bytes)
        assert first_chunk["file_retention"] == 30_000
        assert "chunk_data" in first_chunk
        assert "chunk" not in first_chunk
        assert upload_calls[-2]["expected_sha256"] == hashlib.sha256(image_bytes).hexdigest()
        assert upload_calls[-1] == {"stream_id": first_chunk["stream_id"], "is_complete": True}
        send_params = next(params for action, params in calls if action == "send_group_msg")
        image_seg = next(s for s in send_params["message"] if s["type"] == "image")
        assert image_seg["data"]["file"] == uploaded_path

    def test_remote_image_url_is_not_stream_uploaded(self):
        adapter = self._setup_adapter()
        calls = []

        async def fake_call_action(action, params, *, timeout=None):
            calls.append((action, params))
            return _ok(message_id="img_url")

        adapter.call_action = fake_call_action
        adapter._call_action = fake_call_action
        result = _run(adapter.send_image_file("555", "https://example.test/cat.png"))

        assert result.success is True
        assert [action for action, _ in calls] == ["send_group_msg"]
        image_seg = next(s for s in calls[0][1]["message"] if s["type"] == "image")
        assert image_seg["data"]["file"] == "https://example.test/cat.png"

    def test_stream_upload_failure_is_returned_with_send_failure(self, tmp_path):
        adapter = self._setup_adapter()
        image_path = tmp_path / "cat.png"
        image_path.write_bytes(b"not-on-napcat")

        async def fake_call_action(action, params, *, timeout=None):
            if action == "upload_file_stream":
                return {"status": "failed", "retcode": 404, "message": "STREAM_UNSUPPORTED"}
            if action == "send_group_msg" and any(
                seg["type"] == "image" for seg in params.get("message", [])
            ):
                return {"status": "failed", "retcode": 100, "message": "ENOENT"}
            return _ok(message_id="fallback_text")

        adapter.call_action = fake_call_action
        adapter._call_action = fake_call_action
        adapter._call_action_raw = fake_call_action
        result = _run(adapter.send_image_file("555", str(image_path)))

        assert result.success is False
        assert "STREAM_UNSUPPORTED" in result.error
        assert "ENOENT" in result.error


class TestSendVoice:
    def test_sends_record_segment(self):
        adapter = _make_adapter()
        adapter._mark_connected()
        ws = MagicMock()
        ws.closed = False
        adapter._ws = ws
        adapter._chat_type_map["10001"] = "private"

        captured = {}

        async def fake_call_action(action, params, *, timeout=None):
            captured["action"] = action
            captured["params"] = params
            return _ok(message_id="voice_1")

        adapter.call_action = fake_call_action
        adapter._call_action = fake_call_action
        result = _run(adapter.send_voice("10001", "/tmp/hi.ogg"))

        assert result.success is True
        assert captured["action"] == "send_private_msg"
        assert captured["params"]["user_id"] == 10001
        seg_types = [s["type"] for s in captured["params"]["message"]]
        assert "record" in seg_types


class TestSendVideo:
    def test_sends_video_segment(self):
        adapter = _make_adapter()
        adapter._mark_connected()
        ws = MagicMock()
        ws.closed = False
        adapter._ws = ws
        adapter._chat_type_map["999"] = "group"

        captured = {}

        async def fake_call_action(action, params, *, timeout=None):
            captured["action"] = action
            captured["params"] = params
            return _ok(message_id="vid_1")

        adapter.call_action = fake_call_action
        adapter._call_action = fake_call_action
        result = _run(adapter.send_video("999", "/tmp/clip.mp4"))

        assert result.success is True
        seg_types = [s["type"] for s in captured["params"]["message"]]
        assert "video" in seg_types


class TestSendDocument:
    def test_existing_local_document_is_stream_uploaded_before_private_upload(self, tmp_path):
        adapter = _make_adapter()
        adapter._mark_connected()
        adapter._stream_upload_chunk_size = 4
        ws = MagicMock()
        ws.closed = False
        adapter._ws = ws
        adapter._chat_type_map["10001"] = "private"
        local_file = tmp_path / "hermes_napcat_test.svg"
        file_bytes = b"<svg>napcat</svg>"
        local_file.write_bytes(file_bytes)
        uploaded_path = r"C:\NapCat\Temp\hermes_napcat_test.svg"
        calls = []

        async def fake_call_action(action, params, *, timeout=None):
            calls.append((action, params))
            if action == "upload_file_stream" and params.get("is_complete"):
                return {
                    "status": "ok",
                    "retcode": 0,
                    "data": {
                        "status": "file_complete",
                        "file_path": uploaded_path,
                        "file_size": len(file_bytes),
                        "sha256": hashlib.sha256(file_bytes).hexdigest(),
                    },
                }
            if action == "upload_file_stream":
                return _ok({"received_chunks": params["chunk_index"] + 1}, message_id=None)
            return _ok(message_id="doc_msg")

        adapter.call_action = fake_call_action
        adapter._call_action = fake_call_action
        adapter._call_action_raw = fake_call_action
        result = _run(adapter.send_document("10001", str(local_file)))

        assert result.success is True
        actions = [action for action, _ in calls]
        assert "upload_file_stream" in actions
        assert actions[-1] == "upload_private_file"
        upload_params = calls[-1][1]
        assert upload_params["user_id"] == 10001
        assert upload_params["file"] == uploaded_path
        assert upload_params["name"] == "hermes_napcat_test.svg"

    def test_document_stream_upload_failure_is_reported_with_file_upload_failure(self, tmp_path):
        adapter = _make_adapter()
        adapter._mark_connected()
        ws = MagicMock()
        ws.closed = False
        adapter._ws = ws
        adapter._chat_type_map["10001"] = "private"
        local_file = tmp_path / "hermes_napcat_test.svg"
        local_file.write_text("<svg/>")

        async def fake_call_action(action, params, *, timeout=None):
            if action == "upload_file_stream":
                return {"status": "failed", "retcode": 404, "message": "STREAM_UNSUPPORTED"}
            if action == "upload_private_file":
                return {"status": "failed", "retcode": 1, "message": "识别URL失败"}
            return _ok(message_id="notice")

        adapter.call_action = fake_call_action
        adapter._call_action = fake_call_action
        adapter._call_action_raw = fake_call_action
        result = _run(adapter.send_document("10001", str(local_file)))

        assert result.success is False
        assert result.message_id == "notice"
        assert "STREAM_UNSUPPORTED" in result.error
        assert "识别URL失败" in result.error

    def test_uploads_via_upload_group_file(self):
        adapter = _make_adapter()
        adapter._mark_connected()
        ws = MagicMock()
        ws.closed = False
        adapter._ws = ws
        adapter._chat_type_map["12345"] = "group"

        calls = []

        async def fake_call_action(action, params, *, timeout=None):
            calls.append((action, params))
            return _ok(message_id="file_msg_1")

        adapter.call_action = fake_call_action
        adapter._call_action = fake_call_action
        result = _run(
            adapter.send_document("12345", "/var/log/app.log", file_name="app.log")
        )

        assert result.success is True
        actions = [c[0] for c in calls]
        assert "upload_group_file" in actions
        params = next(p for a, p in calls if a == "upload_group_file")
        assert params["group_id"] == 12345
        assert params["name"] == "app.log"
        # Path must be absolute & expanded.
        assert os.path.isabs(params["file"])

    def test_private_chat_uploads_via_upload_private_file(self):
        adapter = _make_adapter()
        adapter._mark_connected()
        ws = MagicMock()
        ws.closed = False
        adapter._ws = ws
        adapter._chat_type_map["10001"] = "private"

        captured_action = []

        async def fake_call_action(action, params, *, timeout=None):
            captured_action.append(action)
            return _ok(message_id="f1")

        adapter.call_action = fake_call_action
        adapter._call_action = fake_call_action
        _run(adapter.send_document("10001", "/tmp/x.txt"))

        assert "upload_private_file" in captured_action

    def test_falls_back_to_text_notice_on_failure(self):
        adapter = _make_adapter()
        adapter._mark_connected()
        ws = MagicMock()
        ws.closed = False
        adapter._ws = ws
        adapter._chat_type_map["12345"] = "group"

        async def fake_call_action(action, params, *, timeout=None):
            if action.startswith("upload_"):
                return {"status": "failed", "retcode": 1, "message": "QUOTA"}
            # text notice succeeds.
            return _ok(message_id="notice")

        adapter.call_action = fake_call_action
        adapter._call_action = fake_call_action
        result = _run(adapter.send_document("12345", "/tmp/big.bin", file_name="big.bin"))

        # The fallback text notice is sent, but the original upload failure is
        # still returned so the gateway can surface/log media delivery errors.
        assert result.success is False
        assert result.message_id == "notice"
        assert result.error == "QUOTA"


# ---------------------------------------------------------------------------
# Outbound media gating — disconnected adapter
# ---------------------------------------------------------------------------


class TestMediaSendsRequireConnection:
    def test_image_returns_not_connected_when_ws_closed(self):
        adapter = _make_adapter()
        adapter._mark_connected()
        ws = MagicMock()
        ws.closed = True
        adapter._ws = ws

        result = _run(adapter.send_image_file("555", "/tmp/x.png"))
        assert result.success is False
        assert result.retryable is True

    def test_document_returns_not_connected_when_ws_missing(self):
        adapter = _make_adapter()
        adapter._mark_connected()
        adapter._ws = None

        result = _run(adapter.send_document("555", "/tmp/x.txt"))
        assert result.success is False
