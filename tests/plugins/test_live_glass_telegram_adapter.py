"""Tests for the live-glass Telegram adapter (AVA-21)."""
from __future__ import annotations

import base64
import importlib
import json
import sys
from unittest.mock import MagicMock, call

from plugins.observability.live_glass.adapters.telegram import (
    TelegramLiveGlassAdapter,
    _save_data_url_to_tempfile,
    _build_approval_keyboard,
)


def _fresh_module():
    sys.modules.pop("plugins.observability.live_glass.adapters.telegram", None)
    return importlib.import_module(
        "plugins.observability.live_glass.adapters.telegram"
    )


class TestDataUrlToTempfile:
    def test_valid_png_data_url(self):
        b64 = base64.b64encode(b"\x89PNG\r\n\x1a\nfake").decode()
        path = _save_data_url_to_tempfile(f"data:image/png;base64,{b64}")
        assert path is not None
        assert path.endswith(".png")
        import os
        os.unlink(path)

    def test_valid_jpeg_data_url(self):
        b64 = base64.b64encode(b"fakejpeg").decode()
        path = _save_data_url_to_tempfile(f"data:image/jpeg;base64,{b64}")
        assert path is not None
        assert path.endswith(".jpg")
        import os
        os.unlink(path)

    def test_empty_url_returns_none(self):
        assert _save_data_url_to_tempfile("") is None

    def test_non_data_url_returns_none(self):
        assert _save_data_url_to_tempfile("https://example.com/img.png") is None

    def test_invalid_base64_returns_none(self):
        assert _save_data_url_to_tempfile("data:image/png;base64,!!!not-base64!!!") is None


class TestApprovalKeyboard:
    def test_builds_inline_keyboard(self):
        kb = _build_approval_keyboard("tc_123")
        assert kb is not None
        rows = kb["inline_keyboard"]
        assert len(rows) == 2

        # Row 0: Approve Once, Approve Session
        assert rows[0][0]["text"] == "Approve Once"
        assert json.loads(rows[0][0]["callback_data"]) == {"a": "once", "tcid": "tc_123"}
        assert rows[0][1]["text"] == "Approve Session"
        assert json.loads(rows[0][1]["callback_data"]) == {"a": "session", "tcid": "tc_123"}

        # Row 1: Approve Always, Deny
        assert rows[1][0]["text"] == "Approve Always"
        assert json.loads(rows[1][0]["callback_data"]) == {"a": "always", "tcid": "tc_123"}
        assert rows[1][1]["text"] == "Deny"
        assert json.loads(rows[1][1]["callback_data"]) == {"a": "deny", "tcid": "tc_123"}

    def test_empty_tool_call_id_returns_none(self):
        assert _build_approval_keyboard("") is None


class TestTelegramLiveGlassAdapter:
    def _make_deps(self):
        """Return (sender_mock, chat_map, adapter)."""
        sender = MagicMock()
        chat_map = {"s1": 111, "s2": 222}
        router = lambda sid: chat_map.get(sid)
        adapter = TelegramLiveGlassAdapter(sender, router)
        return sender, router, adapter

    def test_start_and_stop(self):
        _, _, adapter = self._make_deps()
        adapter.start()
        assert adapter._unsubscribe is not None
        adapter.stop()
        assert adapter._unsubscribe is None

    def test_log_event_sends_message(self):
        sender, _, adapter = self._make_deps()
        adapter.start()

        from plugins.observability.live_glass import publish
        publish("log", {
            "tool_name": "computer_use",
            "status": "ok",
            "duration_ms": 123,
            "source": "test",
        }, session_id="s1")

        adapter.stop()

        sender.send_message.assert_called_once()
        args, _ = sender.send_message.call_args
        assert args[0] == 111  # chat_id from router
        assert "computer_use" in args[1]
        assert "123ms" in args[1]
        assert "✓" in args[1]

    def test_log_event_with_error(self):
        sender, _, adapter = self._make_deps()
        adapter.start()

        from plugins.observability.live_glass import publish
        publish("log", {
            "tool_name": "click",
            "status": "error",
            "duration_ms": 50,
            "error_message": "denied by user",
            "source": "test",
        }, session_id="s2")

        adapter.stop()

        sender.send_message.assert_called_once()
        args, _ = sender.send_message.call_args
        assert args[0] == 222
        assert "✗" in args[1]
        assert "denied by user" in args[1]

    def test_approval_event_sends_keyboard(self):
        sender, _, adapter = self._make_deps()
        adapter.start()

        from plugins.observability.live_glass import publish
        publish("approval_request", {
            "command": "click element #5",
            "description": "computer_use click",
            "surface": "cli",
            "source": "test",
        }, session_id="s1", tool_call_id="call_1")

        adapter.stop()

        sender.send_message.assert_called_once()
        call_args = sender.send_message.call_args
        assert call_args[0][0] == 111  # first positional: chat_id
        assert "click element #5" in call_args[0][1]  # second: text
        # reply_markup is the third positional arg (may be None)
        reply_markup = call_args[0][2] if len(call_args[0]) > 2 else call_args[1].get("reply_markup")
        assert reply_markup is not None
        assert "inline_keyboard" in reply_markup

    def test_frame_event_sends_photo(self):
        sender, _, adapter = self._make_deps()
        adapter.start()

        b64 = base64.b64encode(b"\x89PNG\r\n\x1a\nfakeframe").decode()

        from plugins.observability.live_glass import publish
        publish("frame", {
            "image_url": f"data:image/png;base64,{b64}",
            "mime_type": "image/png",
            "mode": "som",
            "width": 100,
            "height": 200,
            "summary": "capture mode=som",
            "source": "computer_use",
        }, session_id="s1")

        adapter.stop()

        sender.send_photo.assert_called_once()
        call_args = sender.send_photo.call_args
        assert call_args[0][0] == 111
        assert call_args[0][1].endswith(".png")
        caption = call_args[0][2] if len(call_args[0]) > 2 else call_args[1].get("caption")
        assert "capture mode=som" in (caption or "")
        # Temp file already cleaned up by adapter's _handle_frame()

    def test_skips_events_with_unmapped_session(self):
        sender, _, adapter = self._make_deps()
        adapter.start()

        from plugins.observability.live_glass import publish
        publish("log", {
            "tool_name": "test",
            "status": "ok",
            "duration_ms": 1,
            "source": "test",
        }, session_id="unknown_session")

        adapter.stop()
        sender.send_message.assert_not_called()
        sender.send_photo.assert_not_called()

    def test_skips_events_with_empty_session_id(self):
        sender, router, adapter = self._make_deps()
        adapter.start()

        from plugins.observability.live_glass import publish
        publish("log", {
            "tool_name": "test",
            "status": "ok",
            "duration_ms": 1,
            "source": "test",
        }, session_id="")

        adapter.stop()
        sender.send_message.assert_not_called()

    def test_sender_exceptions_are_caught(self):
        sender, _, adapter = self._make_deps()
        sender.send_message.side_effect = RuntimeError("network down")
        adapter.start()

        from plugins.observability.live_glass import publish
        publish("log", {
            "tool_name": "test",
            "status": "ok",
            "duration_ms": 1,
            "source": "test",
        }, session_id="s1")

        adapter.stop()
        # Should not raise — exception is caught and logged
