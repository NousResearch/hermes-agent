"""Tests for the live-glass BlueBubbles adapter (AVA-23)."""
from __future__ import annotations

import base64
from unittest.mock import MagicMock

from plugins.observability.live_glass.adapters.bluebubbles import (
    BlueBubblesLiveGlassAdapter,
    _save_data_url_to_tempfile,
)


class TestDataUrlToTempfile:
    def test_valid_png(self):
        b64 = base64.b64encode(b"\x89PNG\r\n\x1a\nfakebb").decode()
        path = _save_data_url_to_tempfile(f"data:image/png;base64,{b64}")
        assert path is not None
        assert path.endswith(".png")
        import os; os.unlink(path)

    def test_empty_returns_none(self):
        assert _save_data_url_to_tempfile("") is None

    def test_invalid_base64_returns_none(self):
        assert _save_data_url_to_tempfile("data:image/png;base64,!!!bad!!!") is None


class TestBlueBubblesAdapter:
    ADDR1 = "+15551234567"
    ADDR2 = "user@icloud.com"

    def _make_deps(self):
        sender = MagicMock()
        mapping = {"s1": self.ADDR1, "s2": self.ADDR2}
        router = mapping.get
        adapter = BlueBubblesLiveGlassAdapter(sender, router)
        return sender, adapter

    def test_start_and_stop(self):
        _, adapter = self._make_deps()
        adapter.start()
        assert adapter._unsubscribe is not None
        adapter.stop()
        assert adapter._unsubscribe is None

    def test_log_event_sends_text(self):
        sender, adapter = self._make_deps()
        adapter.start()

        from plugins.observability.live_glass import publish
        publish("log", {"tool_name": "computer_use", "status": "ok",
                        "duration_ms": 234, "source": "test"}, session_id="s1")
        adapter.stop()

        sender.send_message.assert_called_once()
        call_args = sender.send_message.call_args
        assert call_args[0][0] == self.ADDR1
        assert "computer_use" in call_args[0][1]
        assert "234ms" in call_args[0][1]

    def test_log_error_shows_fail(self):
        sender, adapter = self._make_deps()
        adapter.start()

        from plugins.observability.live_glass import publish
        publish("log", {"tool_name": "click", "status": "error",
                        "duration_ms": 10, "error_message": "denied",
                        "source": "test"}, session_id="s2")
        adapter.stop()

        call_args = sender.send_message.call_args
        assert call_args[0][0] == self.ADDR2
        assert "[FAIL]" in call_args[0][1]
        assert "denied" in call_args[0][1]

    def test_approval_sends_reply_prompt_no_buttons(self):
        sender, adapter = self._make_deps()
        adapter.start()

        from plugins.observability.live_glass import publish
        publish("approval_request", {
            "command": "click element #5",
            "description": "computer_use click",
            "surface": "cli",
            "source": "test",
        }, session_id="s1")
        adapter.stop()

        sender.send_message.assert_called_once()
        text = sender.send_message.call_args[0][1]
        assert "approval" in text.lower()
        assert "click element #5" in text
        assert "reply approve or deny" in text.lower()

    def test_frame_sends_image(self):
        sender, adapter = self._make_deps()
        adapter.start()

        b64 = base64.b64encode(b"\x89PNG\r\n\x1a\nbbframe").decode()
        from plugins.observability.live_glass import publish
        publish("frame", {
            "image_url": f"data:image/png;base64,{b64}",
            "summary": "capture mode=som",
            "source": "test",
        }, session_id="s1")
        adapter.stop()

        sender.send_image.assert_called_once()
        call_args = sender.send_image.call_args
        assert call_args[0][0] == self.ADDR1
        assert call_args[0][1].endswith(".png")

    def test_unmapped_session_skips(self):
        sender, adapter = self._make_deps()
        adapter.start()

        from plugins.observability.live_glass import publish
        publish("log", {"tool_name": "x", "status": "ok",
                        "duration_ms": 1, "source": "test"},
                session_id="ghost_session")
        adapter.stop()
        sender.send_message.assert_not_called()

    def test_send_failures_are_caught(self):
        sender, adapter = self._make_deps()
        sender.send_message.side_effect = RuntimeError("no network")
        adapter.start()

        from plugins.observability.live_glass import publish
        publish("log", {"tool_name": "x", "status": "ok",
                        "duration_ms": 1, "source": "test"}, session_id="s1")
        adapter.stop()

    def test_adapter_is_observer_only(self):
        _, adapter = self._make_deps()
        adapter.start()

        from plugins.observability.live_glass import publish
        publish("approval_request", {
            "command": "rm -rf /tmp/test",
            "description": "dangerous",
            "source": "test",
        }, session_id="s1")
        adapter.stop()
        # The adapter has no approve/deny authority
        assert not hasattr(adapter, "approve")
        assert not hasattr(adapter, "deny")
