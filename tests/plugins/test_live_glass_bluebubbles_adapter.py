"""Tests for the live-glass BlueBubbles adapter (AVA-23)."""
from __future__ import annotations

import base64
import importlib
import sys
from unittest.mock import MagicMock

from plugins.observability.live_glass.adapters.bluebubbles import (
    BlueBubblesLiveGlassAdapter,
)


def _fresh_module():
    sys.modules.pop(
        "plugins.observability.live_glass.adapters.bluebubbles", None
    )
    return importlib.import_module(
        "plugins.observability.live_glass.adapters.bluebubbles"
    )


class TestBlueBubblesLiveGlassAdapter:
    def _make_deps(self):
        """Return (sender_mock, router, adapter)."""
        sender = MagicMock()
        chat_map = {"s1": "+15551234567", "s2": "chat_guid_abc"}
        router = lambda sid: chat_map.get(sid)
        adapter = BlueBubblesLiveGlassAdapter(sender, router)
        return sender, router, adapter

    def test_start_and_stop(self):
        _, _, adapter = self._make_deps()
        adapter.start()
        assert adapter._unsubscribe is not None
        adapter.stop()
        assert adapter._unsubscribe is None

    def test_frame_sends_image(self):
        sender, _, adapter = self._make_deps()
        adapter.start()

        b64 = base64.b64encode(b"\x89PNG\r\n\x1a\nfakeframe").decode()

        from plugins.observability.live_glass import publish

        publish(
            "frame",
            {
                "image_url": f"data:image/png;base64,{b64}",
                "mime_type": "image/png",
                "mode": "som",
                "width": 100,
                "height": 200,
                "summary": "capture mode=som",
                "source": "computer_use",
            },
            session_id="s1",
        )

        adapter.stop()

        sender.send_image.assert_called_once()
        call_args = sender.send_image.call_args
        # First arg: chat_address
        assert call_args[0][0] == "+15551234567"
        # Second arg: file_path — should be a .png temp file
        assert call_args[0][1].endswith(".png")
        # Third arg: caption
        caption = (
            call_args[0][2] if len(call_args[0]) > 2
            else call_args[1].get("caption")
        )
        assert "capture mode=som" in (caption or "")
        # Temp file cleaned up by adapter's _handle_frame()

    def test_log_sends_text(self):
        sender, _, adapter = self._make_deps()
        adapter.start()

        from plugins.observability.live_glass import publish

        publish(
            "log",
            {
                "tool_name": "computer_use",
                "status": "ok",
                "duration_ms": 123,
                "source": "test",
            },
            session_id="s1",
        )

        adapter.stop()

        sender.send_message.assert_called_once()
        args, _ = sender.send_message.call_args
        assert args[0] == "+15551234567"  # chat_address from router
        assert "\u2713" in args[1]  # ✓
        assert "computer_use" in args[1]
        assert "123ms" in args[1]

    def test_log_with_error_sends_text(self):
        sender, _, adapter = self._make_deps()
        adapter.start()

        from plugins.observability.live_glass import publish

        publish(
            "log",
            {
                "tool_name": "click",
                "status": "error",
                "duration_ms": 50,
                "error_message": "denied by user",
                "source": "test",
            },
            session_id="s2",
        )

        adapter.stop()

        sender.send_message.assert_called_once()
        args, _ = sender.send_message.call_args
        assert args[0] == "chat_guid_abc"
        assert "\u2717" in args[1]  # ✗
        assert "denied by user" in args[1]

    def test_approval_sends_reply_prompt_text(self):
        sender, _, adapter = self._make_deps()
        adapter.start()

        from plugins.observability.live_glass import publish

        publish(
            "approval_request",
            {
                "command": "click element #5",
                "description": "computer_use click",
                "surface": "cli",
                "source": "test",
            },
            session_id="s1",
            tool_call_id="call_1",
        )

        adapter.stop()

        sender.send_message.assert_called_once()
        call_args = sender.send_message.call_args
        assert call_args[0][0] == "+15551234567"  # chat_address
        text = call_args[0][1]
        assert "Approval needed" in text
        assert "Reply APPROVE or DENY" in text
        assert "click element #5" in text
        assert "computer_use click" in text
        # iMessage limitation: no interactive buttons — just text prompt
        assert len(call_args[0]) == 2  # only chat_address + text, no buttons

    def test_unmapped_session_skips(self):
        sender, _, adapter = self._make_deps()
        adapter.start()

        from plugins.observability.live_glass import publish

        publish(
            "log",
            {
                "tool_name": "test",
                "status": "ok",
                "duration_ms": 1,
                "source": "test",
            },
            session_id="unknown_session",
        )

        adapter.stop()
        sender.send_message.assert_not_called()
        sender.send_image.assert_not_called()

    def test_empty_session_id_skips(self):
        sender, _, adapter = self._make_deps()
        adapter.start()

        from plugins.observability.live_glass import publish

        publish(
            "log",
            {
                "tool_name": "test",
                "status": "ok",
                "duration_ms": 1,
                "source": "test",
            },
            session_id="",
        )

        adapter.stop()
        sender.send_message.assert_not_called()

    def test_send_failures_caught(self):
        sender, _, adapter = self._make_deps()
        sender.send_message.side_effect = RuntimeError("network down")
        adapter.start()

        from plugins.observability.live_glass import publish

        publish(
            "log",
            {
                "tool_name": "test",
                "status": "ok",
                "duration_ms": 1,
                "source": "test",
            },
            session_id="s1",
        )

        adapter.stop()
        # Should not raise — exception is caught and logged

    def test_frame_send_failure_caught(self):
        sender, _, adapter = self._make_deps()
        sender.send_image.side_effect = RuntimeError("image send failed")

        b64 = base64.b64encode(b"\x89PNG\r\n\x1a\nfakeframe").decode()
        adapter.start()

        from plugins.observability.live_glass import publish

        publish(
            "frame",
            {
                "image_url": f"data:image/png;base64,{b64}",
                "mime_type": "image/png",
                "mode": "som",
                "width": 100,
                "height": 200,
                "summary": "capture mode=som",
                "source": "computer_use",
            },
            session_id="s1",
        )

        adapter.stop()
        # Should not raise — exception is caught and logged

    def test_approval_send_failure_caught(self):
        sender, _, adapter = self._make_deps()
        sender.send_message.side_effect = RuntimeError("approval send failed")
        adapter.start()

        from plugins.observability.live_glass import publish

        publish(
            "approval_request",
            {
                "command": "rm -rf /tmp",
                "description": "dangerous command",
                "surface": "cli",
                "source": "test",
            },
            session_id="s1",
            tool_call_id="call_2",
        )

        adapter.stop()
        # Should not raise — exception is caught and logged
