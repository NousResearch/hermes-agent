"""Tests for the live-glass Discord and Slack adapters (AVA-22)."""
from __future__ import annotations

import base64
import json
from unittest.mock import MagicMock

import pytest

from plugins.observability.live_glass.adapters.discord_slack import (
    DiscordLiveGlassAdapter,
    SlackLiveGlassAdapter,
    _build_discord_approval_components,
    _build_slack_approval_blocks,
)


# ── Helpers ──────────────────────────────────────────────────────────────


def _publish_and_wait(adapter, event_type, payload, **context):
    """Publish an event and wait for synchronous subscribers to run."""
    adapter.start()
    from plugins.observability.live_glass import publish

    publish(event_type, payload, **context)
    adapter.stop()


# ── Discord component builder tests ──────────────────────────────────────


class TestDiscordApprovalComponents:
    def test_builds_action_row_components(self):
        components = _build_discord_approval_components("tc_123")
        assert components is not None
        assert len(components) == 2

        # Row 0: Approve Once, Approve Session
        row0 = components[0]
        assert row0["type"] == 1
        btns0 = row0["components"]
        assert len(btns0) == 2
        assert btns0[0]["label"] == "Approve Once"
        assert btns0[0]["style"] == 3  # Success green
        assert json.loads(btns0[0]["custom_id"]) == {"a": "once", "tcid": "tc_123"}
        assert btns0[1]["label"] == "Approve Session"
        assert json.loads(btns0[1]["custom_id"]) == {"a": "session", "tcid": "tc_123"}

        # Row 1: Approve Always, Deny
        row1 = components[1]
        assert row1["type"] == 1
        btns1 = row1["components"]
        assert len(btns1) == 2
        assert btns1[0]["label"] == "Approve Always"
        assert btns1[0]["style"] == 1  # Primary blurple
        assert json.loads(btns1[0]["custom_id"]) == {"a": "always", "tcid": "tc_123"}
        assert btns1[1]["label"] == "Deny"
        assert btns1[1]["style"] == 4  # Danger red
        assert json.loads(btns1[1]["custom_id"]) == {"a": "deny", "tcid": "tc_123"}

    def test_empty_tool_call_id_returns_none(self):
        assert _build_discord_approval_components("") is None


# ── Slack block builder tests ────────────────────────────────────────────


class TestSlackApprovalBlocks:
    def test_builds_block_kit_blocks(self):
        blocks = _build_slack_approval_blocks("tc_456", "⚠️ *Approve?*")
        assert blocks is not None
        assert len(blocks) == 1

        actions = blocks[0]
        assert actions["type"] == "actions"
        elements = actions["elements"]
        assert len(elements) == 4

        assert elements[0]["type"] == "button"
        assert elements[0]["text"]["text"] == "Approve Once"
        assert elements[0]["style"] == "primary"
        assert elements[0]["action_id"] == "approve_once"
        assert json.loads(elements[0]["value"]) == {"a": "once", "tcid": "tc_456"}

        assert elements[1]["text"]["text"] == "Approve Session"
        assert elements[1]["action_id"] == "approve_session"

        assert elements[2]["text"]["text"] == "Approve Always"
        assert elements[2]["action_id"] == "approve_always"

        assert elements[3]["text"]["text"] == "Deny"
        assert elements[3]["style"] == "danger"
        assert elements[3]["action_id"] == "deny"
        assert json.loads(elements[3]["value"]) == {"a": "deny", "tcid": "tc_456"}

    def test_empty_tool_call_id_returns_none(self):
        assert _build_slack_approval_blocks("", "") is None


# ── Discord adapter tests ────────────────────────────────────────────────


class TestDiscordLiveGlassAdapter:
    @staticmethod
    def _make_deps():
        """Return (sender_mock, chat_map, adapter)."""
        sender = MagicMock()
        chat_map = {"s1": 111, "s2": 222}
        router = lambda sid: chat_map.get(sid)
        adapter = DiscordLiveGlassAdapter(sender, router)
        return sender, router, adapter

    def test_start_and_stop(self):
        _, _, adapter = self._make_deps()
        adapter.start()
        assert adapter._unsubscribe is not None
        adapter.stop()
        assert adapter._unsubscribe is None

    def test_log_event_sends_message(self):
        sender, _, adapter = self._make_deps()

        _publish_and_wait(
            adapter,
            "log",
            {
                "tool_name": "computer_use",
                "status": "ok",
                "duration_ms": 123,
                "source": "test",
            },
            session_id="s1",
        )

        sender.send_message.assert_called_once()
        args, _ = sender.send_message.call_args
        assert args[0] == 111  # channel_id from router
        assert "computer_use" in args[1]
        assert "123ms" in args[1]

    def test_log_event_with_error(self):
        sender, _, adapter = self._make_deps()

        _publish_and_wait(
            adapter,
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

        sender.send_message.assert_called_once()
        args, _ = sender.send_message.call_args
        assert args[0] == 222
        assert "denied by user" in args[1]

    def test_approval_event_sends_components(self):
        sender, _, adapter = self._make_deps()

        _publish_and_wait(
            adapter,
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

        sender.send_message.assert_called_once()
        call_args = sender.send_message.call_args
        assert call_args[0][0] == 111
        assert "click element #5" in call_args[0][1]

        # components is the third positional arg
        components = call_args[0][2] if len(call_args[0]) > 2 else call_args[1].get("components")
        assert components is not None
        assert len(components) == 2  # two action rows

    def test_frame_event_sends_file(self):
        sender, _, adapter = self._make_deps()

        b64 = base64.b64encode(b"\x89PNG\r\n\x1a\nfakeframe").decode()

        _publish_and_wait(
            adapter,
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

        sender.send_file.assert_called_once()
        call_args = sender.send_file.call_args
        assert call_args[0][0] == 111
        assert call_args[0][1].endswith(".png")
        caption = call_args[0][2] if len(call_args[0]) > 2 else call_args[1].get("caption")
        assert "capture mode=som" in (caption or "")

    def test_skips_events_with_unmapped_session(self):
        sender, _, adapter = self._make_deps()

        _publish_and_wait(
            adapter,
            "log",
            {"tool_name": "test", "status": "ok", "duration_ms": 1, "source": "test"},
            session_id="unknown_session",
        )

        sender.send_message.assert_not_called()
        sender.send_file.assert_not_called()

    def test_skips_events_with_empty_session_id(self):
        sender, router, adapter = self._make_deps()

        _publish_and_wait(
            adapter,
            "log",
            {"tool_name": "test", "status": "ok", "duration_ms": 1, "source": "test"},
            session_id="",
        )

        sender.send_message.assert_not_called()

    def test_sender_exceptions_are_caught(self):
        sender, _, adapter = self._make_deps()
        sender.send_message.side_effect = RuntimeError("network down")

        _publish_and_wait(
            adapter,
            "log",
            {"tool_name": "test", "status": "ok", "duration_ms": 1, "source": "test"},
            session_id="s1",
        )
        # Should not raise — exception is caught and logged

    def test_frame_send_file_exception_is_caught(self):
        sender, _, adapter = self._make_deps()
        sender.send_file.side_effect = RuntimeError("upload failed")
        b64 = base64.b64encode(b"\x89PNG\r\n\x1a\nfakeframe").decode()

        _publish_and_wait(
            adapter,
            "frame",
            {
                "image_url": f"data:image/png;base64,{b64}",
                "mime_type": "image/png",
                "summary": "test",
                "source": "computer_use",
            },
            session_id="s1",
        )
        # Should not raise

    def test_approval_send_exception_is_caught(self):
        sender, _, adapter = self._make_deps()
        sender.send_message.side_effect = RuntimeError("send failed")

        _publish_and_wait(
            adapter,
            "approval_request",
            {"command": "test", "description": "test", "surface": "cli", "source": "test"},
            session_id="s1",
            tool_call_id="call_1",
        )
        # Should not raise


# ── Slack adapter tests ──────────────────────────────────────────────────


class TestSlackLiveGlassAdapter:
    @staticmethod
    def _make_deps():
        """Return (sender_mock, chat_map, adapter)."""
        sender = MagicMock()
        chat_map = {"s1": "C001", "s2": "C002"}
        router = lambda sid: chat_map.get(sid)
        adapter = SlackLiveGlassAdapter(sender, router)
        return sender, router, adapter

    def test_start_and_stop(self):
        _, _, adapter = self._make_deps()
        adapter.start()
        assert adapter._unsubscribe is not None
        adapter.stop()
        assert adapter._unsubscribe is None

    def test_log_event_sends_message(self):
        sender, _, adapter = self._make_deps()

        _publish_and_wait(
            adapter,
            "log",
            {
                "tool_name": "computer_use",
                "status": "ok",
                "duration_ms": 123,
                "source": "test",
            },
            session_id="s1",
        )

        sender.send_message.assert_called_once()
        args, kwargs = sender.send_message.call_args
        assert args[0] == "C001"  # channel_id from router
        assert "computer_use" in args[1]
        assert "123ms" in args[1]
        # blocks=None for log events (passed as keyword)
        assert kwargs.get("blocks") is None

    def test_log_event_with_error(self):
        sender, _, adapter = self._make_deps()

        _publish_and_wait(
            adapter,
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

        sender.send_message.assert_called_once()
        args, _ = sender.send_message.call_args
        assert args[0] == "C002"
        assert "denied by user" in args[1]

    def test_approval_event_sends_blocks(self):
        sender, _, adapter = self._make_deps()

        _publish_and_wait(
            adapter,
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

        sender.send_message.assert_called_once()
        call_args = sender.send_message.call_args
        assert call_args[0][0] == "C001"
        assert "click element #5" in call_args[0][1]

        # blocks is the third positional arg
        blocks = call_args[0][2] if len(call_args[0]) > 2 else call_args[1].get("blocks")
        assert blocks is not None
        assert len(blocks) == 1
        assert blocks[0]["type"] == "actions"
        assert len(blocks[0]["elements"]) == 4

    def test_frame_event_uploads_file(self):
        sender, _, adapter = self._make_deps()

        b64 = base64.b64encode(b"\x89PNG\r\n\x1a\nfakeframe").decode()

        _publish_and_wait(
            adapter,
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

        sender.upload_file.assert_called_once()
        call_args = sender.upload_file.call_args
        assert call_args[0][0] == "C001"
        assert call_args[0][1].endswith(".png")
        caption = call_args[0][2] if len(call_args[0]) > 2 else call_args[1].get("caption")
        assert "capture mode=som" in (caption or "")

    def test_skips_events_with_unmapped_session(self):
        sender, _, adapter = self._make_deps()

        _publish_and_wait(
            adapter,
            "log",
            {"tool_name": "test", "status": "ok", "duration_ms": 1, "source": "test"},
            session_id="unknown_session",
        )

        sender.send_message.assert_not_called()
        sender.upload_file.assert_not_called()

    def test_skips_events_with_empty_session_id(self):
        sender, router, adapter = self._make_deps()

        _publish_and_wait(
            adapter,
            "log",
            {"tool_name": "test", "status": "ok", "duration_ms": 1, "source": "test"},
            session_id="",
        )

        sender.send_message.assert_not_called()

    def test_sender_exceptions_are_caught(self):
        sender, _, adapter = self._make_deps()
        sender.send_message.side_effect = RuntimeError("network down")

        _publish_and_wait(
            adapter,
            "log",
            {"tool_name": "test", "status": "ok", "duration_ms": 1, "source": "test"},
            session_id="s1",
        )
        # Should not raise

    def test_frame_upload_exception_is_caught(self):
        sender, _, adapter = self._make_deps()
        sender.upload_file.side_effect = RuntimeError("upload failed")
        b64 = base64.b64encode(b"\x89PNG\r\n\x1a\nfakeframe").decode()

        _publish_and_wait(
            adapter,
            "frame",
            {
                "image_url": f"data:image/png;base64,{b64}",
                "mime_type": "image/png",
                "summary": "test",
                "source": "computer_use",
            },
            session_id="s1",
        )
        # Should not raise

    def test_approval_send_exception_is_caught(self):
        sender, _, adapter = self._make_deps()
        sender.send_message.side_effect = RuntimeError("send failed")

        _publish_and_wait(
            adapter,
            "approval_request",
            {"command": "test", "description": "test", "surface": "cli", "source": "test"},
            session_id="s1",
            tool_call_id="call_1",
        )
        # Should not raise
