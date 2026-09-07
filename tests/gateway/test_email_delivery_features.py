"""Tests for config-driven email adapter delivery features:
inbox channel_prompt, escalation-marker routing, duplicate-send suppression,
and internal-block stripping. All behaviors are opt-in via platforms.email.extra.
"""

import asyncio
import os
import unittest
from unittest.mock import MagicMock, patch

from plugins.platforms.email.adapter import EmailAdapter, _strip_internal_prefix


def _make_adapter(extra=None):
    from gateway.config import PlatformConfig
    with patch.dict(os.environ, {
        "EMAIL_ADDRESS": "hermes@test.com",
        "EMAIL_PASSWORD": "secret",
        "EMAIL_IMAP_HOST": "imap.test.com",
        "EMAIL_SMTP_HOST": "smtp.test.com",
    }):
        return EmailAdapter(PlatformConfig(enabled=True, extra=extra or {}))


class TestStripInternalPrefix(unittest.TestCase):
    def test_no_block_passthrough(self):
        self.assertEqual(_strip_internal_prefix("Hello world"), "Hello world")

    def test_reasoning_block_with_fence_stripped(self):
        content = "💭 **Reasoning:**\n```\ninternal thoughts\n```\nReal reply"
        self.assertEqual(_strip_internal_prefix(content), "Real reply")

    def test_reasoning_block_plain_stripped(self):
        content = "💭 internal\nReal reply"
        self.assertEqual(_strip_internal_prefix(content), "Real reply")

    def test_block_only_returns_empty(self):
        self.assertEqual(_strip_internal_prefix("💭 only thinking"), "")

    def test_none_safe(self):
        self.assertEqual(_strip_internal_prefix(None), "")


class TestChannelPrompt(unittest.TestCase):
    def test_extra_channel_prompt(self):
        adapter = _make_adapter({"channel_prompt": "You are an inbox assistant."})
        self.assertEqual(adapter._channel_prompt, "You are an inbox assistant.")

    def test_channel_prompts_inbox_style(self):
        adapter = _make_adapter({"channel_prompts": {"inbox": "Inbox persona"}})
        self.assertEqual(adapter._channel_prompt, "Inbox persona")

    def test_default_none(self):
        adapter = _make_adapter()
        self.assertIsNone(adapter._channel_prompt)

    def test_dispatch_attaches_prompt(self):
        adapter = _make_adapter({"channel_prompt": "persona"})
        captured = {}

        async def fake_handle(event):
            captured["event"] = event

        adapter.handle_message = fake_handle
        adapter._sender_accepted = lambda addr, md: True
        msg_data = {"sender_addr": "user@test.com", "sender_name": "User", "subject": "Hi",
                    "body": "hello", "attachments": [], "message_id": "<m@test>", "in_reply_to": None}
        asyncio.run(adapter._dispatch_message(msg_data))
        self.assertEqual(captured["event"].channel_prompt, "persona")


class TestEscalationRouting(unittest.TestCase):
    def _runner(self, tg_adapter, home=None):
        from gateway.config import Platform, PlatformConfig, HomeChannel
        runner = MagicMock()
        runner.adapters = {Platform.TELEGRAM: tg_adapter}
        runner._profile_adapters = {}
        runner.config.get_home_channel = lambda p: home
        return runner

    def test_escalation_routed_to_platform(self):
        from gateway.config import Platform
        from gateway.platforms.base import SendResult
        tg = MagicMock()
        tg.send = MagicMock(return_value=asyncio.sleep(0, result=SendResult(success=True, message_id="tg-1")))
        adapter = _make_adapter({"escalation_deliver": "telegram", "escalation_chat": "123"})
        adapter.gateway_runner = self._runner(tg)

        result = asyncio.run(adapter.send("customer@test.com", "[ESCALATE] Need human help"))
        self.assertTrue(result.success)
        tg.send.assert_called_once()
        args = tg.send.call_args[0]
        self.assertEqual(args[0], "123")
        self.assertIn("[EMAIL ESCALATION — customer@test.com]", args[1])
        self.assertIn("Need human help", args[1])

    def test_escalation_home_channel_fallback(self):
        from gateway.config import Platform, HomeChannel
        from gateway.platforms.base import SendResult
        tg = MagicMock()
        tg.send = MagicMock(return_value=asyncio.sleep(0, result=SendResult(success=True, message_id="tg-1")))
        adapter = _make_adapter({"escalation_deliver": "telegram"})  # no escalation_chat
        adapter.gateway_runner = self._runner(tg, home=HomeChannel(platform=Platform.TELEGRAM, chat_id="999", name="Home"))

        result = asyncio.run(adapter.send("customer@test.com", "[ESCALATE] help"))
        self.assertTrue(result.success)
        self.assertEqual(tg.send.call_args[0][0], "999")

    def test_no_escalation_deliver_falls_back_to_email(self):
        adapter = _make_adapter()  # marker default, no deliver target
        with patch("smtplib.SMTP") as mock_smtp:
            mock_smtp.return_value = MagicMock()
            result = asyncio.run(adapter.send("customer@test.com", "[ESCALATE] help"))
            self.assertTrue(result.success)
            mock_smtp.return_value.send_message.assert_called_once()

    def test_unknown_platform_error(self):
        adapter = _make_adapter({"escalation_deliver": "carrier-pigeon", "escalation_chat": "x"})
        adapter.gateway_runner = self._runner(MagicMock())
        result = asyncio.run(adapter.send("customer@test.com", "[ESCALATE] help"))
        self.assertFalse(result.success)
        self.assertIn("Unknown platform", result.error)


class TestDuplicateSuppression(unittest.TestCase):
    def test_identical_send_within_window_suppressed(self):
        adapter = _make_adapter({"dedupe_window_seconds": 60})
        with patch("smtplib.SMTP") as mock_smtp:
            mock_smtp.return_value = MagicMock()
            asyncio.run(adapter.send("user@test.com", "same body"))
            asyncio.run(adapter.send("user@test.com", "same body"))
            mock_smtp.return_value.send_message.assert_called_once()

    def test_different_content_not_suppressed(self):
        adapter = _make_adapter({"dedupe_window_seconds": 60})
        with patch("smtplib.SMTP") as mock_smtp:
            mock_smtp.return_value = MagicMock()
            asyncio.run(adapter.send("user@test.com", "body one"))
            asyncio.run(adapter.send("user@test.com", "body two"))
            self.assertEqual(mock_smtp.return_value.send_message.call_count, 2)

    def test_window_zero_disables(self):
        adapter = _make_adapter({"dedupe_window_seconds": 0})
        with patch("smtplib.SMTP") as mock_smtp:
            mock_smtp.return_value = MagicMock()
            asyncio.run(adapter.send("user@test.com", "same body"))
            asyncio.run(adapter.send("user@test.com", "same body"))
            self.assertEqual(mock_smtp.return_value.send_message.call_count, 2)

    def test_non_email_chat_id_skipped(self):
        adapter = _make_adapter()
        with patch("smtplib.SMTP") as mock_smtp:
            result = asyncio.run(adapter.send("7185545230", "bootstrap notice"))
            self.assertTrue(result.success)
            mock_smtp.return_value.send_message.assert_not_called()


import unittest
if __name__ == "__main__":
    unittest.main()
