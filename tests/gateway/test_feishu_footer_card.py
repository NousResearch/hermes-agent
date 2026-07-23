"""Tests for the Feishu footer-card hoisting path in the adapter.

Covers the three delivery scenarios:
1. Non-streamed: footer appended to body → card with body + note.
2. Streamed: footer sent alone as trailing message → card with note only.
3. Card failure: falls back to plain text/post.
"""

from __future__ import annotations

import json
import os
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

from gateway.platforms.base import SendResult


def _make_adapter():
    """Minimal FeishuAdapter with the methods _send_footer_card needs."""
    from plugins.platforms.feishu.adapter import FeishuAdapter

    adapter = object.__new__(FeishuAdapter)
    adapter._client = Mock()  # truthy — passes the "not connected" check
    return adapter


def _ok_response(message_id: str = "om_card_1"):
    """Fake a successful lark SDK response."""
    resp = SimpleNamespace()
    resp.code = 0
    resp.msg = "success"
    resp.success = lambda: True
    resp.data = SimpleNamespace(message_id=message_id)
    return resp


def _err_response(code: int = 230001, msg: str = "card rejected"):
    resp = SimpleNamespace()
    resp.code = code
    resp.msg = msg
    resp.success = lambda: False
    resp.data = None
    return resp


def _cardkit_response(card_id: str = "card_1"):
    """Fake the cardkit create response."""
    resp = SimpleNamespace()
    resp.raw = SimpleNamespace(
        content=json.dumps({"data": {"card_id": card_id}}).encode()
    )
    return resp


class TestFooterCardSend(unittest.TestCase):
    """_send_footer_card returns SendResult on success, None on failure."""

    def test_success_returns_send_result(self):
        adapter = _make_adapter()
        adapter._run_blocking = AsyncMock(return_value=_cardkit_response())
        adapter._feishu_send_with_retry = AsyncMock(return_value=_ok_response())

        import asyncio

        result = asyncio.run(
            adapter._send_footer_card(
                chat_id="oc_1",
                body="Hello",
                footer="Agent: main | Model: k3 | Provider: kimi",
                reply_to=None,
                metadata=None,
            )
        )
        self.assertIsInstance(result, SendResult)
        self.assertTrue(result.success)
        self.assertEqual(result.message_id, "om_card_1")

    def test_unsuccessful_interactive_response_returns_none(self):
        """Interactive response with non-zero code → None (fallback)."""
        adapter = _make_adapter()
        adapter._run_blocking = AsyncMock(return_value=_cardkit_response())
        adapter._feishu_send_with_retry = AsyncMock(return_value=_err_response())

        import asyncio

        result = asyncio.run(
            adapter._send_footer_card(
                chat_id="oc_1",
                body="Hello",
                footer="Agent: main | Model: k3 | Provider: kimi",
                reply_to=None,
                metadata=None,
            )
        )
        self.assertIsNone(result)

    def test_exception_returns_none(self):
        """Any exception → None (fallback)."""
        adapter = _make_adapter()
        adapter._run_blocking = AsyncMock(side_effect=RuntimeError("network"))

        import asyncio

        result = asyncio.run(
            adapter._send_footer_card(
                chat_id="oc_1",
                body="Hello",
                footer="Agent: main | Model: k3 | Provider: kimi",
                reply_to=None,
                metadata=None,
            )
        )
        self.assertIsNone(result)

    def test_cardkit_no_card_id_returns_none(self):
        """Cardkit response missing card_id → None (fallback)."""
        adapter = _make_adapter()
        resp = SimpleNamespace()
        resp.raw = SimpleNamespace(content=b'{"data": {}}')
        adapter._run_blocking = AsyncMock(return_value=resp)

        import asyncio

        result = asyncio.run(
            adapter._send_footer_card(
                chat_id="oc_1",
                body="Hello",
                footer="Agent: main | Model: k3 | Provider: kimi",
                reply_to=None,
                metadata=None,
            )
        )
        self.assertIsNone(result)


class TestSplitBodyAndFooter(unittest.TestCase):
    """_split_body_and_footer handles both \\n\\n-prefixed and bare footers."""

    def _split(self, content: str):
        from plugins.platforms.feishu.adapter import FeishuAdapter

        return FeishuAdapter._split_body_and_footer(content)

    def test_body_with_footer(self):
        body, footer = self._split("Hello world\n\nAgent: main | Model: k3 | Provider: kimi")
        self.assertEqual(body, "Hello world")
        self.assertEqual(footer, "Agent: main | Model: k3 | Provider: kimi")

    def test_bare_footer_streamed(self):
        """Streaming sends the footer as a standalone message (no \\n\\n prefix)."""
        body, footer = self._split("Agent: main | Model: k3 | Provider: kimi")
        self.assertEqual(body, "")
        self.assertEqual(footer, "Agent: main | Model: k3 | Provider: kimi")

    def test_no_footer(self):
        body, footer = self._split("Hello world")
        self.assertEqual(body, "Hello world")
        self.assertEqual(footer, "")

    def test_footer_without_provider(self):
        body, footer = self._split("text\n\nAgent: main | Model: k3")
        self.assertEqual(body, "text")
        self.assertEqual(footer, "Agent: main | Model: k3")


class TestBuildFooterCardJson(unittest.TestCase):
    """Card JSON builder handles body-only, footer-only, and body+footer."""

    def _build(self, body: str, footer: str):
        from plugins.platforms.feishu.adapter import FeishuAdapter

        return FeishuAdapter._build_footer_card_json(body, footer)

    def test_body_and_footer(self):
        card = self._build("Hello", "Agent: a | Model: m | Provider: p")
        elements = card["body"]["elements"]
        self.assertEqual(len(elements), 3)  # markdown + hr + note
        self.assertEqual(elements[0]["tag"], "markdown")
        self.assertEqual(elements[1]["tag"], "hr")
        self.assertIn("grey", elements[2]["content"])

    def test_footer_only_streamed(self):
        """Streamed trailing footer → note element only, no empty markdown."""
        card = self._build("", "Agent: a | Model: m | Provider: p")
        elements = card["body"]["elements"]
        self.assertEqual(len(elements), 1)
        self.assertEqual(elements[0]["tag"], "markdown")
        self.assertIn("grey", elements[0]["content"])


class TestFooterCardEnabled(unittest.TestCase):
    """_footer_card_enabled reads from config.yaml, not just .env."""

    def test_default_enabled(self):
        adapter = _make_adapter()
        with patch.dict(os.environ, {}, clear=True):
            self.assertTrue(adapter._footer_card_enabled())

    def test_env_disable(self):
        adapter = _make_adapter()
        with patch.dict(os.environ, {"FEISHU_FOOTER_CARD": "0"}):
            self.assertFalse(adapter._footer_card_enabled())

    def test_config_disable(self):
        adapter = _make_adapter()
        fake_cfg = {
            "display": {
                "platforms": {
                    "feishu": {"runtime_footer": {"card": False}},
                },
            },
        }
        with patch.dict(os.environ, {}, clear=True):
            with patch("hermes_cli.config.read_raw_config", return_value=fake_cfg):
                self.assertFalse(adapter._footer_card_enabled())

    def test_config_enable_overrides_nothing(self):
        adapter = _make_adapter()
        fake_cfg = {
            "display": {
                "platforms": {
                    "feishu": {"runtime_footer": {"card": True}},
                },
            },
        }
        with patch.dict(os.environ, {}, clear=True):
            with patch("hermes_cli.config.read_raw_config", return_value=fake_cfg):
                self.assertTrue(adapter._footer_card_enabled())

    def test_env_disable_wins_over_config(self):
        """.env escape hatch takes precedence (backward compat)."""
        adapter = _make_adapter()
        fake_cfg = {
            "display": {
                "platforms": {
                    "feishu": {"runtime_footer": {"card": True}},
                },
            },
        }
        with patch.dict(os.environ, {"FEISHU_FOOTER_CARD": "0"}):
            with patch("hermes_cli.config.read_raw_config", return_value=fake_cfg):
                self.assertFalse(adapter._footer_card_enabled())


if __name__ == "__main__":
    unittest.main()
