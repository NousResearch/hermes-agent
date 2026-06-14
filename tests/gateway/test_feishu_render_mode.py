"""Tests for Feishu render_mode (interactive card rendering)."""

import asyncio
import json
import os
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

from gateway.config import PlatformConfig


try:
    import lark_oapi
    _HAS_LARK = True
except ImportError:
    _HAS_LARK = False


@unittest.skipUnless(_HAS_LARK, "lark_oapi not installed")
class TestFeishuRenderModeSettings(unittest.TestCase):
    """Test render_mode config loading and validation."""

    def test_default_render_mode_is_auto(self):
        """Default render_mode should be 'auto'."""
        from gateway.platforms.feishu import FeishuAdapter, FeishuAdapterSettings
        adapter = FeishuAdapter(PlatformConfig())
        self.assertEqual(adapter._render_mode, "auto")

    def test_render_mode_auto_from_config(self):
        """render_mode='auto' loaded from config extra."""
        from gateway.platforms.feishu import FeishuAdapter
        pc = PlatformConfig(enabled=True, extra={"render_mode": "auto"})
        adapter = FeishuAdapter(pc)
        self.assertEqual(adapter._render_mode, "auto")

    def test_render_mode_card_from_config(self):
        """render_mode='card' loaded from config extra."""
        from gateway.platforms.feishu import FeishuAdapter
        pc = PlatformConfig(enabled=True, extra={"render_mode": "card"})
        adapter = FeishuAdapter(pc)
        self.assertEqual(adapter._render_mode, "card")

    def test_render_mode_raw_from_config(self):
        """render_mode='raw' loaded from config extra."""
        from gateway.platforms.feishu import FeishuAdapter
        pc = PlatformConfig(enabled=True, extra={"render_mode": "raw"})
        adapter = FeishuAdapter(pc)
        self.assertEqual(adapter._render_mode, "raw")

    def test_invalid_render_mode_falls_back_to_auto(self):
        """An invalid render_mode falls back to 'auto' with a warning."""
        from gateway.platforms.feishu import FeishuAdapter
        pc = PlatformConfig(enabled=True, extra={"render_mode": "invalid"})
        with self.assertLogs("gateway.platforms.feishu", level="WARNING") as cm:
            adapter = FeishuAdapter(pc)
        self.assertEqual(adapter._render_mode, "auto")
        self.assertTrue(any("Unknown render_mode" in msg for msg in cm.output))

    @patch.dict(os.environ, {"FEISHU_RENDER_MODE": "card"})
    def test_render_mode_from_env_var(self):
        """render_mode can be set via FEISHU_RENDER_MODE env var."""
        from gateway.platforms.feishu import FeishuAdapter
        pc = PlatformConfig(enabled=True, extra={})
        adapter = FeishuAdapter(pc)
        self.assertEqual(adapter._render_mode, "card")

    def test_config_extra_render_mode_case_insensitive(self):
        """render_mode is case-insensitive (e.g. 'CARD' → 'card')."""
        from gateway.platforms.feishu import FeishuAdapter
        pc = PlatformConfig(enabled=True, extra={"render_mode": "CARD"})
        adapter = FeishuAdapter(pc)
        self.assertEqual(adapter._render_mode, "card")


class TestFeishuShouldUseCard(unittest.TestCase):
    """Test _should_use_card detection logic."""

    def setUp(self):
        from gateway.platforms.feishu import FeishuAdapter
        self._should_use_card = FeishuAdapter._should_use_card

    def test_code_fence_triggers_card(self):
        """Content with fenced code blocks should use card."""
        text = "Here is code:\n```python\nprint('hello')\n```\nDone."
        self.assertTrue(self._should_use_card(text))

    def test_markdown_table_triggers_card(self):
        """Content with markdown tables should use card."""
        text = "| Name | Value |\n|------|-------|\n| A    | 1     |"
        self.assertTrue(self._should_use_card(text))

    def test_plain_text_no_card(self):
        """Plain text without code blocks or tables should not use card."""
        text = "Just a simple message with no special formatting."
        self.assertFalse(self._should_use_card(text))

    def test_inline_code_no_card(self):
        """Inline code (single backticks) should not trigger card."""
        text = "Use the `pip install` command."
        self.assertFalse(self._should_use_card(text))


class TestFeishuBuildCardPayload(unittest.TestCase):
    """Test _build_card_payload output format."""

    def setUp(self):
        from gateway.platforms.feishu import FeishuAdapter
        self._build_card_payload = FeishuAdapter._build_card_payload

    def test_card_payload_is_valid_json(self):
        """Card payload should be valid JSON."""
        result = self._build_card_payload("Hello world")
        parsed = json.loads(result)
        self.assertIsInstance(parsed, dict)

    def test_card_payload_schema_2(self):
        """Card payload should use schema 2.0."""
        result = self._build_card_payload("Hello")
        parsed = json.loads(result)
        self.assertEqual(parsed["schema"], "2.0")

    def test_card_payload_width_mode_fill(self):
        """Card payload should set width_mode to fill."""
        result = self._build_card_payload("Hello")
        parsed = json.loads(result)
        self.assertEqual(parsed["config"]["width_mode"], "fill")

    def test_card_payload_markdown_element(self):
        """Card payload should contain a markdown element with the content."""
        result = self._build_card_payload("Test content")
        parsed = json.loads(result)
        elements = parsed["body"]["elements"]
        self.assertEqual(len(elements), 1)
        self.assertEqual(elements[0]["tag"], "markdown")
        self.assertEqual(elements[0]["content"], "Test content")

    def test_card_payload_preserves_unicode(self):
        """Card payload should preserve non-ASCII characters."""
        result = self._build_card_payload("中文测试 🎉")
        parsed = json.loads(result)
        self.assertEqual(parsed["body"]["elements"][0]["content"], "中文测试 🎉")


@unittest.skipUnless(_HAS_LARK, "lark_oapi not installed")
class TestFeishuBuildOutboundPayloadRenderMode(unittest.TestCase):
    """Test _build_outbound_payload with different render_mode settings."""

    def _make_adapter(self, render_mode="auto"):
        from gateway.platforms.feishu import FeishuAdapter
        pc = PlatformConfig(enabled=True, extra={"render_mode": render_mode})
        return FeishuAdapter(pc)

    def test_auto_mode_table_uses_interactive(self):
        """In auto mode, markdown tables should trigger interactive card."""
        adapter = self._make_adapter("auto")
        text = "| Name | Value |\n|------|-------|\n| A    | 1     |"
        msg_type, payload = adapter._build_outbound_payload(text)
        self.assertEqual(msg_type, "interactive")

    def test_auto_mode_code_block_uses_interactive(self):
        """In auto mode, code blocks should trigger interactive card."""
        adapter = self._make_adapter("auto")
        text = "```python\nprint('hello')\n```"
        msg_type, payload = adapter._build_outbound_payload(text)
        self.assertEqual(msg_type, "interactive")

    def test_auto_mode_plain_text_uses_text(self):
        """In auto mode, plain text should use text type."""
        adapter = self._make_adapter("auto")
        msg_type, payload = adapter._build_outbound_payload("Hello world")
        self.assertEqual(msg_type, "text")

    def test_card_mode_always_uses_interactive(self):
        """In card mode, even plain text should use interactive card."""
        adapter = self._make_adapter("card")
        msg_type, payload = adapter._build_outbound_payload("Hello world")
        self.assertEqual(msg_type, "interactive")

    def test_raw_mode_table_uses_text(self):
        """In raw mode, even tables should use text (legacy behaviour)."""
        adapter = self._make_adapter("raw")
        text = "| Name | Value |\n|------|-------|\n| A    | 1     |"
        msg_type, payload = adapter._build_outbound_payload(text)
        self.assertEqual(msg_type, "text")

    def test_raw_mode_plain_text_uses_text(self):
        """In raw mode, plain text should use text type."""
        adapter = self._make_adapter("raw")
        msg_type, payload = adapter._build_outbound_payload("Hello world")
        self.assertEqual(msg_type, "text")


class TestFeishuRenderModeConfigBridge(unittest.TestCase):
    """Test that render_mode is bridged from config.yaml to adapter extra."""

    def test_render_mode_bridged_from_real_config_yaml(self):
        """render_mode in feishu: config is loaded via load_gateway_config()."""
        from gateway.config import Platform, load_gateway_config

        with tempfile.TemporaryDirectory() as tmp:
            config_path = os.path.join(tmp, "config.yaml")
            with open(config_path, "w", encoding="utf-8") as f:
                f.write(
                    "feishu:\n"
                    "  enabled: true\n"
                    "  app_id: cli_test\n"
                    "  app_secret: secret_test\n"
                    "  render_mode: card\n"
                )

            with patch.dict(os.environ, {"HERMES_HOME": tmp}, clear=False):
                config = load_gateway_config()

        feishu_cfg = config.platforms.get(Platform.FEISHU)
        if feishu_cfg is None:
            self.fail("Feishu platform config was not loaded")
        self.assertTrue(feishu_cfg.enabled)
        self.assertEqual(feishu_cfg.extra.get("render_mode"), "card")


@unittest.skipUnless(_HAS_LARK, "lark_oapi not installed")
class TestFeishuRenderModeSendFallback(unittest.TestCase):
    """Test send() fallback when interactive card delivery fails."""

    def _make_adapter(self):
        from gateway.platforms.feishu import FeishuAdapter
        adapter = FeishuAdapter(PlatformConfig(enabled=True, extra={"render_mode": "auto"}))
        adapter._client = Mock()
        return adapter

    @staticmethod
    def _ok_response(message_id="om_fallback"):
        return SimpleNamespace(success=lambda: True, data=SimpleNamespace(message_id=message_id))

    @staticmethod
    def _bad_response():
        return SimpleNamespace(success=lambda: False, code=400, msg="card rejected")

    def test_send_interactive_exception_falls_back_to_post(self):
        adapter = self._make_adapter()
        calls = []

        async def _fake_send(**kwargs):
            calls.append(kwargs)
            if kwargs["msg_type"] == "interactive":
                raise RuntimeError("interactive failed")
            return self._ok_response("om_post_fallback")

        adapter._feishu_send_with_retry = AsyncMock(side_effect=_fake_send)

        result = asyncio.run(
            adapter.send(
                chat_id="oc_chat",
                content="```python\nprint('hello')\n```",
            )
        )

        self.assertTrue(result.success)
        self.assertEqual(result.message_id, "om_post_fallback")
        self.assertEqual(calls[0]["msg_type"], "interactive")
        self.assertEqual(calls[1]["msg_type"], "post")

    def test_send_interactive_rejection_falls_back_to_text_for_table(self):
        adapter = self._make_adapter()
        calls = []

        async def _fake_send(**kwargs):
            calls.append(kwargs)
            if kwargs["msg_type"] == "interactive":
                return self._bad_response()
            return self._ok_response("om_text_fallback")

        adapter._feishu_send_with_retry = AsyncMock(side_effect=_fake_send)

        result = asyncio.run(
            adapter.send(
                chat_id="oc_chat",
                content="| A | B |\n|---|---|\n| 1 | 2 |",
            )
        )

        self.assertTrue(result.success)
        self.assertEqual(result.message_id, "om_text_fallback")
        self.assertEqual(calls[0]["msg_type"], "interactive")
        self.assertEqual(calls[1]["msg_type"], "text")


@unittest.skipUnless(_HAS_LARK, "lark_oapi not installed")
class TestFeishuRenderModeEditPatch(unittest.TestCase):
    """Test edit_message() uses Feishu patch API for interactive cards."""

    def test_edit_interactive_card_uses_patch_not_update(self):
        from gateway.platforms.feishu import FeishuAdapter

        adapter = FeishuAdapter(PlatformConfig(enabled=True, extra={"render_mode": "auto"}))
        response = SimpleNamespace(success=lambda: True, data=SimpleNamespace(message_id="om_existing"))
        message_api = SimpleNamespace(
            patch=Mock(return_value=response),
            update=Mock(return_value=response),
        )
        adapter._client = SimpleNamespace(im=SimpleNamespace(v1=SimpleNamespace(message=message_api)))

        async def _direct(func, *args, **kwargs):
            return func(*args, **kwargs)

        with patch("gateway.platforms.feishu.asyncio.to_thread", side_effect=_direct):
            result = asyncio.run(
                adapter.edit_message(
                    chat_id="oc_chat",
                    message_id="om_existing",
                    content="| A | B |\n|---|---|\n| 1 | 2 |",
                )
            )

        self.assertTrue(result.success)
        self.assertEqual(result.message_id, "om_existing")
        message_api.patch.assert_called_once()
        message_api.update.assert_not_called()
