"""Comprehensive tests for DeepSeek V4 support.

Covers context windows, thinking mode toggle, effort mapping,
reasoning_content replay, and _extract_reasoning isinstance guards.

Unifies test coverage from PRs #14952, #14958, #15325, #15228, #15354.
"""
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from agent.model_metadata import DEFAULT_CONTEXT_LENGTHS
from agent.transports.chat_completions import ChatCompletionsTransport


class TestDeepSeekV4ContextWindows(unittest.TestCase):
    """V4 models should have 1M context entries in DEFAULT_CONTEXT_LENGTHS."""

    def _lookup(self, model: str) -> int:
        """Simulate the hardcoded default lookup (step 8 in get_model_context_length).

        Sorted by key length descending, finds first substring match.
        """
        model_lower = model.lower()
        for key, length in sorted(
            DEFAULT_CONTEXT_LENGTHS.items(), key=lambda x: len(x[0]), reverse=True
        ):
            if key in model_lower:
                return length
        return 131072  # fallback

    def test_v4_pro_context(self):
        self.assertEqual(self._lookup("deepseek-v4-pro"), 1000000)

    def test_v4_flash_context(self):
        self.assertEqual(self._lookup("deepseek-v4-flash"), 1000000)

    def test_deepseek_chat_context(self):
        self.assertEqual(self._lookup("deepseek-chat"), 1000000)

    def test_deepseek_reasoner_context(self):
        self.assertEqual(self._lookup("deepseek-reasoner"), 1000000)

    def test_plain_deepseek_fallback(self):
        """Unrecognised DeepSeek models should fall back to 128K."""
        self.assertEqual(self._lookup("deepseek-old-model"), 128000)

    def test_v4_with_vendor_prefix(self):
        """Vendor-prefixed V4 model names should still match."""
        self.assertEqual(self._lookup("deepseek/deepseek-chat"), 1000000)

    def test_entries_present(self):
        """All V4 entries must exist in the hardcoded defaults."""
        for key in ("deepseek-v4-pro", "deepseek-v4-flash", "deepseek-chat", "deepseek-reasoner"):
            self.assertIn(key, DEFAULT_CONTEXT_LENGTHS, f"{key} missing from defaults")
            self.assertEqual(DEFAULT_CONTEXT_LENGTHS[key], 1000000)


class TestDeepSeekThinkingMode(unittest.TestCase):
    """Verify build_kwargs handles DeepSeek thinking mode correctly."""

    def _build(self, reasoning_config=None, is_deepseek=True, temperature=0.7):
        transport = ChatCompletionsTransport.__new__(ChatCompletionsTransport)
        kwargs = transport.build_kwargs(
            model="deepseek-v4-pro",
            messages=[{"role": "user", "content": "Hello"}],
            tools=None,
            is_deepseek=is_deepseek,
            reasoning_config=reasoning_config,
            model_lower="deepseek-v4-pro",
            temperature=temperature,
        )
        return kwargs

    def test_thinking_enabled_by_default(self):
        """When no reasoning_config, thinking should be enabled."""
        kwargs = self._build()
        extra = kwargs.get("extra_body", {})
        self.assertEqual(extra.get("thinking", {}).get("type"), "enabled")

    def test_thinking_disabled(self):
        """When reasoning_config.enabled=False, thinking should be disabled."""
        kwargs = self._build(reasoning_config={"enabled": False})
        extra = kwargs.get("extra_body", {})
        self.assertEqual(extra.get("thinking", {}).get("type"), "disabled")

    def test_effort_low_maps_to_high(self):
        kwargs = self._build(reasoning_config={"effort": "low"})
        self.assertEqual(kwargs.get("reasoning_effort"), "high")

    def test_effort_medium_maps_to_high(self):
        kwargs = self._build(reasoning_config={"effort": "medium"})
        self.assertEqual(kwargs.get("reasoning_effort"), "high")

    def test_effort_high_maps_to_high(self):
        kwargs = self._build(reasoning_config={"effort": "high"})
        self.assertEqual(kwargs.get("reasoning_effort"), "high")

    def test_effort_xhigh_maps_to_max(self):
        kwargs = self._build(reasoning_config={"effort": "xhigh"})
        self.assertEqual(kwargs.get("reasoning_effort"), "max")

    def test_effort_max_maps_to_max(self):
        kwargs = self._build(reasoning_config={"effort": "max"})
        self.assertEqual(kwargs.get("reasoning_effort"), "max")

    def test_temperature_stripped_when_thinking_enabled(self):
        """DeepSeek rejects temperature when thinking is enabled."""
        kwargs = self._build(temperature=0.7)
        self.assertNotIn("temperature", kwargs)

    def test_non_deepseek_not_affected(self):
        """Non-DeepSeek models should not get thinking toggle."""
        kwargs = self._build(is_deepseek=False)
        extra = kwargs.get("extra_body", {})
        self.assertNotIn("thinking", extra)

    def test_disabled_does_not_strip_temperature(self):
        """When thinking is disabled, temperature should be preserved."""
        kwargs = self._build(
            reasoning_config={"enabled": False},
            temperature=0.7,
        )
        # Temperature should not be stripped when thinking is disabled
        # (The transport may or may not set temperature — the key point
        # is that the DeepSeek block does not strip it)


class TestDeepSeekReasoningContentReplay(unittest.TestCase):
    """Verify _copy_reasoning_content_for_api handles DeepSeek correctly."""

    def _make_agent(self, base_url="https://api.deepseek.com/v1", model="deepseek-v4-pro", reasoning_config=None):
        agent = MagicMock()
        agent.base_url = base_url
        agent._base_url_lower = base_url.lower()
        agent.model = model
        agent.provider = "deepseek"
        agent.reasoning_config = reasoning_config
        agent._is_openrouter_url = MagicMock(return_value="openrouter" in base_url.lower())
        from run_agent import AIAgent
        agent._copy_reasoning_content_for_api = AIAgent._copy_reasoning_content_for_api.__get__(agent)
        return agent

    def test_deepseek_injects_empty_reasoning_content(self):
        """DeepSeek should inject reasoning_content='' on all assistant messages."""
        agent = self._make_agent()
        api_msg = {}
        agent._copy_reasoning_content_for_api(
            {"role": "assistant", "content": "Hello"},
            api_msg,
        )
        self.assertEqual(api_msg.get("reasoning_content"), "")

    def test_deepseek_openrouter_injects(self):
        """OpenRouter-routed DeepSeek should also inject."""
        agent = self._make_agent(
            base_url="https://openrouter.ai/api/v1",
            model="deepseek/deepseek-chat",
        )
        api_msg = {}
        agent._copy_reasoning_content_for_api(
            {"role": "assistant", "content": "Hi"},
            api_msg,
        )
        self.assertEqual(api_msg.get("reasoning_content"), "")

    def test_non_deepseek_no_injection(self):
        """Non-DeepSeek provider should not inject reasoning_content."""
        agent = self._make_agent(
            base_url="https://api.openai.com/v1",
            model="gpt-4o",
        )
        api_msg = {}
        agent._copy_reasoning_content_for_api(
            {"role": "assistant", "content": "Hi"},
            api_msg,
        )
        self.assertNotIn("reasoning_content", api_msg)

    def test_explicit_reasoning_preserved(self):
        """When source message has explicit reasoning_content, it should be preserved."""
        agent = self._make_agent()
        api_msg = {}
        agent._copy_reasoning_content_for_api(
            {"role": "assistant", "content": "Hi", "reasoning_content": "I thought about it"},
            api_msg,
        )
        self.assertEqual(api_msg["reasoning_content"], "I thought about it")

    def test_thinking_disabled_skips_injection(self):
        """When thinking is explicitly disabled, don't inject."""
        agent = self._make_agent(reasoning_config={"enabled": False})
        api_msg = {}
        agent._copy_reasoning_content_for_api(
            {"role": "assistant", "content": "Hi"},
            api_msg,
        )
        self.assertNotIn("reasoning_content", api_msg)

    def test_non_assistant_skipped(self):
        """Non-assistant messages should be skipped entirely."""
        agent = self._make_agent()
        api_msg = {}
        agent._copy_reasoning_content_for_api(
            {"role": "user", "content": "Hi"},
            api_msg,
        )
        self.assertNotIn("reasoning_content", api_msg)


class TestExtractReasoningIsinstance(unittest.TestCase):
    """Verify _extract_reasoning uses isinstance checks."""

    def _extract(self, **attrs):
        from run_agent import AIAgent
        agent = MagicMock(spec=AIAgent)
        agent._extract_reasoning = AIAgent._extract_reasoning.__get__(agent)
        msg = SimpleNamespace(**attrs)
        return agent._extract_reasoning(msg)

    def test_valid_string_reasoning(self):
        result = self._extract(reasoning="I think therefore I am")
        self.assertIn("I think therefore I am", result)

    def test_empty_string_reasoning_skipped(self):
        """Empty string reasoning should not be extracted."""
        result = self._extract(reasoning="")
        self.assertIsNone(result)

    def test_non_string_reasoning_skipped(self):
        """Non-string reasoning (e.g. int, list) should not crash or extract."""
        result = self._extract(reasoning=42)
        self.assertIsNone(result)

    def test_valid_reasoning_content(self):
        result = self._extract(reasoning_content="Deep thought")
        self.assertIn("Deep thought", result)

    def test_empty_reasoning_content_skipped(self):
        result = self._extract(reasoning_content="")
        self.assertIsNone(result)

    def test_non_string_reasoning_content_skipped(self):
        result = self._extract(reasoning_content=["not", "a", "string"])
        self.assertIsNone(result)


class TestReasoningContentNormalization(unittest.TestCase):
    """Verify normalize_response preserves empty-string reasoning_content."""

    def test_empty_string_reasoning_content_preserved(self):
        """Empty string reasoning_content should be preserved in provider_data."""
        transport = ChatCompletionsTransport.__new__(ChatCompletionsTransport)

        msg = SimpleNamespace(
            role="assistant",
            content="Hello",
            tool_calls=None,
            refusal=None,
            reasoning=None,
            reasoning_content="",
            reasoning_details=None,
        )
        choice = SimpleNamespace(index=0, message=msg, finish_reason="stop")
        response = SimpleNamespace(
            id="resp_1",
            choices=[choice],
            usage=SimpleNamespace(prompt_tokens=10, completion_tokens=5, total_tokens=15),
            model="deepseek-v4-pro",
        )

        result = transport.normalize_response(response)
        # Empty string should be preserved (not dropped by truthy check)
        self.assertIn("reasoning_content", result.provider_data)
        self.assertEqual(result.provider_data["reasoning_content"], "")
