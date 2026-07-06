"""Tests for proactive vision-tool-message downgrade (issue #41072).

When a provider supports vision in user messages but rejects list-type
tool message content (e.g. Xiaomi MiMo's 400 "text is not set"),
``_tool_result_content_for_active_model`` should proactively downgrade
to a text summary instead of waiting for a reactive 400 recovery.

The fix adds ``supports_vision_tool_messages`` to ``ProviderProfile``
and checks it in ``_tool_result_content_for_active_model``.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_agent(provider="openrouter", model="gpt-4o"):
    """Create a minimal AIAgent mock with provider/model attributes."""
    from run_agent import AIAgent
    agent = MagicMock(spec=AIAgent)
    agent.provider = provider
    agent.model = model
    agent._no_list_tool_content_models = set()

    def _real_content_has_image_parts(content):
        if not isinstance(content, list):
            return False
        for part in content:
            if isinstance(part, dict) and part.get("type") in {"image_url", "input_image"}:
                return True
        return False

    agent._content_has_image_parts = _real_content_has_image_parts
    agent._model_supports_vision = lambda: AIAgent._model_supports_vision(agent)
    agent._provider_supports_vision_tool_messages = lambda: AIAgent._provider_supports_vision_tool_messages(agent)
    agent._compact_text_tool_result_for_active_model = lambda _name, result: result
    agent._tool_result_content_for_active_model = (
        lambda name, result: AIAgent._tool_result_content_for_active_model(agent, name, result)
    )
    return agent


def _multimodal_result(text="screenshot", image_url="data:image/png;base64,AAAA"):
    return {
        "_multimodal": True,
        "content": [
            {"type": "text", "text": text},
            {"type": "image_url", "image_url": {"url": image_url}},
        ],
        "text_summary": text,
    }


# ---------------------------------------------------------------------------
# _provider_supports_vision_tool_messages
# ---------------------------------------------------------------------------


class TestProviderSupportsVisionToolMessages:
    def test_xiaomi_returns_false(self):
        agent = _make_agent("xiaomi", "mimo-v2.5")
        assert agent._provider_supports_vision_tool_messages() is False

    def test_xiaomi_alias_mimo_returns_false(self):
        agent = _make_agent("mimo", "mimo-v2.5")
        assert agent._provider_supports_vision_tool_messages() is False

    def test_unknown_provider_defaults_true(self):
        agent = _make_agent("some-unknown-provider", "model-v1")
        assert agent._provider_supports_vision_tool_messages() is True

    def test_openrouter_defaults_true(self):
        agent = _make_agent("openrouter", "gpt-4o")
        assert agent._provider_supports_vision_tool_messages() is True

    def test_anthropic_defaults_true(self):
        agent = _make_agent("anthropic", "claude-sonnet-4")
        assert agent._provider_supports_vision_tool_messages() is True

    def test_empty_provider_defaults_true(self):
        agent = _make_agent("", "")
        assert agent._provider_supports_vision_tool_messages() is True


# ---------------------------------------------------------------------------
# _tool_result_content_for_active_model — proactive downgrade
# ---------------------------------------------------------------------------


class TestToolResultContentProactiveDowngrade:
    def test_xiaomi_downgrades_to_text_summary(self):
        """Xiaomi: vision=True but supports_vision_tool_messages=False → text."""
        agent = _make_agent("xiaomi", "mimo-v2.5")
        result = _multimodal_result(text="screenshot captured")

        with patch.object(agent, "_model_supports_vision", return_value=True):
            content = agent._tool_result_content_for_active_model("browser_screenshot", result)

        assert isinstance(content, str)
        assert "screenshot captured" in content

    def test_xiaomi_non_multimodal_passes_through(self):
        """Non-multimodal results should pass through unchanged."""
        agent = _make_agent("xiaomi", "mimo-v2.5")
        result = "plain text result"

        content = agent._tool_result_content_for_active_model("some_tool", result)

        assert content == "plain text result"

    def test_openrouter_vision_keeps_list_content(self):
        """OpenRouter with vision: list content preserved."""
        agent = _make_agent("openrouter", "gpt-4o")
        result = _multimodal_result()

        with patch.object(agent, "_model_supports_vision", return_value=True):
            content = agent._tool_result_content_for_active_model("browser_screenshot", result)

        assert isinstance(content, list)
        assert any(p.get("type") == "image_url" for p in content if isinstance(p, dict))

    def test_non_vision_model_gets_text_summary(self):
        """Non-vision model: text summary regardless of provider."""
        agent = _make_agent("openrouter", "gpt-3.5-turbo")
        result = _multimodal_result(text="screenshot")

        with patch.object(agent, "_model_supports_vision", return_value=False):
            content = agent._tool_result_content_for_active_model("browser_screenshot", result)

        assert isinstance(content, str)
        assert "screenshot" in content

    def test_xiaomi_computer_use_gets_text_summary(self):
        """Xiaomi + computer_use: text summary (not the error dict)."""
        agent = _make_agent("xiaomi", "mimo-v2.5")
        result = _multimodal_result(text="desktop screenshot")

        with patch.object(agent, "_model_supports_vision", return_value=True):
            content = agent._tool_result_content_for_active_model("computer_use", result)

        # Should be a text summary, not the error dict for non-vision models
        assert isinstance(content, str)
        assert "desktop screenshot" in content

    def test_xiaomi_no_image_parts_returns_content(self):
        """Xiaomi tool result with no image parts: returns content list."""
        agent = _make_agent("xiaomi", "mimo-v2.5")
        result = {
            "_multimodal": True,
            "content": [{"type": "text", "text": "just text"}],
        }

        with patch.object(agent, "_model_supports_vision", return_value=True):
            content = agent._tool_result_content_for_active_model("some_tool", result)

        # No image parts → returns content as-is
        assert isinstance(content, list)

    def test_reactive_cache_still_works(self):
        """In-session cache (_no_list_tool_content_models) still triggers."""
        agent = _make_agent("openrouter", "some-model")
        agent._no_list_tool_content_models = {("openrouter", "some-model")}
        result = _multimodal_result(text="cached downgrade")

        with patch.object(agent, "_model_supports_vision", return_value=True):
            content = agent._tool_result_content_for_active_model("browser_screenshot", result)

        assert isinstance(content, str)
        assert "cached downgrade" in content


# ---------------------------------------------------------------------------
# ProviderProfile.supports_vision_tool_messages field
# ---------------------------------------------------------------------------


class TestProviderProfileField:
    def test_default_is_true(self):
        from providers.base import ProviderProfile
        # ProviderProfile uses __init__ with defaults; check via a minimal instance
        # by reading the class-level default from a dataclass-like field
        import dataclasses
        if dataclasses.is_dataclass(ProviderProfile):
            fields = {f.name: f.default for f in dataclasses.fields(ProviderProfile)}
            assert fields.get("supports_vision_tool_messages", True) is True
        else:
            # Class-level attribute default
            assert getattr(ProviderProfile, "supports_vision_tool_messages", True) is True

    def test_xiaomi_profile_has_false(self):
        from providers import get_provider_profile
        profile = get_provider_profile("xiaomi")
        assert profile is not None
        assert profile.supports_vision_tool_messages is False

    def test_xiaomi_alias_mimo_has_false(self):
        from providers import get_provider_profile
        profile = get_provider_profile("mimo")
        assert profile is not None
        assert profile.supports_vision_tool_messages is False

    def test_anthropic_profile_defaults_true(self):
        from providers import get_provider_profile
        profile = get_provider_profile("anthropic")
        if profile is not None:
            assert profile.supports_vision_tool_messages is True
