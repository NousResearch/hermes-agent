"""Tests for the vision-aware image preprocessing in run_agent.py.

Covers:

* ``_prepare_anthropic_messages_for_api`` — passes image parts through
  unchanged when the active model reports ``supports_vision=True`` (the
  adapter handles them natively), and falls back to text-description
  replacement when the model lacks vision.

* ``_prepare_messages_for_non_vision_model`` — the mirror method for the
  chat.completions / codex_responses paths. Same contract.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from run_agent import AIAgent


def _make_agent() -> AIAgent:
    """Build a bare-bones AIAgent instance without running __init__.

    Avoids the heavy provider/credential setup for these pure-method tests.
    """
    agent = object.__new__(AIAgent)
    agent.provider = "anthropic"
    agent.model = "claude-sonnet-4"
    agent._anthropic_image_fallback_cache = {}
    return agent


IMG_PARTS_USER_MSG = {
    "role": "user",
    "content": [
        {"type": "text", "text": "What's in this image?"},
        {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}},
    ],
}

PLAIN_USER_MSG = {"role": "user", "content": "hello, no images here"}


# ─── _prepare_anthropic_messages_for_api ─────────────────────────────────────


class TestPrepareAnthropicMessages:
    def test_no_images_passes_through(self):
        agent = _make_agent()
        msgs = [PLAIN_USER_MSG]
        out = agent._prepare_anthropic_messages_for_api(msgs)
        assert out is msgs  # unchanged reference

    def test_vision_capable_passes_images_through(self):
        """The Anthropic adapter handles image_url/input_image natively."""
        agent = _make_agent()
        with patch.object(agent, "_model_supports_vision", return_value=True):
            out = agent._prepare_anthropic_messages_for_api([IMG_PARTS_USER_MSG])
        # Passes through unchanged — image_url parts still present.
        assert out[0]["content"][1]["type"] == "image_url"

    def test_non_vision_replaces_images_with_text(self):
        agent = _make_agent()
        with patch.object(agent, "_model_supports_vision", return_value=False), \
             patch.object(
                 agent,
                 "_describe_image_for_anthropic_fallback",
                 return_value="[Image description: a cat]",
             ):
            out = agent._prepare_anthropic_messages_for_api([IMG_PARTS_USER_MSG])
        # Content collapsed to a string containing the description + user text.
        content = out[0]["content"]
        assert isinstance(content, str)
        assert "[Image description: a cat]" in content
        assert "What's in this image?" in content
        # No more image parts.
        assert "image_url" not in content


# ─── _prepare_messages_for_non_vision_model ──────────────────────────────────


class TestPrepareMessagesForNonVision:
    def test_no_images_passes_through(self):
        agent = _make_agent()
        msgs = [PLAIN_USER_MSG]
        out = agent._prepare_messages_for_non_vision_model(msgs)
        assert out is msgs

    def test_vision_capable_passes_through(self):
        """For vision-capable models on chat.completions path, provider handles pixels."""
        agent = _make_agent()
        agent.provider = "openrouter"
        agent.model = "anthropic/claude-sonnet-4"
        with patch.object(agent, "_model_supports_vision", return_value=True):
            out = agent._prepare_messages_for_non_vision_model([IMG_PARTS_USER_MSG])
        assert out[0]["content"][1]["type"] == "image_url"

    def test_non_vision_strips_images(self):
        agent = _make_agent()
        agent.provider = "openrouter"
        agent.model = "qwen/qwen3-235b-a22b"
        with patch.object(agent, "_model_supports_vision", return_value=False), \
             patch.object(
                 agent,
                 "_describe_image_for_anthropic_fallback",
                 return_value="[Image description: a dog]",
             ):
            out = agent._prepare_messages_for_non_vision_model([IMG_PARTS_USER_MSG])
        content = out[0]["content"]
        assert isinstance(content, str)
        assert "[Image description: a dog]" in content
        assert "image_url" not in content

    def test_multiple_messages_with_mixed_content(self):
        agent = _make_agent()
        agent.model = "qwen/qwen3-235b"
        msgs = [
            {"role": "user", "content": "first turn"},
            {"role": "assistant", "content": "ack"},
            IMG_PARTS_USER_MSG,
        ]
        with patch.object(agent, "_model_supports_vision", return_value=False), \
             patch.object(
                 agent,
                 "_describe_image_for_anthropic_fallback",
                 return_value="[Image: thing]",
             ):
            out = agent._prepare_messages_for_non_vision_model(msgs)
        # First two messages unchanged (no images), third stripped.
        assert out[0]["content"] == "first turn"
        assert out[1]["content"] == "ack"
        assert isinstance(out[2]["content"], str)
        assert "[Image: thing]" in out[2]["content"]


# ─── _model_supports_vision ──────────────────────────────────────────────────


class TestModelSupportsVision:
    def test_missing_provider_or_model_returns_false(self):
        agent = _make_agent()
        agent.provider = ""
        agent.model = "claude-sonnet-4"
        assert agent._model_supports_vision() is False
        agent.provider = "anthropic"
        agent.model = ""
        assert agent._model_supports_vision() is False

    def test_uses_get_model_capabilities(self):
        agent = _make_agent()
        fake_caps = MagicMock()
        fake_caps.supports_vision = True
        with patch("agent.models_dev.get_model_capabilities", return_value=fake_caps):
            assert agent._model_supports_vision() is True
        fake_caps.supports_vision = False
        with patch("agent.models_dev.get_model_capabilities", return_value=fake_caps):
            assert agent._model_supports_vision() is False

    def test_none_caps_returns_false(self):
        agent = _make_agent()
        with patch("agent.models_dev.get_model_capabilities", return_value=None):
            assert agent._model_supports_vision() is False

    def test_exception_returns_false(self):
        agent = _make_agent()
        with patch("agent.models_dev.get_model_capabilities", side_effect=RuntimeError("boom")):
            assert agent._model_supports_vision() is False


# ─── _build_api_kwargs profile-path call site (regression for #23733) ────────


def _make_agent_for_build_kwargs() -> AIAgent:
    """Bare-bones AIAgent with just enough state to traverse the
    chat_completions profile branch in ``_build_api_kwargs``.

    Mirrors the minimal-setup pattern of ``_make_agent`` above; only the
    attributes actually read on the path are populated.
    """
    agent = object.__new__(AIAgent)
    agent.provider = "opencode-go"  # has a registered ProviderProfile
    agent.model = "deepseek-v4-pro"
    agent.api_mode = "chat_completions"
    agent.base_url = "https://opencode.ai/zen/go/v1"
    agent._base_url_lower = agent.base_url.lower()
    agent._base_url_hostname = "opencode.ai"
    agent.session_id = "test-session"
    agent.tools = []
    agent.max_tokens = None
    agent.reasoning_config = None
    agent.request_overrides = None
    agent._ollama_num_ctx = None
    agent._ephemeral_max_output_tokens = None
    agent.providers_allowed = None
    agent.providers_ignored = None
    agent.providers_order = None
    agent.provider_sort = None
    agent.provider_require_parameters = False
    agent.provider_data_collection = None
    agent.openrouter_min_coding_score = None
    agent._anthropic_image_fallback_cache = {}
    return agent


class TestBuildApiKwargsProfileBranch:
    """Regression coverage for issue #23733.

    The profile branch of ``_build_api_kwargs`` (registered providers like
    ``opencode-go``, ``deepseek``, ``kimi``, ...) must run the same
    non-vision image fallback the legacy branch does. Without it, image
    parts pass through to text-only models and the provider returns
    HTTP 400 ``unknown variant 'image_url'``.
    """

    def test_profile_branch_strips_images_for_non_vision_models(self):
        agent = _make_agent_for_build_kwargs()

        fake_transport = MagicMock()
        fake_transport.build_kwargs = MagicMock(side_effect=lambda **kw: kw)

        msgs = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "describe"},
                    {
                        "type": "image_url",
                        "image_url": {"url": "data:image/png;base64,AAAA"},
                    },
                ],
            }
        ]

        with patch.object(agent, "_get_transport", return_value=fake_transport), \
             patch.object(agent, "_model_supports_vision", return_value=False), \
             patch.object(agent, "_is_qwen_portal", return_value=False), \
             patch.object(agent, "_is_openrouter_url", return_value=False), \
             patch.object(agent, "_max_tokens_param", lambda v: {"max_tokens": v}), \
             patch.object(agent, "_resolved_api_call_timeout", return_value=60.0), \
             patch.object(agent, "_supports_reasoning_extra_body", return_value=False), \
             patch.object(
                 agent,
                 "_describe_image_for_anthropic_fallback",
                 return_value="[Image: thing]",
             ):
            agent._build_api_kwargs(msgs)

        # The transport must have been called via the profile branch.
        fake_transport.build_kwargs.assert_called_once()
        kwargs = fake_transport.build_kwargs.call_args.kwargs
        assert kwargs.get("provider_profile") is not None, \
            "test must exercise the profile branch — got legacy path instead"

        # The contract: image_url parts must be replaced with text before
        # they reach the transport, just like the legacy branch already does.
        sent_messages = kwargs["messages"]
        assert len(sent_messages) == 1
        content = sent_messages[0]["content"]
        assert isinstance(content, str), (
            f"expected stripped content (str), got {type(content).__name__}: "
            f"profile branch is bypassing _prepare_messages_for_non_vision_model"
        )
        assert "image_url" not in content
        assert "[Image: thing]" in content

    def test_profile_branch_passes_through_when_model_supports_vision(self):
        agent = _make_agent_for_build_kwargs()

        fake_transport = MagicMock()
        fake_transport.build_kwargs = MagicMock(side_effect=lambda **kw: kw)

        msgs = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "describe"},
                    {
                        "type": "image_url",
                        "image_url": {"url": "data:image/png;base64,AAAA"},
                    },
                ],
            }
        ]

        with patch.object(agent, "_get_transport", return_value=fake_transport), \
             patch.object(agent, "_model_supports_vision", return_value=True), \
             patch.object(agent, "_is_qwen_portal", return_value=False), \
             patch.object(agent, "_is_openrouter_url", return_value=False), \
             patch.object(agent, "_max_tokens_param", lambda v: {"max_tokens": v}), \
             patch.object(agent, "_resolved_api_call_timeout", return_value=60.0), \
             patch.object(agent, "_supports_reasoning_extra_body", return_value=False):
            agent._build_api_kwargs(msgs)

        kwargs = fake_transport.build_kwargs.call_args.kwargs
        # Vision-capable: pixels must reach the transport unchanged.
        sent_messages = kwargs["messages"]
        assert sent_messages[0]["content"][1]["type"] == "image_url"
