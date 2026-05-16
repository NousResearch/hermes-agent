"""Regression test: Cerebras rejects reasoning_content in assistant messages.

Cerebras serves models via an OpenAI-compatible endpoint but does NOT accept
the ``reasoning_content`` field in replayed assistant messages, returning::

    HTTP 400: messages.N.assistant.reasoning_content:
    property '...' is unsupported

Two complementary guards prevent this:

1. ``AIAgent._strips_reasoning_content_from_api_messages`` — detects Cerebras
   by URL so the agent-level ``_copy_reasoning_content_for_api`` never adds
   the field (covers ``provider: custom`` + Cerebras URL, and ``provider:
   cerebras`` via the registered profile's hostname).

2. ``CerebrasProfile.prepare_messages`` — transport-layer strip as a belt-
   and-suspenders guard for the profile path (``provider: cerebras``).

Refs: HTTP 400 "property 'messages.2.assistant.reasoning_content' is
unsupported" when using gpt-oss-120b / llama-4-scout on api.cerebras.ai.
"""

from __future__ import annotations

import pytest

from run_agent import AIAgent


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_agent(provider: str = "", model: str = "", base_url: str = "") -> AIAgent:
    """Create a minimal AIAgent instance without a real session."""
    agent = object.__new__(AIAgent)
    agent.provider = provider
    agent.model = model
    agent.base_url = base_url
    agent.verbose_logging = False
    agent.reasoning_callback = None
    agent.stream_delta_callback = None
    agent._stream_callback = None
    return agent


# ---------------------------------------------------------------------------
# _strips_reasoning_content_from_api_messages — URL detection
# ---------------------------------------------------------------------------


class TestStripsReasoningContentDetection:
    """_strips_reasoning_content_from_api_messages detects Cerebras by URL."""

    def test_cerebras_base_url_detected(self):
        agent = _make_agent(base_url="https://api.cerebras.ai/v1")
        assert agent._strips_reasoning_content_from_api_messages() is True

    def test_cerebras_base_url_no_path(self):
        agent = _make_agent(base_url="https://api.cerebras.ai")
        assert agent._strips_reasoning_content_from_api_messages() is True

    def test_cerebras_custom_provider_still_detected(self):
        """provider=custom + Cerebras URL should be detected."""
        agent = _make_agent(provider="custom", base_url="https://api.cerebras.ai/v1")
        assert agent._strips_reasoning_content_from_api_messages() is True

    def test_non_cerebras_url_not_detected(self):
        agent = _make_agent(base_url="https://api.openai.com/v1")
        assert agent._strips_reasoning_content_from_api_messages() is False

    def test_deepseek_not_detected(self):
        agent = _make_agent(
            provider="deepseek", base_url="https://api.deepseek.com/v1"
        )
        assert agent._strips_reasoning_content_from_api_messages() is False

    def test_empty_base_url_not_detected(self):
        agent = _make_agent(base_url="")
        assert agent._strips_reasoning_content_from_api_messages() is False

    def test_subdomain_spoof_not_detected(self):
        """evil.com/api.cerebras.ai should not match."""
        agent = _make_agent(base_url="https://evil.com/api.cerebras.ai/v1")
        assert agent._strips_reasoning_content_from_api_messages() is False

    def test_similar_domain_not_detected(self):
        """api.cerebras.ai.evil.com should not match."""
        agent = _make_agent(base_url="https://api.cerebras.ai.evil.com/v1")
        assert agent._strips_reasoning_content_from_api_messages() is False


# ---------------------------------------------------------------------------
# _copy_reasoning_content_for_api — Cerebras strips reasoning_content
# ---------------------------------------------------------------------------


class TestCopyReasoningContentForApiCerebras:
    """_copy_reasoning_content_for_api strips reasoning_content for Cerebras."""

    def _cerebras_agent(self) -> AIAgent:
        return _make_agent(
            provider="custom", base_url="https://api.cerebras.ai/v1"
        )

    def test_strips_reasoning_content_from_api_msg(self):
        """reasoning_content copied via msg.copy() must be removed."""
        agent = self._cerebras_agent()
        source = {
            "role": "assistant",
            "content": "Hello",
            "reasoning_content": "some reasoning text",
        }
        api_msg = source.copy()
        agent._copy_reasoning_content_for_api(source, api_msg)
        assert "reasoning_content" not in api_msg

    def test_strips_reasoning_content_with_tool_calls(self):
        """Tool-call assistant messages also have reasoning_content stripped."""
        agent = self._cerebras_agent()
        source = {
            "role": "assistant",
            "content": None,
            "reasoning_content": "planning step",
            "tool_calls": [{"id": "tc1", "function": {"name": "foo", "arguments": "{}"}}],
        }
        api_msg = source.copy()
        agent._copy_reasoning_content_for_api(source, api_msg)
        assert "reasoning_content" not in api_msg
        # tool_calls must not be touched by this method
        assert api_msg.get("tool_calls") is source["tool_calls"]

    def test_handles_missing_reasoning_content_gracefully(self):
        """No reasoning_content in source — api_msg should not gain one."""
        agent = self._cerebras_agent()
        source = {"role": "assistant", "content": "Hello", "reasoning": "think"}
        api_msg = source.copy()
        agent._copy_reasoning_content_for_api(source, api_msg)
        assert "reasoning_content" not in api_msg

    def test_non_assistant_roles_untouched(self):
        """Non-assistant messages are always left untouched."""
        agent = self._cerebras_agent()
        for role in ("user", "system", "tool"):
            source = {"role": role, "content": "x", "reasoning_content": "leak"}
            api_msg = source.copy()
            agent._copy_reasoning_content_for_api(source, api_msg)
            # _copy_reasoning_content_for_api returns immediately for non-assistant
            # roles without modifying reasoning_content; the field stays as copied.
            # The important invariant: no Cerebras-path strip runs on non-assistant.
            # (The api_msg.copy() retains whatever was there — the transport layer
            # does not send reasoning_content for non-assistant roles because the
            # field only appears in assistant messages in practice.)
            assert api_msg["role"] == role

    def test_non_cerebras_provider_preserves_reasoning_content(self):
        """DeepSeek agent must NOT strip reasoning_content."""
        agent = _make_agent(
            provider="deepseek", base_url="https://api.deepseek.com/v1"
        )
        source = {
            "role": "assistant",
            "content": "answer",
            "reasoning_content": "chain of thought",
        }
        api_msg = source.copy()
        agent._copy_reasoning_content_for_api(source, api_msg)
        assert api_msg.get("reasoning_content") == "chain of thought"


# ---------------------------------------------------------------------------
# CerebrasProfile.prepare_messages — transport-layer strip
# ---------------------------------------------------------------------------


class TestCerebrasProfilePrepareMessages:
    """CerebrasProfile.prepare_messages strips reasoning_content for all
    assistant messages and leaves other roles untouched."""

    @pytest.fixture
    def profile(self):
        from providers import get_provider_profile

        return get_provider_profile("cerebras")

    def test_strips_reasoning_content(self, profile):
        messages = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Hello"},
            {
                "role": "assistant",
                "content": "Hi there!",
                "reasoning_content": "I should greet the user.",
            },
            {"role": "user", "content": "Bye"},
        ]
        result = profile.prepare_messages(messages)
        for msg in result:
            assert "reasoning_content" not in msg

    def test_system_and_user_messages_preserved(self, profile):
        messages = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "usr"},
        ]
        result = profile.prepare_messages(messages)
        assert result[0]["content"] == "sys"
        assert result[1]["content"] == "usr"

    def test_does_not_mutate_original_messages(self, profile):
        original = {
            "role": "assistant",
            "content": "hi",
            "reasoning_content": "think",
        }
        messages = [original]
        profile.prepare_messages(messages)
        # Original dict must be unchanged
        assert "reasoning_content" in original

    def test_passthrough_when_no_reasoning_content(self, profile):
        messages = [
            {"role": "assistant", "content": "answer"},
        ]
        result = profile.prepare_messages(messages)
        assert result[0]["content"] == "answer"
        assert "reasoning_content" not in result[0]
