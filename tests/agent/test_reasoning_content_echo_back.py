"""Tests for reasoning_content echo-back padding on auxiliary paths.

Covers the standalone ``needs_thinking_reasoning_pad`` /
``ensure_reasoning_content_on_messages`` helpers and their integration into
``_build_call_kwargs`` so that:

* require-side providers (DeepSeek V4 thinking, Kimi/Moonshot thinking, Xiaomi
  MiMo) get ``reasoning_content`` padded on replayed assistant messages;
* strict/indifferent providers do NOT receive the field;
* the caller's shared ``messages`` list is never mutated in place — a
  require→strict cross-provider fallback rebuilds its own (unpadded) request
  messages from the untouched original, so a DeepSeek pad never leaks into a
  strict OpenAI/Mistral fallback request (the exact #45655 regression).
"""

import copy
from unittest.mock import MagicMock

import pytest

from agent.agent_runtime_helpers import (
    ensure_reasoning_content_on_messages,
    needs_thinking_reasoning_pad,
)


# ---------------------------------------------------------------------------
# needs_thinking_reasoning_pad — provider-direction classification
# ---------------------------------------------------------------------------


class TestNeedsThinkingReasoningPad:
    """Provider-direction classification matches the rule table."""

    @pytest.mark.parametrize(
        "provider, model, base_url",
        [
            ("deepseek", "deepseek-chat", "https://api.deepseek.com/v1"),
            ("deepseek", "deepseek-reasoner", "https://api.deepseek.com"),
            (None, None, "https://api.deepseek.com/v1"),
            ("kimi", None, "https://api.moonshot.ai/v1"),
            (None, "kimi-k2", "https://api.kimi.com/coding/v1"),
            ("moonshot", "moonshot-v1-auto", "https://api.moonshot.cn/v1"),
            ("mimo", None, "https://platform.xiaomimimo.com/v1"),
        ],
    )
    def test_require_side_providers_detected(self, provider, model, base_url):
        assert needs_thinking_reasoning_pad(provider, model, base_url) is True

    @pytest.mark.parametrize(
        "provider, model, base_url",
        [
            ("openai", "gpt-4o", "https://api.openai.com/v1"),
            ("anthropic", "claude-sonnet-4-6", "https://api.anthropic.com"),
            ("mistral", "mistral-large-latest", "https://api.mistral.ai/v1"),
            ("groq", "llama-3.1-70b", "https://api.groq.com/openai/v1"),
            ("cerebras", None, "https://api.cerebras.ai/v1"),
            (None, None, None),
            ("", "", ""),
        ],
    )
    def test_strict_indifferent_providers_not_detected(self, provider, model, base_url):
        assert needs_thinking_reasoning_pad(provider, model, base_url) is False


# ---------------------------------------------------------------------------
# ensure_reasoning_content_on_messages — no-mutation contract
# ---------------------------------------------------------------------------


def _sample_history():
    """A representative message list with an assistant tool-call turn."""
    return [
        {"role": "system", "content": "You are helpful."},
        {"role": "user", "content": "Run the tool."},
        {
            "role": "assistant",
            "content": "Okay.",
            "tool_calls": [
                {"id": "call_1", "type": "function",
                 "function": {"name": "lookup", "arguments": "{}"}}
            ],
        },
        {"role": "tool", "tool_call_id": "call_1", "content": "result"},
    ]


class TestEnsureReasoningContentNoMutation:
    """The helper must NEVER mutate the caller's messages in place."""

    def test_require_side_returns_new_list_original_untouched(self):
        msgs = _sample_history()
        original = copy.deepcopy(msgs)

        padded = ensure_reasoning_content_on_messages(
            msgs, "deepseek", "deepseek-chat", "https://api.deepseek.com/v1"
        )

        # A new list object is returned.
        assert padded is not msgs
        # The original list is byte-for-byte unchanged.
        assert msgs == original
        # The assistant turn in the *original* has no reasoning_content key.
        assert "reasoning_content" not in msgs[2]

    def test_require_side_does_not_share_assistant_dict(self):
        msgs = _sample_history()

        padded = ensure_reasoning_content_on_messages(
            msgs, "deepseek", "deepseek-chat", "https://api.deepseek.com/v1"
        )

        # The padded assistant dict is a NEW dict, not the original.
        assert padded[2] is not msgs[2]
        assert padded[2]["reasoning_content"] == " "
        # Original dict still has no reasoning_content.
        assert "reasoning_content" not in msgs[2]

    def test_non_assistant_messages_shared_not_copied(self):
        """Non-assistant messages are shared by reference (no needless copy)."""
        msgs = _sample_history()

        padded = ensure_reasoning_content_on_messages(
            msgs, "deepseek", "deepseek-chat", "https://api.deepseek.com/v1"
        )

        assert padded[0] is msgs[0]  # system
        assert padded[1] is msgs[1]  # user
        assert padded[3] is msgs[3]  # tool

    def test_strict_side_returns_new_list_no_pad(self):
        msgs = _sample_history()
        original = copy.deepcopy(msgs)

        result = ensure_reasoning_content_on_messages(
            msgs, "openai", "gpt-4o", "https://api.openai.com/v1"
        )

        # Still a new list (callers can always assign the return value).
        assert result is not msgs
        # No pad on any message.
        assert "reasoning_content" not in result[2]
        # Original untouched.
        assert msgs == original

    def test_existing_reasoning_content_preserved_not_overwritten(self):
        msgs = _sample_history()
        msgs[2]["reasoning_content"] = "genuine reasoning text"

        padded = ensure_reasoning_content_on_messages(
            msgs, "deepseek", "deepseek-chat", "https://api.deepseek.com/v1"
        )

        assert padded[2]["reasoning_content"] == "genuine reasoning text"
        # Original not mutated (the dict was shared since it already had rc).
        assert msgs[2]["reasoning_content"] == "genuine reasoning text"

    def test_empty_reasoning_content_upgraded_to_space_on_new_copy(self):
        """DeepSeek V4 Pro rejects empty-string reasoning_content (refs #17341)."""
        msgs = _sample_history()
        msgs[2]["reasoning_content"] = ""

        padded = ensure_reasoning_content_on_messages(
            msgs, "deepseek", "deepseek-chat", "https://api.deepseek.com/v1"
        )

        # The empty string is upgraded to a single space on the padded copy.
        assert padded[2]["reasoning_content"] == " "
        # The original dict still has the empty string (not mutated).
        assert msgs[2]["reasoning_content"] == ""


# ---------------------------------------------------------------------------
# _build_call_kwargs integration — pad applied to request-local copy
# ---------------------------------------------------------------------------


class TestBuildCallKwargsPadsRequestLocal:
    """_build_call_kwargs pads kwargs['messages'] without touching the input."""

    def test_deepseek_provider_gets_pad_on_kwargs_messages(self):
        from agent.auxiliary_client import _build_call_kwargs

        messages = _sample_history()
        kwargs = _build_call_kwargs(
            "deepseek", "deepseek-chat", messages,
            base_url="https://api.deepseek.com/v1",
        )

        # kwargs['messages'] has the pad.
        assert kwargs["messages"][2]["reasoning_content"] == " "
        # The caller's list is untouched.
        assert "reasoning_content" not in messages[2]
        # kwargs['messages'] is a new list, not the input.
        assert kwargs["messages"] is not messages

    def test_strict_provider_no_pad_on_kwargs_messages(self):
        from agent.auxiliary_client import _build_call_kwargs

        messages = _sample_history()
        kwargs = _build_call_kwargs(
            "openai", "gpt-4o", messages,
            base_url="https://api.openai.com/v1",
        )

        assert "reasoning_content" not in kwargs["messages"][2]
        assert "reasoning_content" not in messages[2]


# ---------------------------------------------------------------------------
# Require → strict fallback: pad must NOT leak into the fallback request
# ---------------------------------------------------------------------------


class _DummyResponse:
    def __init__(self, text="ok"):
        self.choices = [MagicMock(message=MagicMock(content=text))]


class TestRequireToStrictFallbackNoLeak:
    """teknium1's review: a DeepSeek pad must not leak into a strict fallback.

    The scenario: a require-side provider (DeepSeek) is the primary auxiliary
    backend; its request is built (padding applied to kwargs['messages']); the
    call fails and ``call_llm`` falls back to a strict provider (OpenAI). The
    fallback re-enters ``_build_call_kwargs`` with the SAME shared ``messages``
    list. Because the pad was applied to a request-local copy (not in place),
    the fallback's request messages are clean — no stale reasoning_content to
    trigger a 422.
    """

    @staticmethod
    def _messages_with_assistant_turn():
        return [
            {"role": "user", "content": "summarize this"},
            {
                "role": "assistant",
                "content": "here is a summary",
                "tool_calls": [
                    {"id": "c1", "type": "function",
                     "function": {"name": "f", "arguments": "{}"}}
                ],
            },
        ]

    def test_sync_fallback_request_has_no_leaked_pad(self):
        """Simulate the require→strict fallback by calling _build_call_kwargs twice
        with the same shared messages list — once for DeepSeek, once for OpenAI.
        """
        from agent.auxiliary_client import _build_call_kwargs

        messages = self._messages_with_assistant_turn()

        # Primary: DeepSeek (require-side) — builds a padded request-local copy.
        deepseek_kwargs = _build_call_kwargs(
            "deepseek", "deepseek-chat", messages,
            base_url="https://api.deepseek.com/v1",
        )
        assert deepseek_kwargs["messages"][1]["reasoning_content"] == " "

        # Fallback: OpenAI (strict) — re-enters with the SAME shared messages.
        # The shared list was NOT mutated by the DeepSeek call, so the OpenAI
        # request must have NO reasoning_content on the assistant turn.
        openai_kwargs = _build_call_kwargs(
            "openai", "gpt-4o", messages,
            base_url="https://api.openai.com/v1",
        )
        assert "reasoning_content" not in openai_kwargs["messages"][1]

        # The shared messages list is still clean.
        assert "reasoning_content" not in messages[1]

    def test_async_fallback_request_has_no_leaked_pad(self):
        """Same invariant via the async _build_call_kwargs path (async_call_llm
        uses the same _build_call_kwargs)."""
        from agent.auxiliary_client import _build_call_kwargs

        messages = self._messages_with_assistant_turn()

        # Primary: Kimi/Moonshot (require-side).
        kimi_kwargs = _build_call_kwargs(
            "kimi", "kimi-k2", messages,
            base_url="https://api.moonshot.ai/v1",
        )
        assert kimi_kwargs["messages"][1]["reasoning_content"] == " "

        # Fallback: Mistral (strict — rejects reasoning_content with 422).
        mistral_kwargs = _build_call_kwargs(
            "mistral", "mistral-large-latest", messages,
            base_url="https://api.mistral.ai/v1",
        )
        assert "reasoning_content" not in mistral_kwargs["messages"][1]

        # Shared list untouched.
        assert "reasoning_content" not in messages[1]

    def test_repeated_require_side_calls_are_idempotent(self):
        """Calling _build_call_kwargs twice for the same require-side provider
        does not double-pad or corrupt the shared list."""
        from agent.auxiliary_client import _build_call_kwargs

        messages = self._messages_with_assistant_turn()

        first = _build_call_kwargs(
            "deepseek", "deepseek-chat", messages,
            base_url="https://api.deepseek.com/v1",
        )
        second = _build_call_kwargs(
            "deepseek", "deepseek-chat", messages,
            base_url="https://api.deepseek.com/v1",
        )

        assert first["messages"][1]["reasoning_content"] == " "
        assert second["messages"][1]["reasoning_content"] == " "
        # Shared list still clean.
        assert "reasoning_content" not in messages[1]
