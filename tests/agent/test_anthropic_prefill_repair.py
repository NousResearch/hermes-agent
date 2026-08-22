"""Tests for trailing-assistant prefill repair on Claude 4.6+/Fable.

Claude Sonnet/Opus 4.6+ and the Fable family reject requests whose final
message is role=assistant with a non-retryable HTTP 400:
"This model does not support assistant message prefill. The conversation
must end with a user message."

Hermes can produce that tail when streamed preamble/status text is persisted
as an assistant turn before the next request (interrupt, gateway restart,
cron resume). The adapter must append a synthetic user continuation for the
rejecting model families and leave genuine prefill behaviour alone elsewhere.
"""

import pytest

from agent.anthropic_adapter import (
    build_anthropic_kwargs,
    ensure_user_tail_for_no_prefill,
    model_rejects_assistant_prefill,
)


class TestModelRejectsAssistantPrefill:

    @pytest.mark.parametrize("model", [
        "claude-sonnet-4-6",
        "claude-sonnet-4.6",
        "claude-opus-4-6",
        "claude-opus-4-7",
        "claude-sonnet-5",
        "claude-haiku-4-6",
        "claude-fable-5",
        "anthropic/claude-sonnet-4-6",
    ])
    def test_rejecting_models(self, model):
        assert model_rejects_assistant_prefill(model) is True

    @pytest.mark.parametrize("model", [
        "claude-sonnet-4-5",
        "claude-opus-4-1",
        "claude-haiku-4-5-20251001",
        "claude-3-5-sonnet-20241022",  # legacy naming, no (sonnet)-N suffix ≥4.6
        "gpt-4o",
        "gemini-3-pro",
        "",
        None,
    ])
    def test_prefill_capable_models(self, model):
        assert model_rejects_assistant_prefill(model) is False


class TestEnsureUserTail:

    def test_appends_user_turn_after_text_assistant_tail(self):
        msgs = [
            {"role": "user", "content": [{"type": "text", "text": "go"}]},
            {"role": "assistant", "content": [{"type": "text", "text": "Checking logs."}]},
        ]
        ensure_user_tail_for_no_prefill(msgs)
        assert msgs[-1]["role"] == "user"
        assert msgs[-2]["role"] == "assistant"
        assert "Continue" in msgs[-1]["content"][0]["text"]

    def test_noop_when_tail_is_user(self):
        msgs = [{"role": "user", "content": [{"type": "text", "text": "hi"}]}]
        ensure_user_tail_for_no_prefill(msgs)
        assert len(msgs) == 1

    def test_noop_when_tail_assistant_has_tool_use(self):
        msgs = [
            {"role": "user", "content": [{"type": "text", "text": "run"}]},
            {"role": "assistant", "content": [
                {"type": "text", "text": "Running."},
                {"type": "tool_use", "id": "toolu_1", "name": "terminal", "input": {}},
            ]},
        ]
        ensure_user_tail_for_no_prefill(msgs)
        assert len(msgs) == 2  # tool_use tails need real tool results, not fakes

    def test_noop_on_empty(self):
        msgs = []
        ensure_user_tail_for_no_prefill(msgs)
        assert msgs == []


class TestBuildKwargsIntegration:

    def _msgs(self):
        return [
            {"role": "system", "content": "be useful"},
            {"role": "user", "content": "diagnose the cron"},
            {"role": "assistant", "content": "Smoking gun hunt, checking configs."},
        ]

    def test_sonnet_46_gets_user_tail(self):
        kw = build_anthropic_kwargs(
            model="claude-sonnet-4-6",
            messages=self._msgs(),
            tools=None,
            max_tokens=1024,
            reasoning_config=None,
        )
        roles = [m["role"] for m in kw["messages"]]
        assert roles[-2:] == ["assistant", "user"]

    def test_fable_gets_user_tail(self):
        kw = build_anthropic_kwargs(
            model="claude-fable-5",
            messages=self._msgs(),
            tools=None,
            max_tokens=1024,
            reasoning_config=None,
        )
        assert kw["messages"][-1]["role"] == "user"

    def test_older_claude_keeps_prefill(self):
        kw = build_anthropic_kwargs(
            model="claude-sonnet-4-5",
            messages=self._msgs(),
            tools=None,
            max_tokens=1024,
            reasoning_config=None,
        )
        assert kw["messages"][-1]["role"] == "assistant"

    def test_user_tail_already_present_unchanged(self):
        msgs = self._msgs() + [{"role": "user", "content": "and?"}]
        kw = build_anthropic_kwargs(
            model="claude-sonnet-4-6",
            messages=msgs,
            tools=None,
            max_tokens=1024,
            reasoning_config=None,
        )
        # No double user tail beyond what _merge_consecutive_roles produces
        assert kw["messages"][-1]["role"] == "user"
        assert kw["messages"][-2]["role"] == "assistant"
