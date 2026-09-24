"""Regression for https://github.com/NousResearch/hermes-agent/issues/120828.

A custom Ollama provider using the OpenAI-compatible ``/v1/chat/completions``
endpoint saw HTTP 500 from Ollama's Qwen renderer on a follow-up payload
whose ``messages`` list carried no ``role:\"user\"`` row (system +
assistant tool_call + tool result, no user). Every native OpenAI-compat
endpoint expects at least one user turn; Hermes's continue-after-tool-call
and retry-of-failed-stream paths can drop the last user row under a few
known conditions, and the transport used to ship the empty-message list
verbatim.

The fix: ``ChatCompletionsTransport.build_kwargs`` runs every incoming
``messages`` through ``_ensure_has_user_message`` first, which appends
a synthetic continuation user turn when no real one is present. Real user
turns pass through untouched.
"""

from providers.base import ProviderProfile

import pytest

from agent.transports import get_transport
from agent.transports.chat_completions import (
    _ensure_has_user_message,
    _injected_user_continuation_marker,
)


@pytest.fixture
def transport():
    import agent.transports.chat_completions  # noqa: F401
    return get_transport("chat_completions")


class TestEnsureUserMessageHelper:
    def test_tool_only_messages_get_user_injected(self):
        messages = [
            {"role": "system", "content": "sys"},
            {"role": "assistant", "content": "", "tool_calls": [
                {"id": "call_1", "type": "function",
                 "function": {"name": "x", "arguments": "{}"}},
            ]},
            {"role": "tool", "tool_call_id": "call_1", "content": "result"},
        ]
        out = _ensure_has_user_message(messages)
        roles = [m["role"] for m in out]
        assert roles[-1] == "user", f"expected last role=user, got {roles}"
        assert out[-1].get(_injected_user_continuation_marker) is True

    def test_real_user_message_is_pass_through(self):
        messages = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "ok"},
        ]
        out = _ensure_has_user_message(messages)
        assert out == messages, "real user message must not trigger injection"
        assert not any(m.get(_injected_user_continuation_marker) for m in out)

    def test_empty_list_is_pass_through(self):
        assert _ensure_has_user_message([]) == []

    def test_only_system_no_user_gets_user_injected(self):
        messages = [{"role": "system", "content": "sys"}]
        out = _ensure_has_user_message(messages)
        roles = [m["role"] for m in out]
        assert "user" in roles, f"missing user injection, got {roles}"


class TestBuildKwargsRequiresUserMessage:
    def test_legacy_kwargs_path_injects_user(self, transport):
        messages = [
            {"role": "system", "content": "sys"},
            {"role": "assistant", "content": "", "tool_calls": [
                {"id": "c1", "type": "function",
                 "function": {"name": "x", "arguments": "{}"}},
            ]},
            {"role": "tool", "tool_call_id": "c1", "content": "r"},
        ]
        out = transport.build_kwargs(
            model="huihui_ai/Qwen3.8-abliterated:27b",
            messages=messages,
            tools=None,
            base_url="http://localhost:11434/v1",
        )
        roles = [m.get("role") for m in out["messages"]]
        assert "user" in roles, f"legacy kwargs path must inject user, got {roles}"

    def test_profile_path_injects_user(self, transport):
        profile = ProviderProfile(name="ollama-custom")
        messages = [
            {"role": "assistant", "content": "", "tool_calls": [
                {"id": "c1", "type": "function",
                 "function": {"name": "x", "arguments": "{}"}},
            ]},
            {"role": "tool", "tool_call_id": "c1", "content": "r"},
        ]
        out = transport.build_kwargs(
            model="huihui_ai/Qwen3.8-abliterated:27b",
            messages=messages,
            tools=None,
            base_url="http://localhost:11434/v1",
            provider_profile=profile,
        )
        roles = [m.get("role") for m in out["messages"]]
        assert "user" in roles, f"profile path must inject user, got {roles}"

    def test_real_user_message_unchanged_in_profile_path(self, transport):
        profile = ProviderProfile(name="ollama-custom")
        messages = [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "ok"},
        ]
        out = transport.build_kwargs(
            model="huihui_ai/Qwen3.8-abliterated:27b",
            messages=messages,
            tools=None,
            base_url="http://localhost:11434/v1",
            provider_profile=profile,
        )
        roles = [m.get("role") for m in out["messages"]]
        assert roles == ["user", "assistant"], f"unexpected roles {roles}"
        assert not any(m.get(_injected_user_continuation_marker) for m in out["messages"])