"""The max-iterations summary call treats ``reasoning_details`` per api_mode: the
anthropic_messages converter rebuilds signed thinking blocks from it, so the summary messages
must keep it; a strict chat-completions route drops it on the wire via the same kwargs
builder the main loop uses (hermes-agent#70233)."""

import json

import pytest

from agent.chat_completion_helpers import _build_api_kwargs_for_mode, _iteration_summary_api_messages
from run_agent import AIAgent

_HISTORY = [
    {"role": "user", "content": "q"},
    {"role": "assistant", "content": "", "tool_calls": [{"id": "t1", "type": "function", "function": {"name": "f", "arguments": "{}"}}],
     "reasoning_details": [{"type": "thinking", "thinking": "x", "signature": "SIG"}]},
    {"role": "tool", "tool_call_id": "t1", "content": "r"},
]


@pytest.fixture
def make_agent(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))

    def _make(base_url, provider):
        agent = AIAgent(api_key="k", base_url=base_url, provider=provider, model="m", quiet_mode=True,
                        skip_context_files=True, skip_memory=True)
        agent._cached_system_prompt = "SYS"
        return agent
    return _make


def test_anthropic_summary_messages_keep_reasoning_details(make_agent):
    agent = make_agent("https://api.anthropic.com", "anthropic")
    assert agent.api_mode == "anthropic_messages"
    out = _iteration_summary_api_messages(agent, [dict(m) for m in _HISTORY])
    assistant = next(m for m in out if m.get("role") == "assistant")
    assert assistant["reasoning_details"] == _HISTORY[1]["reasoning_details"]


def test_strict_chat_route_summary_wire_drops_reasoning_details(make_agent):
    agent = make_agent("https://api.groq.com/openai/v1", "custom")
    assert agent.api_mode == "chat_completions"
    api_messages = _iteration_summary_api_messages(agent, [dict(m) for m in _HISTORY])
    kwargs = _build_api_kwargs_for_mode(agent, api_messages)
    assert all("reasoning_details" not in m for m in kwargs["messages"])


class TestSummaryPrefixParity:
    """The summary request must close with the same normalization the main send
    path applies (assemble_api_request): canonical tool-call argument JSON and
    stripped string content — otherwise the summary's prefix diverges from every
    prior request and prefix-caching providers re-read the whole conversation
    (hermes-agent#123002)."""

    def test_canonicalizes_tool_call_arguments_and_strips_content(self, make_agent):
        agent = make_agent("https://api.groq.com/openai/v1", "custom")
        history = [
            {"role": "user", "content": "  q  "},
            {"role": "assistant", "content": "  thinking out loud  ", "tool_calls": [
                {"id": "t1", "type": "function",
                 "function": {"name": "f", "arguments": '{"b": 1, "a": 2}'}},
            ]},
            {"role": "tool", "tool_call_id": "t1", "content": "  r  "},
        ]
        snapshot = [dict(m) for m in history]
        out = _iteration_summary_api_messages(agent, history)

        assistant = next(m for m in out if m.get("role") == "assistant")
        args = assistant["tool_calls"][0]["function"]["arguments"]
        assert args == json.dumps({"a": 2, "b": 1}, separators=(",", ":"), sort_keys=True)

        # Same whitespace contract as the main path: string content stripped on the
        # wire copy, history untouched.
        by_role = {m["role"]: m for m in out if m.get("role") in {"user", "tool"}}
        assert by_role["user"]["content"] == "q"
        assert by_role["tool"]["content"] == "r"
        assert history == snapshot  # send-path rewrites never reach the transcript
