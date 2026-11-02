"""The max-iterations summary call treats ``reasoning_details`` per api_mode: the
anthropic_messages converter rebuilds signed thinking blocks from it, so the summary messages
must keep it; a strict chat-completions route drops it on the wire via the same kwargs
builder the main loop uses (hermes-agent#70233)."""

import copy
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
        snapshot = copy.deepcopy(history)
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

    def test_strips_lone_surrogates_like_the_send_path(self, make_agent):
        agent = make_agent("https://api.groq.com/openai/v1", "custom")
        history = [
            {"role": "user", "content": "clip \ud800 paste"},
            {"role": "assistant", "content": "ok \ud83d", "tool_calls": [
                {"id": "t1", "type": "function",
                 "function": {"name": "f", "arguments": '{"k": "v\ud800"}'}},
            ]},
            {"role": "tool", "tool_call_id": "t1", "content": "r"},
            {"role": "user", "content": [{"type": "text", "text": "part \ud800"}]},
        ]
        snapshot = copy.deepcopy(history)
        out = _iteration_summary_api_messages(agent, history)
        # Main path's third pass rewrites lone surrogates to U+FFFD; anything else diverges the
        # prefix or makes the SDK's ensure_ascii=False utf-8 encode raise.
        assert next(m for m in out if m["role"] == "user")["content"] == "clip \ufffd paste"
        assistant = next(m for m in out if m["role"] == "assistant")
        assert assistant["content"] == "ok \ufffd"
        # Canonicalization runs first, so argument surrogates are already ASCII \udXXX escapes.
        assert assistant["tool_calls"][0]["function"]["arguments"] == '{"k":"v\\ud800"}'
        assert out[-1]["content"][0]["text"] == "part \ufffd"
        json.dumps(out, ensure_ascii=False).encode("utf-8")
        # The sanitizer is in-place and list-content parts are shared nested dicts:
        # history must keep its stored bytes.
        assert history == snapshot
