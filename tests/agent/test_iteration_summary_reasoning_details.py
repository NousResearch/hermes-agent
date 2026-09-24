"""The max-iterations summary call treats ``reasoning_details`` per api_mode: the
anthropic_messages converter rebuilds signed thinking blocks from it, so the summary messages
must keep it; a strict chat-completions route drops it on the wire via the same kwargs
builder the main loop uses (hermes-agent#70233)."""

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


def test_portal_summary_wire_replays_only_the_newest_assistant_details(make_agent):
    agent = make_agent("https://inference-api.nousresearch.com/v1", "nous")
    history = [
        {"role": "user", "content": "first"},
        {"role": "assistant", "content": "first answer", "reasoning_details": [{"type": "thinking", "thinking": "old", "signature": "OLD"}]},
        {"role": "user", "content": "second"},
        {"role": "assistant", "content": "second answer", "reasoning_details": [{"type": "thinking", "thinking": "new", "signature": "NEW"}]},
        {"role": "user", "content": "continue"},
    ]

    api_messages = _iteration_summary_api_messages(agent, [dict(message) for message in history])
    kwargs = _build_api_kwargs_for_mode(agent, api_messages)
    replayed = [message for message in kwargs["messages"] if message.get("role") == "assistant"]

    assert "reasoning_details" not in replayed[0]
    assert replayed[1]["reasoning_details"] == history[3]["reasoning_details"]
