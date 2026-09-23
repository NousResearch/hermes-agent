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


def test_gemini_summary_messages_inject_thought_signature_sentinel(make_agent):
    """Mirror transport sentinel injection for max-iterations summary path (#66587)."""
    agent = make_agent("https://api.example.com/openai/v1", "custom")
    agent.model = "google/gemini-3-pro-preview"
    assert agent.api_mode == "chat_completions"
    api_messages = _iteration_summary_api_messages(agent, [dict(m) for m in _HISTORY])
    assistant = next(m for m in api_messages if m.get("role") == "assistant")
    tc = assistant["tool_calls"][0]
    assert tc["extra_content"] == {"google": {"thought_signature": "skip_thought_signature_validator"}}
