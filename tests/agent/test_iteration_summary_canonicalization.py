"""The max-iterations summary messages must carry the same send-path normalization the main
loop applies in ``assemble_api_request`` (whitespace strip + canonical tool-call argument
JSON). Without it the summary request re-sends the conversation in stored key order, so a
local server with prefix caching misses the whole warmed prefix (hermes-agent#123002)."""

import pytest

from agent.chat_completion_helpers import _iteration_summary_api_messages
from run_agent import AIAgent


@pytest.fixture
def make_agent(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))

    def _make():
        return AIAgent(api_key="k", base_url="https://api.groq.com/openai/v1", provider="custom",
                       model="m", quiet_mode=True, skip_context_files=True, skip_memory=True)
    return _make


def test_summary_messages_canonicalize_tool_call_args_and_strip_content(make_agent):
    agent = make_agent()
    agent._cached_system_prompt = "SYS"
    history = [
        {"role": "user", "content": "q"},
        {
            "role": "assistant",
            "content": "  thinking out loud  ",
            "tool_calls": [
                {"id": "t1", "type": "function",
                 "function": {"name": "f", "arguments": '{"zeta": 1, "alpha": 2}'}},
            ],
        },
        {"role": "tool", "tool_call_id": "t1", "content": "r"},
    ]
    out = _iteration_summary_api_messages(agent, history)
    assistant = next(m for m in out if m.get("role") == "assistant")
    # Same canonical wire form the main send path emits: sorted keys, compact separators.
    assert assistant["tool_calls"][0]["function"]["arguments"] == '{"alpha":2,"zeta":1}'
    assert assistant["content"] == "thinking out loud"


def test_summary_normalization_is_copy_on_write(make_agent):
    agent = make_agent()
    agent._cached_system_prompt = "SYS"
    history = [
        {"role": "assistant", "content": "  pad  ", "tool_calls": [
            {"id": "t1", "type": "function",
             "function": {"name": "f", "arguments": '{"zeta": 1, "alpha": 2}'}},
        ]},
    ]
    _iteration_summary_api_messages(agent, history)
    # The persisted transcript keeps its stored bytes; only the send-path copy is rewritten.
    assert history[0]["tool_calls"][0]["function"]["arguments"] == '{"zeta": 1, "alpha": 2}'
    assert history[0]["content"] == "  pad  "
