"""The iteration-limit consumer must preserve actual 413 image evictions (#148146)."""
from copy import deepcopy

from agent.chat_completion_helpers import _iteration_summary_api_messages
from tests.agent.test_413_compression import _new_test_agent


def test_iteration_summary_keeps_recovered_image_evictions():
    agent = _new_test_agent()
    agent.provider, agent.model = "openrouter", "test/model"
    image = "data:image/png;base64," + "a" * 2000
    history = [
        {"role": "user", "content": "Inspect the page"},
        {"role": "assistant", "content": None, "tool_calls": [{"id": "old", "type": "function",
         "function": {"name": "browser_vision", "arguments": "{}"}}]},
        {"role": "tool", "tool_call_id": "old", "content": [
            {"type": "text", "text": "Visible page text"},
            {"type": "image_url", "image_url": {"url": image}},
        ]},
        {"role": "user", "content": "Summarize"},
    ]
    original = deepcopy(history)
    assert agent._try_strip_image_parts_from_tool_messages(
        deepcopy(history), remember_model=False, remember_bytes=True,
    )
    wire = _iteration_summary_api_messages(agent, history)
    assert image not in str(wire)
    assert "Visible page text" in str(wire)
    assert history == original
