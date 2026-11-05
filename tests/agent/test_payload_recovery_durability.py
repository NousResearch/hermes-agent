"""A 413 evicts rejected tool images without disabling subsequent vision (#148146)."""

from copy import deepcopy
from unittest.mock import patch

import pytest

from agent.turn_context import build_api_messages
from tests.agent.test_413_compression import _make_413_error, _mock_response, _new_test_agent


def _tool_round(call_id, image, *, part_type="image_url"):
    part = {"type": part_type, "image_url": {"url": image}}
    if part_type == "input_image":
        part["image_url"] = image
    return [
        {"role": "assistant", "content": None, "tool_calls": [{
            "id": call_id, "type": "function",
            "function": {"name": "browser_vision", "arguments": "{}"},
        }]},
        {"role": "tool", "tool_call_id": call_id, "content": [
            {"type": "text", "text": "Visible page text"}, part,
        ]},
    ]


@pytest.fixture
def agent():
    instance = _new_test_agent()
    instance.provider = "openrouter"
    instance.model = "test/model"
    instance._current_turn_timestamp = "2026-11-05T00:00:00"
    return instance


def _build(agent, history):
    return build_api_messages(
        agent, history, current_turn_user_idx=len(history) - 1,
        ext_prefetch_cache=None, plugin_user_context=None, moa_config=None,
        active_system_prompt="You are helpful.",
    )[0]


@pytest.mark.parametrize("part_type", ["image_url", "input_image"])
def test_recovery_replays_old_eviction_but_accepts_fresh_vision(agent, part_type):
    old = "data:image/png;base64," + "a" * 2000
    new = "data:image/png;base64," + "b" * 2000
    history = [{"role": "user", "content": "Inspect"}, *_tool_round("old", old, part_type=part_type)]
    original = deepcopy(history)
    requests = []
    response = _mock_response(content="Recovered", finish_reason="stop")

    def provider(**kwargs):
        requests.append(deepcopy(kwargs["messages"]))
        if len(requests) == 1:
            raise _make_413_error()
        return response

    agent.client.chat.completions.create.side_effect = provider
    with (
        patch.object(agent, "_model_supports_vision", return_value=True),
        patch.object(agent, "_compress_context", side_effect=lambda msgs, *a, **kw: (msgs, "You are helpful.")),
        patch.object(agent, "_persist_session"),
        patch.object(agent, "_save_trajectory"),
        patch.object(agent, "_cleanup_task_resources"),
    ):
        first = agent.run_conversation("continue", conversation_history=history)
        assert first["completed"] is True
        assert old in str(requests[0])
        assert old not in str(requests[1])
        assert len(requests) == 2
        # Rebuild from unchanged image-bearing history, including a fresh tool result.
        next_history = [*original, *_tool_round("new", new, part_type=part_type)]
        second = agent.run_conversation("Inspect the new page", conversation_history=next_history)

    assert second["completed"] is True
    assert len(requests) == 3
    assert old not in str(requests[2])
    assert new in str(requests[2])
    assert "Visible page text" in str(requests[2])
    assert all(request[0] == {"role": "system", "content": "You are helpful."} for request in requests)
    assert history[2]["content"] == original[2]["content"]
    assert next_history[2]["content"] == original[2]["content"]
    assert not getattr(agent, "_no_list_tool_content_models", set())


@pytest.mark.parametrize("change", [
    "none", "model", "provider", "base_url", "api_mode", "tool_call_id", "payload",
    "no_strip", "no_model", "text_only_change", "mixed_payload",
])
def test_eviction_is_bound_to_rejected_tool_payload_and_route(agent, change):
    image = "data:image/png;base64," + "a" * 2000
    history = [{"role": "user", "content": "Inspect"}, *_tool_round("old", image),
               {"role": "user", "content": "continue"}]
    before = deepcopy(history)
    if change == "no_model":
        agent.model = ""
    wire = _build(agent, history)
    if change == "no_strip":
        assert not agent._try_strip_image_parts_from_tool_messages(
            [{"role": "tool", "content": "Already plain text"}],
            remember_model=False, remember_bytes=True,
        )
    else:
        assert agent._try_strip_image_parts_from_tool_messages(wire, remember_model=False, remember_bytes=True)
    if change in {"model", "provider", "base_url", "api_mode"}:
        previous = getattr(agent, change)
        setattr(agent, change, previous + "/other")
    call_id = {"tool_call_id": "new"}.get(change, "old")
    history[1]["tool_calls"][0]["id"] = call_id
    history[2]["tool_call_id"] = call_id
    history[2]["content"][0]["text"] = {
        "text_only_change": "Updated Visible page text",
    }.get(change, "Visible page text")
    replacement = image.replace("aaaa", "bbbb")
    history[2]["content"][1]["image_url"]["url"] = replacement if change == "payload" else image
    if change == "mixed_payload":
        history[2]["content"].append({"type": "image_url", "image_url": {"url": replacement}})
    image = replacement if change in {"payload", "mixed_payload"} else image
    result = _build(agent, history)
    assert (image in str(result)) is (change not in {"none", "text_only_change"})
    assert "Visible page text" in str(result)
    if change == "mixed_payload":
        assert before[2]["content"][1]["image_url"]["url"] not in str(result)
    if change in {"model", "provider", "base_url", "api_mode"}:
        setattr(agent, change, previous)
        assert image not in str(_build(agent, history))
    if change in {"none", "model", "provider", "base_url", "api_mode", "no_strip"}:
        assert history[2]["content"] == before[2]["content"]
