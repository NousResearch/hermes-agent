"""Cross-protocol projection must discard Codex interims before empty-row repair."""
from copy import deepcopy
import logging
from types import SimpleNamespace

import pytest

from agent.conversation_loop import _CODEX_INCOMPLETE_NUDGE
from agent.turn_context import _reset_per_turn_agent_state
from agent.turn_request_assembly import assemble_api_request
from tests.agent.test_run_agent_codex_responses import _build_agent


def _request(agent, rows):
    _reset_per_turn_agent_state(agent)
    return assemble_api_request(
        agent, messages=rows, current_turn_user_idx=0, _ext_prefetch_cache="",
        _plugin_user_context="", moa_config=None, active_system_prompt="",
        original_user_message=rows[0]["content"], pending_moa_prepared_request=None,
        request_logger=logging.getLogger(__name__),
    ).api_messages


@pytest.mark.parametrize("selection", [False, True], ids=["built-wire", "canonical-selection"])
@pytest.mark.parametrize("visible_sidecar", [False, True], ids=["visible-content", "visible-api-content"])
@pytest.mark.parametrize("hidden", [
    {"codex_reasoning_items": [{"type": "reasoning", "encrypted_content": "opaque"}]},
    {"reasoning_content": "private Codex trace"},
], ids=["encrypted-only", "readable-only"])
def test_chat_projection_drops_foreign_codex_interims_before_healing(monkeypatch, selection, visible_sidecar, hidden):
    agent = _build_agent(monkeypatch)
    interim = agent._build_assistant_message(SimpleNamespace(
        content="", tool_calls=None, **hidden,
    ), "length")
    visible = {"role": "assistant", "content": "Visible progress."}
    if visible_sidecar:
        visible = {**interim, "api_content": "Visible progress."}
    rows = [
        {"role": "user", "content": "do it"}, interim,
        {"role": "user", "content": _CODEX_INCOMPLETE_NUDGE}, deepcopy(interim),
        {"role": "user", "content": _CODEX_INCOMPLETE_NUDGE},
        visible,
        {"role": "user", "content": "finish"},
    ]
    before = deepcopy(rows)
    agent.api_mode = "chat_completions"
    if selection:
        monkeypatch.setattr("agent.conversation_loop._apply_context_engine_selection",
                            lambda *args, **kwargs: deepcopy(rows))
    wire = _request(agent, rows)
    assert [{"role": m["role"], "content": m.get("content")} for m in wire] == [
        {"role": "user", "content": "do it"},
        {"role": "assistant", "content": "Visible progress."},
        {"role": "user", "content": "finish"},
    ]
    assert not any(key in m for m in wire for key in (
        "api_content", "_reasoning_route", "codex_reasoning_items", "reasoning_content", "reasoning",
    ))
    assert rows == before


@pytest.mark.parametrize("content", [None, "", " \n ", []])
def test_chat_projection_heals_genuinely_empty_nonfinal_rows(monkeypatch, content):
    agent = _build_agent(monkeypatch)
    rows = [
        {"role": "user", "content": "do it"},
        {"role": "assistant", "content": content},
        {"role": "user", "content": "finish"},
    ]
    before = deepcopy(rows)
    agent.api_mode = "chat_completions"
    assert _request(agent, rows) == [
        {"role": "user", "content": "do it"},
        {"role": "assistant", "content": "[response interrupted]"},
        {"role": "user", "content": "finish"},
    ]
    assert rows == before


@pytest.mark.parametrize("require_pad", [False, True], ids=["strict-chat", "require-side-chat"])
def test_chat_projection_preserves_visible_answer_and_parallel_tool_group(monkeypatch, require_pad):
    agent = _build_agent(monkeypatch)
    hidden = [{"type": "reasoning", "encrypted_content": "opaque"}]
    tool_calls = [
        SimpleNamespace(id=f"call_{i}", type="function",
                        function=SimpleNamespace(name="terminal", arguments="{}"))
        for i in range(2)
    ]
    tool_row = agent._build_assistant_message(SimpleNamespace(
        content="", tool_calls=tool_calls, codex_reasoning_items=hidden,
        reasoning_content="private Codex trace",
    ), "tool_calls")
    answer = agent._build_assistant_message(SimpleNamespace(
        content="Visible answer.", tool_calls=None, codex_reasoning_items=hidden,
        reasoning_content="private Codex trace",
    ), "stop")
    rows = [
        {"role": "user", "content": "do it"}, tool_row,
        {"role": "tool", "tool_call_id": "call_0", "name": "terminal", "content": "first result"},
        {"role": "tool", "tool_call_id": "call_1", "name": "terminal", "content": "second result"},
        answer, {"role": "user", "content": "finish"},
    ]
    before = deepcopy(rows)
    agent.api_mode = "chat_completions"
    if require_pad:
        agent.provider = "deepseek"
        agent.model = "deepseek-chat"
        agent.base_url = "https://api.deepseek.com"
    wire = _request(agent, rows)
    assert [m["role"] for m in wire] == ["user", "assistant", "tool", "tool", "assistant", "user"]
    assert [m["content"] for m in wire if m["role"] == "tool"] == ["first result", "second result"]
    assert [tc["id"] for tc in wire[1]["tool_calls"]] == ["call_0", "call_1"]
    assert wire[4]["content"] == "Visible answer."
    for message in wire:
        assert not any(key in message for key in ("_reasoning_route", "codex_reasoning_items", "reasoning", "reasoning_details"))
        if message["role"] == "assistant" and require_pad:
            assert message["reasoning_content"] == " "
        else:
            assert "reasoning_content" not in message
    assert rows == before
