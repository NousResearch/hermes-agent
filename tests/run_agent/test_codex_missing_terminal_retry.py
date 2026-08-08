from types import SimpleNamespace

import pytest

from tests.run_agent.test_run_agent_codex_responses import (
    _FakeCreateStream,
    _build_agent,
    _codex_reasoning_only_response,
    _codex_request_kwargs,
    _codex_tool_call_response,
)


def test_missing_terminal_stream_records_transport_invariant(monkeypatch):
    agent = _build_agent(monkeypatch)
    output_item = SimpleNamespace(
        type="message",
        status="completed",
        content=[SimpleNamespace(type="output_text", text="no terminal frame")],
    )

    def _fake_create(**kwargs):
        assert kwargs.get("stream") is True
        return _FakeCreateStream([
            SimpleNamespace(type="response.created"),
            SimpleNamespace(type="response.output_item.done", item=output_item),
        ])

    agent.client = SimpleNamespace(
        responses=SimpleNamespace(create=_fake_create),
    )

    response = agent._run_codex_stream(_codex_request_kwargs())

    assert response.output == [output_item]
    assert response.terminal_event_received is False


@pytest.mark.parametrize("missing_terminal_shape", ["reasoning_item", "empty_stream"])
def test_post_tool_missing_terminal_is_bounded_and_does_not_persist_incomplete_turns(
    monkeypatch, missing_terminal_shape,
):
    """A truncated post-tool stream gets one retry without polluting history."""
    agent = _build_agent(monkeypatch)
    agent.provider = "custom"
    agent.base_url = "http://127.0.0.1:11434/v1"

    def _missing_terminal_reasoning(index):
        response = _codex_reasoning_only_response(
            encrypted_content=f"enc_missing_{index}",
            summary_text=f"Still reasoning {index}",
        )
        response.terminal_event_received = False
        return response

    request_count = 0

    def _fake_api_call(api_kwargs):
        nonlocal request_count
        request_count += 1
        if request_count == 1:
            return _codex_tool_call_response()
        if missing_terminal_shape == "reasoning_item":
            return _missing_terminal_reasoning(request_count - 1)
        from agent.codex_runtime import CodexMissingTerminalResponseError

        raise CodexMissingTerminalResponseError(
            "Codex Responses stream did not emit a terminal response"
        )

    monkeypatch.setattr(agent, "_interruptible_api_call", _fake_api_call)

    def _fake_execute_tool_calls(assistant_message, messages, effective_task_id, *_args):
        for call in assistant_message.tool_calls:
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": call.id,
                    "content": '{"ok":true}',
                }
            )

    monkeypatch.setattr(agent, "_execute_tool_calls", _fake_execute_tool_calls)
    persisted = []
    agent._persist_session = lambda messages, history=None: persisted.append(list(messages))

    result = agent.run_conversation("inspect memory and summarize it")

    expected_error = (
        "Codex Responses stream ended without a terminal response after one "
        "finalization retry"
    )
    assert result["completed"] is False
    assert result["partial"] is True
    assert result["error"] == expected_error
    assert result["final_response"] == expected_error
    assert request_count == 3
    assert any(
        msg.get("role") == "tool" and msg.get("tool_call_id") == "call_1"
        for msg in result["messages"]
    )
    assert not any(
        msg.get("role") == "assistant" and msg.get("finish_reason") == "incomplete"
        for msg in result["messages"]
    )
    assert persisted
    assert not any(
        msg.get("role") == "assistant" and msg.get("finish_reason") == "incomplete"
        for msg in persisted[-1]
    )


def test_exhausted_reasoning_only_continuations_roll_back_incomplete_turns(monkeypatch):
    """Terminal reasoning-only responses remain resumable but are transient on failure."""
    agent = _build_agent(monkeypatch)
    responses = [
        _codex_tool_call_response(),
        _codex_reasoning_only_response(encrypted_content="enc_1"),
        _codex_reasoning_only_response(encrypted_content="enc_2"),
        _codex_reasoning_only_response(encrypted_content="enc_3"),
    ]
    monkeypatch.setattr(agent, "_interruptible_api_call", lambda api_kwargs: responses.pop(0))

    def _fake_execute_tool_calls(assistant_message, messages, effective_task_id, *_args):
        messages.extend(
            {
                "role": "tool",
                "tool_call_id": call.id,
                "content": '{"ok":true}',
            }
            for call in assistant_message.tool_calls
        )

    monkeypatch.setattr(agent, "_execute_tool_calls", _fake_execute_tool_calls)

    result = agent.run_conversation("inspect memory and summarize it")

    assert result["error"] == (
        "Codex response remained incomplete after 3 continuation attempts"
    )
    assert result["api_calls"] == 4
    assert any(msg.get("role") == "tool" for msg in result["messages"])
    assert not any(
        msg.get("role") == "assistant" and msg.get("finish_reason") == "incomplete"
        for msg in result["messages"]
    )


def test_codex_incomplete_rollback_preserves_intervening_nontransient_messages():
    from agent.conversation_loop import (
        _CODEX_INCOMPLETE_NUDGE,
        _rollback_codex_incomplete_chain,
    )

    tool = {"role": "tool", "tool_call_id": "call_1", "content": "ok"}
    steer = {"role": "user", "content": "Use the shorter answer."}
    incomplete = {
        "role": "assistant",
        "content": "",
        "finish_reason": "incomplete",
    }
    synthetic_nudge = {"role": "user", "content": _CODEX_INCOMPLETE_NUDGE}
    messages = [
        tool,
        incomplete.copy(),
        synthetic_nudge,
        steer,
        incomplete.copy(),
    ]
    agent = SimpleNamespace(
        _codex_incomplete_checkpoint=1,
        _codex_incomplete_retries=3,
        _codex_missing_terminal_retries=1,
    )

    _rollback_codex_incomplete_chain(agent, messages)

    assert messages == [tool, steer]
    assert agent._session_messages is messages
