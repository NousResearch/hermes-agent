"""Tests for the kanban worker clean-exit protocol guard."""

from types import SimpleNamespace

from agent import conversation_loop


def _tool_call(name: str):
    return {
        "id": f"call_{name}",
        "type": "function",
        "function": {"name": name, "arguments": "{}"},
    }


def test_kanban_worker_terminal_tool_detection():
    assert not conversation_loop._kanban_worker_called_terminal_tool(
        [{"role": "assistant", "content": "done"}]
    )
    assert conversation_loop._kanban_worker_called_terminal_tool(
        [{"role": "assistant", "tool_calls": [_tool_call("kanban_complete")]}]
    )
    assert conversation_loop._kanban_worker_called_terminal_tool(
        [{"role": "assistant", "tool_calls": [_tool_call("kanban_block")]}]
    )


def test_kanban_worker_text_only_final_response_auto_blocks(monkeypatch):
    calls = []

    fake_run_agent = SimpleNamespace(
        handle_function_call=lambda name, args, task_id=None: calls.append(
            (name, args, task_id)
        )
    )
    monkeypatch.setattr(conversation_loop, "_ra", lambda: fake_run_agent)
    monkeypatch.setenv("HERMES_KANBAN_TASK", "t_test1234")

    blocked = conversation_loop._auto_block_kanban_worker_protocol_violation(
        messages=[{"role": "assistant", "content": "I am done"}],
        final_response="I am done",
        api_call_count=2,
        effective_task_id="loop-task",
    )

    assert blocked is True
    assert calls == [
        (
            "kanban_block",
            {
                "task_id": "t_test1234",
                "reason": (
                    "protocol-guard: worker produced a final text response without "
                    "calling kanban_complete or kanban_block; auto-blocked to prevent "
                    "clean-exit protocol violation. api_calls=2; "
                    "final_response_preview='I am done'"
                ),
            },
            "loop-task",
        )
    ]


def test_kanban_worker_guard_skips_when_terminal_tool_was_called(monkeypatch):
    calls = []
    fake_run_agent = SimpleNamespace(
        handle_function_call=lambda name, args, task_id=None: calls.append(
            (name, args, task_id)
        )
    )
    monkeypatch.setattr(conversation_loop, "_ra", lambda: fake_run_agent)
    monkeypatch.setenv("HERMES_KANBAN_TASK", "t_test1234")

    blocked = conversation_loop._auto_block_kanban_worker_protocol_violation(
        messages=[{"role": "assistant", "tool_calls": [_tool_call("kanban_complete")]}],
        final_response="completed",
        api_call_count=3,
        effective_task_id=None,
    )

    assert blocked is False
    assert calls == []
