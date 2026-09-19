"""Execution barriers for in-turn model selection."""

from contextlib import nullcontext
from types import SimpleNamespace

from agent import tool_executor


def _call(name: str, call_id: str):
    return SimpleNamespace(
        id=call_id,
        function=SimpleNamespace(name=name, arguments="{}"),
    )


def test_model_selection_stops_later_terminal_approval_runs(monkeypatch):
    select_call = _call("select_model", "select")
    terminal_call = _call("terminal", "terminal")
    assistant = SimpleNamespace(tool_calls=[select_call, terminal_call])
    agent = SimpleNamespace(_incremental_persistence_failed=False)
    executed = []
    skipped = []

    monkeypatch.setattr(
        "agent.terminal_approval_batch.terminal_approval_runs",
        lambda _agent, _calls: [[select_call], [terminal_call]],
    )
    monkeypatch.setattr(
        "agent.terminal_approval_batch.terminal_approval_batch",
        lambda *_args, **_kwargs: nullcontext(),
    )

    def run(_agent, segment, *_args, **_kwargs):
        executed.extend(call.id for call in segment.tool_calls)
        _agent._model_selection_attempted_in_tool_batch = True

    monkeypatch.setattr(tool_executor, "_execute_tool_calls_sequential", run)
    monkeypatch.setattr(
        tool_executor,
        "_append_skipped_tool_results",
        lambda _agent, _messages, calls, *_args, **_kwargs: skipped.extend(
            call.id for call in calls
        ) or True,
    )
    monkeypatch.setattr(tool_executor, "_finalize_tool_batch", lambda *_args, **_kwargs: None)

    tool_executor.execute_tool_calls_sequential(agent, assistant, [], "task")

    assert executed == ["select"]
    assert skipped == ["terminal"]


def test_model_selection_stops_later_segmented_calls(monkeypatch):
    select_call = _call("select_model", "select")
    terminal_call = _call("terminal", "terminal")
    assistant = SimpleNamespace(tool_calls=[select_call, terminal_call])
    agent = SimpleNamespace(_incremental_persistence_failed=False)
    executed = []
    skipped = []

    def run(_agent, segment, *_args, **_kwargs):
        executed.extend(call.id for call in segment.tool_calls)
        _agent._model_selection_attempted_in_tool_batch = True

    monkeypatch.setattr(tool_executor, "_execute_tool_calls_sequential", run)
    monkeypatch.setattr(
        tool_executor,
        "_append_skipped_tool_results",
        lambda _agent, _messages, calls, *_args, **_kwargs: skipped.extend(
            call.id for call in calls
        ) or True,
    )
    monkeypatch.setattr(tool_executor, "_finalize_tool_batch", lambda *_args, **_kwargs: None)

    tool_executor.execute_tool_calls_segmented(
        agent,
        assistant,
        [],
        "task",
        segments=[("sequential", [select_call]), ("sequential", [terminal_call])],
    )

    assert executed == ["select"]
    assert skipped == ["terminal"]
