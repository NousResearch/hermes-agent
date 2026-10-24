"""Structured delegation payloads and child progress over ACP."""

import json
from unittest.mock import patch

import pytest

from acp_adapter.events import make_step_cb, make_tool_progress_cb
from acp_adapter.tools import build_tool_complete, build_tool_start


@pytest.mark.parametrize("arguments", [
    {"goal": "Review parser", "model": "test/reviewer", "role": "reviewer"},
    {"tasks": [{"goal": "Inspect routing"}, {"goal": "Run tests", "model": "test/tester"}]},
])
def test_delegation_start_preserves_arguments(arguments):
    update = build_tool_start("tc-delegate", "delegate_task", arguments)
    assert update.raw_input == arguments
    assert update.content  # Keep the stock human-readable rendering.
    assert update.model_dump(by_alias=True)["rawInput"] == arguments


@pytest.mark.parametrize("result", [
    {"results": [{"task_index": 0, "goal": "Review parser", "model": "test/reviewer",
                  "status": "completed", "duration_seconds": 1.5, "summary": "Reviewed."}]},
    {"results": [{"task_index": 0, "goal": "Inspect routing", "status": "completed", "summary": "OK"},
                 {"task_index": 1, "goal": "Run tests", "status": "error", "error": "Failed"}],
     "total_duration_seconds": 2},
    {"error": "Delegation unavailable"},
])
def test_delegation_complete_preserves_parsed_result(result):
    update = build_tool_complete("tc-delegate", "delegate_task", json.dumps(result))
    assert update.raw_output == result
    assert update.content
    assert update.model_dump(by_alias=True)["rawOutput"] == result
    assert update.status == ("failed" if "error" in result else "completed")


def test_delegation_complete_keeps_non_json_failure():
    update = build_tool_complete("tc-delegate", "delegate_task", "Worker unavailable")
    assert update.raw_output == "Worker unavailable"


def test_other_polished_tools_still_suppress_raw_payloads():
    assert build_tool_start("tc-terminal", "terminal", {"command": "pwd"}).raw_input is None
    assert build_tool_complete("tc-terminal", "terminal", '{"output":"/tmp"}').raw_output is None


def test_delegation_start_preserves_arguments_on_render_failure():
    arguments = {"goal": "Review parser", "model": "test/reviewer"}
    with patch("acp_adapter.tools.build_tool_title", side_effect=ValueError("bad display")):
        update = build_tool_start("tc-delegate", "delegate_task", arguments)
    assert update.raw_input == arguments


@pytest.fixture
def relay_harness():
    ids, meta = {}, {}
    cb = make_tool_progress_cb(None, "session", None, ids, meta)
    with patch("acp_adapter.events._send_update") as send:
        yield cb, make_step_cb(None, "session", None, ids, meta), ids, meta, send


def _start_in_context(cb, arguments):
    from contextvars import Context, copy_context

    def start():
        cb("tool.started", "delegate_task", None, arguments)
        return copy_context()

    return Context().run(start)


def _updates(send):
    return [call.args[3] for call in send.call_args_list]


def test_overlapping_identical_delegations_route_by_inherited_context(relay_harness):
    cb, step, ids, meta, send = relay_harness
    arguments = {"goal": "Same goal"}
    first = _start_in_context(cb, arguments)
    second = _start_in_context(cb, arguments)
    first_id, second_id = ids["delegate_task"]
    send.reset_mock()
    second.run(cb, "subagent.start", None, "Same goal", None,
               task_index=0, goal="Same goal", delegation_id="batch-b", subagent_id="child-b")
    first.run(cb, "subagent.start", None, "Same goal", None,
              task_index=0, goal="Same goal", delegation_id="batch-a", subagent_id="child-a")
    assert [u.tool_call_id for u in _updates(send)] == [second_id, first_id]

    # Dispatch has returned and the original metadata is gone; detached children
    # still own their captured parent context, not the latest outstanding call.
    step(1, [{"name": "delegate_task", "result": '{"status":"dispatched"}'}] * 2)
    assert not ids and not meta
    send.reset_mock()
    first.run(cb, "subagent.complete", None, "Done", None, task_index=0,
              delegation_id="batch-a", subagent_id="child-a", status="completed", summary="Done")
    assert _updates(send)[0].tool_call_id == first_id
    assert _updates(send)[0].status == "in_progress"


def test_contextless_events_require_proven_unambiguous_identity(relay_harness):
    from contextvars import Context

    cb, _, ids, _, send = relay_harness
    first = _start_in_context(cb, {"goal": "Same"})
    second = _start_in_context(cb, {"goal": "Same"})
    first_id = ids["delegate_task"][0]
    send.reset_mock()
    # Matching goals, indices and even parent-shaped args are NOT evidence.
    Context().run(cb, "subagent.tool", "terminal", "preview", {"goal": "Same"},
                  task_index=0, goal="Same", delegation_id="unknown")
    assert not send.called
    first.run(cb, "subagent.start", task_index=0, delegation_id="known", subagent_id="child")
    Context().run(cb, "subagent.text", None, "hello", None, task_index=0, delegation_id="known")
    assert _updates(send)[-1].tool_call_id == first_id
    send.reset_mock()
    second.run(cb, "subagent.start", task_index=0, delegation_id="known", subagent_id="child")
    Context().run(cb, "subagent.text", None, "ambiguous", None, task_index=0, delegation_id="known")
    assert not send.called


def test_per_child_bounded_snapshots_and_allowlisted_metadata(relay_harness):
    cb, _, _, _, send = relay_harness
    context = _start_in_context(cb, {"tasks": [{"goal": "A"}, {"goal": "B"}]})
    send.reset_mock()
    context.run(cb, "subagent.text", None, "hello ", None, task_index=0, goal="A", model="test/a")
    context.run(cb, "subagent.text", None, "other child", None, task_index=1, goal="B", model="test/b")
    context.run(cb, "subagent.text", None, "world", None, task_index=0)
    assert [u.raw_output["hermesDelegation"]["text"] for u in _updates(send)] == [
        "hello ", "other child", "hello world",
    ]
    context.run(cb, "subagent.thinking", None, "thinking", None, task_index=0)
    context.run(cb, "subagent.text", None, "x" * 4000, None, task_index=0)
    assert _updates(send)[-1].raw_output["hermesDelegation"]["text"] == "x" * 2000
    context.run(cb, "subagent.tool", "terminal", "preview", {"secret": "CHILD_SECRET"},
                task_index=0, goal="A", model="test/a", task_count=2, tool_count=1,
                secret="KWARG_SECRET", output_tail="PRIVATE_OUTPUT", files_read=["private-file"],
                toolsets=["private-toolset"])
    payload = _updates(send)[-1].raw_output["hermesDelegation"]
    assert payload == {"event": "subagent.tool", "task_index": 0, "goal": "A", "model": "test/a",
                       "task_count": 2, "tool_count": 1, "tool": "terminal", "text": "preview"}
    context.run(cb, "subagent.complete", None, "Failed", None, task_index=1, status="error",
                duration_seconds=1.25, summary="s" * 3000, error="e" * 3000)
    complete = _updates(send)[-1]
    payload = complete.raw_output["hermesDelegation"]
    assert complete.status == "in_progress"
    assert payload["goal"] == "B" and payload["model"] == "test/b"
    assert payload["duration_seconds"] == 1.25 and payload["status"] == "error"
    assert payload["summary"] == "s" * 2000 and payload["error"] == "e" * 2000


@pytest.mark.parametrize("kwargs", [
    {}, {"task_index": -1}, {"task_index": True}, {"task_index": "0"},
    {"task_index": 0, "depth": 1}, {"task_index": 0, "parent_id": "nested-parent"},
])
def test_missing_or_ambiguous_child_identity_is_dropped(relay_harness, kwargs):
    cb, _, _, _, send = relay_harness
    context = _start_in_context(cb, {"goal": "A"})
    send.reset_mock()
    context.run(cb, "subagent.text", None, "text", None, **kwargs)
    assert not send.called


def test_nonfinite_duration_and_nonscalar_fields_are_not_forwarded(relay_harness):
    cb, _, _, _, send = relay_harness
    context = _start_in_context(cb, {"goal": "A"})
    send.reset_mock()
    for index, duration in enumerate((float("nan"), float("inf"), -1, True, {"secret": "value"})):
        context.run(cb, "subagent.complete", task_index=index, duration_seconds=duration,
                    model={"secret": "value"}, summary=["secret"], delegation_id="x" * 257)
        assert _updates(send)[-1].raw_output["hermesDelegation"] == {
            "event": "subagent.complete", "task_index": index, "text": "",
        }


def test_real_relay_and_thread_context_propagation(relay_harness):
    from concurrent.futures import ThreadPoolExecutor
    from types import SimpleNamespace

    from agent.tool_executor import _ToolCallRef, _begin_tool_execution
    from tools.delegate_tool_progress import _build_child_progress_callback
    from tools.thread_context import propagate_context_to_thread

    cb, _, ids, _, send = relay_harness
    parent = SimpleNamespace(
        tool_progress_callback=cb, tool_start_callback=None, _delegate_spinner=None,
        quiet_mode=True, _touch_activity=lambda _: None, _checkpoint_mgr=SimpleNamespace(enabled=False),
    )

    def invoke():
        _begin_tool_execution(parent, _ToolCallRef("delegate_task", {"goal": "Review"}, "task", "call", []), None)
        relay = _build_child_progress_callback(
            0, "Review", parent, subagent_id="child-0", depth=0, model="test/reviewer",
            session_ref={"delegation_id": "batch-0", "session_id": "child-session"},
        )

        def child():
            relay("subagent.start")
            relay("tool.started", "terminal", "Inspecting", {"secret": "not forwarded"})
            relay("subagent.text", preview="hello ")
            relay("subagent.text", preview="world")
            relay("subagent.complete", preview="Reviewed", status="completed", duration_seconds=2,
                  summary="Reviewed")

        # Match async dispatch -> child-conversation context propagation, using
        # real Hermes helper and real relay (no providers or external services).
        with ThreadPoolExecutor(max_workers=1) as pool:
            pool.submit(propagate_context_to_thread(child)).result()

    with ThreadPoolExecutor(max_workers=1) as pool:
        pool.submit(propagate_context_to_thread(invoke)).result()
    updates = _updates(send)
    parent_id = ids["delegate_task"][0]
    assert all(update.tool_call_id == parent_id for update in updates)
    progress = [u.raw_output["hermesDelegation"] for u in updates[1:]]
    assert [p["event"] for p in progress] == [
        "subagent.start", "subagent.tool", "subagent.text", "subagent.text", "subagent.complete",
    ]
    assert progress[3]["text"] == "hello world"
    assert all(p["child_session_id"] == "child-session" for p in progress)
    assert all(u.status == "in_progress" for u in updates[1:])
    assert "not forwarded" not in json.dumps(progress)


def test_contextless_followup_does_not_establish_new_ownership(relay_harness):
    from contextvars import Context

    cb, _, _, _, send = relay_harness
    context = _start_in_context(cb, {"goal": "A"})
    context.run(cb, "subagent.start", task_index=0, delegation_id="proven")
    Context().run(cb, "subagent.text", None, "known", None, task_index=0,
                  delegation_id="proven", subagent_id="unproven")
    send.reset_mock()
    Context().run(cb, "subagent.text", None, "unknown", None, task_index=0, subagent_id="unproven")
    assert not send.called


def test_callbacks_are_session_and_turn_isolated(relay_harness):
    cb, _, ids, _, send = relay_harness
    old_context = _start_in_context(cb, {"goal": "A"})
    old_id = ids["delegate_task"][0]
    other_ids = {}
    other = make_tool_progress_cb(None, "other-session", None, other_ids, {})
    new_context = _start_in_context(other, {"goal": "A"})
    new_id = other_ids["delegate_task"][0]
    send.reset_mock()
    old_context.run(cb, "subagent.text", None, "old", None, task_index=0, delegation_id="same-id")
    new_context.run(other, "subagent.text", None, "new", None, task_index=0, delegation_id="same-id")
    assert [(c.args[1], c.args[3].tool_call_id) for c in send.call_args_list] == [
        ("session", old_id), ("other-session", new_id),
    ]


def test_child_count_is_bounded(relay_harness):
    cb, _, _, _, send = relay_harness
    context = _start_in_context(cb, {"goal": "A"})
    send.reset_mock()
    for index in range(130):
        context.run(cb, "subagent.text", None, "text", None, task_index=index)
    assert send.call_count == 128


def test_background_dispatch_result_is_preserved_without_finishing_children():
    result = {"status": "dispatched", "mode": "background", "delegation_id": "batch",
              "count": 1, "goals": ["A"], "subagent_ids": ["child"]}
    update = build_tool_complete("tc-delegate", "delegate_task", json.dumps(result))
    assert update.status == "completed"  # The dispatch tool, not its children.
    assert update.raw_output == result


def test_child_terminal_event_cannot_be_reopened_by_late_worker(relay_harness):
    cb, _, _, _, send = relay_harness
    context = _start_in_context(cb, {"goal": "A"})
    context.run(cb, "subagent.complete", task_index=0, subagent_id="child", status="interrupted",
                summary="Registry finalized stalled child")
    send.reset_mock()
    context.run(cb, "subagent.text", None, "late text", None, task_index=0, subagent_id="child")
    context.run(cb, "subagent.complete", task_index=0, subagent_id="child", status="completed")
    assert not send.called


def test_parent_fallback_registry_is_bounded_without_losing_live_context():
    from contextvars import Context, copy_context
    from acp_adapter.events import _DelegationProgress

    progress = _DelegationProgress()

    def start(index):
        progress.start(f"tc-{index}", "delegate_task")
        progress.update("subagent.text", None, "first", {"task_index": 0, "delegation_id": f"batch-{index}"})
        return copy_context()

    old_context = Context().run(start, 0)
    for index in range(1, 300):
        Context().run(start, index)
    assert len(progress.parents) == 256
    assert len(progress.bindings) == 256
    assert "tc-0" not in progress.parents
    assert ("delegation_id", "batch-0") not in progress.bindings

    # The fallback forgets evicted parents rather than attaching to the newest
    # invocation. The real background worker still owns its copied context.
    assert Context().run(progress.update, "subagent.text", None, "unknown", {
        "task_index": 0, "delegation_id": "batch-0",
    }) is None
    update = old_context.run(progress.update, "subagent.text", None, " second", {
        "task_index": 0, "delegation_id": "batch-0",
    })
    assert update.tool_call_id == "tc-0"
    assert update.raw_output["hermesDelegation"]["text"] == "first second"
    assert len(progress.parents) == 256 and len(progress.bindings) == 256
    assert ("delegation_id", "batch-0") not in progress.bindings

    update = Context().run(progress.update, "subagent.text", None, " latest", {
        "task_index": 0, "delegation_id": "batch-299",
    })
    assert update.tool_call_id == "tc-299"
    assert update.raw_output["hermesDelegation"]["text"] == "first latest"
    # Defensive handling of a stale alias must never raise KeyError.
    progress.bindings[("delegation_id", "stale")] = {"tc-0"}
    assert Context().run(progress.update, "subagent.text", None, "drop", {
        "task_index": 0, "delegation_id": "stale",
    }) is None


def test_parent_eviction_drops_ambiguous_binding_instead_of_guessing_survivor():
    from contextvars import Context
    from acp_adapter.events import _DelegationProgress

    progress = _DelegationProgress()
    def start(index):
        progress.start(f"tc-{index}", "delegate_task")
        if index < 2:
            progress.update("subagent.start", None, "", {"task_index": 0, "delegation_id": "conflict"})
    for index in range(257):
        Context().run(start, index)
    assert "tc-1" in progress.parents and "tc-0" not in progress.parents
    assert ("delegation_id", "conflict") not in progress.bindings
    assert Context().run(progress.update, "subagent.text", None, "ambiguous", {
        "task_index": 0, "delegation_id": "conflict",
    }) is None


def test_executor_tool_completed_closes_parent_with_structured_result(relay_harness):
    cb, _, ids, meta, send = relay_harness
    arguments = {"goal": "A", "model": "test/model"}
    result = {"status": "dispatched", "mode": "background", "delegation_id": "batch", "subagent_ids": ["child"]}
    context = _start_in_context(cb, arguments)
    parent_id = ids["delegate_task"][0]
    send.reset_mock()
    # Upstream closes the call from the executor's ``tool.completed``, outside the delegation context.
    cb("tool.completed", "delegate_task", None, None, result=json.dumps(result), is_error=False)
    closed, = _updates(send)
    assert closed.tool_call_id == parent_id and closed.status == "completed"
    assert closed.raw_output == result and not ids and not meta
    send.reset_mock()
    context.run(cb, "subagent.text", None, "still live", None, task_index=0, delegation_id="batch")
    assert _updates(send)[0].tool_call_id == parent_id
    assert _updates(send)[0].status == "in_progress"
