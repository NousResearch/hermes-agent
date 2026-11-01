"""``tool_decision``: exactly one structured allow/refuse event per tool call, never raw arguments."""

import hashlib
import json
import logging
import threading
from collections import OrderedDict
from functools import partial
from types import SimpleNamespace
from unittest.mock import patch

import pytest

import model_tools
from agent import tool_decision
from agent.inline_tool_executors import emit_terminal_post_tool_call
from tools import approval, approval_prompt
from tools import approval_context

SECRET = "sk-live-do-not-leak-4242"
ARGS = {"path": f"/tmp/{SECRET}.txt", "nested": {"b": 2, "a": [1, SECRET]}}
IDS = dict(task_id="task-1", tool_call_id="call-1", session_id="sess-1", turn_id="turn-1")


def _digest(args):
    canonical = json.dumps(args, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


@pytest.fixture
def hooks(monkeypatch):
    """Capture tool_decision events; ``rec.pre`` is what pre_tool_call subscribers return."""
    from hermes_cli import lifecycle

    rec = SimpleNamespace(decisions=[], pre=[], fail=False, events=[])

    def fake_invoke(name, **kwargs):
        rec.events.append((name, kwargs))
        if name == "tool_decision":
            if rec.fail:
                raise RuntimeError("subscriber exploded")
            rec.decisions.append(kwargs)
        return list(rec.pre) if name == "pre_tool_call" else []

    monkeypatch.setattr(lifecycle, "invoke_hook", fake_invoke)
    monkeypatch.setattr(lifecycle, "has_hook", lambda name: name in ("tool_decision", "pre_tool_call"))
    monkeypatch.setattr(tool_decision, "_pending", OrderedDict())
    monkeypatch.setattr(tool_decision, "_emitted", OrderedDict())
    monkeypatch.setattr(tool_decision, "_abandoned", OrderedDict())
    yield rec
    # Raw arguments never reach a subscriber, on any path.
    assert SECRET not in repr(rec.decisions)


@pytest.fixture
def cli_prompt(monkeypatch):
    """An interactive CLI approver whose answer the test sets; nothing pre-approved, no yolo."""
    answer = SimpleNamespace(choice="deny")
    for module in (approval, approval_context):
        monkeypatch.setattr(module, "get_current_session_key", lambda default="default": "test-session")
        monkeypatch.setattr(module, "_is_gateway_approval_context", lambda: False)
    monkeypatch.setattr(approval, "is_approved", lambda sk, pk: False)
    monkeypatch.setattr(approval, "_yolo_active", lambda: False)
    monkeypatch.setattr(approval, "_is_interactive_cli", lambda: True)
    monkeypatch.setattr("tools.terminal_tool._get_approval_callback", lambda: None, raising=False)
    for module in (approval, approval_prompt):
        monkeypatch.setattr(module, "prompt_dangerous_approval", lambda *a, **k: answer.choice)
    return answer


def _call(name="read_file", args=None, dispatch=None, ids=None):
    with patch("model_tools.registry.dispatch", side_effect=dispatch or (lambda *a, **k: '{"ok": true}')) as mock:
        result = model_tools.handle_function_call(name, dict(args or ARGS), **(ids or IDS))
    return result, mock


def test_plain_allowed_call_emits_one_allowed_event(hooks):
    _call()
    [event] = hooks.decisions
    assert event["verdict"] == "allowed"
    assert event["tool_name"] == "read_file"
    assert event["args_sha256"] == _digest(ARGS)
    assert {k: event[k] for k in IDS} == IDS
    assert event["initiator"] == "session"
    assert "args" not in event


def test_hook_blocked_call_emits_one_blocked_by_hook_event(hooks):
    hooks.pre = [{"action": "block", "message": "not today"}]
    result, dispatch = _call()
    dispatch.assert_not_called()
    assert "not today" in result
    [event] = hooks.decisions
    assert (event["verdict"], event["source"]) == ("blocked_by_hook", "pre_tool_call")


def test_user_denied_approval_emits_one_denied_by_user_event(hooks, cli_prompt):
    hooks.pre = [{"action": "approve", "message": "needs a human"}]
    cli_prompt.choice = "deny"
    result, dispatch = _call()
    dispatch.assert_not_called()
    assert "BLOCKED" in result
    [event] = hooks.decisions
    assert (event["verdict"], event["source"]) == ("denied_by_user", "approval_prompt")


def test_user_approved_call_emits_one_user_approved_event(hooks, cli_prompt):
    hooks.pre = [{"action": "approve", "message": "needs a human"}]
    cli_prompt.choice = "once"
    _, dispatch = _call()
    dispatch.assert_called_once()
    [event] = hooks.decisions
    assert (event["verdict"], event["source"]) == ("user_approved", "approval_prompt")


def test_auto_approved_call_emits_one_auto_approved_event(hooks, cli_prompt, monkeypatch):
    monkeypatch.setattr(approval, "is_approved", lambda sk, pk: True)  # session/permanent allowlist hit
    hooks.pre = [{"action": "approve", "message": "needs a human"}]
    _, dispatch = _call()
    dispatch.assert_called_once()
    [event] = hooks.decisions
    assert (event["verdict"], event["source"]) == ("auto_approved", "allowlist")


def test_deny_rule_inside_the_handler_emits_one_denied_by_rule_event(hooks, monkeypatch):
    monkeypatch.setattr(approval, "_match_user_deny_rule", lambda command: "git push --force*")

    def terminal_handler(name, args, **kwargs):
        verdict = approval.check_all_command_guards(args["command"], "local")
        return json.dumps({"error": verdict["message"]}) if not verdict["approved"] else '{"ok": true}'

    _call("terminal", {"command": f"git push --force {SECRET}"}, dispatch=terminal_handler)
    [event] = hooks.decisions
    assert (event["verdict"], event["source"]) == ("denied_by_rule", "approvals.deny")


def test_executor_owned_terminal_event_is_the_only_one(hooks, monkeypatch):
    """Sequential executor path: the inner post hook is suppressed, the executor emits once, and the
    verdict a gate noted during the suppressed dispatch is still the one reported."""
    monkeypatch.setattr(approval, "_match_user_deny_rule", lambda command: "rm*")

    def terminal_handler(name, args, **kwargs):
        approval.check_all_command_guards(args["command"], "local")
        return '{"error": "blocked"}'

    ids = dict(IDS, turn_id=tool_decision.DispatchTurn(IDS["turn_id"]))
    with model_tools.suppress_post_tool_call_hook():
        _call("terminal", {"command": "rm notes.txt"}, dispatch=terminal_handler, ids=ids)
    assert hooks.decisions == []
    agent = SimpleNamespace(session_id=IDS["session_id"], _current_turn_id=IDS["turn_id"], _current_api_request_id="")
    emit_terminal_post_tool_call(agent, function_name="terminal", function_args={"command": "rm notes.txt"},
                                 result='{"error": "blocked"}', effective_task_id=IDS["task_id"],
                                 tool_call_id=IDS["tool_call_id"], dispatch_turn_id=ids["turn_id"])
    [event] = hooks.decisions
    assert event["verdict"] == "denied_by_rule"
    assert tool_decision._pending == {}


def _agent():
    return SimpleNamespace(session_id=IDS["session_id"], _current_turn_id=IDS["turn_id"], _current_api_request_id="")


def _terminal_emit(**outcome):
    emit_terminal_post_tool_call(_agent(), function_name="write_file", function_args={"path": "a"}, result="r",
                                 effective_task_id=IDS["task_id"], tool_call_id=IDS["tool_call_id"], **outcome)


def test_a_second_terminal_emission_for_the_same_call_is_dropped(hooks):
    """A worker finishing after its timeout result was published must not report the call twice."""
    _terminal_emit(status="timeout", error_type="tool_timeout")
    _terminal_emit()
    [event] = hooks.decisions
    assert event["verdict"] == "abandoned"


def test_abandoned_call_is_not_re_reported_when_its_worker_settles_late(hooks, monkeypatch):
    """The late worker may report in a later turn and after many other calls have been reported."""
    monkeypatch.setattr(tool_decision, "_MAX_EMITTED", 2)
    _terminal_emit(status="timeout", error_type="tool_timeout")
    agent = _agent()
    for n in range(5):  # evicts the abandoned call from the general repeat table
        emit_terminal_post_tool_call(agent, function_name="read_file", function_args={}, result="r",
                                     effective_task_id="task-1", tool_call_id=f"other-{n}")
    agent._current_turn_id = "turn-2"
    emit_terminal_post_tool_call(agent, function_name="write_file", function_args={"path": "a"}, result="r",
                                 effective_task_id=IDS["task_id"], tool_call_id=IDS["tool_call_id"],
                                 dispatch_turn_id=IDS["turn_id"])
    assert [e["tool_call_id"] for e in hooks.decisions].count(IDS["tool_call_id"]) == 1
    assert len(hooks.decisions) == 6


def test_dispatch_without_a_tool_call_id_fires_nothing(hooks, monkeypatch):
    """execute_code RPC callbacks dispatch with only a task id: their verdicts cannot be attributed,
    so they report nothing rather than a false ``allowed``."""
    monkeypatch.setattr(approval, "_match_user_deny_rule", lambda command: "rm*")

    def terminal_handler(name, args, **kwargs):
        approval.check_all_command_guards(args["command"], "local")
        return '{"error": "blocked"}'

    with patch("model_tools.registry.dispatch", side_effect=terminal_handler):
        model_tools.handle_function_call("terminal", {"command": "rm notes.txt"}, task_id="task-1")
    assert hooks.decisions == []
    assert tool_decision._pending == {}


def test_block_inside_a_suppressed_dispatch_reaches_the_outer_emission(hooks):
    """Sequential path: an ACP edit denial inside handle_function_call is reported by the executor's
    single terminal emission, which carries no blocked status of its own."""
    ids = dict(IDS, turn_id=tool_decision.DispatchTurn(IDS["turn_id"]))
    with patch("acp_adapter.edit_approval.maybe_require_edit_approval", return_value='{"error": "rejected"}'), \
            model_tools.suppress_post_tool_call_hook():
        _, dispatch = _call("write_file", {"path": "a", "content": SECRET}, ids=ids)
    dispatch.assert_not_called()
    assert hooks.decisions == []
    _terminal_emit(dispatch_turn_id=ids["turn_id"])
    [event] = hooks.decisions
    assert (event["verdict"], event["source"]) == ("denied_by_user", "acp_edit_approval")


def test_notes_made_while_deferred_are_dropped(hooks):
    tokens = approval_context.set_current_observability_context(tool_call_id="call-1", session_id="sess-1")
    try:
        with tool_decision.notes_deferred():
            tool_decision.note("denied_by_user", "approval_prompt")
        assert tool_decision._pending == {}
        tool_decision.note("user_approved", "approval_prompt")
    finally:
        approval_context.reset_current_observability_context(tokens)
    assert tool_decision._pending == {("sess-1", "call-1", ""): ("user_approved", "approval_prompt")}


def test_call_skipped_by_interrupt_emits_no_decision(hooks):
    agent = SimpleNamespace(session_id="sess-1", _current_turn_id="turn-1", _current_api_request_id="")
    emit_terminal_post_tool_call(agent, function_name="read_file", function_args={}, result="skipped",
                                 effective_task_id="task-1", tool_call_id="call-9", status="cancelled",
                                 error_type="user_interrupt")
    assert hooks.decisions == []


def test_failing_subscriber_never_breaks_the_call(hooks, caplog):
    hooks.fail = True
    with caplog.at_level(logging.WARNING, logger="agent.tool_decision"):
        result, dispatch = _call()
    dispatch.assert_called_once()
    assert json.loads(result) == {"ok": True}
    [record] = [r for r in caplog.records if r.name == "agent.tool_decision"]
    assert record.exc_info is not None


def test_no_subscriber_records_nothing(monkeypatch):
    from hermes_cli import lifecycle

    monkeypatch.setattr(lifecycle, "has_hook", lambda name: False)
    monkeypatch.setattr(tool_decision, "_pending", OrderedDict())
    tokens = approval_context.set_current_observability_context(tool_call_id="call-1", session_id="sess-1")
    try:
        tool_decision.note("denied_by_user", "approval_prompt")
    finally:
        approval_context.reset_current_observability_context(tokens)
    assert tool_decision._pending == {}


def test_a_refusal_outranks_an_earlier_approval_for_the_same_call():
    merged = tool_decision._merge(("auto_approved", "allowlist"), ("denied_by_user", "approval_prompt"))
    assert merged == ("denied_by_user", "approval_prompt")
    assert tool_decision._merge(merged, ("user_approved", "approval_prompt")) == merged
    assert tool_decision._merge(("auto_approved", "allowlist"), ("user_approved", "approval_prompt"))[0] == "user_approved"


def test_reused_call_id_in_a_new_dispatch_turn_is_not_an_old_abandoned_call(hooks):
    ids = dict(IDS, tool_call_id="call_0", turn_id=tool_decision.DispatchTurn("turn-1"))  # ids recur after compaction
    model_tools._emit_post_tool_call_hook(function_name="read_file", function_args={}, result="timeout",
                                         status="timeout", error_type="tool_timeout", **ids)
    with patch.object(model_tools.registry, "dispatch", return_value="ok"):
        model_tools.handle_function_call("read_file", {}, **dict(ids, turn_id="turn-2"))
    assert [e["turn_id"] for e in hooks.decisions] == ["turn-1", "turn-2"]
    assert ("sess-1", "call_0", ids["turn_id"].dispatch_id) in tool_decision._abandoned
    model_tools._emit_post_tool_call_hook(function_name="read_file", function_args={}, result="late", **ids)
    assert len(hooks.decisions) == 2


@pytest.mark.parametrize("name", ["read_file", "todo_list"])
def test_timeout_and_late_concurrent_worker_keep_the_dispatch_turn(hooks, monkeypatch, name):
    """A queued worker starts after T2; both its event attempt and timeout still carry T1."""
    from agent import tool_executor as te
    from agent.inline_tool_executors import INLINE_TOOL_EXECUTORS
    from agent.agent_runtime_helpers import invoke_tool

    agent = _agent()
    agent._interrupt_requested = False
    agent._tool_worker_threads_lock = threading.Lock()
    agent._tool_worker_threads = set()
    agent._memory_manager = None
    agent.valid_tool_names = {name}
    agent._touch_activity = lambda *args: None
    agent._invoke_tool = partial(invoke_tool, agent)
    pc = te._ParsedCall(SimpleNamespace(id="call_0"), name, {}, [], None, None)
    batch = te._ConcurrentBatch(agent, [], "task-1", [pc], 1.0)
    queued = []
    executor = SimpleNamespace(submit=lambda fn, *args: queued.append(partial(fn, *args)))

    def execute_middleware(_agent, *, execute, function_args, **kwargs):
        return te._ManagedToolResult(result=execute(function_args), args=function_args, middleware_trace=[],
                                     blocked=False, dispatched=True)

    monkeypatch.setattr(te, "_run_agent_tool_execution_middleware", execute_middleware)
    monkeypatch.setattr(te, "_commit_tool_result", lambda *args, **kwargs: None)
    monkeypatch.setitem(INLINE_TOOL_EXECUTORS, "todo_list", lambda *args: "ok")
    with patch.object(model_tools.registry, "dispatch", return_value="ok"), \
            patch.object(tool_decision, "emit_for_call", wraps=tool_decision.emit_for_call) as attempts, \
            patch.object(model_tools, "_emit_post_tool_call_hook", wraps=model_tools._emit_post_tool_call_hook) as posts:
        batch.submit_all(executor, [0])
        batch.timed_out_indices = {0}
        te._append_batch_results(agent, [], "task-1", batch, None)
        agent._current_turn_id = "turn-2"
        queued[0]()
    assert [c.kwargs["turn_id"] for c in attempts.call_args_list] == ["turn-1", "turn-1"]
    assert [c.kwargs["turn_id"] for c in posts.call_args_list] == ["turn-1", "turn-1"]
    [event] = hooks.decisions
    assert (event["turn_id"], event["verdict"], event["source"]) == ("turn-1", "abandoned", "tool_timeout")


@pytest.mark.parametrize("error_type", ["tool_timeout", "tool_interrupted", "keyboard_interrupt", "thread_missing_result"])
def test_abandoned_calls_report_the_reason_unless_a_gate_refused(hooks, error_type):
    _terminal_emit(status="error", error_type=error_type)
    [event] = hooks.decisions
    assert (event["verdict"], event["source"]) == ("abandoned", error_type)
    hooks.decisions.clear()
    tool_decision._emitted.clear()
    tool_decision._abandoned.clear()
    tokens = approval_context.set_current_observability_context(**{k: IDS[k] for k in ("session_id", "tool_call_id", "turn_id")})
    try:
        tool_decision.note("denied_by_user", "approval_prompt")
    finally:
        approval_context.reset_current_observability_context(tokens)
    _terminal_emit(status="error", error_type=error_type)
    assert hooks.decisions[0]["verdict"] == "denied_by_user"


@pytest.mark.parametrize("context", ["cron", "single_query"])
@pytest.mark.parametrize("dangerous", [True, False])
def test_unattended_approve_reports_only_flagged_commands_as_auto_approved(hooks, monkeypatch, context, dangerous):
    from gateway.session_context import clear_session_vars, set_session_vars

    monkeypatch.setattr("hermes_cli.config.load_config_readonly", lambda: {"approvals": {f"{context}_mode": "approve"}})
    monkeypatch.setattr(approval, "_yolo_active", lambda: False)
    monkeypatch.setattr(approval, "_command_matches_permanent_allowlist", lambda command: False)
    monkeypatch.setattr(approval, "_presence", lambda cb: (None, False, False, False))
    monkeypatch.setenv("HERMES_SINGLE_QUERY_SESSION", "1" if context == "single_query" else "")
    tokens = set_session_vars(cron_session="1" if context == "cron" else "")

    def handler(name, args, **kwargs):
        assert approval.check_all_command_guards(args["command"], "local")["approved"]
        return "ok"

    try:
        _call("terminal", {"command": "rm -rf ./fixture" if dangerous else "echo ok"}, handler)
    finally:
        clear_session_vars(tokens)
    [event] = hooks.decisions
    expected = ("auto_approved", "unattended_approve") if dangerous else ("allowed", "")
    assert (event["verdict"], event["source"]) == expected


@pytest.mark.parametrize("circular", [True, False])
def test_unserializable_arguments_still_emit_a_digest(hooks, circular):
    args = {"value": SECRET} if circular else {1: SECRET, "value": 2}
    if circular:
        args["self"] = args
    model_tools._emit_post_tool_call_hook(function_name="read_file", function_args=args, result="ok", **IDS)
    [event] = hooks.decisions
    assert event["args_sha256"] == hashlib.sha256(repr(args).encode("utf-8")).hexdigest()


def _dispatch_ref(turn_id="turn-1"):
    from agent.tool_executor import _ParsedCall

    pc = _ParsedCall(SimpleNamespace(id="call-1"), "read_file", {}, [], None, None)
    return pc.ref("task-1", turn_id)


def _note_for_ref(ref, verdict="denied_by_user"):
    ids = model_tools._CallIds(ref.task_id, "sess-1", ref.call_id, ref.dispatch_turn_id)
    with model_tools._approval_observability(ids):
        tool_decision.note(verdict, "approval_prompt")


def test_repeated_id_after_compaction_in_one_turn_emits_both_dispatches(hooks):
    messages = [{"role": "assistant", "tool_calls": [{"id": "call-1"}]}]
    _call()
    messages.clear()  # compaction removed the old id; the provider can reuse it
    _call()
    assert [event["turn_id"] for event in hooks.decisions] == ["turn-1", "turn-1"]
    assert all(type(event["turn_id"]) is str and "dispatch_id" not in event for event in hooks.decisions)


@pytest.mark.parametrize("next_turn", ["turn-1", "turn-2"])
def test_timeout_late_worker_and_reused_id_are_distinct_dispatches(hooks, next_turn):
    old = _dispatch_ref()
    old.emit_post(_agent(), "timeout", status="timeout", error_type="tool_timeout")
    current = _dispatch_ref(next_turn)
    current.emit_post(_agent(), "ok")
    old.emit_post(_agent(), "late")
    assert [(e["turn_id"], e["verdict"]) for e in hooks.decisions] == [
        ("turn-1", "abandoned"), (next_turn, "allowed"),
    ]


def test_late_abandoned_refusal_does_not_contaminate_reused_id(hooks):
    old = _dispatch_ref()
    old.emit_post(_agent(), "timeout", status="timeout", error_type="tool_timeout")
    _note_for_ref(old)
    current = _dispatch_ref("turn-2")
    current.emit_post(_agent(), "ok")
    old.emit_post(_agent(), "late")
    assert [e["verdict"] for e in hooks.decisions] == ["abandoned", "allowed"]


def test_late_abandoned_worker_cannot_consume_new_dispatch_denial(hooks):
    old = _dispatch_ref()
    old.emit_post(_agent(), "timeout", status="timeout", error_type="tool_timeout")
    current = _dispatch_ref("turn-2")
    _note_for_ref(current)
    old.emit_post(_agent(), "late")  # dropped before consuming any gate note
    current.emit_post(_agent(), "blocked")
    assert [e["verdict"] for e in hooks.decisions] == ["abandoned", "denied_by_user"]


def test_unattended_approve_does_not_detect_danger_without_subscriber(monkeypatch):
    monkeypatch.setattr(tool_decision, "_has_subscriber", lambda: False)
    monkeypatch.setattr(approval, "_floor_block", lambda *a, **k: None)
    monkeypatch.setattr(approval, "_yolo_active", lambda: False)
    monkeypatch.setattr(approval_context, "_get_approval_mode", lambda: "manual")
    monkeypatch.setattr(approval, "_command_matches_permanent_allowlist", lambda command: False)
    monkeypatch.setattr(approval, "_presence", lambda cb: (None, False, False, False))
    monkeypatch.setattr(approval, "_unattended_contexts", lambda: ["cron"])
    monkeypatch.setattr(approval, "_unattended_deny", lambda *a: None)
    with patch.object(approval, "detect_dangerous_command", return_value=(True, "fixture", "fixture")) as detect:
        assert approval.check_all_command_guards("fixture command", "local")["approved"]
    detect.assert_not_called()


@pytest.mark.parametrize("turn_id", ["", "turn-1"])
def test_dispatch_notes_survive_empty_turn_and_stay_private(hooks, cli_prompt, turn_id, monkeypatch):
    monkeypatch.setattr("hermes_cli.lifecycle.has_hook", lambda name: True)
    hooks.pre = [{"action": "approve", "message": "needs a human"}]
    cli_prompt.choice = "once"
    for _ in range(2):
        _call(ids=dict(IDS, turn_id=turn_id))
    assert [(e["turn_id"], e["verdict"]) for e in hooks.decisions] == [
        (turn_id, "user_approved"), (turn_id, "user_approved"),
    ]
    assert all(type(e["turn_id"]) is str and "dispatch_id" not in e for _, e in hooks.events)
    assert tool_decision._pending == {}
