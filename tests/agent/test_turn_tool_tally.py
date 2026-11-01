"""Cron refusal evidence comes from real turns, including middleware replacements (#135544)."""
import json
from copy import deepcopy
from types import SimpleNamespace
from threading import Event, Thread
from unittest.mock import MagicMock

import pytest

from tests.agent.test_run_agent import _mock_response, _mock_tool_call
from tests.agent.test_turn_user_intervened import _PROBE, _loop_agent, _run


@pytest.mark.parametrize("policy", ["plugin", "middleware", "allowed"])
def test_turn_counts_attempts_and_real_dispatch(monkeypatch, policy):
    agent = _loop_agent()
    dispatch = MagicMock(return_value="probe ok")
    monkeypatch.setattr("model_tools.handle_function_call", dispatch)
    if policy == "plugin":
        monkeypatch.setattr("hermes_cli.plugins._dispatch_pre_tool_call_hooks",
                            lambda *_a, **_k: ("HTTP 401 invalid API key", None))
    elif policy == "middleware":
        from hermes_cli.plugins import get_plugin_manager
        monkeypatch.setattr(get_plugin_manager(), "_middleware",
                            {"tool_request": [], "tool_execution": [lambda **kw: "refused"]})
    replies = MagicMock(side_effect=[
        _mock_response(content=None, finish_reason="tool_calls", tool_calls=[_mock_tool_call(name=_PROBE)]),
        _mock_response(content="done"),
    ])
    result = _run(agent, replies)
    assert result["completed"] is True
    assert result["tool_calls_attempted"] == 1
    assert result["tool_calls_refused"] == (0 if policy == "allowed" else 1)
    assert result["tool_calls_refused_names"] == ([] if policy == "allowed" else [_PROBE])
    assert dispatch.call_count == (1 if policy == "allowed" else 0)
    # Cached agents start each turn clean, including a text-only turn after a refusal.
    next_result = _run(agent, lambda _: _mock_response(content="hello"))
    assert not any(key.startswith("tool_calls_") for key in next_result)


def test_text_only_turn_has_no_tally():
    result = _run(_loop_agent(), lambda _: _mock_response(content="hello"))
    assert result["completed"] is True
    assert not any(key.startswith("tool_calls_") for key in result)


@pytest.mark.parametrize("finish", ["refused", "exception"])
def test_stale_worker_updates_only_its_original_tally(finish):
    from agent import turn_tool_tally as tally
    agent = SimpleNamespace()
    tally.reset(agent)
    original = agent._turn_tool_tally
    started, release = Event(), Event()
    errors = []

    @tally.track_attempt
    def run(_agent, **kwargs):
        started.set()
        assert release.wait(5)
        if finish == "exception":
            raise RuntimeError("middleware failed")
        return SimpleNamespace(blocked=True, dispatched=False, result="refused")

    def worker():
        try:
            run(agent, function_name="terminal")
        except RuntimeError as exc:
            errors.append(str(exc))

    thread = Thread(target=worker)
    thread.start()
    try:
        assert started.wait(5)
        tally.reset(agent)
        current = deepcopy(agent._turn_tool_tally)
    finally:
        release.set()
        thread.join(5)
    assert not thread.is_alive()
    assert agent._turn_tool_tally == current
    assert original["attempted"] == 1
    assert original["refused"] == (1 if finish == "refused" else 0)
    assert errors == (["middleware failed"] if finish == "exception" else [])


def test_exception_and_abandoned_attempt_are_not_positive_refusals():
    from agent import turn_tool_tally as tally
    from cron.scheduler_turn_verdict import refused_turn_error
    agent = SimpleNamespace()
    tally.reset(agent)
    started, release = Event(), Event()

    @tally.track_attempt
    def run(_agent, **kwargs):
        started.set()
        assert release.wait(5)
        raise RuntimeError("middleware failed")

    def worker():
        with pytest.raises(RuntimeError, match="middleware failed"):
            run(agent, function_name="terminal")

    thread = Thread(target=worker)
    thread.start()
    try:
        assert started.wait(5)
        result = {}
        tally.export(agent, result)
        assert result["tool_calls_attempted"] == 1
        assert refused_turn_error(result, "done") is None
    finally:
        release.set()
        thread.join(5)
    result = {}
    tally.export(agent, result)
    assert refused_turn_error(result, "done") is None


@pytest.mark.parametrize("output, refused", [
    ('{"output":"","exit_code":-1,"error":"denied","status":"blocked"}', True),
    (' {"output":"","exit_code":-1,"error":"","status":"pending_approval"}', True),
    ('{"output":"","exit_code":-1,"error":"","approval_pending":true}', True),
    ('{"status":"error","exit_code":1}', False),
    (json.dumps({"output": '{"status":"blocked"}', "exit_code": 1}), False),
    ('{"approval_pending":false}', False), ('["blocked"]', False),
    ('{"status":', False), ('ordinary output', False),
    ('{"status":"blocked","padding":"' + 'x' * 65536 + '"}', False),
], ids=["blocked", "pending", "flag", "error", "nested", "false", "array", "invalid", "text", "oversized"])
def test_structured_guard_refusals_only(output, refused):
    from agent import turn_tool_tally as tally
    agent = SimpleNamespace()
    tally.reset(agent)

    @tally.track_attempt
    def run(_agent, **kwargs):
        return SimpleNamespace(blocked=False, dispatched=True, result=output)

    run(agent, function_name="terminal")
    result = {}
    tally.export(agent, result)
    assert result["tool_calls_refused"] == int(refused)


@pytest.mark.parametrize("function_name, body", [
    ("plugin_record", {"status": "blocked"}),
    ("plugin_record", {"output": "", "exit_code": -1, "error": "denied", "status": "blocked"}),
    ("terminal", {"status": "blocked"}),
    ("terminal", {"exit_code": -1, "error": "denied", "status": "blocked"}),
    ("terminal", {"output": "", "error": "denied", "status": "blocked"}),
    ("terminal", {"output": "", "exit_code": -1, "status": "blocked"}),
    ("terminal", {"output": "ran", "exit_code": -1, "error": "failed", "status": "blocked"}),
    ("terminal", {"output": "", "exit_code": 1, "error": "failed", "status": "blocked"}),
    ("terminal", {"output": "", "exit_code": 0, "error": None, "approval_pending": True}),
], ids=["plugin-status", "plugin-envelope", "status-only", "missing-output", "missing-exit",
        "missing-error", "ran-output", "ran-nonzero", "ran-zero"])
def test_dispatched_results_outside_terminal_guard_envelope_are_not_refused(function_name, body):
    from agent import turn_tool_tally as tally
    from cron.scheduler_turn_verdict import refused_turn_error
    agent = SimpleNamespace()
    tally.reset(agent)

    @tally.track_attempt
    def run(_agent, **kwargs):
        return SimpleNamespace(blocked=False, dispatched=True, result=json.dumps(body))

    run(agent, function_name=function_name)
    result = {}
    tally.export(agent, result)
    assert result["tool_calls_attempted"] == 1
    assert result["tool_calls_refused"] == 0
    assert refused_turn_error(result, "done") is None
