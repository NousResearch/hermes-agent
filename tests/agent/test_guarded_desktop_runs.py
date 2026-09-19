"""Guarded ordered desktop runs — RFC #112639 revised critical path.

Behavior contracts (not snapshots): admission decides a bounded, single-target
run of computer_use inputs; the duplicate remover spares admitted repeats while
still collapsing duplicates elsewhere; the sequential executor stops the run at
the first step whose effect is unconfirmed and skips the rest.
"""

from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from agent.guarded_desktop_runs import (
    find_guarded_desktop_runs,
    guarded_run_continues,
    guarded_run_preserve_indices,
    guarded_run_stop,
)
from run_agent import AIAgent


def _tc(action=None, tool="computer_use", call_id=None, **args):
    payload = dict(args)
    if action is not None:
        payload["action"] = action
    return SimpleNamespace(
        id=call_id or f"call-{tool}-{action or 'x'}",
        function=SimpleNamespace(name=tool, arguments=json.dumps(payload)),
    )


def _verdict(decision, text=""):
    return json.dumps({"verdict": {"decision": decision}, "text": text})


def _managed(result, blocked=False):
    return SimpleNamespace(result=result, blocked=blocked, args={}, middleware_trace=[])


# -- admission -----------------------------------------------------------


class TestAdmission:
    def test_keyboard_run_one_target(self):
        calls = [
            _tc("click", element=3, call_id="a"),
            _tc("type", text="hi", call_id="b"),
            _tc("key", key="Tab", call_id="c"),
            _tc("key", key="Tab", call_id="d"),
        ]
        assert find_guarded_desktop_runs(calls) == [(0, 4)]

    def test_single_call_not_a_run(self):
        assert find_guarded_desktop_runs([_tc("type", text="hi", call_id="a")]) == []

    def test_mixed_targets_not_admitted(self):
        calls = [
            _tc("click", app="Safari", call_id="a"),
            _tc("type", app="Finder", text="x", call_id="b"),
        ]
        assert find_guarded_desktop_runs(calls) == []

    def test_second_element_use_ends_run(self):
        calls = [
            _tc("click", element=3, call_id="a"),
            _tc("type", text="x", call_id="b"),
            _tc("click", element=7, call_id="c"),
        ]
        assert find_guarded_desktop_runs(calls) == [(0, 2)]

    def test_run_capped_at_four_ops(self):
        calls = [_tc("key", key="Tab", call_id=f"a{i}") for i in range(5)]
        assert find_guarded_desktop_runs(calls) == [(0, 4)]

    def test_observation_breaks_run(self):
        calls = [
            _tc("click", call_id="a"),
            _tc("capture", call_id="b"),
            _tc("type", text="x", call_id="c"),
        ]
        assert find_guarded_desktop_runs(calls) == []

    def test_preserve_indices_covers_run_only(self):
        calls = [
            _tc("key", key="Tab", call_id="a"),
            _tc("key", key="Tab", call_id="b"),
            _tc("click", call_id="c"),
            _tc("web_search", call_id="d", query="x"),
        ]
        assert guarded_run_preserve_indices(calls) == frozenset({0, 1, 2})


# -- continuation rule ---------------------------------------------------


class TestContinuation:
    def test_done_verdict_continues(self):
        assert guarded_run_continues(_managed(_verdict("done")))[0] is True

    @pytest.mark.parametrize(
        "decision", ["verify_fresh_state", "escalate", "suspected_noop", "rejected", "unknown"]
    )
    def test_unconfirmed_verdict_stops(self, decision):
        ok, _ = guarded_run_continues(_managed(_verdict(decision)))
        assert ok is False

    def test_missing_verdict_stops(self):
        assert guarded_run_continues(_managed("not json at all"))[0] is False

    def test_blocked_stops(self):
        assert guarded_run_continues(_managed(_verdict("done"), blocked=True))[0] is False

    def test_timeout_result_stops(self):
        class _ToolTimeoutResult:
            pass
        assert guarded_run_continues(_managed(_ToolTimeoutResult()))[0] is False

    def test_stop_fires_only_inside_run(self):
        calls = [_tc("key", key="Tab", call_id=f"a{i}") for i in range(3)]
        assert guarded_run_stop(calls, 2, _managed(_verdict("verify_fresh_state"))) is None

    def test_stop_reports_run_end_and_reason(self):
        calls = [_tc("key", key="Tab", call_id=f"a{i}") for i in range(3)]
        end, reason = guarded_run_stop(calls, 0, _managed(_verdict("verify_fresh_state")))
        assert end == 3 and reason

    def test_stop_none_when_confirmed(self):
        calls = [_tc("key", key="Tab", call_id=f"a{i}") for i in range(3)]
        assert guarded_run_stop(calls, 0, _managed(_verdict("done"))) is None


# -- duplicate preservation -----------------------------------------------


class TestDedup:
    def test_preserves_intentional_repeats_in_run(self):
        calls = [
            _tc("click", element=3, call_id="a"),
            _tc("key", key="Tab", call_id="b"),
            _tc("key", key="Tab", call_id="c"),
        ]
        keep = guarded_run_preserve_indices(calls)
        assert len(AIAgent._deduplicate_tool_calls(calls, preserve_indices=keep)) == 3

    def test_duplicates_collapse_outside_runs(self):
        calls = [
            _tc("key", key="Tab", call_id="a"),
            _tc("web_search", call_id="b", query="x"),
            _tc("web_search", call_id="c", query="x"),
        ]
        keep = guarded_run_preserve_indices(calls)
        assert len(AIAgent._deduplicate_tool_calls(calls, preserve_indices=keep)) == 2


# -- executor wiring -------------------------------------------------------


def _run_sequential(calls, verdicts, blocked=(False, False, False)):
    import agent.tool_executor as te

    agent = MagicMock()
    agent._interrupt_requested = False
    agent._incremental_persistence_failed = False
    assistant = SimpleNamespace(tool_calls=calls)
    messages = []
    dispatched = []

    def _parse(_agent, tool_call, flatten_probe=True):
        ref = SimpleNamespace(name="computer_use", args={}, trace=[], call_id=tool_call.id)
        return SimpleNamespace(parse_error=None, scope_block=None, ref=lambda _t: ref)

    def _run_call(_agent, dispatch, ref, **kw):
        dispatched.append(ref.call_id)
        idx = len(dispatched) - 1
        return (
            SimpleNamespace(result=verdicts[idx], blocked=blocked[idx], args={},
                            middleware_trace=[], dispatched=True),
            0.01,
        )

    with (
        patch.object(te, "_parse_tool_call", side_effect=_parse),
        patch.object(te, "_resolve_sequential_dispatch", return_value=MagicMock()),
        patch.object(te, "_run_sequential_call", side_effect=_run_call),
        patch.object(te, "_publish_sequential_result", return_value=True),
        patch.object(te, "_budget_for_agent", return_value=MagicMock()),
        patch.object(te, "_finalize_tool_batch"),
        patch.object(te, "_flush_session_db_after_tool_progress", return_value=True),
    ):
        te._execute_tool_calls_sequential(agent, assistant, messages, "task-1")
    return dispatched, messages


class TestExecutorWiring:
    def test_uncertain_first_result_skips_run(self):
        calls = [_tc("key", key="Tab", call_id=f"a{i}") for i in range(3)]
        dispatched, messages = _run_sequential(calls, [_verdict("verify_fresh_state")] * 3)
        assert dispatched == ["a0"]
        skipped = [m for m in messages if m["role"] == "tool"]
        assert [m["tool_call_id"] for m in skipped] == ["a1", "a2"]
        assert "Guarded desktop run stopped" in skipped[0]["content"]

    def test_confirmed_results_complete_run(self):
        calls = [_tc("key", key="Tab", call_id=f"a{i}") for i in range(3)]
        dispatched, messages = _run_sequential(calls, [_verdict("done")] * 3)
        assert dispatched == ["a0", "a1", "a2"]
        assert not any("Guarded desktop run stopped" in str(m.get("content")) for m in messages)

    def test_approval_block_stops_run(self):
        calls = [_tc("key", key="Tab", call_id=f"a{i}") for i in range(3)]
        verdicts = [_verdict("done"), _verdict("done"), _verdict("done")]
        dispatched, messages = _run_sequential(calls, verdicts, blocked=(False, True, False))
        assert dispatched == ["a0", "a1"]
        assert any("Guarded desktop run stopped" in str(m.get("content")) for m in messages)
