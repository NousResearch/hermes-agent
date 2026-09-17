"""Guarded-run executor — RFC #112639.

Behavior contracts: after a mid-run step whose verdict allowed continuation,
a bounded readiness check confirms the step; ``unsatisfied``/``error`` stops
the run with the evidence attached, ``unknown`` and missing signal degrade to
verdict-only behavior (fail open, never fail closed, on missing signal).
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from agent.guarded_run_executor import (
    GUARDED_RUN_READINESS_TIMEOUT_MS,
    confirm_run_step,
    guarded_run_readiness_stop,
    run_step_needs_confirmation,
)
from tools.computer_use.readiness import ReadinessResult


def _tc(action=None, tool="computer_use", call_id=None, **args):
    payload = dict(args)
    if action is not None:
        payload["action"] = action
    return SimpleNamespace(
        id=call_id or f"call-{tool}-{action or 'x'}",
        function=SimpleNamespace(name=tool, arguments=json.dumps(payload)),
    )


def _run_calls():
    return [
        _tc("click", coordinate=[260, 116], call_id="a"),
        _tc("type", text="hello", call_id="b"),
        _tc("click", coordinate=[260, 168], call_id="c"),
    ]


class _Backend:
    """Fake backend with a sticky target and a canned readiness verdict."""

    def __init__(self, status="satisfied", detail="driver status: satisfied"):
        self._active_pid = 1234
        self._active_window_id = 5678
        self._last_target = {"pid": 1234, "window_id": 5678}
        self.seen_kwargs = None
        self._result = ReadinessResult(status=status, detail=detail, duration_ms=41.2)

    def verify_readiness(self, **kwargs):
        self.seen_kwargs = kwargs
        return self._result


class TestNeedsConfirmation:
    def test_mid_run_input_needs_confirmation(self):
        assert run_step_needs_confirmation(_run_calls(), 0) is True
        assert run_step_needs_confirmation(_run_calls(), 1) is True

    def test_last_call_and_outsiders_need_nothing(self):
        assert run_step_needs_confirmation(_run_calls(), 2) is False
        assert run_step_needs_confirmation([_tc("type", text="x", call_id="solo")], 0) is False
        mixed = [_tc("click", coordinate=[1, 1], call_id="a"), _tc("capture", mode="vision", call_id="b")]
        assert run_step_needs_confirmation(mixed, 0) is False


class TestReadinessStop:
    def test_unsatisfied_stops_run_with_evidence(self):
        backend = _Backend(status="unsatisfied", detail="driver status: unsatisfied (#0:unsatisfied)")
        stopped = guarded_run_readiness_stop(_run_calls(), 0, backend)
        assert stopped is not None
        run_end, reason = stopped
        assert run_end == 3
        assert "unsatisfied" in reason and "#0:unsatisfied" in reason and "41ms" in reason

    def test_error_stops_run(self):
        backend = _Backend(status="error", detail="verify_state call failed: boom")
        assert guarded_run_readiness_stop(_run_calls(), 1, backend)[0] == 3

    @pytest.mark.parametrize("status", ["satisfied", "unknown"])
    def test_satisfied_and_unknown_continue(self, status):
        assert guarded_run_readiness_stop(_run_calls(), 0, _Backend(status=status)) is None

    def test_missing_signal_fails_open(self):
        no_target = _Backend()
        no_target._active_pid = no_target._active_window_id = None
        no_target._last_target = {}
        assert guarded_run_readiness_stop(_run_calls(), 0, no_target) is None
        assert guarded_run_readiness_stop(_run_calls(), 0, SimpleNamespace()) is None

    def test_last_call_never_stops(self):
        assert guarded_run_readiness_stop(_run_calls(), 2, _Backend(status="error")) is None


class TestConfirmStep:
    def test_check_is_bounded_and_screenshotted_off(self):
        backend = _Backend()
        confirm_run_step(backend, timeout_ms=500)
        assert backend.seen_kwargs["timeout_ms"] == 500
        assert backend.seen_kwargs["include_screenshot"] is False
        assert backend.seen_kwargs["pid"] == 1234
        assert backend.seen_kwargs["window_id"] == 5678

    def test_default_deadline_is_the_named_constant(self):
        backend = _Backend()
        confirm_run_step(backend)
        assert backend.seen_kwargs["timeout_ms"] == GUARDED_RUN_READINESS_TIMEOUT_MS

    def test_last_target_fallback(self):
        backend = _Backend()
        backend._active_pid = backend._active_window_id = None
        confirm_run_step(backend)
        assert (backend.seen_kwargs["pid"], backend.seen_kwargs["window_id"]) == (1234, 5678)


class TestPerActionExpect:
    """The confirmation postcondition follows the executed step's action.

    Red on the old code: every one of these sent the bare ``window.exists``
    expect regardless of the step.
    """

    SOM = {1: ("Name", "AXTextField"), 2: ("Submit", "AXButton")}

    def _two(self, first):
        return [first, _tc("click", coordinate=[1, 1], call_id="b")]

    def test_type_with_som_label_builds_value_predicate(self):
        backend = _Backend()
        guarded_run_readiness_stop(
            self._two(_tc("type", text="hello", element=1, call_id="a")),
            0, backend, som_labels=self.SOM,
        )
        assert backend.seen_kwargs["expect"] == [{
            "element": {
                "selector": {"label_contains": "Name", "role": "AXTextField"},
                "exists": True,
                "value_equals": "hello",
            }
        }]

    def test_type_without_som_labels_falls_back(self):
        backend = _Backend()
        guarded_run_readiness_stop(self._two(_tc("type", text="hello", call_id="a")), 0, backend)
        assert backend.seen_kwargs["expect"] == [{"window": {"exists": True}}]

    def test_click_with_som_label_builds_element_predicate(self):
        backend = _Backend()
        guarded_run_readiness_stop(
            self._two(_tc("click", element=2, call_id="a")),
            0, backend, som_labels=self.SOM,
        )
        assert backend.seen_kwargs["expect"] == [{
            "element": {
                "selector": {"label_contains": "Submit", "role": "AXButton"},
                "exists": True,
            }
        }]

    def test_click_without_label_falls_back(self):
        backend = _Backend()
        guarded_run_readiness_stop(
            self._two(_tc("click", element=2, call_id="a")), 0, backend
        )
        assert backend.seen_kwargs["expect"] == [{"window": {"exists": True}}]

    def test_drag_falls_back_even_with_som_labels(self):
        backend = _Backend()
        guarded_run_readiness_stop(
            self._two(_tc("drag", from_element=1, to_element=2, call_id="a")),
            0, backend, som_labels=self.SOM,
        )
        assert backend.seen_kwargs["expect"] == [{"window": {"exists": True}}]

    def test_empty_text_fails_open_to_fallback(self):
        backend = _Backend()
        guarded_run_readiness_stop(
            self._two(_tc("type", text="", element=1, call_id="a")),
            0, backend, som_labels=self.SOM,
        )
        assert backend.seen_kwargs["expect"] == [{"window": {"exists": True}}]

    def test_stop_rule_fires_with_sharp_predicate(self):
        backend = _Backend(status="unsatisfied", detail="driver status: unsatisfied (#0:unsatisfied)")
        stopped = guarded_run_readiness_stop(
            self._two(_tc("type", text="hello", element=1, call_id="a")),
            0, backend, som_labels=self.SOM,
        )
        assert stopped is not None and stopped[0] == 2
        assert "unsatisfied" in stopped[1]
        assert backend.seen_kwargs["expect"][0]["element"]["value_equals"] == "hello"

    def test_satisfied_sharp_predicate_continues(self):
        backend = _Backend(status="satisfied")
        assert guarded_run_readiness_stop(
            self._two(_tc("type", text="hello", element=1, call_id="a")),
            0, backend, som_labels=self.SOM,
        ) is None

    def test_confirm_without_tool_call_stays_fallback(self):
        backend = _Backend()
        confirm_run_step(backend)
        assert backend.seen_kwargs["expect"] == [{"window": {"exists": True}}]
