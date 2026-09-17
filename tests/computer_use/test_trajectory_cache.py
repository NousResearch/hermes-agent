"""Tests for trajectory replay cache (RFC #112639 autoresearch)."""

from __future__ import annotations

import json

import pytest

from tools.computer_use.decide_loop import run_decide_loop
from tools.computer_use.trajectory_cache import (
    CachedStep,
    cached_step_to_decision,
    invalidate_trajectory,
    load_trajectory,
    save_trajectory,
)


@pytest.fixture()
def cache_home(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes"))
    yield tmp_path


def test_save_load_roundtrip(cache_home):
    steps = [
        CachedStep(action="click", target_element=1, backend="jev"),
        CachedStep(action="click", target_element=2, backend="jev"),
        CachedStep(action="done", backend="jev"),
    ]
    path = save_trajectory("Open settings", "Settings", steps)
    assert path
    loaded = load_trajectory("Open settings", "Settings")
    assert loaded is not None
    assert len(loaded) == 3
    assert loaded[0].target_element == 1


def test_invalidate_removes_cache(cache_home):
    save_trajectory("goal", None, [CachedStep(action="click", target_element=1)])
    invalidate_trajectory("goal", None)
    assert load_trajectory("goal", None) is None


def test_cached_step_to_decision_marks_from_cache():
    decision = cached_step_to_decision(CachedStep(action="click", target_element=3))
    assert decision["from_cache"] is True
    assert decision["target_element"] == 3
    assert decision["action"] == "click"


def test_loop_replays_cache_before_decide(cache_home):
    save_trajectory(
        "finish wizard",
        None,
        [
            CachedStep(action="click", target_element=1),
            CachedStep(action="done"),
        ],
    )
    decide_calls = {"n": 0}

    def handle(args: dict):
        action = args.get("action")
        if action == "capture":
            return json.dumps({"ok": True, "total_elements": 2})
        if action == "decide":
            decide_calls["n"] += 1
            return json.dumps({"ok": True, "fail_open": False, "decision": {"action": "done", "done": True}})
        if action == "click":
            return json.dumps({"ok": True})
        return json.dumps({"error": action})

    result = run_decide_loop(
        "finish wizard",
        handle,
        max_steps=5,
        step_pause_s=0,
        use_trajectory_cache=True,
    )
    assert result.ok is True
    assert result.status == "done"
    assert result.cache_hit_steps == 2
    assert decide_calls["n"] == 0
    assert result.steps[0].from_cache is True


def test_loop_invalidates_cache_on_fail_open(cache_home):
    save_trajectory("x", None, [CachedStep(action="click", target_element=1)])

    def handle(args: dict):
        action = args.get("action")
        if action == "capture":
            return json.dumps({"ok": True, "total_elements": 1})
        if action == "decide":
            return json.dumps({"ok": True, "fail_open": True})
        return json.dumps({"ok": True})

    result = run_decide_loop(
        "x", handle, max_steps=5, stuck_threshold=2, step_pause_s=0, use_trajectory_cache=True
    )
    assert result.status == "fail_open"
    assert load_trajectory("x", None) is None
