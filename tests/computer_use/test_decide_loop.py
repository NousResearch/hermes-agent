"""Tests for the bounded decide loop (#113850)."""

from __future__ import annotations

import json

from tools.computer_use.decide_loop import run_decide_loop


def _handle_from_script(script: list[dict]) -> callable:
    state = {"elements": 3}

    def handle(args: dict):
        action = args.get("action")
        if action == "capture":
            return json.dumps({"ok": True, "total_elements": state["elements"]})
        if action == "decide":
            payload = script.pop(0)
            return json.dumps(payload)
        if action == "click":
            state["elements"] = max(0, state["elements"] - 1)
            return json.dumps({"ok": True})
        return json.dumps({"error": f"unexpected {action}"})

    return handle


def test_loop_stops_on_done():
    handle = _handle_from_script([
        {
            "ok": True,
            "fail_open": False,
            "decision": {"action": "click", "target_element": 2, "confidence": 0.9, "backend": "jev"},
        },
        {
            "ok": True,
            "fail_open": False,
            "decision": {"action": "done", "done": True, "confidence": 1.0, "backend": "jev"},
        },
    ])
    result = run_decide_loop("finish the wizard", handle, max_steps=5, step_pause_s=0)
    assert result.ok is True
    assert result.status == "done"
    assert len(result.steps) == 2
    assert result.steps[0].executed is True
    assert result.steps[1].executed is False


def test_loop_fail_open_becomes_stuck():
    handle = _handle_from_script([
        {"ok": True, "fail_open": True},
        {"ok": True, "fail_open": True},
        {"ok": True, "fail_open": True},
    ])
    result = run_decide_loop("x", handle, max_steps=5, stuck_threshold=3, step_pause_s=0)
    assert result.ok is False
    assert result.status == "fail_open"
    assert len(result.steps) == 3


def test_loop_external_done_at_start():
    handle = _handle_from_script([
        {
            "ok": True,
            "fail_open": False,
            "decision": {"action": "click", "target_element": 1, "confidence": 0.9, "backend": "aux"},
        },
    ])
    result = run_decide_loop(
        "two-step wizard",
        handle,
        max_steps=5,
        step_pause_s=0,
        external_done=lambda: True,
    )
    assert result.ok is True
    assert result.status == "completed"
    assert len(result.steps) == 1
    assert result.steps[0].executed is False


def test_loop_external_done_after_first_action():
    seen = {"done": False}

    def handle(args: dict):
        if args.get("action") == "capture":
            return json.dumps({"ok": True, "total_elements": 2})
        if args.get("action") == "decide":
            return json.dumps(
                {
                    "ok": True,
                    "fail_open": False,
                    "decision": {"action": "click", "target_element": 1, "confidence": 0.9, "backend": "aux"},
                }
            )
        if args.get("action") == "click":
            seen["done"] = True
            return json.dumps({"ok": True})
        return json.dumps({"error": "unexpected"})

    result = run_decide_loop(
        "two-step wizard",
        handle,
        max_steps=5,
        step_pause_s=0,
        external_done=lambda: seen["done"],
    )
    assert result.ok is True
    assert result.status == "completed"
    assert len(result.steps) == 1
    assert result.steps[0].executed is True


def test_loop_detects_no_progress_stuck():
    script = [
        {
            "ok": True,
            "fail_open": False,
            "decision": {"action": "click", "target_element": 2, "confidence": 0.9, "backend": "jev"},
        },
    ] * 4
    idx = {"i": 0}
    frozen = {"n": 3}

    def handle(args: dict):
        action = args.get("action")
        if action == "capture":
            return json.dumps({"ok": True, "total_elements": frozen["n"]})
        if action == "decide":
            payload = script[idx["i"]]
            idx["i"] = min(idx["i"] + 1, len(script) - 1)
            return json.dumps(payload)
        if action == "click":
            return json.dumps({"ok": True})
        return json.dumps({"error": f"unexpected {action}"})

    result = run_decide_loop("stuck fixture", handle, max_steps=8, stuck_threshold=3, step_pause_s=0)
    assert result.ok is False
    assert result.status == "stuck"
