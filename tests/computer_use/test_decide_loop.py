"""Bounded decide→act loop (#113850) — no frontier model in the loop."""

from __future__ import annotations

import json

from tools.computer_use.backend import ActionResult, CaptureResult, UIElement
from tools.computer_use.decide_loop import run_decide_loop
from tools.computer_use import tool as cu_tool


def test_loop_clicks_then_done():
    clicks = []
    queue = [
        {"ok": True, "fail_open": False, "decision": {
            "action": "click", "target_element": 1, "done": False, "backend": "jev", "confidence": 0.9,
        }},
        {"ok": True, "fail_open": False, "decision": {
            "action": "click", "target_element": 2, "done": False, "backend": "jev", "confidence": 0.91,
        }},
        {"ok": True, "fail_open": False, "decision": {
            "action": "done", "done": True, "backend": "jev", "confidence": 0.99,
        }},
    ]

    def handle(args):
        if args["action"] == "decide":
            return queue.pop(0)
        if args["action"] == "click":
            clicks.append(args["element"])
            return {"ok": True}
        return {"ok": False, "error": args["action"]}

    result = run_decide_loop("finish wizard", handle)
    assert result.ok is True
    assert result.status == "done"
    assert clicks == [1, 2]
    assert [s.decide_backend for s in result.steps] == ["jev", "jev", "jev"]


def test_loop_fail_open_returns_to_planner():
    def handle(args):
        return {"ok": True, "fail_open": True, "decision": None}

    result = run_decide_loop("vague", handle, max_steps=3)
    assert result.ok is False
    assert result.status == "fail_open"
    assert result.steps[0].fail_open is True


def test_loop_type_without_text_fail_opens_generation():
    def handle(args):
        if args["action"] == "decide":
            return {"ok": True, "fail_open": False, "decision": {
                "action": "type", "target_element": 1, "needs_generation": True,
                "done": False, "backend": "jev", "confidence": 0.9,
            }}
        return {"ok": True}

    result = run_decide_loop("fill name", handle)
    assert result.ok is False
    assert "needs_generation" in (result.steps[0].error or "")

def test_loop_type_compiles_to_set_value_on_chosen_target():
    calls = []
    queue = [
        {"ok": True, "fail_open": False, "decision": {
            "action": "type", "target_element": 2, "done": False, "backend": "jev", "confidence": 0.9,
        }},
        {"ok": True, "fail_open": False, "decision": {
            "action": "done", "done": True, "backend": "jev", "confidence": 0.99,
        }},
    ]

    def handle(args):
        if args["action"] == "decide":
            return queue.pop(0)
        calls.append(dict(args))
        return {"ok": True}

    result = run_decide_loop("Enter Kevin into textbox B", handle, text="Kevin")
    assert result.ok is True
    assert calls == [{"action": "set_value", "element": 2, "value": "Kevin"}]


def test_run_goal_action_on_noop(monkeypatch, tmp_path):
    monkeypatch.setenv("HERMES_COMPUTER_USE_BACKEND", "noop")
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    scopes = []
    monkeypatch.setattr(cu_tool, "_request_approval", lambda action, args: scopes.append(action) or None)
    cu_tool.reset_backend_for_tests()
    backend = cu_tool._get_backend("")
    clicks = []
    backend.click = lambda **kw: (clicks.append(kw.get("element")) or ActionResult(ok=True, action="click"))
    n = {"i": 0}

    def fake_decide(_backend, _action, _args, session_id=None, **_):
        n["i"] += 1
        if n["i"] == 1:
            return json.dumps({"ok": True, "fail_open": False, "decision": {
                "action": "click", "target_element": 1, "done": False, "backend": "jev", "confidence": 0.9,
            }})
        return json.dumps({"ok": True, "fail_open": False, "decision": {
            "action": "done", "done": True, "backend": "jev", "confidence": 0.99,
        }})

    spec = cu_tool._ACTIONS["decide"]
    monkeypatch.setitem(cu_tool._ACTIONS, "decide", spec._replace(handler=fake_decide))
    raw = cu_tool._do_run_goal(backend, "run_goal", {"goal": "Submit", "max_steps": 4})
    payload = json.loads(raw)
    assert payload["action"] == "run_goal"
    assert payload["ok"] is True
    assert payload["status"] == "done"
    assert clicks == [1]
    assert "click" in scopes
    cu_tool.reset_backend_for_tests()


def test_run_goal_type_preserves_target_element(monkeypatch, tmp_path):
    monkeypatch.setenv("HERMES_COMPUTER_USE_BACKEND", "noop")
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setattr(cu_tool, "_request_approval", lambda *a, **k: None)
    cu_tool.reset_backend_for_tests()
    backend = cu_tool._get_backend("")
    fields = {1: "", 2: ""}

    def set_value(value, element=None, **_):
        fields[int(element)] = value
        return ActionResult(ok=True, action="set_value")

    backend.set_value = set_value
    n = {"i": 0}

    def fake_decide(_backend, _action, _args, session_id=None, **_):
        n["i"] += 1
        if n["i"] == 1:
            return json.dumps({"ok": True, "fail_open": False, "decision": {
                "action": "type", "target_element": 2, "done": False, "backend": "jev", "confidence": 0.9,
            }})
        return json.dumps({"ok": True, "fail_open": False, "decision": {
            "action": "done", "done": True, "backend": "jev", "confidence": 0.99,
        }})

    spec = cu_tool._ACTIONS["decide"]
    monkeypatch.setitem(cu_tool._ACTIONS, "decide", spec._replace(handler=fake_decide))
    raw = cu_tool._do_run_goal(
        backend, "run_goal",
        {"goal": "Enter Kevin into textbox B", "text": "Kevin", "max_steps": 4},
    )
    payload = json.loads(raw)
    assert payload["ok"] is True
    assert payload["status"] == "done"
    assert fields == {1: "", 2: "Kevin"}
    cu_tool.reset_backend_for_tests()



def test_same_state_ab_rules_vs_jev():
    from tools.computer_use.decision_lane import Decision, ElementCandidate, SemanticState, run_decision_lane

    state = SemanticState(goal_hint="Submit")
    cands = (ElementCandidate("1", "Submit", role="button"),)
    d_rules, p_rules = run_decision_lane(state, cands)
    d_jev, p_jev = run_decision_lane(
        state, cands,
        jev=lambda s, c: Decision(action="click", target_ref="1", confidence=0.99, backend="jev"),
    )
    assert p_rules.state_summary == p_jev.state_summary
    assert p_rules.candidates == p_jev.candidates
    assert d_rules is not None and d_rules.backend == "rules"
    # rules fire first; Jev is only in scores if rules abstain. Same state is still A/B-able:
    assert "screenshot" not in repr(p_jev.to_dict()).lower()


def test_loop_suspected_noop_fail_opens():
    queue = [
        {"ok": True, "fail_open": False, "decision": {
            "action": "click", "target_element": 1, "done": False, "backend": "jev", "confidence": 0.9,
        }},
    ]

    def handle(args):
        if args["action"] == "decide":
            return queue.pop(0)
        return {"ok": True, "verdict": {"decision": "escalate"}}

    result = run_decide_loop("click it", handle)
    assert result.ok is False
    assert result.status == "fail_open"
    assert result.steps[0].fail_open is True


def test_loop_unverifiable_does_fresh_decide_not_retry():
    decides = []
    clicks = []
    n = {"i": 0}

    def handle(args):
        if args["action"] == "decide":
            decides.append(1)
            n["i"] += 1
            if n["i"] == 1:
                return {"ok": True, "fail_open": False, "decision": {
                    "action": "click", "target_element": 1, "done": False, "backend": "jev", "confidence": 0.9,
                }}
            return {"ok": True, "fail_open": False, "decision": {
                "action": "done", "done": True, "backend": "jev", "confidence": 0.99,
            }}
        if args["action"] == "click":
            clicks.append(args["element"])
            return {"ok": True, "verdict": {"decision": "verify_fresh_state"}}
        return {"ok": False, "error": args["action"]}

    result = run_decide_loop("click it", handle)
    assert result.ok is True
    assert clicks == [1]
    assert len(decides) == 2


def test_capture_url_does_not_navigate(monkeypatch, tmp_path):
    monkeypatch.setenv("HERMES_COMPUTER_USE_BACKEND", "noop")
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setattr(cu_tool, "_request_approval", lambda *a, **k: None)
    cu_tool.reset_backend_for_tests()
    backend = cu_tool._get_backend("")
    navs = []
    backend.navigate = lambda url, **kw: navs.append(url) or ActionResult(ok=True, action="navigate")
    cu_tool.handle_computer_use({"action": "capture", "mode": "ax", "url": "https://evil.example"})
    assert navs == []
    cu_tool.reset_backend_for_tests()


def test_run_goal_url_requests_navigate_approval(monkeypatch, tmp_path):
    monkeypatch.setenv("HERMES_COMPUTER_USE_BACKEND", "noop")
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    scopes = []
    monkeypatch.setattr(cu_tool, "_request_approval", lambda action, args: scopes.append(action) or None)
    cu_tool.reset_backend_for_tests()
    backend = cu_tool._get_backend("")
    backend.navigate = lambda url, **kw: ActionResult(ok=True, action="navigate")
    n = {"i": 0}

    def fake_decide(_backend, _action, _args, session_id=None, **_):
        n["i"] += 1
        return json.dumps({"ok": True, "fail_open": False, "decision": {
            "action": "done", "done": True, "backend": "jev", "confidence": 0.99,
        }})

    spec = cu_tool._ACTIONS["decide"]
    monkeypatch.setitem(cu_tool._ACTIONS, "decide", spec._replace(handler=fake_decide))
    raw = cu_tool._do_run_goal(
        backend, "run_goal",
        {"goal": "open then done", "url": "https://example.com", "max_steps": 2},
    )
    payload = json.loads(raw)
    assert payload.get("ok") is True or "error" not in payload
    assert "navigate" in scopes
    cu_tool.reset_backend_for_tests()


def test_run_goal_itself_is_not_an_approval_scope(monkeypatch, tmp_path):
    monkeypatch.setenv("HERMES_COMPUTER_USE_BACKEND", "noop")
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    scopes = []
    monkeypatch.setattr(cu_tool, "_request_approval", lambda action, args: scopes.append(action) or None)
    cu_tool.reset_backend_for_tests()

    def fake_decide(_backend, _action, _args, session_id=None, **_):
        return json.dumps({"ok": True, "fail_open": False, "decision": {
            "action": "done", "done": True, "backend": "jev", "confidence": 0.99,
        }})

    spec = cu_tool._ACTIONS["decide"]
    monkeypatch.setitem(cu_tool._ACTIONS, "decide", spec._replace(handler=fake_decide))
    cu_tool.handle_computer_use({"action": "run_goal", "goal": "already done", "max_steps": 2})
    assert "run_goal" not in scopes
    cu_tool.reset_backend_for_tests()

