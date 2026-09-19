"""Tests for decision stages and the computer_use decide action (#113850)."""

from __future__ import annotations

import json
import os

import pytest

from tools.computer_use.backend import CaptureResult, UIElement
from tools.computer_use.decision_lane import ElementCandidate, SemanticState
from tools.computer_use.decision_stages import (
    _parse_aux_json,
    aux_stage,
    build_jev_questions,
    jev_stage,
    parse_jev_response,
    reranker_stage,
)
from tools.computer_use.system_one import ParsedResponse, Usage, Answer
from tools.computer_use import tool as cu_tool


def test_reranker_fuzzy_match():
    state = SemanticState(goal_hint="Open Settings panel")
    cands = (
        ElementCandidate("1", "Cancel", role="button"),
        ElementCandidate("2", "Settings", role="button"),
    )
    decision = reranker_stage(state, cands)
    assert decision is not None
    assert decision.target_ref == "2"
    assert decision.action == "click"


def test_reranker_abstains_on_ambiguity():
    state = SemanticState(goal_hint="Save")
    cands = (
        ElementCandidate("1", "Save draft", role="button"),
        ElementCandidate("2", "Save copy", role="button"),
    )
    assert reranker_stage(state, cands) is None


def test_build_jev_questions_factorized():
    cands = (ElementCandidate("3", "Submit", role="button"),)
    qs = build_jev_questions(cands)
    assert set(qs) == {"action", "target", "needs_vision", "needs_generation", "done"}
    assert "click" in qs["action"]["criteria"]
    assert "3" in qs["target"]["criteria"]


def test_parse_jev_response_done():
    parsed = ParsedResponse(
        model="jev-latest",
        answers={
            "done": Answer(
                id="done", kind="noul", raw={}, probabilities={"true": 0.9, "false": 0.1}, noul=0.9
            )
        },
        usage=Usage(),
    )
    decision = parse_jev_response(parsed, (ElementCandidate("1", "x"),))
    assert decision is not None
    assert decision.action == "done" and decision.done is True


def test_parse_aux_json_maps_target_ref():
    cands = (ElementCandidate("2", "Submit", role="button"),)
    decision = _parse_aux_json(
        '{"action":"click","target_ref":"2","confidence":0.91}', cands,
    )
    assert decision is not None
    assert decision.backend == "aux"
    assert decision.action == "click" and decision.target_ref == "2"


def test_parse_aux_json_abstains_on_invalid():
    cands = (ElementCandidate("1", "OK", role="button"),)
    assert _parse_aux_json("not json", cands) is None
    assert _parse_aux_json('{"action":"click","confidence":0.9}', cands) is None


def test_aux_stage_uses_call_llm(monkeypatch):
    cands = (ElementCandidate("5", "Submit", role="button"),)
    state = SemanticState(goal_hint="press submit", elements=cands)

    class _Msg:
        content = '{"action":"click","target_ref":"5","confidence":0.88}'

    class _Choice:
        message = _Msg()

    class _Resp:
        choices = [_Choice()]

    monkeypatch.setattr(
        "agent.auxiliary_client.call_llm",
        lambda **kw: _Resp(),
    )
    decision = aux_stage(state, cands)
    assert decision is not None
    assert decision.backend == "aux" and decision.target_ref == "5"


def test_jev_stage_uses_injected_transport(monkeypatch):
    monkeypatch.setenv("TYPESAFE_API_KEY", "test-key")

    def fake_transport(_payload):
        return {
            "model": "jev-latest",
            "answers": {
                "done": {"type": "noul", "noul": 0.1, "probabilities": {"true": 0.1, "false": 0.9}},
                "action": {
                    "type": "choice",
                    "choice": "click",
                    "probabilities": {
                        "click": 0.8, "type": 0.05, "key": 0.05, "scroll": 0.03,
                        "wait": 0.03, "done": 0.02, "escalate": 0.02,
                    },
                    "confidence": 0.8,
                },
                "target": {
                    "type": "choice",
                    "choice": "7",
                    "probabilities": {"7": 0.85, "none": 0.15},
                    "confidence": 0.85,
                },
                "needs_vision": {"type": "noul", "noul": 0.05, "probabilities": {"true": 0.05, "false": 0.95}},
                "needs_generation": {"type": "noul", "noul": 0.05, "probabilities": {"true": 0.05, "false": 0.95}},
            },
            "usage": {"input_tokens": 100, "output_tokens": 10},
        }

    monkeypatch.setattr("tools.computer_use.decision_stages.http_transport", fake_transport)
    state = SemanticState(goal_hint="press submit")
    cands = (ElementCandidate("7", "Submit", role="button"),)
    decision = jev_stage(state, cands)
    assert decision is not None
    assert decision.backend == "jev"
    assert decision.action == "click" and decision.target_ref == "7"


def test_handle_decide_with_noop_backend(monkeypatch, tmp_path):
    monkeypatch.setenv("HERMES_COMPUTER_USE_BACKEND", "noop")
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    cu_tool.reset_backend_for_tests()
    elements = [UIElement(index=1, role="button", label="Submit", bounds=(0, 0, 10, 10), app="TestApp")]
    backend = cu_tool._get_backend("")
    backend.capture = lambda **kw: CaptureResult(
        mode="ax", width=100, height=100, png_b64=None, elements=elements, app="TestApp", window_title="Win"
    )
    raw = cu_tool.handle_computer_use({"action": "decide", "goal": "Submit"})
    payload = json.loads(raw)
    assert payload["ok"] is True
    assert payload["decision"]["action"] == "click"
    assert payload["decision"]["target_element"] == 1
    assert payload["decision"]["backend"] == "rules"
    cu_tool.reset_backend_for_tests()


def test_handle_decide_requires_goal():
    cu_tool.reset_backend_for_tests()
    os.environ["HERMES_COMPUTER_USE_BACKEND"] = "noop"
    raw = cu_tool.handle_computer_use({"action": "decide"})
    assert "error" in json.loads(raw)
    cu_tool.reset_backend_for_tests()


def test_run_goal_action_invokes_loop(monkeypatch):
    captured = {}

    def fake_loop(goal, handle, **kwargs):
        captured["goal"] = goal
        captured["kwargs"] = kwargs
        from tools.computer_use.decide_loop import LoopResult

        return LoopResult(ok=True, status="done", goal=goal, max_steps=kwargs.get("max_steps", 8))

    monkeypatch.setattr("tools.computer_use.decide_loop.run_decide_loop", fake_loop)
    backend = type("B", (), {"capture": lambda *a, **k: type("C", (), {"pid": 1, "window_id": 2})()})()
    raw = cu_tool._do_run_goal(backend, "run_goal", {"goal": "test goal", "use_cache": True, "use_prepared": True})
    payload = json.loads(raw)
    assert payload["ok"] is True
    assert payload["action"] == "run_goal"
    assert captured["goal"] == "test goal"
    assert captured["kwargs"]["use_trajectory_cache"] is True
    assert captured["kwargs"]["use_prepared_actions"] is True
