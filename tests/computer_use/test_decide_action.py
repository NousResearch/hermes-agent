"""Tests for decision stages and the computer_use decide action (#113850)."""

from __future__ import annotations

import json

from tools.computer_use import tool as cu_tool
from tools.computer_use.backend import CaptureResult, UIElement
from tools.computer_use.decision_lane import ElementCandidate, SemanticState
from tools.computer_use.decision_stages import (
    build_jev_questions,
    jev_stage,
    parse_jev_response,
    reranker_stage,
)
from tools.computer_use.system_one import Answer, ParsedResponse, Usage


def test_reranker_fuzzy_match():
    state = SemanticState(goal_hint="Open Settings panel")
    cands = (
        ElementCandidate("1", "Cancel", role="button"),
        ElementCandidate("2", "Settings", role="button"),
    )
    decision = reranker_stage(state, cands)
    assert decision is not None
    assert decision.action == "click"
    assert decision.target_ref == "2"
    assert decision.backend == "reranker"


def test_reranker_abstains_on_ambiguity():
    state = SemanticState(goal_hint="save")
    cands = (
        ElementCandidate("1", "Confirm save", role="button"),
        ElementCandidate("2", "Cancel save", role="button"),
    )
    assert reranker_stage(state, cands) is None


def test_build_jev_questions_factorized():
    cands = (ElementCandidate("3", "Submit", role="button"),)
    qs = build_jev_questions(cands)
    assert set(qs) == {"action", "target", "needs_vision", "needs_generation", "done"}
    assert "click" in qs["action"]["criteria"]
    assert "3" in qs["target"]["criteria"]
    assert "none" in qs["target"]["criteria"]


def test_parse_jev_response_done():
    parsed = ParsedResponse(
        model="jev-latest",
        answers={
            "done": Answer(
                id="done", kind="noul", raw={},
                probabilities={"true": 0.91, "false": 0.09}, noul=0.91,
            ),
        },
        usage=Usage(),
    )
    decision = parse_jev_response(parsed, (ElementCandidate("1", "OK"),))
    assert decision is not None
    assert decision.action == "done" and decision.done is True


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
    assert payload["fail_open"] is False
    cu_tool.reset_backend_for_tests()


def test_handle_decide_requires_goal(monkeypatch):
    monkeypatch.setenv("HERMES_COMPUTER_USE_BACKEND", "noop")
    cu_tool.reset_backend_for_tests()
    raw = cu_tool.handle_computer_use({"action": "decide"})
    assert "error" in json.loads(raw)
    cu_tool.reset_backend_for_tests()


def test_handle_decide_fail_open_without_match(monkeypatch):
    monkeypatch.setenv("HERMES_COMPUTER_USE_BACKEND", "noop")
    monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)
    monkeypatch.delenv("JEV_API_KEY", raising=False)
    cu_tool.reset_backend_for_tests()
    backend = cu_tool._get_backend("")
    backend.capture = lambda **kw: CaptureResult(
        mode="ax", width=100, height=100, png_b64=None,
        elements=[UIElement(index=1, role="button", label="Cancel")],
        app="TestApp", window_title="Win",
    )
    payload = json.loads(cu_tool.handle_computer_use({"action": "decide", "goal": "do something vague"}))
    assert payload["ok"] is True
    assert payload["fail_open"] is True
    assert payload["verdict"]["decision"] == "escalate"
    cu_tool.reset_backend_for_tests()
