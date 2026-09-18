"""Tests for the System-One decision lane substrate (#113850, Phase 0)."""

import pytest

from tools.computer_use.decision_lane import (
    Decision,
    DecisionPacket,
    ElementCandidate,
    SemanticState,
    jev_available,
    run_decision_lane,
)


def _cand(ref, label="", enabled=True):
    return ElementCandidate(ref=ref, label=label, enabled=enabled)


def test_unknown_action_rejected():
    with pytest.raises(ValueError):
        Decision(action="teleport", confidence=1.0)


def test_rules_escalate_when_nothing_enabled():
    decision, packet = run_decision_lane(
        SemanticState(), (_cand("a", enabled=False),))
    assert decision is not None and decision.action == "escalate"
    assert packet.chosen_backend == "rules"


def test_rules_wait_when_busy():
    decision, _ = run_decision_lane(
        SemanticState(busy=True), (_cand("a", "Save"),))
    assert decision is not None and decision.action == "wait"


def test_rules_click_on_exact_goal_match():
    decision, _ = run_decision_lane(
        SemanticState(goal_hint="Submit"), (_cand("b1", "Submit"), _cand("b2", "Cancel")))
    assert decision is not None
    assert (decision.action, decision.target_ref) == ("click", "b1")


def test_rules_abstain_falls_through_to_reranker():
    reranker = lambda state, cands: Decision(
        action="click", target_ref="b2", confidence=0.8, backend="reranker")
    decision, packet = run_decision_lane(
        SemanticState(), (_cand("b1", "x"), _cand("b2", "y")), reranker=reranker)
    assert decision is not None and decision.backend == "reranker"
    assert packet.scores == {"reranker": 0.8}


def test_low_confidence_abstains_fail_open():
    shy = lambda state, cands: Decision(action="click", target_ref="b1", confidence=0.1)
    decision, packet = run_decision_lane(
        SemanticState(), (_cand("b1", "x"),), reranker=shy)
    assert decision is None and packet.chosen_backend is None


def test_broken_stage_abstains():
    def _boom(state, cands):
        raise RuntimeError("reranker down")

    decision, _ = run_decision_lane(
        SemanticState(), (_cand("b1", "x"),), reranker=_boom)
    assert decision is None


def test_verifier_veto_fails_open():
    decider = lambda state, cands: Decision(
        action="click", target_ref="b1", confidence=0.95, backend="reranker")
    decision, packet = run_decision_lane(
        SemanticState(), (_cand("b1", "x"),), reranker=decider, verifier=lambda s, d: False)
    assert decision is None
    assert packet.verifier_outcome == "veto"


def test_jev_stage_skipped_without_key_or_callable(monkeypatch):
    monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)
    monkeypatch.delenv("JEV_API_KEY", raising=False)
    decision, packet = run_decision_lane(SemanticState(), (_cand("b1", "x"),))
    assert decision is None  # same path as today: rules abstain, nothing else plugged in


def test_packet_round_trip_and_no_pixels_or_secrets():
    decider = lambda state, cands: Decision(
        action="type", target_ref="f1", needs_generation=True,
        confidence=0.9, backend="reranker")
    _, packet = run_decision_lane(
        SemanticState(goal_hint="name"), (_cand("f1", "Name"),), reranker=decider)
    clone = DecisionPacket.from_dict(packet.to_dict())
    assert clone == packet
    blob = repr(packet.to_dict()).lower()
    assert "screenshot" not in blob and "secret" not in blob and "api_key" not in blob
