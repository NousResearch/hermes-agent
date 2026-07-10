"""Isolated tests for the AMDP planning layer.

Hermetic: model calls (``_call``) and state intake (``_intake``) are
monkeypatched, and the audit path is redirected to a tmp dir, so no test hits a
real model, the proprioception collector, or the live HERMES_HOME.
"""

from __future__ import annotations

import json

import pytest

from agent.amdp import loop
from agent.amdp.config import AmdpConfigError, resolve_amdp_config

COA_JSON = json.dumps({
    "coas": [
        {"coa_id": "A", "summary": "observe first", "dispatches": [
            {"task": "look at the state", "kind": "observe", "irreversible": False,
             "success_criteria": ["state understood"]}]},
        {"coa_id": "B", "summary": "act fast", "dispatches": [
            {"task": "delete the thing", "kind": "act", "irreversible": True}]},
    ]
})
REVIEW_JSON = json.dumps({
    "alignment_1to10": 7, "fragility_0to1": 0.4,
    "risks": [{"desc": "could fail", "severity_1to5": 3}],
})

ENABLED_CFG = {
    "amdp": {
        "enabled": True,
        "planner": {"provider": "planner-prov", "model": "pm"},
        "reviewer": {"provider": "reviewer-prov", "model": "rm"},
        "n_coas": 2,
        "gate": {"min_estimated_steps": 1},
        "staleness_max_s": 120,
    }
}
MULTISTEP_PROMPT = "Migrate the database and then verify it end-to-end."


@pytest.fixture
def audit_tmp(tmp_path, monkeypatch):
    path = tmp_path / "amdp_audit.jsonl"
    monkeypatch.setattr(loop, "_audit_path", lambda cfg: str(path))
    return path


@pytest.fixture
def healthy_state(monkeypatch):
    monkeypatch.setattr(loop, "_intake", lambda config, timeout_s=None: {
        "brief": "all calm", "sensors_down": [], "staleness_s": 0.0,
        "verdict": "ok", "gateway_state": "running", "system_count": 3,
    })


@pytest.fixture
def mock_models(monkeypatch):
    calls = []

    def fake_call(slot, messages, *, temperature, max_tokens, json_mode=False):
        calls.append(slot.get("provider"))
        if slot.get("provider") == "planner-prov":
            return COA_JSON, ""
        return REVIEW_JSON, ""

    monkeypatch.setattr(loop, "_call", fake_call)
    return calls


# --------------------------------------------------------------------------- #
# Config validation
# --------------------------------------------------------------------------- #
def test_absent_config_returns_none():
    assert resolve_amdp_config({}) is None
    assert resolve_amdp_config({"amdp": {"enabled": False}}) is None


def test_enabled_but_empty_reviewer_hard_fails():
    with pytest.raises(AmdpConfigError):
        resolve_amdp_config({"amdp": {"enabled": True, "planner": {"provider": "p", "model": "m"}, "reviewer": {}}})


def test_enabled_but_missing_planner_hard_fails():
    with pytest.raises(AmdpConfigError):
        resolve_amdp_config({"amdp": {"enabled": True, "reviewer": {"provider": "r", "model": "m"}}})


def test_valid_config_parses():
    cfg = resolve_amdp_config(ENABLED_CFG)
    assert cfg is not None and cfg.enabled
    assert cfg.planner["provider"] == "planner-prov"
    assert cfg.n_coas == 2


# --------------------------------------------------------------------------- #
# Default-OFF: nothing changes, no model calls (key acceptance)
# --------------------------------------------------------------------------- #
def test_disabled_injects_nothing_and_calls_no_model(mock_models):
    out = loop.maybe_amdp_context(MULTISTEP_PROMPT, [], {})
    assert out == ""
    assert mock_models == []  # no model calls when AMDP is off


def test_gate_skips_trivial_turns(mock_models, healthy_state):
    cfg = dict(ENABLED_CFG)
    cfg["amdp"] = dict(ENABLED_CFG["amdp"], gate={"min_estimated_steps": 5})
    out = loop.maybe_amdp_context("hi", [], cfg)
    assert out == ""
    assert mock_models == []  # gate rejected before any model call


# --------------------------------------------------------------------------- #
# Happy path: produces a plan + audit
# --------------------------------------------------------------------------- #
def test_enabled_produces_plan_and_audit(mock_models, healthy_state, audit_tmp):
    out = loop.maybe_amdp_context(MULTISTEP_PROMPT, [], ENABLED_CFG)
    assert out and "AMDP plan" in out
    assert "IRREVERSIBLE" in out or "observe" in out
    # planner called once, reviewer called for each COA
    assert mock_models.count("planner-prov") == 1
    assert mock_models.count("reviewer-prov") == 2
    rec = json.loads(audit_tmp.read_text(encoding="utf-8").strip())
    assert rec["refused"] is False and rec["chosen"] in ("A", "B")
    assert rec["coas"] and rec["scores"]


# --------------------------------------------------------------------------- #
# Refuse gate: blind state -> no plan, no model calls, audited
# --------------------------------------------------------------------------- #
def test_blind_state_refuses(mock_models, monkeypatch, audit_tmp):
    # Truly blind = no gateway status. (A down DASHBOARD alone does NOT refuse —
    # see test_amdp_hardening.test_dashboard_down_still_plans.)
    monkeypatch.setattr(loop, "_intake", lambda config, timeout_s=None: {
        "brief": "", "sensors_down": ["gateway-status"], "staleness_s": 0.0,
        "verdict": "unknown", "gateway_state": "unknown", "system_count": 0,
    })
    out = loop.maybe_amdp_context(MULTISTEP_PROMPT, [], ENABLED_CFG)
    assert out == ""
    assert mock_models == []  # refused before any model call
    rec = json.loads(audit_tmp.read_text(encoding="utf-8").strip())
    assert rec["refused"] is True and "gateway status" in rec["refuse_reason"]


def test_stale_state_refuses(mock_models, monkeypatch, audit_tmp):
    monkeypatch.setattr(loop, "_intake", lambda config, timeout_s=None: {
        "brief": "", "sensors_down": [], "staleness_s": 999.0,
        "verdict": "ok", "gateway_state": "running", "system_count": 3,
    })
    out = loop.maybe_amdp_context(MULTISTEP_PROMPT, [], ENABLED_CFG)
    assert out == ""
    assert mock_models == []


# --------------------------------------------------------------------------- #
# gpt-oss reasoning_content fallback
# --------------------------------------------------------------------------- #
def test_extract_text_reasoning_fallback():
    class _Msg(dict):
        pass

    class _Resp:
        def __init__(self, msg):
            self.choices = [type("C", (), {"message": msg})()]

    assert loop._extract_text(_Resp({"content": "answer"})) == "answer"
    assert loop._extract_text(_Resp({"content": "", "reasoning_content": "thought"})) == "thought"
    assert loop._extract_text(_Resp({"content": None, "reasoning": "r"})) == "r"


# --------------------------------------------------------------------------- #
# Fail-closed: an exception anywhere yields "" not a raised error
# --------------------------------------------------------------------------- #
def test_planner_exception_is_swallowed(monkeypatch, healthy_state):
    def boom(*a, **k):
        raise RuntimeError("kaboom")

    monkeypatch.setattr(loop, "_generate_coas", boom)
    out = loop.maybe_amdp_context(MULTISTEP_PROMPT, [], ENABLED_CFG)
    assert out == ""  # never raises into the turn
