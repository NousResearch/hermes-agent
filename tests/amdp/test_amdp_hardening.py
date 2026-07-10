"""Regression tests for the review-driven hardening pass.

Each test locks in a fix for a finding from the multi-agent adversarial review,
so a future change can't silently reintroduce the defect. Same hermetic style.
"""

from __future__ import annotations

import json
import logging

import pytest

from agent.amdp import loop
from agent.amdp.config import AmdpConfigError, resolve_amdp_config

ENABLED_CFG = {
    "amdp": {
        "enabled": True,
        "planner": {"provider": "planner-prov", "model": "pm"},
        "reviewer": {"provider": "reviewer-prov", "model": "rm"},
        "n_coas": 2,
        "gate": {"min_estimated_steps": 1},
    }
}
COA_JSON = json.dumps({"coas": [
    {"coa_id": "A", "summary": "s1", "dispatches": [{"task": "t1", "kind": "observe"}]},
    {"coa_id": "B", "summary": "s2", "dispatches": [{"task": "t2", "kind": "act"}]},
]})
REVIEW_JSON = json.dumps({"alignment_1to10": 7, "fragility_0to1": 0.4,
                          "risks": [{"desc": "r", "severity_1to5": 3}]})
MULTISTEP = "Migrate the database and then verify it end-to-end."


@pytest.fixture
def audit_tmp(tmp_path, monkeypatch):
    monkeypatch.setattr(loop, "_audit_path", lambda cfg: str(tmp_path / "a.jsonl"))


@pytest.fixture
def healthy_state(monkeypatch):
    monkeypatch.setattr(loop, "_intake", lambda config, timeout_s=None: {
        "brief": "calm", "sensors_down": [], "staleness_s": 0.0,
        "verdict": "ok", "gateway_state": "running", "system_count": 3})


@pytest.fixture(autouse=True)
def _reset_amdp_cache():
    """The hook-path config cache is process-global; reset around every test so a
    None-path test can't leak a cached config into another."""
    loop.reset_cache_for_tests()
    yield
    loop.reset_cache_for_tests()


# CRITICAL-1: misconfig is fail-LOUD (ERROR logged) but still fail-SAFE (no raise, no plan)
def test_misconfig_is_loud_but_safe(caplog):
    bad = {"amdp": {"enabled": True, "planner": {"provider": "p", "model": "m"}, "reviewer": {}}}
    with caplog.at_level(logging.ERROR):
        out = loop.maybe_amdp_context(MULTISTEP, [], bad)
    assert out == ""  # safe: no raise, no plan
    assert any("MISCONFIGURED" in r.message for r in caplog.records)  # loud


# R1-MEDIUM-1: non-string slot types are rejected, not stringified
def test_non_string_slot_hard_fails():
    with pytest.raises(AmdpConfigError):
        resolve_amdp_config({"amdp": {"enabled": True,
                                      "planner": {"provider": ["p"], "model": "m"},
                                      "reviewer": {"provider": "r", "model": "m"}}})


# R2-HIGH#2: unknown decision_profile is caught at config time -> default, no KeyError later
def test_decision_profile_invalid_defaults_to_v1():
    cfg = resolve_amdp_config({"amdp": {"enabled": True,
                                        "planner": {"provider": "p", "model": "m"},
                                        "reviewer": {"provider": "r", "model": "m"},
                                        "decision_profile": "bogus"}})
    assert cfg.decision_profile == "v1_balanced"


# R1-MEDIUM-2: a bad reviewer_max_tokens defaults instead of raising (which would silent-disable)
def test_reviewer_max_tokens_bad_value_defaults():
    cfg = resolve_amdp_config({"amdp": {"enabled": True,
                                        "planner": {"provider": "p", "model": "m"},
                                        "reviewer": {"provider": "r", "model": "m"},
                                        "reviewer_max_tokens": "lots"}})
    assert cfg.reviewer_max_tokens == 1800


# R2-HIGH#3: timeout / deadline knobs exist and parse
def test_timeout_fields():
    cfg = resolve_amdp_config(ENABLED_CFG)
    assert cfg.call_timeout_s == 90.0 and cfg.episode_deadline_s == 240.0
    cfg2 = resolve_amdp_config({"amdp": dict(ENABLED_CFG["amdp"], call_timeout_s=30, episode_deadline_s=120)})
    assert cfg2.call_timeout_s == 30.0 and cfg2.episode_deadline_s == 120.0


# R1-MEDIUM-3: duplicate model-emitted coa_ids are relabeled unique so reviews can't collide
def test_duplicate_coa_ids_are_relabeled(monkeypatch):
    dup = json.dumps({"coas": [
        {"coa_id": "A", "summary": "s1", "dispatches": [{"task": "t1", "kind": "observe"}]},
        {"coa_id": "A", "summary": "s2", "dispatches": [{"task": "t2", "kind": "act"}]},
    ]})
    monkeypatch.setattr(loop, "_call", lambda *a, **k: (dup, ""))
    cfg = resolve_amdp_config(ENABLED_CFG)
    coas = loop._generate_coas(cfg, "intent", "brief", [])
    assert [c["coa_id"] for c in coas] == ["A", "B"]  # not ["A", "A"]


# R3 policy: when ALL reviewers are down, the plan is emitted but LABELED unvetted
def test_all_reviewers_down_labels_plan_unvetted(monkeypatch, healthy_state, audit_tmp):
    def fc(slot, messages, *, temperature, max_tokens, json_mode=False):
        return (COA_JSON, "") if slot.get("provider") == "planner-prov" else ("", "down")
    monkeypatch.setattr(loop, "_call", fc)
    out = loop.maybe_amdp_context(MULTISTEP, [], ENABLED_CFG)
    assert "WARNING: war-gaming unavailable" in out


# R2-HIGH#1: reasoning_content arriving via the SDK object's model_extra is extracted
def test_extract_text_reads_model_extra():
    class _Msg:
        content = ""
        model_extra = {"reasoning_content": "the verdict json"}

    class _Resp:
        choices = [type("C", (), {"message": _Msg()})()]

    assert loop._extract_text(_Resp()) == "the verdict json"


# R2-MEDIUM#7 / R1-LOW-1: bare discourse markers no longer trip the gate
def test_gate_ignores_bare_discourse_markers():
    assert loop._estimate_steps("First, thanks — then can you tell me what you think?", []) < 3


# R2-MEDIUM#4: a commander that errors on the json_mode call recovers via the prompt-only repair
def test_commander_json_mode_error_recovers_via_prompt_only_repair(monkeypatch, healthy_state, audit_tmp):
    seen = []

    def fc(slot, messages, *, temperature, max_tokens, json_mode=False):
        if slot.get("provider") == "planner-prov":
            seen.append(json_mode)
            if json_mode:   # first call: pretend the endpoint rejected response_format
                return "", "HTTP 400: response_format unsupported"
            return COA_JSON, ""  # repair (prompt-only) succeeds
        return REVIEW_JSON, ""

    monkeypatch.setattr(loop, "_call", fc)
    out = loop.maybe_amdp_context(MULTISTEP, [], ENABLED_CFG)
    assert out and "AMDP plan" in out
    assert seen == [True, False]  # json_mode primary, prompt-only repair


# NEW-1: the hook path (config=None) resolves via load_config ONCE and caches,
# so default-off/enabled is not a fresh deepcopy every turn.
def test_hook_path_resolves_config_once_and_caches(monkeypatch, tmp_path):
    import hermes_cli.config as hc

    calls = {"n": 0}

    def fake_load():
        calls["n"] += 1
        return {"amdp": {"enabled": True,
                         "planner": {"provider": "planner-prov", "model": "pm"},
                         "reviewer": {"provider": "reviewer-prov", "model": "rm"},
                         "gate": {"min_estimated_steps": 1}}}

    monkeypatch.setattr(hc, "load_config", fake_load)
    monkeypatch.setattr(loop, "_intake", lambda c, timeout_s=None: {
        "brief": "x", "sensors_down": [], "staleness_s": 0.0,
        "verdict": "ok", "gateway_state": "running", "system_count": 1})
    monkeypatch.setattr(loop, "_call",
                        lambda slot, m, **k: (COA_JSON if slot.get("provider") == "planner-prov" else REVIEW_JSON, ""))
    monkeypatch.setattr(loop, "_audit_path", lambda cfg: str(tmp_path / "a.jsonl"))

    out1 = loop.maybe_amdp_context(MULTISTEP, [])   # config=None -> cached hook path
    out2 = loop.maybe_amdp_context(MULTISTEP, [])
    assert out1 and out2 and "AMDP plan" in out1
    assert calls["n"] == 1  # load_config called once; cached thereafter


# NEW-1 (cont.): a misconfig on the hook path is cached-disabled + logged loud, once.
def test_hook_path_misconfig_caches_disabled_loud(monkeypatch, caplog):
    import hermes_cli.config as hc

    calls = {"n": 0}

    def fake_load():
        calls["n"] += 1
        return {"amdp": {"enabled": True, "planner": {"provider": "p", "model": "m"}, "reviewer": {}}}

    monkeypatch.setattr(hc, "load_config", fake_load)
    with caplog.at_level(logging.ERROR):
        assert loop.maybe_amdp_context(MULTISTEP, []) == ""
        assert loop.maybe_amdp_context(MULTISTEP, []) == ""
    assert calls["n"] == 1  # resolved+cached once, not re-attempted every turn
    assert any("MISCONFIGURED" in r.message for r in caplog.records)


def _counting_call():
    calls = []

    def fc(slot, messages, *, temperature, max_tokens, json_mode=False):
        calls.append(slot.get("provider"))
        return (COA_JSON, "") if slot.get("provider") == "planner-prov" else (REVIEW_JSON, "")
    return calls, fc


# TUNING-1: plan ONCE per user turn — a repeat call for the same turn (tool-loop
# iteration) is a cache hit with zero new model calls.
def test_once_per_turn_caches(monkeypatch, healthy_state, audit_tmp):
    calls, fc = _counting_call()
    monkeypatch.setattr(loop, "_call", fc)
    msgs = [{"role": "user", "content": "do the migration end-to-end and deploy"}]
    out1 = loop.maybe_amdp_context(MULTISTEP, msgs, ENABLED_CFG)
    out2 = loop.maybe_amdp_context(MULTISTEP, msgs, ENABLED_CFG)  # same turn -> cache hit
    assert out1 and out1 == out2 and "AMDP plan" in out1
    assert calls.count("planner-prov") == 1  # planned once, NOT re-planned per iteration


# TUNING-1 (cont.): a genuinely new user turn re-plans.
def test_new_user_turn_replans(monkeypatch, healthy_state, audit_tmp):
    calls, fc = _counting_call()
    monkeypatch.setattr(loop, "_call", fc)
    loop.maybe_amdp_context("Migrate pipeline A end-to-end and deploy it", [{"role": "user", "content": "A"}], ENABLED_CFG)
    loop.maybe_amdp_context("Migrate pipeline B end-to-end and deploy it", [{"role": "user", "content": "B"}], ENABLED_CFG)
    assert calls.count("planner-prov") == 2  # different turns -> two episodes


# TUNING-2: a DOWN dashboard (but gateway status present) does NOT refuse — AMDP
# plans on gateway status alone rather than declining under load.
def test_dashboard_down_still_plans(monkeypatch, audit_tmp):
    monkeypatch.setattr(loop, "_intake", lambda config, timeout_s=None: {
        "brief": "gateway: running; dashboard unavailable — planning on gateway status alone",
        "sensors_down": [], "staleness_s": 0.0, "verdict": "unknown",
        "gateway_state": "running", "system_count": 0, "dashboard_up": False})
    calls, fc = _counting_call()
    monkeypatch.setattr(loop, "_call", fc)
    out = loop.maybe_amdp_context(MULTISTEP, [], ENABLED_CFG)
    assert out and "AMDP plan" in out  # planned despite the dashboard being down
    assert calls.count("planner-prov") == 1


# TUNING-2 (cont.): the configured intake timeout is passed through to intake.
def test_intake_timeout_is_passed(monkeypatch, audit_tmp):
    seen = {}

    def fake_intake(config, timeout_s=None):
        seen["t"] = timeout_s
        return {"brief": "x", "sensors_down": [], "staleness_s": 0.0,
                "verdict": "ok", "gateway_state": "running", "system_count": 1}

    monkeypatch.setattr(loop, "_intake", fake_intake)
    monkeypatch.setattr(loop, "_call",
                        lambda slot, m, **k: (COA_JSON if slot.get("provider") == "planner-prov" else REVIEW_JSON, ""))
    loop.maybe_amdp_context(MULTISTEP, [], ENABLED_CFG)
    assert seen["t"] == 4.0  # default intake_timeout_s
