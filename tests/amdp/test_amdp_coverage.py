"""Coverage-gap tests for the AMDP planning layer.

These complement ``test_amdp.py`` by exercising the code paths that the
original 11 tests leave unproven: the COA parse-failure + repair-retry path,
the empty-planner-output audit branch, the reviewer-failed sentinel, schema
coercion/clamping, the scoring tie-break and the v0 defect the report claims,
``_estimate_steps`` scoring, config coercion of bad scalar types, the plan
renderer, and the stale-refuse audit record.

Same hermetic style as ``test_amdp.py``: model calls (``_call``) and state
intake (``_intake``) are monkeypatched, and ``_audit_path`` is redirected to a
tmp file, so nothing hits a real model, the proprioception collector, or the
live HERMES_HOME.
"""

from __future__ import annotations

import json

import pytest

from agent.amdp import loop, schemas, scoring
from agent.amdp.config import AmdpConfigError, resolve_amdp_config

# --------------------------------------------------------------------------- #
# Shared fixtures / constants (mirrors test_amdp.py)
# --------------------------------------------------------------------------- #
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


# --------------------------------------------------------------------------- #
# G3 partial: planner COA parse failure -> repair retry SUCCEEDS on 2nd call
# --------------------------------------------------------------------------- #
def test_planner_repair_retry_succeeds(monkeypatch, healthy_state, audit_tmp):
    """First planner reply is unparseable prose; the repair prompt returns valid
    JSON. AMDP should recover, plan, and audit — and the planner is called
    exactly twice (initial + one repair), reviewer once per surviving COA."""
    slots = []

    def fake_call(slot, messages, *, temperature, max_tokens, json_mode=False):
        prov = slot.get("provider")
        slots.append(prov)
        if prov == "planner-prov":
            # 1st planner call -> junk (no JSON); 2nd -> valid.
            n_planner = slots.count("planner-prov")
            return ("sorry, here are some thoughts" if n_planner == 1 else COA_JSON), ""
        return REVIEW_JSON, ""

    monkeypatch.setattr(loop, "_call", fake_call)
    out = loop.maybe_amdp_context(MULTISTEP_PROMPT, [], ENABLED_CFG)
    assert out and "AMDP plan" in out
    assert slots.count("planner-prov") == 2   # initial + repair
    assert slots.count("reviewer-prov") == 2  # one per COA
    rec = json.loads(audit_tmp.read_text(encoding="utf-8").strip())
    assert rec["refused"] is False
    # The retry path records the parse error even though it ultimately recovered.
    assert any("parse failed" in e for e in rec["errors"])


# --------------------------------------------------------------------------- #
# G3 negative: planner NEVER produces valid COAs -> no plan, audited coas:0,
# reviewer never called
# --------------------------------------------------------------------------- #
def test_planner_all_garbage_produces_no_plan_and_audits_zero(monkeypatch, healthy_state, audit_tmp):
    calls = []

    def fake_call(slot, messages, *, temperature, max_tokens, json_mode=False):
        calls.append(slot.get("provider"))
        return "still not json", ""  # both initial and repair fail to parse

    monkeypatch.setattr(loop, "_call", fake_call)
    out = loop.maybe_amdp_context(MULTISTEP_PROMPT, [], ENABLED_CFG)
    assert out == ""  # fail-closed, no plan injected
    assert calls == ["planner-prov", "planner-prov"]  # 2 planner tries, NO reviewer
    rec = json.loads(audit_tmp.read_text(encoding="utf-8").strip())
    assert rec["coas"] == 0
    assert any("parse failed after repair" in e for e in rec["errors"])


def test_planner_empty_output_audits_zero(monkeypatch, healthy_state, audit_tmp):
    """Planner returns empty text both times -> commander failed, no repair COAs."""
    def fake_call(slot, messages, *, temperature, max_tokens, json_mode=False):
        return "", "TimeoutError: x"

    monkeypatch.setattr(loop, "_call", fake_call)
    out = loop.maybe_amdp_context(MULTISTEP_PROMPT, [], ENABLED_CFG)
    assert out == ""
    rec = json.loads(audit_tmp.read_text(encoding="utf-8").strip())
    assert rec["coas"] == 0
    assert any("commander failed" in e for e in rec["errors"])


# --------------------------------------------------------------------------- #
# Reviewer-failed sentinel: a COA whose reviewer never returns valid JSON is
# NOT dropped — it gets a max-penalty unvetted verdict and is scored/audited.
# --------------------------------------------------------------------------- #
def test_reviewer_failure_yields_unvetted_sentinel(monkeypatch):
    def empty_call(slot, messages, *, temperature, max_tokens, json_mode=False):
        return "", "boom"  # reviewer fails on both the initial and repair call

    monkeypatch.setattr(loop, "_call", empty_call)
    cfg = resolve_amdp_config(ENABLED_CFG)
    verdict = loop._review_one(cfg, "intent", "brief", {"coa_id": "A"})
    assert verdict["_review_failed"] is True
    assert verdict["alignment_1to10"] == 0.0
    assert verdict["fragility_0to1"] == 1.0
    assert verdict["risks"] and verdict["risks"][0]["severity_1to5"] == 5


def test_reviewer_failure_does_not_crash_full_turn(monkeypatch, healthy_state, audit_tmp):
    """Planner OK but reviewer always fails: turn still produces a plan (from the
    worst-case sentinel verdicts) and audits — reviewer failure is fail-soft."""
    def fake_call(slot, messages, *, temperature, max_tokens, json_mode=False):
        if slot.get("provider") == "planner-prov":
            return COA_JSON, ""
        return "", "reviewer down"  # every reviewer call fails

    monkeypatch.setattr(loop, "_call", fake_call)
    out = loop.maybe_amdp_context(MULTISTEP_PROMPT, [], ENABLED_CFG)
    assert out and "AMDP plan" in out
    rec = json.loads(audit_tmp.read_text(encoding="utf-8").strip())
    assert all(r.get("_review_failed") for r in rec["reviews"])
    assert rec["chosen"] in ("A", "B")


# --------------------------------------------------------------------------- #
# G4 partial fix: stale-state refusal MUST also write an audit record
# (test_amdp.py::test_stale_state_refuses never checks the audit).
# --------------------------------------------------------------------------- #
def test_stale_state_refusal_is_audited(monkeypatch, audit_tmp):
    monkeypatch.setattr(loop, "_intake", lambda config, timeout_s=None: {
        "brief": "", "sensors_down": [], "staleness_s": 999.0,
        "verdict": "ok", "gateway_state": "running", "system_count": 3,
    })
    out = loop.maybe_amdp_context(MULTISTEP_PROMPT, [], ENABLED_CFG)
    assert out == ""
    rec = json.loads(audit_tmp.read_text(encoding="utf-8").strip())
    assert rec["refused"] is True
    assert "staleness" in rec["refuse_reason"]
    assert rec["believed_state"]["staleness_s"] == 999.0


# --------------------------------------------------------------------------- #
# _should_refuse direct unit coverage (both branches + the pass case)
# --------------------------------------------------------------------------- #
def test_should_refuse_branches():
    # Only a missing gateway status (truly blind) refuses — a down dashboard does not.
    down, reason = loop._should_refuse(
        {"sensors_down": ["gateway-status"], "staleness_s": 0}, staleness_max_s=120)
    assert down and "gateway status" in reason
    # A down dashboard alone (not in the blinding set) does NOT refuse.
    nodash, _ = loop._should_refuse(
        {"sensors_down": [], "staleness_s": 0}, staleness_max_s=120)
    assert nodash is False
    stale, reason = loop._should_refuse(
        {"sensors_down": [], "staleness_s": 500}, staleness_max_s=120)
    assert stale and "staleness" in reason
    ok, reason = loop._should_refuse(
        {"sensors_down": [], "staleness_s": 10}, staleness_max_s=120)
    assert ok is False and reason == ""


# --------------------------------------------------------------------------- #
# Gate estimator: dispatch-worthiness heuristic (G2 internal)
# --------------------------------------------------------------------------- #
def test_estimate_steps_signals():
    # trivial one-liner scores 0
    assert loop._estimate_steps("hi there", []) == 0
    # multistep hints fire
    assert loop._estimate_steps("migrate then deploy the pipeline", []) >= 2
    # markdown list items count
    assert loop._estimate_steps("do:\n- a\n- b", []) >= 2
    # long prompt gets a bump
    assert loop._estimate_steps("x" * 500, []) >= 1
    # >=3 tool messages in history bumps the score
    tool_hist = [{"role": "tool"}] * 3
    assert loop._estimate_steps("short", tool_hist) >= 1


def test_gate_uses_estimate_below_threshold(monkeypatch):
    """A qualifying-content prompt still skips if min_estimated_steps is high,
    and no model or intake call happens."""
    called = {"intake": 0, "model": 0}
    monkeypatch.setattr(loop, "_intake", lambda c, timeout_s=None: called.__setitem__("intake", called["intake"] + 1) or {})
    monkeypatch.setattr(loop, "_call", lambda *a, **k: called.__setitem__("model", called["model"] + 1) or ("", ""))
    cfg = {"amdp": dict(ENABLED_CFG["amdp"], gate={"min_estimated_steps": 99})}
    out = loop.maybe_amdp_context(MULTISTEP_PROMPT, [], cfg)
    assert out == ""
    assert called == {"intake": 0, "model": 0}  # gate is BEFORE intake and model


# --------------------------------------------------------------------------- #
# schemas.extract_json — the three shapes + the failure
# --------------------------------------------------------------------------- #
def test_extract_json_shapes():
    assert schemas.extract_json('{"a": 1}') == {"a": 1}                       # clean
    assert schemas.extract_json('```json\n{"a": 1}\n```') == {"a": 1}          # fenced
    assert schemas.extract_json('```\n{"a": 1}\n```') == {"a": 1}             # bare fence
    assert schemas.extract_json('blah {"coas": [1]} tail') == {"coas": [1]}   # prose-embedded
    with pytest.raises(ValueError):
        schemas.extract_json("")
    with pytest.raises(ValueError):
        schemas.extract_json("no json here")


# --------------------------------------------------------------------------- #
# schemas.coerce_coas — bare list, wrong wrapper key, all-invalid, kind coercion
# --------------------------------------------------------------------------- #
def test_coerce_coas_accepts_bare_list_and_wrong_key():
    bare = schemas.coerce_coas([{"dispatches": [{"task": "t"}]}])
    assert [c["coa_id"] for c in bare] == ["A"]
    # model wrapped the list under a non-"coas" key -> defensive grab
    wrapped = schemas.coerce_coas({"plans": [{"dispatches": [{"task": "t"}]}]})
    assert wrapped[0]["dispatches"][0]["task"] == "t"


def test_coerce_coas_all_invalid_raises():
    with pytest.raises(ValueError):
        schemas.coerce_coas({"coas": [{"summary": "no dispatches here"}]})
    with pytest.raises(ValueError):
        schemas.coerce_coas({"coas": []})


def test_coerce_coa_kind_and_irreversible_coercion():
    coa = schemas.coerce_coa(
        {"coa_id": "X", "dispatches": [
            {"task": "t1", "kind": "WEIRD"},          # unknown kind -> "act"
            {"task": "t2", "kind": "observe", "irreversible": 1},  # truthy -> True
            {"kind": "act"},                           # no task -> dropped
        ]},
        "fallback",
    )
    assert coa["coa_id"] == "X"
    assert len(coa["dispatches"]) == 2  # the taskless one is dropped
    assert coa["dispatches"][0]["kind"] == "act"      # unknown coerced
    assert coa["dispatches"][1]["irreversible"] is True


def test_coerce_coa_no_dispatches_raises():
    with pytest.raises(ValueError):
        schemas.coerce_coa({"coa_id": "X", "dispatches": []}, "f")
    with pytest.raises(ValueError):
        schemas.coerce_coa("not a dict", "f")


# --------------------------------------------------------------------------- #
# schemas.coerce_review — clamping + defaults + risk filtering
# --------------------------------------------------------------------------- #
def test_coerce_review_clamps_and_defaults():
    r = schemas.coerce_review(
        {"alignment_1to10": 99, "fragility_0to1": 5.0,
         "risks": [{"desc": "x", "severity_1to5": 42},
                   {"desc": "", "severity_1to5": 3},   # blank desc -> dropped
                   "not a dict"]},                      # non-dict -> dropped
        "A",
    )
    assert r["alignment_1to10"] == 10          # clamped hi
    assert r["fragility_0to1"] == 1.0          # clamped hi
    assert len(r["risks"]) == 1 and r["risks"][0]["severity_1to5"] == 5  # sev clamped

    lo = schemas.coerce_review({"alignment_1to10": -3, "fragility_0to1": -1}, "B")
    assert lo["alignment_1to10"] == 1 and lo["fragility_0to1"] == 0.0

    d = schemas.coerce_review({}, "Z")
    assert d["alignment_1to10"] == 5 and d["fragility_0to1"] == 0.5 and d["risks"] == []
    assert d["coa_id"] == "Z"  # falls back to passed id when obj omits it


def test_coerce_review_non_dict_raises():
    with pytest.raises(ValueError):
        schemas.coerce_review(["nope"], "A")


# --------------------------------------------------------------------------- #
# scoring — the v0 defect the report claims, the v1 fix, tie-break, unknown prof
# --------------------------------------------------------------------------- #
def test_v0_paranoid_penalizes_thoroughness_but_v1_does_not():
    """A well-aligned COA that surfaces MANY risks is punished into last place by
    v0_paranoid (Σseverity² blows up), but correctly wins under v1_balanced."""
    aligned_thorough = {"alignment_1to10": 9, "fragility_0to1": 0.1,
                        "risks": [{"desc": str(i), "severity_1to5": 3} for i in range(7)]}
    weak_lazy = {"alignment_1to10": 6, "fragility_0to1": 0.1,
                 "risks": [{"desc": "a", "severity_1to5": 3}]}
    v0_thorough = scoring.score(aligned_thorough, staleness_norm=0, profile="v0_paranoid").score
    v0_lazy = scoring.score(weak_lazy, staleness_norm=0, profile="v0_paranoid").score
    v1_thorough = scoring.score(aligned_thorough, staleness_norm=0, profile="v1_balanced").score
    v1_lazy = scoring.score(weak_lazy, staleness_norm=0, profile="v1_balanced").score
    # v0 inverts the correct ranking (defect); v1 restores it.
    assert v0_thorough < v0_lazy
    assert v1_thorough > v1_lazy


def test_scoring_unknown_profile_falls_back_to_v1():
    review = {"alignment_1to10": 6, "fragility_0to1": 0.1, "risks": [{"desc": "a", "severity_1to5": 3}]}
    a = scoring.score(review, staleness_norm=0, profile="does-not-exist").score
    b = scoring.score(review, staleness_norm=0, profile="v1_balanced").score
    assert a == b


def test_scoring_ignores_malformed_severities():
    review = {"alignment_1to10": 5, "fragility_0to1": 0.0,
              "risks": [{"desc": "a", "severity_1to5": "high"},   # non-int -> dropped
                        {"desc": "b", "severity_1to5": 0},         # 0 -> dropped (>0 filter)
                        {"desc": "c", "severity_1to5": 4}]}
    comp = scoring.score_v1_params(review, staleness_norm=0).components
    assert comp["n_risks"] == 1  # only the valid, positive severity survives


def test_decide_tie_break_prefers_fewer_dispatches(monkeypatch):
    cfg = resolve_amdp_config(ENABLED_CFG)
    coas = [
        {"coa_id": "A", "summary": "", "dispatches": [{"task": "1", "kind": "act"},
                                                      {"task": "2", "kind": "act"}]},
        {"coa_id": "B", "summary": "", "dispatches": [{"task": "1", "kind": "act"}]},
    ]
    reviews = [
        {"coa_id": "A", "alignment_1to10": 7, "risks": [], "fragility_0to1": 0.0},
        {"coa_id": "B", "alignment_1to10": 7, "risks": [], "fragility_0to1": 0.0},
    ]
    best, _ = loop._decide(cfg, coas, reviews, 0.0)
    assert best["coa"]["coa_id"] == "B"  # equal score -> the shorter plan wins


def test_decide_handles_missing_review_for_coa():
    """A COA with no matching review gets a synthesized worst-case verdict rather
    than crashing the decision."""
    cfg = resolve_amdp_config(ENABLED_CFG)
    coas = [{"coa_id": "A", "summary": "", "dispatches": [{"task": "1", "kind": "act"}]}]
    best, scored = loop._decide(cfg, coas, reviews=[], staleness_norm=0.0)
    assert best["coa"]["coa_id"] == "A"
    assert scored[0]["review"]["alignment_1to10"] == 0.0


# --------------------------------------------------------------------------- #
# _render_plan — irreversible flag, success criteria, risk sort, assumptions
# --------------------------------------------------------------------------- #
def test_render_plan_marks_irreversible_and_sorts_risks():
    cfg = resolve_amdp_config(ENABLED_CFG)
    best = {
        "coa": {
            "coa_id": "B", "summary": "act fast",
            "dispatches": [
                {"task": "delete prod", "kind": "act", "irreversible": True,
                 "success_criteria": ["gone"]},
                {"task": "log it", "kind": "observe", "irreversible": False},
            ],
            "assumptions": ["db exists", "creds valid"],
        },
        "review": {
            "coa_id": "B", "alignment_1to10": 7, "fragility_0to1": 0.4,
            "risks": [{"desc": "minor", "severity_1to5": 2},
                      {"desc": "catastrophic", "severity_1to5": 5}],
        },
    }
    txt = loop._render_plan(best, cfg)
    assert "IRREVERSIBLE" in txt
    assert "delete prod" in txt and "success: gone" in txt
    assert "Assumptions: db exists; creds valid" in txt
    # highest-severity risk is listed first
    watch = txt.split("Watch for:")[1]
    assert watch.index("catastrophic") < watch.index("minor")


# --------------------------------------------------------------------------- #
# config coercion — bad scalar types fall back to defaults; n_coas floor;
# reviewer_max_tokens sentinel handling
# --------------------------------------------------------------------------- #
def test_config_coerces_bad_scalar_types_to_defaults():
    cfg = resolve_amdp_config({"amdp": {
        "enabled": True,
        "planner": {"provider": "p", "model": "m"},
        "reviewer": {"provider": "r", "model": "m"},
        "n_coas": "lots",                       # bad -> default 3
        "gate": {"min_estimated_steps": "many"},  # bad -> default 3
        "staleness_max_s": "soon",              # bad -> default 120.0
    }})
    assert cfg.n_coas == 3
    assert cfg.min_estimated_steps == 3
    assert cfg.staleness_max_s == 120.0


def test_config_allows_single_plan_and_floors_min_steps_at_one():
    cfg = resolve_amdp_config({"amdp": {
        "enabled": True,
        "planner": {"provider": "p", "model": "m"},
        "reviewer": {"provider": "r", "model": "m"},
        "n_coas": 1,
        "gate": {"min_estimated_steps": 0},  # below floor -> 1
    }})
    assert cfg.n_coas == 1
    assert cfg.min_estimated_steps == 1


def test_background_intents_are_excluded_before_state_or_model_calls(monkeypatch):
    cfg = {"amdp": {
        "enabled": True,
        "planner": {"provider": "p", "model": "m"},
        "reviewer": {"provider": "r", "model": "m"},
        "n_coas": 1,
        "gate": {"min_estimated_steps": 1},
        "exclude_background": True,
    }}
    monkeypatch.setattr(loop, "_intake", lambda *a, **k: pytest.fail("state intake should be bypassed"))
    monkeypatch.setattr(loop, "_call", lambda *a, **k: pytest.fail("models should be bypassed"))
    prompt = "[/learn] Review the conversation and update the skill library end-to-end."
    assert loop.maybe_amdp_context(prompt, [], cfg) == ""


def test_single_routine_plan_skips_reviewer(monkeypatch, healthy_state, audit_tmp):
    cfg = resolve_amdp_config({"amdp": {
        "enabled": True,
        "planner": {"provider": "p", "model": "m"},
        "reviewer": {"provider": "r", "model": "m"},
        "n_coas": 1,
        "gate": {"min_estimated_steps": 1},
        "review_only_on_risk_or_disagreement": True,
    }})
    coa = {"coa_id": "A", "summary": "safe plan", "dispatches": [
        {"task": "read and reconcile files", "kind": "act", "irreversible": False}
    ]}
    monkeypatch.setattr(loop, "_generate_coas", lambda *a, **k: [coa])
    monkeypatch.setattr(loop, "_review_all", lambda *a, **k: pytest.fail("reviewer should be skipped"))
    out = loop._plan_turn(cfg, {"amdp": {}}, "reconcile each file end-to-end", [])
    assert "safe plan" in out
    rec = json.loads(audit_tmp.read_text(encoding="utf-8").splitlines()[-1])
    assert rec["review_invoked"] is False


def test_irreversible_single_plan_invokes_reviewer(monkeypatch, healthy_state, audit_tmp):
    cfg = resolve_amdp_config({"amdp": {
        "enabled": True,
        "planner": {"provider": "p", "model": "m"},
        "reviewer": {"provider": "r", "model": "m"},
        "n_coas": 1,
        "gate": {"min_estimated_steps": 1},
        "review_only_on_risk_or_disagreement": True,
    }})
    coa = {"coa_id": "A", "summary": "risky plan", "dispatches": [
        {"task": "delete production records", "kind": "act", "irreversible": True}
    ]}
    review = {"coa_id": "A", "alignment_1to10": 8, "risks": [], "fragility_0to1": 0.2}
    seen = {"reviewed": False}
    monkeypatch.setattr(loop, "_generate_coas", lambda *a, **k: [coa])
    monkeypatch.setattr(loop, "_review_all", lambda *a, **k: seen.__setitem__("reviewed", True) or [review])
    out = loop._plan_turn(cfg, {"amdp": {}}, "delete production records safely end-to-end", [])
    assert seen["reviewed"] is True and "risky plan" in out


def test_config_reviewer_max_tokens_sentinels():
    def mk(rmt):
        return resolve_amdp_config({"amdp": {
            "enabled": True,
            "planner": {"provider": "p", "model": "m"},
            "reviewer": {"provider": "r", "model": "m"},
            "reviewer_max_tokens": rmt,
        }}).reviewer_max_tokens

    assert mk(0) is None
    assert mk(None) is None
    assert mk("none") is None
    assert mk("") is None
    assert mk(2048) == 2048


def test_config_whitespace_only_slot_hard_fails():
    """A provider/model of pure whitespace is NOT a valid slot — must hard-fail,
    not silently pass through to a default/cloud model."""
    with pytest.raises(AmdpConfigError):
        resolve_amdp_config({"amdp": {
            "enabled": True,
            "planner": {"provider": "   ", "model": "m"},
            "reviewer": {"provider": "r", "model": "m"},
        }})


def test_config_non_dict_amdp_block_returns_none():
    assert resolve_amdp_config({"amdp": "enabled please"}) is None
    assert resolve_amdp_config({"amdp": None}) is None
    assert resolve_amdp_config(None) is None


# --------------------------------------------------------------------------- #
# G7 fail-closed: exceptions at OTHER sites than the planner are also swallowed.
# --------------------------------------------------------------------------- #
def test_intake_exception_is_swallowed(monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("intake blew up")

    monkeypatch.setattr(loop, "_intake", boom)
    out = loop.maybe_amdp_context(MULTISTEP_PROMPT, [], ENABLED_CFG)
    assert out == ""


def test_decide_exception_is_swallowed(monkeypatch, healthy_state):
    def fake_call(slot, messages, *, temperature, max_tokens, json_mode=False):
        if slot.get("provider") == "planner-prov":
            return COA_JSON, ""
        return REVIEW_JSON, ""

    def boom(*a, **k):
        raise RuntimeError("decide blew up")

    monkeypatch.setattr(loop, "_call", fake_call)
    monkeypatch.setattr(loop, "_decide", boom)
    out = loop.maybe_amdp_context(MULTISTEP_PROMPT, [], ENABLED_CFG)
    assert out == ""  # exception downstream of planning still yields "" not a raise


def test_render_exception_is_swallowed(monkeypatch, healthy_state):
    def fake_call(slot, messages, *, temperature, max_tokens, json_mode=False):
        if slot.get("provider") == "planner-prov":
            return COA_JSON, ""
        return REVIEW_JSON, ""

    def boom(*a, **k):
        raise RuntimeError("render blew up")

    monkeypatch.setattr(loop, "_call", fake_call)
    monkeypatch.setattr(loop, "_render_plan", boom)
    out = loop.maybe_amdp_context(MULTISTEP_PROMPT, [], ENABLED_CFG)
    assert out == ""


# --------------------------------------------------------------------------- #
# _extract_text robustness (beyond the happy fallback in test_amdp.py)
# --------------------------------------------------------------------------- #
def test_extract_text_handles_bad_shapes():
    class _Resp:
        def __init__(self, choices):
            self.choices = choices

    assert loop._extract_text(_Resp([])) == ""          # no choices
    assert loop._extract_text(object()) == ""            # no .choices attr
    # all candidate fields empty/blank -> ""
    class _C:
        message = {"content": "   ", "reasoning_content": "", "reasoning": None}
    assert loop._extract_text(_Resp([_C()])) == ""
