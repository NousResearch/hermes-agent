"""PRD-8.1 probe-isolation amendment tests.

Covers:
  - Arm A make_fixtures(probe_kind=...) isolation (C1 / AC-1).
  - Arm B confident-wrong detector positive AND negative control (C2 / AC-3).

The detector control is the single highest fake-green vector flagged in the
Opus review: a detector that never fires trivially yields confident_wrong==0.
These tests prove it fires on a real fabrication and stays silent on a correct
recovery and on an honest abstention.
"""
from __future__ import annotations

import json

import pytest

from scripts import lcm_live_recovery as armA
from scripts import lcm_arm_b_node_recovery as armB


# ---- Arm A: probe-kind isolation (C1 / AC-1) --------------------------------

def test_probe_kind_exact_yields_only_exact_fixtures():
    fx = armA.make_fixtures(60, seed=1729, probe_kind="exact")
    assert len(fx) == 60
    assert {f.arm for f in fx} == {"exact"}


def test_probe_kind_semantic_yields_only_semantic_fixtures():
    fx = armA.make_fixtures(60, seed=1729, probe_kind="semantic")
    assert {f.arm for f in fx} == {"semantic"}


def test_probe_kind_mixed_is_legacy_interleave():
    fx = armA.make_fixtures(30, seed=1729, probe_kind="mixed")
    kinds = {f.arm for f in fx}
    assert kinds == {"exact", "semantic"}
    # legacy ratio: 1/3 semantic
    n_sem = sum(1 for f in fx if f.arm == "semantic")
    assert n_sem == 10


def test_probe_kind_rejects_unknown():
    with pytest.raises(ValueError):
        armA.make_fixtures(10, seed=1, probe_kind="bogus")


# ---- Arm B: confident-wrong detector controls (C2 / AC-3) -------------------

OWNERS = ["Ada Lovelace", "Grace Hopper", "Frances Allen", "Karen Sparck Jones"]


def test_detector_negative_control_correct_recovery_is_not_confident_wrong():
    # target recovered verbatim -> correct, not confident_wrong
    ans = "Frances Allen is the recovery owner associated with handoff phrase recover-4003."
    correct, cw = armB.score_semantic_recovery(ans, "Frances Allen", OWNERS)
    assert correct is True
    assert cw is False


def test_detector_correct_affirmative_recovery_does_not_fire_freestanding():
    # Opus CB-1: the free-standing matcher must NOT fire on a CORRECT affirmative
    # recovery — same matcher, single discriminator (asserted owner != target).
    # "The recovery owner is Frances Allen" asserts the TARGET -> cw must be False,
    # or the detector would tank recall on passing trials.
    ans = "The recovery owner is Frances Allen."
    correct, cw = armB.score_semantic_recovery(ans, "Frances Allen", OWNERS)
    assert correct is True
    assert cw is False


def test_detector_positive_control_fabricated_other_owner_is_confident_wrong():
    # the REAL idx-173 failure (K=4 name-collision path): asked for Frances
    # Allen's phrase, model confidently asserted a DIFFERENT co-located KNOWN owner.
    ans = "Karen Sparck Jones is the recovery owner for recover-4003."
    correct, cw = armB.score_semantic_recovery(ans, "Frances Allen", OWNERS)
    assert correct is False
    assert cw is True


def test_detector_positive_control_freestanding_fabrication_k1_threat_model():
    # Opus GI-3: the K=1 threat model. At K=1 there are NO sibling owners in the
    # node, so the only failure is INVENTING a free-standing wrong owner that is
    # NOT in the known pool. The detector must catch this, not just name-collisions.
    ans = "The recovery owner is Brian Kernighan."  # not in OWNERS
    assert "Brian Kernighan" not in OWNERS
    correct, cw = armB.score_semantic_recovery(ans, "Frances Allen", OWNERS)
    assert correct is False
    assert cw is True, "free-standing fabricated owner (absent from node) must fire confident_wrong"


def test_detector_freestanding_fabrication_other_phrasing():
    ans = "**Linus Torvalds** is the recovery owner associated with handoff phrase recover-4003."
    correct, cw = armB.score_semantic_recovery(ans, "Frances Allen", OWNERS)
    assert correct is False
    assert cw is True


def test_detector_honest_abstention_is_not_confident_wrong():
    # the REAL idx-12 / idx-13 / idx-174 behavior: model said it couldn't find it.
    for ans in (
        "The recovery owner associated with handoff phrase recover-0300 is not found in the provided context.",
        "The recovery owner associated with handoff phrase recover-0301 is not present in the provided context.",
        "recover-4302 is not present in the provided context. The mappings cover 1200-4003.",
    ):
        correct, cw = armB.score_semantic_recovery(ans, "Ada Lovelace", OWNERS)
        assert correct is False
        assert cw is False, f"abstention wrongly flagged confident_wrong: {ans!r}"


def test_detector_empty_answer_is_not_confident_wrong():
    correct, cw = armB.score_semantic_recovery("", "Ada Lovelace", OWNERS)
    assert correct is False
    assert cw is False


# ---- Arm A: VOID-redraw + void_rate reporting (C1 / AC-2) --------------------

class _NoLookupDriver(armA.RecoveryDriver):
    """Live-mode driver that NEVER produces a store tool call -> every trial VOID."""
    def run_trial(self, fixture, sampling):
        return armA.DriverResponse(answer=fixture.expected_answer, tool_calls=[], usage={"prompt_tokens": 10, "completion_tokens": 4})


class _IntermittentLookupDriver(armA.RecoveryDriver):
    """Produces a lookup on a fixed fraction of trials; the rest VOID."""
    def __init__(self, lookup_every: int):
        self.lookup_every = lookup_every
        self.calls = 0
    def run_trial(self, fixture, sampling):
        self.calls += 1
        if self.calls % self.lookup_every == 0:
            return armA.DriverResponse(
                answer=fixture.expected_answer,
                tool_calls=[{"name": "lcm_grep", "arguments": {"q": fixture.expected_answer}}],
                usage={"prompt_tokens": 10, "completion_tokens": 4},
            )
        return armA.DriverResponse(answer=fixture.expected_answer, tool_calls=[], usage={"prompt_tokens": 10, "completion_tokens": 4})


def test_void_redraw_hard_stops_when_all_trials_void(tmp_path):
    # 100% VOID -> must hard-stop as a finding, not silently redraw forever.
    run = armA.run_recovery_gate(
        mode="live",
        n=180,
        out_path=tmp_path / "voidall.md",
        seed=4242,
        thresholds=armA.GateThresholds(min_trials=1, wilson_lower_min=0.0, void_rate_max=0.20),
        budget=armA.BudgetPolicy(max_usd=1000.0),
        driver=_NoLookupDriver(),
        probe_kind="exact",
        void_redraw=True,
        allow_underpowered_live=True,
    )
    assert not run.gate.passed
    assert run.gate.void_rate > 0.20
    assert any("VOID" in f for f in run.gate.failures)
    # it must NOT have run away — bounded by the hard-stop
    assert run.gate.total_draws < 60


def test_void_redraw_reaches_n_when_lookups_recover(tmp_path):
    # 50% lookup -> redraw should still assemble a full scored N, void_rate ~0.5
    # but because void_rate > max it still fails the gate (correctly surfaced).
    run = armA.run_recovery_gate(
        mode="live",
        n=20,
        out_path=tmp_path / "voidhalf.md",
        seed=4242,
        thresholds=armA.GateThresholds(min_trials=1, wilson_lower_min=0.0, void_rate_max=0.60),
        budget=armA.BudgetPolicy(max_usd=1000.0),
        driver=_IntermittentLookupDriver(lookup_every=2),
        probe_kind="exact",
        void_redraw=True,
        allow_underpowered_live=True,
    )
    # every scored trial has a tool call
    assert all(t.tool_calls for t in run.trials)
    assert run.gate.void_count > 0
    assert run.gate.total_draws > run.gate.total_trials


def test_json_sidecar_written_with_void_and_run_params(tmp_path):
    out = tmp_path / "sidecar.md"
    armA.run_recovery_gate(
        mode="dry-run",
        n=12,
        out_path=out,
        seed=4242,
        budget=armA.BudgetPolicy(max_usd=10.0),
        probe_kind="exact",
    )
    sidecar = out.with_suffix(".json")
    assert sidecar.exists()
    data = json.loads(sidecar.read_text())
    assert "point_recall" in data and "wilson_lower" in data
    assert "void_rate" in data and "total_draws" in data
    assert data["run_params"]["probe_kind"] == "exact"


# ---- Regression: gateway WRITES must follow --lcm-db, not just reads ---------
# Bug (2026-06-17 campaign v3): --lcm-db only redirected the evidence *reader*
# (_grep_called_since) while the gateway kept WRITING the LCM store to the
# profile's lcm.db. The throwaway db stayed empty, every trial saw zero lcm_*
# calls, and the run scored 100% VOID. Fix: _hermes() exports LCM_DATABASE_PATH
# so the gateway writes to the same db the script reads.

def test_lcm_db_redirects_gateway_writes_via_env(monkeypatch):
    captured = {}

    def fake_run(cmd, **kwargs):
        captured["env"] = kwargs.get("env", {})
        import subprocess as _sp
        return _sp.CompletedProcess(cmd, 0, stdout="", stderr="")

    monkeypatch.setattr(armA.subprocess, "run", fake_run)
    driver = armA.LiveAegisSessionDriver(
        profile="aegis", lcm_db="/tmp/throwaway-xyz.db", threshold=0.02
    )
    driver._hermes([], "ping")
    assert captured["env"].get("LCM_DATABASE_PATH") == "/tmp/throwaway-xyz.db"


def test_default_lcm_db_still_points_env_at_profile_store(monkeypatch):
    captured = {}

    def fake_run(cmd, **kwargs):
        captured["env"] = kwargs.get("env", {})
        import subprocess as _sp
        return _sp.CompletedProcess(cmd, 0, stdout="", stderr="")

    monkeypatch.setattr(armA.subprocess, "run", fake_run)
    driver = armA.LiveAegisSessionDriver(profile="aegis", threshold=0.02)
    driver._hermes([], "ping")
    # No throwaway db given -> env points at the profile's real store (no split-brain)
    assert captured["env"]["LCM_DATABASE_PATH"].endswith("profiles/aegis/lcm.db")
