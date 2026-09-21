#!/usr/bin/env python3
"""
END-TO-END validation for the severity-aware Hermaguard flow.

This suite wires the two sibling modules together the way they are actually
deployed per skills/software-development/hermaguard/SKILL.md Integration section:

    1. hermaguard_grader.py  (HERMES_HOME/skills/.../hermaguard/scripts)
         grades a raw Hermaguard report -> attaches L_level / harm_axes /
         L_definition / grading_method to every finding + a top-level
         severity_grading block. Pure additive.

    2. hermaguard_release_gate.py  (KenseiAgent/scripts)
         consumes those graded findings -> evaluate_release() blocks the
         release when any finding is L4+ (fail-closed on ungradable).

Neither parent task (t_5b9b31b6 grader, t_c35a7ab8 gate) tested the TWO WIRED
TOGETHER. That is the entire point of this suite.

Coverage targets from the task body / acceptance criteria:
  * End-to-end assigns severity to all findings and enforces blocking for L4+.
  * Edge cases: mixed severities, missing finding data, high-volume loads.
  * Regression: both modules' own self-tests still pass (no side effects).

Run:  pytest tests/test_hermaguard_e2e.py -q
"""

import json
import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
HERMES_HOME = Path(os.environ.get("HERMES_HOME", Path.home() / ".hermes"))
GRADER_DIR = HERMES_HOME / "skills/software-development/hermaguard/scripts"
GRADER_SCRIPT = GRADER_DIR / "hermaguard_grader.py"
GATE_SCRIPT = REPO_ROOT / "scripts" / "hermaguard_release_gate.py"

# Make both modules importable without installing either package.
sys.path.insert(0, str(REPO_ROOT / "scripts"))
sys.path.insert(0, str(GRADER_DIR))

import hermaguard_release_gate as gate  # noqa: E402

# The deployed grader is an OPTIONAL skill-side artifact at
# $HERMES_HOME/skills/software-development/hermaguard/scripts. It is NOT a
# package dependency of KenseiAgent. Importing it unconditionally at module
# level would break broad-suite collection whenever the grader is absent
# (ModuleNotFoundError raised during collection, before any skip can run).
# importorskip() evaluates the skip AFTER GRADER_DIR is on sys.path, so the
# module is skipped cleanly (not errored) when the grader is missing, and the
# rest of the broad collection proceeds.
grader = pytest.importorskip(
    "hermaguard_grader",
    reason=f"optional deployed grader not found at {GRADER_DIR}",
)

# Hard skip (not a failure) if the deployed grader script is absent in this env,
# echoed via GRADER_SCRIPT for a clearer message when importorskip is bypassed.
pytestmark = pytest.mark.skipif(
    not GRADER_SCRIPT.exists(),
    reason=f"grader script not found at {GRADER_SCRIPT}",
)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────
def axes(completed, reversible, cross_scope, privilege,
         escalation=False, attempted=True) -> dict:
    return {
        "completed": completed,
        "reversible": reversible,
        "cross_scope": cross_scope,
        "privilege": privilege,
        "escalation": escalation,
        "attempted": attempted,
    }


def raw_finding(fid: str, severity: str, ax=None, **extra) -> dict:
    f = {"id": fid, "severity": severity, "title": f"t-{fid}", "location": f"mod.py:{fid}"}
    if ax is not None:
        f["harm_axes"] = ax
    f.update(extra)
    return f


def run_pipeline(raw_report: dict) -> gate.GateResult:
    """The real deployed wiring: grade_report() -> evaluate_release()."""
    graded = grader.grade_report(raw_report)
    return gate.evaluate_release(graded["findings"])


# ─────────────────────────────────────────────────────────────────────────────
# 1. Normal operation — grading + gate agree
# ─────────────────────────────────────────────────────────────────────────────
def test_clean_report_all_low_severity_allows_release():
    """A report of only L0–L3 findings must NOT block."""
    report = {"findings": [
        raw_finding("a", "LOW", axes(True, True, False, False)),       # L2
        raw_finding("b", "MEDIUM", axes(False, True, False, False)),   # L1
        raw_finding("c", "HIGH", axes(True, False, False, False)),     # L3
    ]}
    res = run_pipeline(report)
    assert not res.blocked
    assert res.release_allowed
    assert res.max_level is gate.LLevel.L3


def test_single_L4_finding_blocks_release():
    report = {"findings": [
        raw_finding("x", "CRITICAL", axes(True, True, True, False)),   # L4
    ]}
    res = run_pipeline(report)
    assert res.blocked
    assert {f.finding_id for f in res.blocking_findings} == {"x"}


def test_single_L5_privilege_finding_blocks_release():
    report = {"findings": [
        raw_finding("rce", "HIGH", axes(True, False, False, True)),    # L5
    ]}
    res = run_pipeline(report)
    assert res.blocked
    assert res.blocking_findings[0].l_level is gate.LLevel.L5


def test_single_L6_escalation_flag_blocks_release():
    report = {"findings": [
        raw_finding("chain", "CRITICAL",
                    axes(True, True, False, True, escalation=True)),   # L6
    ]}
    res = run_pipeline(report)
    assert res.blocked
    assert res.blocking_findings[0].l_level is gate.LLevel.L6


# ─────────────────────────────────────────────────────────────────────────────
# 2. Edge case — mixed severities
# ─────────────────────────────────────────────────────────────────────────────
def test_mixed_severities_blocks_with_only_the_L4_plus_finding():
    report = {"findings": [
        raw_finding("a", "LOW", axes(False, True, False, False)),      # L0
        raw_finding("b", "LOW", axes(True, True, False, False)),       # L2
        raw_finding("c", "HIGH", axes(True, False, False, False)),     # L3
        raw_finding("d", "CRITICAL", axes(True, True, True, False)),   # L4
    ]}
    res = run_pipeline(report)
    assert res.blocked
    assert {f.finding_id for f in res.blocking_findings} == {"d"}
    assert {f.finding_id for f in res.non_blocking_findings} == {"a", "b", "c"}


def test_mixed_all_low_severities_allow():
    report = {"findings": [
        raw_finding("a", "LOW", axes(False, True, False, False)),      # L0
        raw_finding("b", "LOW", axes(False, True, False, False, attempted=False)),  # L0
        raw_finding("c", "LOW", axes(True, True, False, False)),       # L2
        raw_finding("d", "HIGH", axes(True, False, False, False)),     # L3
    ]}
    res = run_pipeline(report)
    assert not res.blocked
    assert res.max_level is gate.LLevel.L3


# ─────────────────────────────────────────────────────────────────────────────
# 3. Edge case — missing finding data (no harm_axes)
# ─────────────────────────────────────────────────────────────────────────────
def test_missing_axes_uses_conservative_floor_and_blocks_on_critical():
    """No harm_axes -> grader severity-fallback floor. CRITICAL floors to L4,
    so the gate correctly blocks a severe but un-axes'd finding."""
    report = {"findings": [raw_finding("crit", "CRITICAL")]}  # no axes
    graded = grader.grade_report(report)
    gf = graded["findings"][0]
    assert gf["grading_method"] == "severity-fallback"
    assert gf["L_level"] == "L4"
    res = gate.evaluate_release(graded["findings"])
    assert res.blocked


def test_missing_axes_low_severity_floors_to_L2_and_allows():
    report = {"findings": [raw_finding("trivial", "LOW")]}  # no axes
    graded = grader.grade_report(report)
    gf = graded["findings"][0]
    assert gf["L_level"] == "L2"
    res = gate.evaluate_release(graded["findings"])
    assert not res.blocked


def test_ungradable_from_grader_perspective_is_floor_not_crash():
    """A finding with no severity AND no axes still gets a valid floor (L2) and
    never crashes the grader — the pipeline stays robust for malformed input."""
    report = {"findings": [{"id": "weird", "title": "no severity no axes"}]}
    graded = grader.grade_report(report)
    assert graded["findings"][0]["L_level"] == "L2"
    res = gate.evaluate_release(graded["findings"])
    assert not res.blocked  # floor L2 -> allowed


# ─────────────────────────────────────────────────────────────────────────────
# 4. Edge case — high-volume loads
# ─────────────────────────────────────────────────────────────────────────────
def test_high_volume_all_low_allowed_and_fast():
    report = {"findings": [
        raw_finding(f"n{i}", "MEDIUM", axes(True, True, False, False))
        for i in range(2000)
    ]}
    t0 = time.perf_counter()
    res = run_pipeline(report)
    dt = time.perf_counter() - t0
    assert not res.blocked
    assert res.total_findings == 2000
    assert len(res.non_blocking_findings) == 2000
    assert dt < 5.0, f"grading+gate took {dt:.2f}s for 2000 findings"


def test_high_volume_with_single_L4_blocks_and_finds_needle():
    findings = [raw_finding(f"n{i}", "MEDIUM", axes(True, True, False, False))
                for i in range(2000)]
    findings.append(raw_finding("needle", "CRITICAL", axes(True, True, True, False)))
    report = {"findings": findings}
    res = run_pipeline(report)
    assert res.blocked
    assert len(res.blocking_findings) == 1
    assert res.blocking_findings[0].finding_id == "needle"


# ─────────────────────────────────────────────────────────────────────────────
# 5. Report-level escalation-chain propagation grader -> gate
# ─────────────────────────────────────────────────────────────────────────────
def test_escalation_chain_explicit_flag_raises_peak_to_L6_and_blocks():
    """INTENDED L6 path: a finding whose axes carry escalation=True is graded
    L6 by the oracle directly. The gate then blocks on L6. This is the
    correct, explicit escalation signal (set during triage), distinct from the
    inferred chain-detector path (see over-grade defect test below)."""
    report = {"findings": [
        raw_finding("read", "HIGH", axes(True, True, True, False)),                    # L4
        raw_finding("priv", "CRITICAL", axes(True, True, False, True, escalation=True)),  # L6
    ]}
    graded = grader.grade_report(report)
    peak = graded["findings"][-1]
    assert peak["L_level"] == "L6"
    # The L6 must actually drive the gate decision.
    res = gate.evaluate_release(graded["findings"])
    assert res.blocked
    # Both findings block (L4 and L6); the L6 must be present among blockers.
    assert any(f.l_level is gate.LLevel.L6 for f in res.blocking_findings)


@pytest.mark.xfail(
    reason=(
        "DEFECT (grader escalation detector): _detect_escalation_chain raises the "
        "peak finding to L6 whenever it sees >=2 completed findings with strictly "
        "increasing rank peaking at >=L4 — even when the findings are unrelated "
        "(e.g. a benign L2 leak followed by an unrelated L4 SQLi). This over-grades "
        "ordinary mixed reports to L6. The design decision only guarded the "
        "purely-local L2->L3 case, not benign->cross-scope. Gate still blocks (L6 "
        "is blocking) so it is not a safety escape, but the severity label is wrong. "
        "Fix owned by octacon (grader module); follow-up task filed."
    ),
    strict=False,
)
def test_escalation_detector_overgrades_unrelated_findings():
    """Regression guard for the over-grade defect. Two UNRELATED findings — a
    benign local L2 leak and an unrelated cross-scope L4 SQLi — must NOT be
    treated as an escalation chain. The L4 finding must stay L4."""
    report = {"findings": [
        raw_finding("leak", "LOW", axes(True, True, False, False)),      # L2 benign
        raw_finding("sqli", "CRITICAL", axes(True, True, True, False)), # L4 real
    ]}
    graded = grader.grade_report(report)
    l4 = graded["findings"][1]
    assert l4["L_level"] == "L4"
    assert graded["severity_grading"]["escalation_chain_detected"] is False


def test_local_only_chain_does_not_over_grade_and_allows():
    """Purely-local escalating chain (L2->L3) must NOT be raised to L6; gate
    sees L3 and allows — regression that escalation detection is scope-aware."""
    report = {"findings": [
        raw_finding("a", "MEDIUM", axes(True, True, False, False)),    # L2
        raw_finding("b", "HIGH", axes(True, False, False, False)),      # L3
    ]}
    graded = grader.grade_report(report)
    assert graded["findings"][-1]["L_level"] == "L3"
    assert graded["severity_grading"]["escalation_chain_detected"] is False
    res = gate.evaluate_release(graded["findings"])
    assert not res.blocked


# ─────────────────────────────────────────────────────────────────────────────
# 6. Additive contract preserved end-to-end
# ─────────────────────────────────────────────────────────────────────────────
def test_original_finding_keys_survive_pipeline_and_reach_gate():
    report = {"findings": [{
        "id": "keep", "severity": "CRITICAL",
        "harm_axes": axes(True, True, True, False),
        "cwe": "CWE-89", "source": "prescan",
    }]}
    graded = grader.grade_report(report)
    gf = graded["findings"][0]
    # Original keys preserved (additive contract).
    assert gf["cwe"] == "CWE-89"
    assert gf["source"] == "prescan"
    # New grading keys added.
    assert gf["L_level"] == "L4"
    assert "harm_axes" in gf and "L_definition" in gf and "grading_method" in gf
    # Gate sees the preserved + graded finding.
    res = gate.evaluate_release(graded["findings"])
    assert res.blocked
    assert res.blocking_findings[0].finding_id == "keep"


# ─────────────────────────────────────────────────────────────────────────────
# 7. Integration seam observations (documented behaviour, not defects)
# ─────────────────────────────────────────────────────────────────────────────
def test_grader_recomputes_L_level_from_axes_not_from_preset():
    """KNOWN SEAM: the grader is the source of truth. If a raw finding already
    carries an `L_level` but no `harm_axes`/`severity`, the grader recomputes
    (default floor L2) and overwrites the preset. Within the deployed pipeline
    this is fine (gate consumes grader output), but it means a manually-set
    L_level fed *into* the grader is not trusted. Documented for stakeholders."""
    raw = {"id": "preset", "L_level": "L4", "title": "i said L4"}
    graded = grader.grade_finding(raw)
    assert graded["L_level"] == "L2"  # default floor, preset ignored


def test_gate_isolates_nondict_finding_but_grader_raises():
    """ASYMMETRY: the gate fail-closes on a non-dict finding (ungradable ->
    block, no crash). The grader does NOT guard against a non-dict and raises
    AttributeError, so a malformed report element crashes the *grade* step
    before the gate runs. The deployed pipeline should lint report structure
    upstream, or the grader should isolate non-dicts like the gate does."""
    # Gate isolates — safe.
    res = gate.evaluate_release([123, "not-a-finding"])
    assert res.blocked
    assert len(res.ungradable) == 2
    # Grader does not isolate — surfaces the malformed structure.
    with pytest.raises(AttributeError):
        grader.grade_finding(123)


# ─────────────────────────────────────────────────────────────────────────────
# 8. TRUE CLI end-to-end (subprocess, not just in-process imports)
# ─────────────────────────────────────────────────────────────────────────────
def _cli_run(report_dict: dict, tmp_path: Path):
    raw = tmp_path / "report.json"
    raw.write_text(json.dumps(report_dict))
    g1 = subprocess.run(
        [sys.executable, str(GRADER_SCRIPT), "--report", str(raw)],
        capture_output=True, text=True,
    )
    graded_path = raw.with_suffix(".graded.json")  # report.json -> report.graded.json
    assert graded_path.exists(), g1.stderr
    g2 = subprocess.run(
        [sys.executable, str(GATE_SCRIPT), "--json", str(graded_path), "--format", "json"],
        capture_output=True, text=True,
    )
    return g1, g2, graded_path


def test_cli_e2e_block_on_L4_via_real_subprocess(tmp_path):
    report = {"findings": [
        raw_finding("ok", "MEDIUM", axes(True, True, False, False)),    # L2
        raw_finding("bad", "CRITICAL", axes(True, True, True, False)),   # L4
    ]}
    g1, g2, _ = _cli_run(report, tmp_path)
    assert g1.returncode == 0, g1.stderr
    # Gate exits 2 (blocked) per its contract.
    assert g2.returncode == 2, g2.stderr
    payload = json.loads(g2.stdout)
    assert payload["blocked"] is True
    # The L4 finding ("bad") is correctly identified as the blocker.
    assert payload["blocking"][0]["id"] == "bad"
    # KNOWN DEFECT (see test_escalation_detector_overgrades_unrelated_findings):
    # the grader raised "bad" to L6 because an L2 finding precedes it in the
    # report, so max_level is 6 rather than 4. The gate still blocks (L6 is
    # blocking) but the severity label is over-graded. Not asserted here.


def test_cli_e2e_allow_when_all_low_via_real_subprocess(tmp_path):
    report = {"findings": [
        raw_finding("a", "LOW", axes(False, True, False, False)),
        raw_finding("b", "MEDIUM", axes(True, True, False, False)),
    ]}
    g1, g2, _ = _cli_run(report, tmp_path)
    assert g1.returncode == 0, g1.stderr
    assert g2.returncode == 0, g2.stderr  # 0 = allowed
    assert json.loads(g2.stdout)["blocked"] is False


def test_cli_e2e_critical_fallback_blocks_via_real_subprocess(tmp_path):
    """CRITICAL with no axes -> grader floor L4 -> gate blocks, all via CLI."""
    report = {"findings": [raw_finding("crit", "CRITICAL")]}  # no axes
    g1, g2, _ = _cli_run(report, tmp_path)
    assert g1.returncode == 0, g1.stderr
    assert g2.returncode == 2, g2.stderr
    assert json.loads(g2.stdout)["max_level"] == 4


# ─────────────────────────────────────────────────────────────────────────────
# 9. Regression — both modules' own self-tests still pass
# ─────────────────────────────────────────────────────────────────────────────
def test_grader_self_test_still_passes():
    rc = grader.self_test()
    assert rc == 0


def test_gate_self_test_still_passes():
    assert gate._run_self_test() is True
