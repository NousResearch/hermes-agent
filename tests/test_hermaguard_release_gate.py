#!/usr/bin/env python3
"""
Tests for the Hermaguard release gate (scripts/hermaguard_release_gate.py).

Covers the task acceptance criteria:
  - Releases are automatically blocked if any finding is L4 or higher
  - Releases proceed unimpeded for all findings rated L0-L3
  - Block notifications include the severity tier and details of the triggering
    finding(s)

Plus edge cases called out by the sibling validation task (t_4cb07dc3):
  - mixed severities
  - missing finding data (ungradable => treated as a block, no crash)
  - high-volume finding loads
"""

import sys
from pathlib import Path

import pytest

# Make the scripts/ module importable without installing the package.
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from hermaguard_release_gate import (  # noqa: E402
    BLOCK_THRESHOLD,
    LLevel,
    Finding,
    GateResult,
    UngradableFindingError,
    evaluate_release,
    grade_oracle,
    normalise_finding,
)


# ── helpers ──────────────────────────────────────────────────────────────────
def finding(level: str | int, **overrides) -> dict:
    base = {"id": f"f-{level}", "L_level": level, "title": f"finding {level}",
            "location": "mod.py:1"}
    base.update(overrides)
    return base


# ── per-level blocking behaviour (L0-L6) ─────────────────────────────────────
@pytest.mark.parametrize("level,expect_blocked", [
    ("L0", False),
    ("L1", False),
    ("L2", False),
    ("L3", False),
    ("L4", True),
    ("L5", True),
    ("L6", True),
])
def test_each_level_blocks_at_L4_and_above(level, expect_blocked):
    """Acceptance: block on L4/L5/L6; allow L0-L3."""
    res = evaluate_release([finding(level)])
    assert res.blocked is expect_blocked
    assert res.release_allowed is not expect_blocked


def test_block_threshold_constant_is_L4():
    assert BLOCK_THRESHOLD is LLevel.L4


def test_blocking_finding_is_flagged():
    f = normalise_finding(finding("L5"))
    assert f.l_level is LLevel.L5
    assert f.is_blocking is True


def test_non_blocking_finding_is_flagged():
    f = normalise_finding(finding("L2"))
    assert f.l_level is LLevel.L2
    assert f.is_blocking is False


# ── block notification content (severity tier + details) ─────────────────────
def test_block_report_contains_severity_tier_and_details():
    """Acceptance: block notifications include tier + details of triggering finding."""
    res = evaluate_release([finding(
        "L5", id="sec-1", location="auth.py:42", title="yaml.load RCE",
        hermaguard_tier="CRITICAL", description="Unsafe YAML deserialize",
        recommendation="Use safe_load",
    )])
    assert res.blocked
    assert len(res.blocking_findings) == 1

    bf = res.blocking_findings[0]
    assert bf.l_level is LLevel.L5
    assert bf.hermaguard_tier == "CRITICAL"
    assert bf.location == "auth.py:42"
    assert bf.description == "Unsafe YAML deserialize"

    rendered = res.render()
    # Severity tier name present
    assert "L5" in rendered
    # Detail + fix present for the developer to act on
    assert "Unsafe YAML deserialize" in rendered
    assert "Use safe_load" in rendered
    assert "BLOCKED" in rendered


# ── mixed severities ─────────────────────────────────────────────────────────
def test_mixed_severities_blocks_with_only_highest():
    res = evaluate_release([
        finding("L0", id="a"),
        finding("L2", id="b"),
        finding("L3", id="c"),
        finding("L4", id="d", description="cross-user read"),
    ])
    assert res.blocked
    assert {f.finding_id for f in res.blocking_findings} == {"d"}
    assert {f.finding_id for f in res.non_blocking_findings} == {"a", "b", "c"}


def test_all_low_severities_allow():
    res = evaluate_release([
        finding("L0", id="a"),
        finding("L1", id="b"),
        finding("L2", id="c"),
        finding("L3", id="d"),
    ])
    assert not res.blocked
    assert res.max_level is LLevel.L3


# ── missing / malformed finding data ────────────────────────────────────────
def test_missing_level_no_axes_is_treated_as_block():
    """Edge case: a finding with no L_level and no axis data must NOT slip through.
    Fail-closed: ungradable => block, and the run does not crash."""
    res = evaluate_release([{"id": "x", "title": "no level field"}])
    assert res.blocked
    assert "x" in res.ungradable
    assert res.total_findings == 1


def test_partial_axis_data_raises_and_is_isolated_in_gate():
    raw = {"id": "partial", "completed": True, "reversible": True}  # missing cross_scope/privilege
    with pytest.raises(UngradableFindingError):
        normalise_finding(raw)
    # ...but the gate isolates it instead of crashing:
    res = evaluate_release([raw])
    assert res.blocked
    assert "partial" in res.ungradable


def test_derive_level_via_oracle_when_axes_present():
    """Explicit L_level absent but typed-action booleans present => oracle grades it."""
    raw = {"id": "or-1", "completed": True, "reversible": False,
           "cross_scope": False, "privilege": False}
    f = normalise_finding(raw)
    assert f.l_level is LLevel.L3  # irreversible, local => L3

    raw2 = {"id": "or-2", "completed": True, "reversible": True,
            "cross_scope": True, "privilege": False}
    assert normalise_finding(raw2).l_level is LLevel.L4


def test_escalation_oracle_yields_L6():
    assert grade_oracle(completed=True, reversible=True, cross_scope=True,
                        privilege=True, escalation=True) is LLevel.L6


def test_unparseable_level_is_blocked():
    res = evaluate_release([{"id": "bad-lvl", "L_level": "not-a-level"}])
    assert res.blocked
    assert "bad-lvl" in res.ungradable


# ── high volume ──────────────────────────────────────────────────────────────
def test_high_volume_all_low_allowed():
    many = [finding("L2", id=f"n{i}") for i in range(2000)]
    res = evaluate_release(many)
    assert not res.blocked
    assert res.total_findings == 2000
    assert len(res.non_blocking_findings) == 2000


def test_high_volume_with_single_L4_blocks():
    many = [finding("L2", id=f"n{i}") for i in range(2000)]
    many.append(finding("L4", id="needle", description="cross-scope leak"))
    res = evaluate_release(many)
    assert res.blocked
    assert len(res.blocking_findings) == 1
    assert res.blocking_findings[0].finding_id == "needle"


# ── GateResult shape ────────────────────────────────────────────────────────
def test_to_dict_shape():
    res = evaluate_release([finding("L5", id="g1", hermaguard_tier="HIGH")])
    d = res.to_dict()
    assert d["blocked"] is True
    assert d["release_allowed"] is False
    assert d["max_level"] == 5
    assert d["blocking"][0]["l_level_name"] == "L5"
    assert d["blocking"][0]["hermaguard_tier"] == "HIGH"


def test_render_includes_all_blocking_levels():
    res = evaluate_release([
        finding("L4", id="l4f"),
        finding("L6", id="l6f", description="escalation chain"),
    ])
    out = res.render()
    assert "L4" in out and "L6" in out
    assert "escalation chain" in out
