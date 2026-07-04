from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

SCRIPT_PATH = Path(__file__).resolve().parents[2] / "scripts" / "hq_harness_validator.py"
SPEC = importlib.util.spec_from_file_location("hq_harness_validator", SCRIPT_PATH)
hq_harness_validator = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = hq_harness_validator
SPEC.loader.exec_module(hq_harness_validator)


def test_default_manifest_is_allowed():
    decision = hq_harness_validator.validate_manifest(hq_harness_validator.DEFAULT_MANIFEST)

    assert decision.actual_decision == "allow"
    assert decision.passed is True
    assert "needs_review" in hq_harness_validator.DEFAULT_MANIFEST["memory_trust_gate"]["decisions"]


def test_default_fixture_set_meets_minimum_coverage():
    assert len(hq_harness_validator.DEFAULT_MEMORY_FIXTURES) >= 6
    assert len(hq_harness_validator.DEFAULT_POLICY_FIXTURES) >= 5
    assert len(hq_harness_validator.DEFAULT_RED_TEAM_FIXTURES) >= 4


def test_missing_required_manifest_field_is_rejected():
    manifest = dict(hq_harness_validator.DEFAULT_MANIFEST)
    manifest.pop("audit")

    decision = hq_harness_validator.validate_manifest(manifest)

    assert decision.actual_decision == "reject"
    assert decision.passed is False
    assert "audit" in decision.reason


def test_default_memory_fixtures_match_expected_decisions():
    decisions = [
        hq_harness_validator.evaluate_memory_fixture(fixture)
        for fixture in hq_harness_validator.DEFAULT_MEMORY_FIXTURES
    ]

    assert {decision.fixture_id: decision.actual_decision for decision in decisions} == {
        "memory-safe-relevant": "allow",
        "memory-stale-operational": "summarize",
        "memory-cross-domain-private": "reject",
        "memory-prompt-injection": "quarantine",
        "memory-secret-like": "reject",
        "memory-contradictory-operational": "needs_review",
    }
    assert all(decision.passed for decision in decisions)


def test_secret_like_memory_reason_is_redacted():
    fixture = {
        "fixture_id": "memory-secret-redaction-regression",
        "kind": "memory",
        "snippet": "this synthetic token api_key=not-a-real-secret must not appear in the reason",
        "expected_decision": "reject",
    }

    decision = hq_harness_validator.evaluate_memory_fixture(fixture)

    assert decision.actual_decision == "reject"
    assert "not-a-real-secret" not in decision.reason
    assert "redacted" in decision.reason


def test_default_policy_fixtures_match_expected_decisions():
    decisions = [
        hq_harness_validator.evaluate_policy_fixture(fixture)
        for fixture in hq_harness_validator.DEFAULT_POLICY_FIXTURES
    ]

    assert {decision.fixture_id: decision.actual_decision for decision in decisions} == {
        "policy-readonly-inspect": "allow",
        "policy-doc-write-skill-reference": "allow",
        "policy-branch-worktree-code": "allow",
        "policy-external-unapproved": "confirm",
        "policy-destructive-delete": "reject",
        "policy-admin-helper-status": "allow",
        "policy-arbitrary-admin": "reject",
        "policy-live-enforcement": "confirm",
    }
    assert all(decision.passed for decision in decisions)


def test_default_red_team_fixtures_match_expected_decisions():
    decisions = [
        hq_harness_validator.evaluate_red_team_fixture(fixture)
        for fixture in hq_harness_validator.DEFAULT_RED_TEAM_FIXTURES
    ]

    assert {decision.fixture_id: decision.actual_decision for decision in decisions} == {
        "redteam-gateway-injection-as-data": "confirm",
        "redteam-secret-exfiltration-rejected": "reject",
        "redteam-fixture-intake-policy-reviewed": "allow",
        "redteam-live-gate-approval-recorded": "allow",
    }
    assert all(decision.passed for decision in decisions)


def test_sandbox_eval_gate_mode_includes_red_team_fixtures():
    report = hq_harness_validator.run_eval(
        hq_harness_validator.DEFAULT_MANIFEST,
        [
            *hq_harness_validator.DEFAULT_MEMORY_FIXTURES,
            *hq_harness_validator.DEFAULT_POLICY_FIXTURES,
            *hq_harness_validator.DEFAULT_RED_TEAM_FIXTURES,
        ],
        "synthetic-test",
        mode="sandbox_eval_gate",
    )

    assert report["mode"] == "sandbox_eval_gate"
    assert report["failed"] == 0
    assert any(d["fixture_id"] == "redteam-live-gate-approval-recorded" for d in report["decisions"])


def test_run_eval_and_report_writer(tmp_path):
    report = hq_harness_validator.run_eval(
        hq_harness_validator.DEFAULT_MANIFEST,
        [*hq_harness_validator.DEFAULT_MEMORY_FIXTURES, *hq_harness_validator.DEFAULT_POLICY_FIXTURES],
        "synthetic-test",
    )

    assert report["mode"] == "dry_run_only"
    assert report["failed"] == 0
    assert report["safety_invariants"] == {
        "synthetic_only": True,
        "no_live_tool_execution": True,
        "no_secret_access": True,
        "no_cron_mutation": True,
        "no_gateway_runtime_change": True,
    }

    json_path, md_path = hq_harness_validator.write_reports(report, tmp_path)

    assert json.loads(json_path.read_text(encoding="utf-8"))["failed"] == 0
    markdown = md_path.read_text(encoding="utf-8")
    assert "# HQ Harness Eval Latest" in markdown
    assert "policy-live-enforcement" in markdown
