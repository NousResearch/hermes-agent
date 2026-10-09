from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

SCRIPT_PATH = Path(__file__).resolve().parents[2] / "scripts" / "hq_harness_dashboard_adapter.py"
SPEC = importlib.util.spec_from_file_location("hq_harness_dashboard_adapter", SCRIPT_PATH)
hq_harness_dashboard_adapter = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = hq_harness_dashboard_adapter
SPEC.loader.exec_module(hq_harness_dashboard_adapter)


def valid_report() -> dict:
    return {
        "run_id": "hq-harness-eval-test",
        "mode": "dry_run_only",
        "source_manifest": "synthetic",
        "total_fixtures": 2,
        "passed": 2,
        "failed": 0,
        "failures": [],
        "decisions": [
            {
                "fixture_id": "manifest",
                "expected_decision": "allow",
                "actual_decision": "allow",
                "passed": True,
                "reason": "complete dry-run manifest",
            },
            {
                "fixture_id": "policy-live-enforcement",
                "expected_decision": "confirm",
                "actual_decision": "confirm",
                "passed": True,
                "reason": "runtime behavior change requires exact approval",
            },
        ],
        "safety_invariants": {
            "synthetic_only": True,
            "no_live_tool_execution": True,
            "no_secret_access": True,
            "no_cron_mutation": True,
            "no_gateway_runtime_change": True,
        },
    }


def test_valid_dry_run_report_maps_to_ok():
    summary = hq_harness_dashboard_adapter.summarize_report(valid_report())

    assert summary.status == "OK"
    assert summary.passed == 2
    assert summary.total_fixtures == 2
    assert summary.failed_fixtures == []
    assert "continue dry-run review" in summary.next_action


def test_failed_fixture_maps_to_review():
    report = valid_report()
    report["failed"] = 1
    report["passed"] = 1
    report["failures"] = [
        {
            "fixture_id": "memory-stale-operational",
            "passed": False,
            "reason": "regression",
        }
    ]
    report["decisions"][0]["passed"] = False

    summary = hq_harness_dashboard_adapter.summarize_report(report)

    assert summary.status == "REVIEW"
    assert summary.failed_fixtures == ["memory-stale-operational"]
    assert "inspect failed fixtures" in summary.next_action


def test_false_safety_invariant_maps_to_blocked():
    report = valid_report()
    report["safety_invariants"]["no_live_tool_execution"] = False

    summary = hq_harness_dashboard_adapter.summarize_report(report)

    assert summary.status == "BLOCKED"
    assert summary.false_invariants == ["no_live_tool_execution"]
    assert "stop capability expansion" in summary.next_action


def test_missing_report_path_maps_to_recover(tmp_path):
    missing_path = tmp_path / "missing.json"

    summary = hq_harness_dashboard_adapter.summarize_path(missing_path)

    assert summary.status == "RECOVER"
    assert summary.run_id == "unknown"
    assert "report missing" in summary.reason


def test_malformed_json_maps_to_recover(tmp_path):
    report_path = tmp_path / "malformed.json"
    report_path.write_text("{not-json", encoding="utf-8")

    summary = hq_harness_dashboard_adapter.summarize_path(report_path)

    assert summary.status == "RECOVER"
    assert "could not parse" in summary.reason


def test_missing_required_report_field_maps_to_review():
    report = valid_report()
    report.pop("decisions")

    summary = hq_harness_dashboard_adapter.summarize_report(report)

    assert summary.status == "REVIEW"
    assert "missing report fields" in summary.reason


def test_markdown_renderer_is_review_dispatch_friendly():
    summary = hq_harness_dashboard_adapter.summarize_report(valid_report())

    markdown = hq_harness_dashboard_adapter.render_markdown(summary)

    assert "HQ Harness Dashboard Adapter" in markdown
    assert "Status: `OK`" in markdown
    assert "Review Dispatch Input" in markdown
    assert "No live Gateway/cron/tool enforcement changed" in markdown


def test_json_renderer_outputs_compact_status():
    summary = hq_harness_dashboard_adapter.summarize_report(valid_report())

    payload = json.loads(hq_harness_dashboard_adapter.render_json(summary))

    assert payload["status"] == "OK"
    assert payload["passed"] == 2
    assert payload["total_fixtures"] == 2
    assert payload["safety_invariants"]["no_secret_access"] is True
