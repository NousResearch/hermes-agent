from __future__ import annotations

import importlib.util
import json
import shutil
import sys
from pathlib import Path

DASHBOARD_SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "hq_health_dashboard.py"
ADAPTER_SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "hq_harness_dashboard_adapter.py"
SPEC = importlib.util.spec_from_file_location("hq_health_dashboard", DASHBOARD_SCRIPT)
hq_health_dashboard = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = hq_health_dashboard
SPEC.loader.exec_module(hq_health_dashboard)


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
                "fixture_id": "memory-trust-gate-ok",
                "expected_decision": "allow",
                "actual_decision": "allow",
                "passed": True,
                "reason": "trusted dry-run memory",
            },
            {
                "fixture_id": "pre-dispatch-policy-live-enforcement",
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


def configure_temp_dashboard(tmp_path, monkeypatch):
    scripts = tmp_path / "scripts"
    reports = tmp_path / "reports"
    logs = tmp_path / "logs"
    scripts.mkdir()
    reports.mkdir()
    logs.mkdir()
    shutil.copyfile(ADAPTER_SCRIPT, scripts / "hq_harness_dashboard_adapter.py")

    monkeypatch.setattr(hq_health_dashboard, "HERMES_HOME", tmp_path)
    monkeypatch.setattr(hq_health_dashboard, "SCRIPTS", scripts)
    monkeypatch.setattr(hq_health_dashboard, "REPORTS", reports)
    monkeypatch.setattr(hq_health_dashboard, "LOGS", logs)
    monkeypatch.setattr(hq_health_dashboard, "DASHBOARD_MD", reports / "hq_health_dashboard_latest.md")
    monkeypatch.setattr(hq_health_dashboard, "DASHBOARD_JSON", reports / "hq_health_dashboard_latest.json")
    monkeypatch.setattr(hq_health_dashboard, "HARNESS_SPECS", [])
    return reports


def write_report(reports: Path, report: dict) -> Path:
    path = reports / "hq_harness_eval_latest.json"
    path.write_text(json.dumps(report, ensure_ascii=False), encoding="utf-8")
    return path


def test_harness_evidence_ok_is_in_dashboard_payload(tmp_path, monkeypatch):
    reports = configure_temp_dashboard(tmp_path, monkeypatch)
    write_report(reports, valid_report())

    harnesses = hq_health_dashboard.run_harnesses()
    evidence = harnesses["harness_evidence_dashboard"]

    assert evidence["status"] == "OK"
    assert evidence["classification"] == "OK"
    assert evidence["return_code"] == 0
    assert evidence["summary"]["passed"] == 2
    assert evidence["summary"]["mode"] == "dry_run_only"

    payload = hq_health_dashboard.write_reports(harnesses, {"best_effort_unresolved_from_logs": 0})
    assert payload["harnesses"]["harness_evidence_dashboard"]["summary"]["total_fixtures"] == 2
    assert payload["summary"]["overall_status"] == "OK"
    assert (reports / "hq_health_dashboard_latest.json").exists()


def test_harness_evidence_failed_fixture_maps_to_review(tmp_path, monkeypatch):
    reports = configure_temp_dashboard(tmp_path, monkeypatch)
    report = valid_report()
    report["passed"] = 1
    report["failed"] = 1
    report["failures"] = [{"fixture_id": "memory-stale-operational", "passed": False}]
    report["decisions"][0]["passed"] = False
    write_report(reports, report)

    evidence = hq_health_dashboard.run_harness_evidence_adapter()

    assert evidence["status"] == "REVIEW"
    assert evidence["classification"] == "REVIEW"
    assert evidence["summary"]["failed_fixtures"] == ["memory-stale-operational"]


def test_harness_evidence_false_invariant_maps_to_fail(tmp_path, monkeypatch):
    reports = configure_temp_dashboard(tmp_path, monkeypatch)
    report = valid_report()
    report["safety_invariants"]["no_live_tool_execution"] = False
    write_report(reports, report)

    harnesses = hq_health_dashboard.run_harnesses()
    evidence = harnesses["harness_evidence_dashboard"]

    assert evidence["status"] == "BLOCKED"
    assert evidence["classification"] == "FAIL"
    assert evidence["return_code"] == 2
    assert hq_health_dashboard.overall_status(harnesses, {"best_effort_unresolved_from_logs": 0}) == "FAIL"


def test_harness_evidence_missing_report_maps_to_review_not_live_mutation(tmp_path, monkeypatch):
    reports = configure_temp_dashboard(tmp_path, monkeypatch)

    harnesses = hq_health_dashboard.run_harnesses()
    evidence = harnesses["harness_evidence_dashboard"]

    assert evidence["status"] == "RECOVER"
    assert evidence["classification"] == "REVIEW"
    assert "report missing" in evidence["summary"]["reason"]
    assert not (reports / "hq_harness_eval_latest.json").exists()


def test_harness_evidence_corrupt_report_maps_to_recover_review(tmp_path, monkeypatch):
    reports = configure_temp_dashboard(tmp_path, monkeypatch)
    (reports / "hq_harness_eval_latest.json").write_text("{not-json", encoding="utf-8")

    evidence = hq_health_dashboard.run_harness_evidence_adapter()

    assert evidence["status"] == "RECOVER"
    assert evidence["classification"] == "REVIEW"
    assert "could not parse" in evidence["summary"]["reason"]
