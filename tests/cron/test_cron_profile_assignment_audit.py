import json
import importlib.util
import sys
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parents[2] / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

spec = importlib.util.spec_from_file_location(
    "cron_profile_assignment_audit", SCRIPTS_DIR / "cron_profile_assignment_audit.py"
)
assert spec and spec.loader
audit = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = audit
spec.loader.exec_module(audit)
assignment_for_job = audit.assignment_for_job
build_report = audit.build_report
load_job_store = audit.load_job_store


def write_jobs(path: Path, jobs, *, wrapped: bool = True):
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"jobs": jobs, "updated_at": "2026-06-30T00:00:00Z"} if wrapped else jobs
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_report_counts_wrapped_active_jobs_and_paused_followup(tmp_path):
    jobs_file = tmp_path / "cron" / "jobs.json"
    write_jobs(
        jobs_file,
        [
            {"id": "j1", "name": "crm-lead-check", "script": "lead.py", "no_agent": True},
            {"id": "j2", "name": "gbrain-daily-check", "prompt": "check gbrain"},
            {"id": "j3", "name": "stale-draft-alert", "enabled": False},
        ],
    )

    jobs, metadata = load_job_store(jobs_file)
    report = build_report(jobs)

    assert metadata["shape"] == "wrapped"
    assert report["summary"]["active_total"] == 2
    assert report["summary"]["paused_total"] == 1
    assert report["summary"]["target_counts"]["rva-leads"] == 1
    assert report["summary"]["target_counts"]["default"] == 1
    assert report["paused"][0]["target_profile"] == "rva-firm-ops"


def test_legacy_list_form_jobs_are_supported(tmp_path):
    jobs_file = tmp_path / "cron" / "jobs.json"
    write_jobs(
        jobs_file,
        [
            {"id": "tax", "name": "tax-research-archive-lint", "script": "tax.py", "no_agent": True},
            {"id": "old", "name": "old-paused", "enabled": False},
        ],
        wrapped=False,
    )

    jobs, metadata = load_job_store(jobs_file)
    report = build_report(jobs)

    assert metadata["shape"] == "list"
    assert report["summary"]["active_total"] == 1
    assert report["active"][0]["target_profile"] == "cpa-tax-researcher"
    assert report["summary"]["paused_total"] == 1


def test_unknown_job_defaults_to_default_with_review_needed():
    assignment = assignment_for_job({"id": "mystery", "name": "mystery-job"})

    assert assignment.target_profile == "default"
    assert assignment.review_needed is True
    assert "unknown" in assignment.reason


def test_rule_based_classification_covers_business_domains():
    assert assignment_for_job({"id": "1", "name": "crm-new-thing"}).target_profile == "rva-leads"
    assert assignment_for_job({"id": "2", "name": "square-metrics"}).target_profile == "rva-profit-pulse"
    assert assignment_for_job({"id": "3", "name": "tax_research_extra"}).target_profile == "cpa-tax-researcher"
    assert assignment_for_job({"id": "4", "name": "gmail-maintenance"}).target_profile == "rva-firm-ops"


def test_audit_main_json_smoke(tmp_path, capsys):
    jobs_file = tmp_path / "cron" / "jobs.json"
    write_jobs(jobs_file, [{"id": "j1", "name": "crm-lead-check", "script": "lead.py", "no_agent": True}])

    assert audit.main(["--jobs-file", str(jobs_file), "--format", "json"]) == 0
    payload = json.loads(capsys.readouterr().out)

    assert payload["summary"]["active_total"] == 1
    assert payload["active"][0]["target_profile"] == "rva-leads"
