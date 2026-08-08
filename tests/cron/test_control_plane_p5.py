from __future__ import annotations

import json
from pathlib import Path

from cron_control.report import build_shadow_diff_report, main as report_main


FIXTURE_DIR = Path(__file__).resolve().parents[2] / "docs" / "cron-control" / "p0" / "examples"


def _load_verdict(name: str) -> dict:
    return json.loads((FIXTURE_DIR / name).read_text(encoding="utf-8"))


def _make_verdict(*, verdict_id: str, incident_id: str, job_id: str, state: str, action: str) -> dict:
    verdict = _load_verdict("verdict.example.json")
    verdict.update(
        {
            "verdict_id": verdict_id,
            "incident_id": incident_id,
            "job_id": job_id,
            "state": state,
            "recommended_action": action,
        }
    )
    return verdict


def test_build_shadow_diff_report_detects_added_removed_changed_verdicts() -> None:
    baseline = {
        "verdicts": [
            _make_verdict(
                verdict_id="vd_a",
                incident_id="inc_a",
                job_id="job-a",
                state="stale_running",
                action="reset_job",
            ),
            _make_verdict(
                verdict_id="vd_b",
                incident_id="inc_b",
                job_id="job-b",
                state="healthy",
                action="none",
            ),
            _make_verdict(
                verdict_id="vd_c",
                incident_id="inc_c",
                job_id="job-c",
                state="quarantined",
                action="escalate_to_human",
            ),
            _make_verdict(
                verdict_id="vd_d",
                incident_id="inc_d",
                job_id="job-d",
                state="suspect",
                action="none",
            ),
        ]
    }
    current = {
        "verdicts": [
            _make_verdict(
                verdict_id="vd_a2",
                incident_id="inc_a",
                job_id="job-a",
                state="healthy",
                action="none",
            ),
            _make_verdict(
                verdict_id="vd_b",
                incident_id="inc_b",
                job_id="job-b",
                state="healthy",
                action="none",
            ),
            _make_verdict(
                verdict_id="vd_d",
                incident_id="inc_d",
                job_id="job-d",
                state="suspect",
                action="none",
            ),
            _make_verdict(
                verdict_id="vd_e",
                incident_id="inc_e",
                job_id="job-e",
                state="recoverable",
                action="switch_provider",
            ),
        ]
    }

    report = build_shadow_diff_report(current, baseline)

    assert report["baseline"]["verdict_count"] == 4
    assert report["current"]["verdict_count"] == 4
    assert report["summary"]["added"] == 1
    assert report["summary"]["removed"] == 1
    assert report["summary"]["changed"] == 1
    assert report["summary"]["unchanged"] == 2
    assert report["summary"]["state_transitions"] == {"stale_running -> healthy": 1}
    assert report["summary"]["action_transitions"] == {"reset_job -> none": 1}

    changed = report["changes"]["changed"]
    assert len(changed) == 1
    assert changed[0]["key"] == {"incident_id": "inc_a", "job_id": "job-a"}
    assert changed[0]["changed_fields"]["state"] == {"baseline": "stale_running", "current": "healthy"}
    assert changed[0]["changed_fields"]["recommended_action"] == {"baseline": "reset_job", "current": "none"}


def test_report_main_reads_json_files_and_prints_diff(tmp_path, capsys) -> None:
    baseline_path = tmp_path / "baseline.json"
    current_path = tmp_path / "current.json"

    baseline_path.write_text(
        json.dumps(
            {
                "verdicts": [
                    _make_verdict(
                        verdict_id="vd_a",
                        incident_id="inc_a",
                        job_id="job-a",
                        state="stale_running",
                        action="reset_job",
                    ),
                    _make_verdict(
                        verdict_id="vd_b",
                        incident_id="inc_b",
                        job_id="job-b",
                        state="healthy",
                        action="none",
                    ),
                ]
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    current_path.write_text(
        json.dumps(
            {
                "verdicts": [
                    _make_verdict(
                        verdict_id="vd_a2",
                        incident_id="inc_a",
                        job_id="job-a",
                        state="healthy",
                        action="none",
                    ),
                    _make_verdict(
                        verdict_id="vd_c",
                        incident_id="inc_c",
                        job_id="job-c",
                        state="recoverable",
                        action="switch_provider",
                    ),
                ]
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    assert report_main(["--baseline-json", str(baseline_path), "--current-json", str(current_path)]) == 0
    captured = json.loads(capsys.readouterr().out)

    assert captured["baseline"]["verdict_count"] == 2
    assert captured["current"]["verdict_count"] == 2
    assert captured["summary"]["added"] == 1
    assert captured["summary"]["removed"] == 1
    assert captured["summary"]["changed"] == 1
