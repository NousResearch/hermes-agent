from __future__ import annotations

import importlib.util
import json
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "cron_stale_monitor.py"


def _module():
    spec = importlib.util.spec_from_file_location("cron_stale_monitor", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_receipt_scope_reports_only_restored_p13_jobs(
    tmp_path: Path,
    capsys,
    monkeypatch,
) -> None:
    module = _module()
    jobs_file = tmp_path / "jobs.json"
    monkeypatch.setattr(module, "JOBS_FILE", jobs_file, raising=False)
    monkeypatch.setattr(module, "OUTPUT_DIR", tmp_path / "output", raising=False)
    jobs_file.write_text(
        json.dumps(
            {
                "jobs": [
                    {
                        "id": "p13-job",
                        "name": "P13 restored check",
                        "enabled": True,
                        "state": "scheduled",
                        "created_at": "2020-01-01T00:00:00+00:00",
                        "last_run_at": None,
                        "schedule": {"display": "every 1h"},
                    },
                    {
                        "id": "legacy-job",
                        "name": "Unrelated legacy check",
                        "enabled": True,
                        "state": "scheduled",
                        "created_at": "2020-01-01T00:00:00+00:00",
                        "last_run_at": None,
                        "schedule": {"display": "every 1h"},
                    },
                ]
            }
        ),
        encoding="utf-8",
    )
    receipt = tmp_path / "p13-receipt.json"
    receipt.write_text(json.dumps({"jobs": [{"target_id": "p13-job"}]}), encoding="utf-8")

    assert module.main(receipt) == 0

    output = capsys.readouterr().out
    assert "P13 restored check" in output
    assert "Unrelated legacy check" not in output


def test_cli_accepts_receipt_path(tmp_path: Path) -> None:
    module = _module()
    receipt = tmp_path / "p13-receipt.json"

    assert module._parse_args(["--receipt", str(receipt)]).receipt == receipt


def test_zero_findings_writes_no_output_and_is_silent(
    tmp_path: Path, capsys, monkeypatch
) -> None:
    """Zero findings: no output dir created, empty stdout (silent rule)."""
    import json
    from datetime import datetime, timezone
    module = _module()
    jobs_file = tmp_path / "jobs.json"
    output_dir = tmp_path / "output"
    monkeypatch.setattr(module, "JOBS_FILE", jobs_file, raising=False)
    monkeypatch.setattr(module, "OUTPUT_DIR", output_dir, raising=False)
    # A healthy job: last_run ~1h ago (well within the 26h daily threshold).
    recent_iso = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S+00:00")
    jobs_file.write_text(
        json.dumps({"jobs": [{
            "id": "healthy", "name": "Healthy daily", "enabled": True,
            "state": "scheduled", "created_at": "2020-01-01T00:00:00+00:00",
            "last_run_at": recent_iso,
            "schedule": {"display": "daily", "expr": "0 10 * * *"},
        }]}),
        encoding="utf-8",
    )

    assert module.main(None) == 0
    assert capsys.readouterr().out == "", "expected silent stdout"
    assert not output_dir.exists(), "output dir created despite zero findings"


def test_findings_writes_only_under_temp_output(
    tmp_path: Path, capsys, monkeypatch
) -> None:
    """Findings case: report written only under temp OUTPUT_DIR, stdout present."""
    import json
    import datetime as _dt
    module = _module()
    jobs_file = tmp_path / "jobs.json"
    output_dir = tmp_path / "output"
    monkeypatch.setattr(module, "JOBS_FILE", jobs_file, raising=False)
    monkeypatch.setattr(module, "OUTPUT_DIR", output_dir, raising=False)
    # A stale never-ran job created >24h ago (real now is far past 2026-07-02).
    jobs_file.write_text(
        json.dumps({"jobs": [{
            "id": "stale-job", "name": "Stale never ran", "enabled": True,
            "state": "scheduled", "created_at": "2026-07-01T00:00:00+00:00",
            "last_run_at": None,
            "schedule": {"display": "daily", "expr": "0 10 * * *"},
        }]}),
        encoding="utf-8",
    )

    assert module.main(None) == 0
    out = capsys.readouterr().out
    assert out, "expected non-empty stdout for findings"
    assert "cron-stale-watchdog" in out
    assert "Stale never ran" in out
    assert "1 stale cron(s) detected" in out
    # Output dir created with exactly one timestamped txt report
    reports = sorted(output_dir.glob("*.txt"))
    assert len(reports) == 1, f"expected 1 report, got {reports}"
    report_text = reports[0].read_text()
    assert "stale-job" in report_text
    assert "Stale never ran" in report_text
