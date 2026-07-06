from __future__ import annotations

import json
import sys
from pathlib import Path


SCRIPTS_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SCRIPTS_DIR))

from torben_health_backfill import backfill_missing_health
from torben_job_contract import health_path


def _write_jobs(profile_home: Path, jobs: list[dict]) -> None:
    jobs_path = profile_home / "cron" / "jobs.json"
    jobs_path.parent.mkdir(parents=True)
    jobs_path.write_text(json.dumps({"jobs": jobs}), encoding="utf-8")


def test_backfills_missing_enabled_health_from_scheduler_output(tmp_path: Path) -> None:
    job = {
        "id": "abc123",
        "name": "torben-example",
        "enabled": True,
        "last_status": "ok",
        "last_run_at": "2026-07-05T10:00:00-04:00",
    }
    _write_jobs(tmp_path, [job])
    output_dir = tmp_path / "cron" / "output" / "abc123"
    output_dir.mkdir(parents=True)
    (output_dir / "2026-07-05_10-00-00.md").write_text("ok\n", encoding="utf-8")

    result = backfill_missing_health(profile_home=tmp_path, apply=True)
    health = json.loads(health_path("torben-example", tmp_path).read_text(encoding="utf-8"))

    assert result["backfilled"] == 1
    assert health["status"] == "ok"
    assert health["source"] == "scheduler_history_backfill"
    assert health["scheduler_job_id"] == "abc123"
    assert health["cron_output_path"] == "cron/output/abc123/2026-07-05_10-00-00.md"


def test_backfill_skips_missing_output_and_disabled_jobs(tmp_path: Path) -> None:
    _write_jobs(
        tmp_path,
        [
            {
                "id": "missing-output",
                "name": "torben-missing-output",
                "enabled": True,
                "last_status": "ok",
                "last_run_at": "2026-07-05T10:00:00-04:00",
            },
            {
                "id": "disabled",
                "name": "torben-disabled",
                "enabled": False,
                "last_status": "ok",
                "last_run_at": "2026-07-05T10:00:00-04:00",
            },
        ],
    )

    result = backfill_missing_health(profile_home=tmp_path, apply=True)
    reasons = {record["job"]: record["reason"] for record in result["records"]}

    assert result["backfilled"] == 0
    assert reasons["torben-missing-output"] == "missing_cron_output"
    assert reasons["torben-disabled"] == "disabled"
