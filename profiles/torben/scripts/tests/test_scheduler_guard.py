from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

from scripts.torben_scheduler_guard_check import (
    build_report,
    inspect_liveness,
    inspect_patch,
    validate_registry,
)


def _write_registry(path: Path, jobs: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"jobs": jobs}, indent=2), encoding="utf-8")


def _job(name: str, schedule: dict, *, last_run_at: str | None = None, enabled: bool = True) -> dict:
    return {
        "id": name,
        "name": name,
        "enabled": enabled,
        "schedule": schedule,
        "last_run_at": last_run_at,
        "script": f"{name}.py",
    }


def _desk_jobs() -> list[dict]:
    return [
        _job(f"torben-desk-v2-{index}", {"kind": "cron", "expr": "0 9 * * 1-5"})
        for index in range(10)
    ]


def test_validate_registry_accepts_valid_jobs_and_desk_count(tmp_path: Path) -> None:
    path = tmp_path / "cron" / "jobs.json"
    _write_registry(
        path,
        [
            _job("torben-gmail-pubsub-pull", {"kind": "interval", "minutes": 1}),
            _job("torben-morning-brief", {"kind": "cron", "expr": "0 8 * * *"}),
        ]
        + _desk_jobs(),
    )

    result = validate_registry(path)

    assert result["status"] == "pass"
    assert result["desk_v2_count"] == 10
    assert result["shortest_interval_minutes"] == 1


def test_validate_registry_rejects_malformed_schedule(tmp_path: Path) -> None:
    path = tmp_path / "cron" / "jobs.json"
    _write_registry(path, [_job("bad", {"kind": "cron"})] + _desk_jobs())

    result = validate_registry(path)

    assert result["status"] == "failed"
    assert "bad: cron schedule requires expr or cron" in result["errors"]


def test_inspect_patch_requires_scheduler_markers(tmp_path: Path) -> None:
    agent_root = tmp_path / "agent"
    (agent_root / "cron").mkdir(parents=True)
    (agent_root / "cron" / "jobs.py").write_text(
        "Skipping cron job during due-check normalization\n"
        "Skipping cron job '%s' during due-check; malformed schedule or run state\n",
        encoding="utf-8",
    )
    (agent_root / "cron" / "scheduler.py").write_text(
        "Cron tick could not load due jobs; skipping this tick\n",
        encoding="utf-8",
    )

    assert inspect_patch(agent_root)["status"] == "pass"
    (agent_root / "cron" / "scheduler.py").write_text("old scheduler", encoding="utf-8")
    assert inspect_patch(agent_root)["status"] == "failed"


def test_liveness_flags_interval_drift(tmp_path: Path) -> None:
    now = datetime(2026, 7, 6, 12, 0, tzinfo=timezone.utc)
    path = tmp_path / "cron" / "jobs.json"
    _write_registry(
        path,
        [
            _job(
                "torben-meeting-prep-watch",
                {"kind": "interval", "minutes": 5},
                last_run_at=(now - timedelta(minutes=20)).isoformat(),
            ),
        ]
        + _desk_jobs(),
    )

    result = inspect_liveness(path, now=now)

    assert result["status"] == "failed"
    assert result["interval_drifts"][0]["name"] == "torben-meeting-prep-watch"


def test_build_report_passes_for_valid_registry_and_patch(tmp_path: Path) -> None:
    profile = tmp_path / "profile"
    agent = tmp_path / "agent"
    (agent / "cron").mkdir(parents=True)
    (agent / "cron" / "jobs.py").write_text(
        "Skipping cron job during due-check normalization\n"
        "Skipping cron job '%s' during due-check; malformed schedule or run state\n",
        encoding="utf-8",
    )
    (agent / "cron" / "scheduler.py").write_text(
        "Cron tick could not load due jobs; skipping this tick\n",
        encoding="utf-8",
    )
    now = datetime.now(timezone.utc)
    _write_registry(
        profile / "cron" / "jobs.json",
        [
            _job(
                "torben-gmail-pubsub-pull",
                {"kind": "interval", "minutes": 1},
                last_run_at=now.isoformat(),
            )
        ]
        + _desk_jobs(),
    )

    result = build_report(profile_home=profile, agent_root=agent)

    assert result["status"] == "pass"
    assert result["wakeAgent"] is False
