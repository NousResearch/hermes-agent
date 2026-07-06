from __future__ import annotations

import json
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]


def _job(name: str) -> dict:
    jobs = json.loads((REPO_ROOT / "cron" / "jobs.json").read_text(encoding="utf-8"))["jobs"]
    return next(job for job in jobs if job.get("name") == name)


def _live_soul_text() -> str:
    path = REPO_ROOT / "SOUL.md"
    if not path.exists():
        pytest.skip("SOUL.md is live-only and intentionally ignored by the profile snapshot")
    return path.read_text(encoding="utf-8")


def test_weekly_reset_is_scheduled_agent_waking_friday_signal_packet() -> None:
    job = _job("torben-weekly-reset")

    assert job["enabled"] is True
    assert job["deliver"] == "signal"
    assert job["no_agent"] is False
    assert job["script"] == "torben_weekly_reset_job.py"
    assert job["schedule"]["expr"] == "0 16 * * 5"
    assert "exactly one concise Signal packet" in job["prompt"]


def test_pattern_miner_is_scheduled_sunday_silent_no_agent() -> None:
    job = _job("torben-pattern-miner")

    assert job["enabled"] is True
    assert job["deliver"] == "signal"
    assert job["no_agent"] is True
    assert job["script"] == "torben_pattern_miner_job.py"
    assert job["schedule"]["expr"] == "0 18 * * 0"
    assert job["prompt"] == ""


def test_soul_instructs_track_replies_through_resolver() -> None:
    soul = _live_soul_text()

    assert "explicit `track ...` Signal reply" in soul
    assert "hermes torben resolve-reply --sender <Eric number> track ..." in soul
    assert "state/torben-open-loops.csv" in soul


def test_soul_routes_personal_ops_requests_to_packet_only_skills() -> None:
    soul = _live_soul_text()

    assert "appointment/service-call" in soul
    assert "paperwork/reimbursement" in soul
    assert "packet-only skill" in soul
    assert "do not invent new mutation" in soul
