from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from scripts.torben_job_contract import (
    delivery_failed,
    health_path,
    record_alert_outcome,
    run_job,
)


def _health(tmp_path: Path, job: str) -> dict:
    return json.loads(health_path(job, tmp_path).read_text(encoding="utf-8"))


def test_run_job_records_success_and_failure_counter(tmp_path: Path) -> None:
    assert run_job("canary", lambda: 0, profile_home=tmp_path) == 0
    assert _health(tmp_path, "canary")["status"] == "ok"
    assert _health(tmp_path, "canary")["consecutive_failures"] == 0

    assert run_job("canary", lambda: (_ for _ in ()).throw(RuntimeError("boom")), profile_home=tmp_path) == 1
    failed = _health(tmp_path, "canary")
    assert failed["status"] == "failed"
    assert failed["consecutive_failures"] == 1
    assert failed["last_error"]["type"] == "RuntimeError"

    assert run_job("canary", lambda: 0, profile_home=tmp_path) == 0
    assert _health(tmp_path, "canary")["consecutive_failures"] == 0


def test_empty_floor_is_failure(tmp_path: Path) -> None:
    code = run_job("empty-floor", lambda: [], expect_non_empty=bool, profile_home=tmp_path)
    assert code == 1
    health = _health(tmp_path, "empty-floor")
    assert health["status"] == "failed"
    assert health["last_error"]["message"] == "empty_result"


def test_delivery_failure_increments_health(tmp_path: Path) -> None:
    delivery_failed("delivery-canary", "signal-cli refused connection", profile_home=tmp_path)
    health = _health(tmp_path, "delivery-canary")
    assert health["status"] == "failed"
    assert health["consecutive_failures"] == 1
    assert health["last_delivery_status"] == "failed"
    assert "signal-cli" in health["last_delivery_error"]


def test_alert_dedupe_ttl_and_realert(tmp_path: Path) -> None:
    sent: list[tuple[str, str]] = []

    def deliver(recipient: str, text: str) -> None:
        sent.append((recipient, text))

    now = datetime(2026, 7, 5, 12, 0, tzinfo=timezone.utc)
    first = record_alert_outcome(
        job_name="alert-canary",
        text="same alert",
        now=now,
        dedupe_seconds=60,
        deliver=deliver,
        profile_home=tmp_path,
    )
    duplicate = record_alert_outcome(
        job_name="alert-canary",
        text="same alert",
        now=now + timedelta(seconds=10),
        dedupe_seconds=60,
        deliver=deliver,
        profile_home=tmp_path,
    )
    realert = record_alert_outcome(
        job_name="alert-canary",
        text="same alert",
        now=now + timedelta(seconds=70),
        dedupe_seconds=60,
        deliver=deliver,
        profile_home=tmp_path,
    )
    assert first["status"] == "sent"
    assert duplicate["status"] == "skipped_duplicate"
    assert realert["status"] == "sent"
    assert len(sent) == 2


def test_alert_rate_limit(tmp_path: Path) -> None:
    now = datetime(2026, 7, 5, 12, 0, tzinfo=timezone.utc)
    one = record_alert_outcome(
        job_name="rate-canary",
        text="first",
        now=now,
        rate_limit_per_hour=1,
        profile_home=tmp_path,
    )
    two = record_alert_outcome(
        job_name="rate-canary",
        text="second",
        now=now + timedelta(seconds=1),
        rate_limit_per_hour=1,
        profile_home=tmp_path,
    )
    assert one["status"] == "sent"
    assert two["status"] == "skipped_rate_limited"


def test_timeout_records_failure(tmp_path: Path) -> None:
    def slow() -> int:
        import time

        time.sleep(2)
        return 0

    code = run_job("timeout-canary", slow, timeout_seconds=1, profile_home=tmp_path)
    assert code == 1
    assert _health(tmp_path, "timeout-canary")["last_error"]["type"] in {"JobTimeoutError", "TimeoutError"}
