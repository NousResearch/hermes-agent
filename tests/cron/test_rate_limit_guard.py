from __future__ import annotations

import json

import cron.scheduler as scheduler
from cron.rate_limit_guard import FleetRateLimitCircuitBreaker


class Clock:
    def __init__(self, now: float = 1_000.0) -> None:
        self.now = now

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


def test_rate_limit_retry_uses_exponential_backoff_with_jitter(monkeypatch) -> None:
    attempts = iter(
        [
            (False, "failed-1", "", "HTTPStatusError: 429 Too Many Requests"),
            (False, "failed-2", "", "HTTPStatusError: 429 Too Many Requests"),
            (True, "success", "done", None),
        ]
    )
    sleeps: list[float] = []
    monkeypatch.setattr(scheduler, "run_job", lambda _job: next(attempts))

    result = scheduler._run_job_with_rate_limit_backoff(
        {"id": "job-1"},
        sleep=sleeps.append,
        jitter=lambda: 0.25,
        base_delay=60.0,
        max_retries=3,
    )

    assert result == (True, "success", "done", None)
    assert sleeps == [75.0, 150.0]


def test_non_rate_limit_failure_is_not_retried(monkeypatch) -> None:
    calls = 0
    sleeps: list[float] = []

    def fail(_job):
        nonlocal calls
        calls += 1
        return False, "failed", "", "RuntimeError: provider timeout"

    monkeypatch.setattr(scheduler, "run_job", fail)

    result = scheduler._run_job_with_rate_limit_backoff(
        {"id": "job-1"},
        sleep=sleeps.append,
        jitter=lambda: 0.25,
    )

    assert result[0] is False
    assert calls == 1
    assert sleeps == []


def test_retry_stops_when_rate_limit_callback_opens_circuit(monkeypatch) -> None:
    calls = 0
    sleeps: list[float] = []

    def rate_limited(_job):
        nonlocal calls
        calls += 1
        return False, "failed", "", "RuntimeError: HTTP 429"

    monkeypatch.setattr(scheduler, "run_job", rate_limited)

    result = scheduler._run_job_with_rate_limit_backoff(
        {"id": "job-1"},
        sleep=sleeps.append,
        on_rate_limit=lambda _error: True,
    )

    assert result[0] is False
    assert calls == 1
    assert sleeps == []


def test_breaker_trips_after_fleet_threshold_within_window(tmp_path) -> None:
    clock = Clock()
    breaker = FleetRateLimitCircuitBreaker(
        tmp_path / "cron" / "rate_limit_breaker.json",
        threshold=3,
        window_seconds=300,
        cooldown_seconds=600,
        clock=clock,
    )

    for job_id in ("job-a", "job-b"):
        permit = breaker.before_run(job_id)
        assert permit.allowed is True
        assert breaker.record_rate_limit(permit).tripped is False
        clock.advance(30)

    permit = breaker.before_run("job-c")
    update = breaker.record_rate_limit(permit)

    assert update.tripped is True
    assert breaker.before_run("job-d").allowed is False
    state = json.loads((tmp_path / "cron" / "rate_limit_breaker.json").read_text())
    assert state["state"] == "open"
    assert len(state["rate_limit_failures"]) == 3


def test_breaker_allows_only_one_half_open_probe_after_cooldown(tmp_path) -> None:
    clock = Clock()
    breaker = FleetRateLimitCircuitBreaker(
        tmp_path / "cron" / "rate_limit_breaker.json",
        threshold=1,
        cooldown_seconds=120,
        clock=clock,
    )
    breaker.record_rate_limit(breaker.before_run("job-a"))

    clock.advance(120)
    probe = breaker.before_run("job-b")
    blocked = breaker.before_run("job-c")

    assert probe.allowed is True
    assert probe.half_open_probe is True
    assert blocked.allowed is False
    assert "half-open probe" in blocked.reason


def test_successful_half_open_probe_closes_breaker(tmp_path) -> None:
    clock = Clock()
    breaker = FleetRateLimitCircuitBreaker(
        tmp_path / "cron" / "rate_limit_breaker.json",
        threshold=1,
        cooldown_seconds=60,
        clock=clock,
    )
    breaker.record_rate_limit(breaker.before_run("job-a"))
    clock.advance(60)
    probe = breaker.before_run("job-b")

    breaker.record_non_rate_limit(probe)

    next_permit = breaker.before_run("job-c")
    assert next_permit.allowed is True
    assert next_permit.half_open_probe is False
    state = json.loads((tmp_path / "cron" / "rate_limit_breaker.json").read_text())
    assert state["state"] == "closed"
    assert state["rate_limit_failures"] == []


def test_half_open_rate_limit_retrips_breaker(tmp_path) -> None:
    clock = Clock()
    breaker = FleetRateLimitCircuitBreaker(
        tmp_path / "cron" / "rate_limit_breaker.json",
        threshold=1,
        cooldown_seconds=60,
        clock=clock,
    )
    breaker.record_rate_limit(breaker.before_run("job-a"))
    clock.advance(60)
    probe = breaker.before_run("job-b")

    update = breaker.record_rate_limit(probe)

    assert update.tripped is True
    assert breaker.before_run("job-c").allowed is False


def test_only_open_transition_reports_tripped_for_in_flight_failures(tmp_path) -> None:
    clock = Clock()
    breaker = FleetRateLimitCircuitBreaker(
        tmp_path / "cron" / "rate_limit_breaker.json",
        threshold=1,
        cooldown_seconds=60,
        clock=clock,
    )
    first = breaker.before_run("job-a")
    already_in_flight = breaker.before_run("job-b")

    first_update = breaker.record_rate_limit(first)
    later_update = breaker.record_rate_limit(already_in_flight)

    assert first_update.tripped is True
    assert later_update.tripped is False
    assert breaker.before_run("job-c").allowed is False


def test_stale_in_flight_success_does_not_close_open_breaker(tmp_path) -> None:
    clock = Clock()
    breaker = FleetRateLimitCircuitBreaker(
        tmp_path / "cron" / "rate_limit_breaker.json",
        threshold=1,
        cooldown_seconds=60,
        clock=clock,
    )
    failing = breaker.before_run("job-a")
    eventually_successful = breaker.before_run("job-b")

    breaker.record_rate_limit(failing)
    breaker.record_non_rate_limit(eventually_successful)

    assert breaker.before_run("job-c").allowed is False


def test_run_one_job_records_breaker_skip_without_delivery(tmp_path, monkeypatch) -> None:
    clock = Clock()
    breaker = FleetRateLimitCircuitBreaker(
        tmp_path / "cron" / "rate_limit_breaker.json",
        threshold=1,
        cooldown_seconds=600,
        clock=clock,
    )
    breaker.record_rate_limit(breaker.before_run("job-a"))
    saved: list[str] = []
    marked: list[tuple[str, bool, str | None]] = []
    monkeypatch.setattr(scheduler, "_get_rate_limit_breaker", lambda: breaker)
    monkeypatch.setattr(scheduler, "run_job", lambda _job: (_ for _ in ()).throw(AssertionError("must not run")))
    monkeypatch.setattr(scheduler, "save_job_output", lambda _job_id, output: saved.append(output) or tmp_path / "run.md")
    monkeypatch.setattr(
        scheduler,
        "mark_job_run",
        lambda job_id, success, error=None, delivery_error=None: marked.append((job_id, success, error)),
    )
    monkeypatch.setattr(scheduler, "_deliver_result", lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("must not deliver")))

    processed = scheduler.run_one_job({"id": "job-b", "name": "blocked job", "schedule_display": "daily"})

    assert processed is True
    assert len(saved) == 1
    assert "## Skipped" in saved[0]
    assert "rate-limit circuit breaker" in saved[0]
    assert marked == [("job-b", False, "Skipped: fleet rate-limit circuit breaker is open")]


def test_no_agent_job_runs_while_breaker_is_open(tmp_path, monkeypatch) -> None:
    clock = Clock()
    breaker = FleetRateLimitCircuitBreaker(
        tmp_path / "cron" / "rate_limit_breaker.json",
        threshold=1,
        cooldown_seconds=600,
        clock=clock,
    )
    breaker.record_rate_limit(breaker.before_run("agent-job"))
    hermes_home = tmp_path / ".hermes"
    scripts_dir = hermes_home / "scripts"
    scripts_dir.mkdir(parents=True)
    marker = tmp_path / "sync-ran"
    (scripts_dir / "kanban-sync.sh").write_text(
        f"#!/bin/bash\necho 'Sync complete'\ntouch {marker}\n"
    )
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    marked: list[tuple[str, bool, str | None]] = []
    deliveries: list[str] = []
    monkeypatch.setattr(scheduler, "_get_rate_limit_breaker", lambda: breaker)
    monkeypatch.setattr(
        scheduler,
        "save_job_output",
        lambda _job_id, _output: tmp_path / "run.md",
    )
    monkeypatch.setattr(
        scheduler,
        "mark_job_run",
        lambda job_id, success, error=None, delivery_error=None: marked.append(
            (job_id, success, error)
        ),
    )
    monkeypatch.setattr(
        scheduler,
        "_deliver_result",
        lambda _job, content, **_kwargs: deliveries.append(content),
    )

    processed = scheduler.run_one_job(
        {
            "id": "script-job",
            "name": "kanban-sync",
            "no_agent": True,
            "script": "kanban-sync.sh",
        }
    )

    assert processed is True
    assert marker.exists()
    assert marked == [("script-job", True, None)]
    assert deliveries == ["Sync complete"]
    assert breaker.before_run("next-agent-job").allowed is False


def test_no_agent_job_does_not_consume_or_close_half_open_probe(
    tmp_path, monkeypatch
) -> None:
    clock = Clock()
    breaker = FleetRateLimitCircuitBreaker(
        tmp_path / "cron" / "rate_limit_breaker.json",
        threshold=1,
        cooldown_seconds=60,
        clock=clock,
    )
    breaker.record_rate_limit(breaker.before_run("agent-job"))
    clock.advance(60)
    probe = breaker.before_run("agent-probe")
    calls: list[str] = []
    monkeypatch.setattr(scheduler, "_get_rate_limit_breaker", lambda: breaker)
    monkeypatch.setattr(
        scheduler,
        "run_job",
        lambda job: calls.append(job["id"])
        or (True, "script output", "Sync complete", None),
    )
    monkeypatch.setattr(
        scheduler,
        "save_job_output",
        lambda _job_id, _output: tmp_path / "run.md",
    )
    monkeypatch.setattr(scheduler, "mark_job_run", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(scheduler, "_deliver_result", lambda *_args, **_kwargs: None)

    processed = scheduler.run_one_job(
        {"id": "script-job", "name": "kanban-sync", "no_agent": True}
    )

    assert probe.half_open_probe is True
    assert processed is True
    assert calls == ["script-job"]
    blocked = breaker.before_run("next-agent-job")
    assert blocked.allowed is False
    state = json.loads((tmp_path / "cron" / "rate_limit_breaker.json").read_text())
    assert state["state"] == "half_open"
    assert state["probe_token"] == probe.token


def test_breaker_trip_emits_one_notification_and_later_skips_are_silent(tmp_path, monkeypatch) -> None:
    clock = Clock()
    breaker = FleetRateLimitCircuitBreaker(
        tmp_path / "cron" / "rate_limit_breaker.json",
        threshold=1,
        cooldown_seconds=600,
        clock=clock,
    )
    deliveries: list[str] = []
    monkeypatch.setattr(scheduler, "_get_rate_limit_breaker", lambda: breaker)
    monkeypatch.setattr(
        scheduler,
        "run_job",
        lambda _job: (False, "failed", "", "RuntimeError: HTTP 429: usage limit reached"),
    )
    monkeypatch.setattr(scheduler, "save_job_output", lambda *_args: tmp_path / "run.md")
    monkeypatch.setattr(scheduler, "mark_job_run", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        scheduler,
        "_deliver_result",
        lambda _job, content, **_kwargs: deliveries.append(content),
    )

    scheduler.run_one_job({"id": "job-a", "name": "first"})
    scheduler.run_one_job({"id": "job-b", "name": "second"})

    assert len(deliveries) == 1
    assert "circuit breaker tripped" in deliveries[0].lower()
