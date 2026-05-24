from __future__ import annotations

from datetime import datetime, timezone

from hermes_cli.usage_guard import (
    RETRYABLE_EXIT_CODE,
    clear_state,
    detect_rate_limit,
    load_state,
    run_guarded_command,
)


def test_detect_rate_limit_retry_after_seconds():
    now = datetime(2026, 5, 24, 12, 0, tzinfo=timezone.utc)

    pause = detect_rate_limit("HTTP 429 rate limit. Retry-After: 90", attempts=0, now=now)

    assert pause is not None
    assert pause.source == "retry-after"
    assert pause.resume_at.isoformat() == "2026-05-24T12:01:30+00:00"


def test_detect_rate_limit_try_again_minutes():
    now = datetime(2026, 5, 24, 12, 0, tzinfo=timezone.utc)

    pause = detect_rate_limit("quota exceeded, try again in 12 minutes", attempts=0, now=now)

    assert pause is not None
    assert pause.source == "retry-after"
    assert pause.resume_at.isoformat() == "2026-05-24T12:12:00+00:00"


def test_detect_rate_limit_uses_exponential_backoff_when_no_reset_hint():
    now = datetime(2026, 5, 24, 12, 0, tzinfo=timezone.utc)

    pause = detect_rate_limit("too many requests from provider", attempts=2, now=now)

    assert pause is not None
    assert pause.source == "exponential-backoff"
    assert pause.resume_at.isoformat() == "2026-05-24T13:00:00+00:00"


def test_run_guarded_command_persists_pause_and_skips_until_resume(monkeypatch, tmp_path):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes"))
    script = tmp_path / "rate_limited.py"
    script.write_text("import sys; print('429 rate limit. Retry-After: 3600', file=sys.stderr); sys.exit(1)\n")
    now = datetime(2026, 5, 24, 12, 0, tzinfo=timezone.utc)

    first_code = run_guarded_command(["python", str(script)], task_id="night-job", now=now)
    state = load_state("night-job")
    second_code = run_guarded_command(["python", str(script)], task_id="night-job", now=now)

    assert first_code == RETRYABLE_EXIT_CODE
    assert second_code == RETRYABLE_EXIT_CODE
    assert state is not None
    assert state.status == "paused"
    assert state.attempts == 1
    assert state.resume_at is not None
    assert state.resume_at.isoformat() == "2026-05-24T13:00:00+00:00"
    assert clear_state("night-job") is True


def test_run_guarded_command_records_completed_state(monkeypatch, tmp_path):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes"))
    script = tmp_path / "ok.py"
    script.write_text("print('done')\n")

    code = run_guarded_command(["python", str(script)], task_id="ok-job", now=datetime(2026, 5, 24, 12, 0, tzinfo=timezone.utc))
    state = load_state("ok-job")

    assert code == 0
    assert state is not None
    assert state.status == "completed"
    assert state.last_exit_code == 0
