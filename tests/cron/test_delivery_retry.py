"""Tests for cron delivery-only retry on transient platform failures (#8846).

When a cron job's task runs successfully but delivery to the messaging
platform fails (Telegram 502, schema validation, etc.) the scheduler must:

1. Leave `task_status = ok` so monitoring isn't poisoned by Telegram outages.
2. Persist a small retry envelope keyed off the cached output file under
   `~/.hermes/cron/output/<job_id>/`.
3. Re-attempt delivery on subsequent ticks with bounded backoff
   (60s, 300s, 1800s; 3 attempts total).
4. Never re-run the agent - the cached output is the source of truth.
"""

from datetime import timedelta
from pathlib import Path
from unittest.mock import patch

import pytest

from cron.jobs import (
    DELIVERY_RETRY_BACKOFF_SECONDS,
    DELIVERY_RETRY_MAX_ATTEMPTS,
    clear_delivery_retry,
    create_job,
    get_due_delivery_retries,
    load_jobs,
    mark_delivery_retry_attempt,
    save_jobs,
    save_job_output,
    schedule_delivery_retry,
)
from hermes_time import now as _hermes_now


@pytest.fixture(autouse=True)
def tmp_cron_dir(tmp_path, monkeypatch):
    """Redirect cron storage to a per-test temp directory.

    Mirrors the fixture in `tests/cron/test_jobs.py`. Module-level path
    constants in `cron.jobs` are cached at import time, so the autouse
    `_hermetic_environment` fixture (which only sets `HERMES_HOME`) is
    not enough on its own.
    """
    monkeypatch.setattr("cron.jobs.CRON_DIR", tmp_path / "cron")
    monkeypatch.setattr("cron.jobs.JOBS_FILE", tmp_path / "cron" / "jobs.json")
    monkeypatch.setattr("cron.jobs.OUTPUT_DIR", tmp_path / "cron" / "output")
    return tmp_path


def _make_job():
    """Create a minimal cron job in the hermetic HERMES_HOME."""
    return create_job(
        prompt="say hi",
        schedule="every 1h",
        deliver="telegram:12345",
        repeat=None,
    )


# ─── Backoff schedule ──────────────────────────────────────────────────────────


def test_backoff_schedule_constants():
    """Backoff matches the contract in #8846: 60s, 300s, 1800s."""
    assert DELIVERY_RETRY_BACKOFF_SECONDS == [60, 300, 1800]
    assert DELIVERY_RETRY_MAX_ATTEMPTS == 3


# ─── schedule_delivery_retry ──────────────────────────────────────────────────


def test_schedule_initial_retry_records_envelope():
    job = _make_job()
    output_file = save_job_output(job["id"], "hello world")

    scheduled = schedule_delivery_retry(
        job["id"], str(output_file), "telegram returned 502",
    )
    assert scheduled is True

    stored = next(j for j in load_jobs() if j["id"] == job["id"])
    envelope = stored["last_delivery_retry"]
    assert envelope is not None
    assert envelope["output_file"] == str(output_file)
    assert envelope["attempts"] == 0
    assert envelope["last_error"] == "telegram returned 502"
    # next_attempt_at should be ~60s from now
    next_at = envelope["next_attempt_at"]
    assert next_at  # non-empty ISO timestamp


def test_schedule_returns_false_for_unknown_job():
    assert schedule_delivery_retry("does-not-exist", "/tmp/x", "boom") is False


# ─── mark_delivery_retry_attempt: success path ────────────────────────────────


def test_success_clears_envelope_and_error():
    job = _make_job()
    output_file = save_job_output(job["id"], "payload")
    schedule_delivery_retry(job["id"], str(output_file), "first failure")

    # Mark success
    still = mark_delivery_retry_attempt(job["id"], None)
    assert still is False  # nothing pending

    stored = next(j for j in load_jobs() if j["id"] == job["id"])
    assert stored["last_delivery_retry"] is None
    assert stored["last_delivery_error"] is None


# ─── mark_delivery_retry_attempt: backoff progression ─────────────────────────


def test_backoff_advances_through_attempts():
    job = _make_job()
    output_file = save_job_output(job["id"], "p")
    schedule_delivery_retry(job["id"], str(output_file), "fail-0")

    # Attempt 1 fails -> attempts=1, next backoff = 300s
    still = mark_delivery_retry_attempt(job["id"], "fail-1")
    assert still is True
    env = next(j for j in load_jobs() if j["id"] == job["id"])["last_delivery_retry"]
    assert env["attempts"] == 1
    assert env["last_error"] == "fail-1"

    # Attempt 2 fails -> attempts=2, next backoff = 1800s
    still = mark_delivery_retry_attempt(job["id"], "fail-2")
    assert still is True
    env = next(j for j in load_jobs() if j["id"] == job["id"])["last_delivery_retry"]
    assert env["attempts"] == 2

    # Attempt 3 fails -> budget exhausted, envelope cleared
    still = mark_delivery_retry_attempt(job["id"], "fail-3")
    assert still is False
    stored = next(j for j in load_jobs() if j["id"] == job["id"])
    assert stored["last_delivery_retry"] is None
    # Last error stays visible to the user
    assert stored["last_delivery_error"] == "fail-3"


def test_clear_delivery_retry_drops_envelope():
    job = _make_job()
    output_file = save_job_output(job["id"], "p")
    schedule_delivery_retry(job["id"], str(output_file), "boom")

    clear_delivery_retry(job["id"])
    stored = next(j for j in load_jobs() if j["id"] == job["id"])
    assert stored["last_delivery_retry"] is None
    assert stored["last_delivery_error"] is None


# ─── get_due_delivery_retries ─────────────────────────────────────────────────


def test_no_retries_due_when_envelope_in_future():
    job = _make_job()
    output_file = save_job_output(job["id"], "p")
    schedule_delivery_retry(job["id"], str(output_file), "boom")
    # Default schedule is now+60s, which is in the future
    due = get_due_delivery_retries()
    assert all(j["id"] != job["id"] for j in due)


def test_retry_picked_up_when_due():
    job = _make_job()
    output_file = save_job_output(job["id"], "p")
    schedule_delivery_retry(job["id"], str(output_file), "boom")

    # Backdate the next_attempt_at so it is due now
    jobs = load_jobs()
    for j in jobs:
        if j["id"] == job["id"]:
            past = (_hermes_now() - timedelta(seconds=5)).isoformat()
            j["last_delivery_retry"]["next_attempt_at"] = past
    save_jobs(jobs)

    due = get_due_delivery_retries()
    assert any(j["id"] == job["id"] for j in due)


def test_retry_envelope_with_no_next_at_is_skipped():
    job = _make_job()
    output_file = save_job_output(job["id"], "p")
    schedule_delivery_retry(job["id"], str(output_file), "boom")
    jobs = load_jobs()
    for j in jobs:
        if j["id"] == job["id"]:
            j["last_delivery_retry"]["next_attempt_at"] = ""
    save_jobs(jobs)

    due = get_due_delivery_retries()
    assert all(j["id"] != job["id"] for j in due)


# ─── _drain_due_delivery_retries integration ──────────────────────────────────


def test_drain_replays_delivery_from_cached_output():
    """The retry pass must read the cached file and call _deliver_result.

    It must NOT re-run the agent.
    """
    from cron import scheduler as sched

    job = _make_job()
    output_file = save_job_output(job["id"], "the cached agent output")
    schedule_delivery_retry(job["id"], str(output_file), "first failure")

    # Make it due
    jobs = load_jobs()
    for j in jobs:
        if j["id"] == job["id"]:
            j["last_delivery_retry"]["next_attempt_at"] = (
                _hermes_now() - timedelta(seconds=5)
            ).isoformat()
    save_jobs(jobs)

    # Stub _deliver_result so we can assert it was called with the cached
    # content, and have it return success this time.
    captured = {}

    def fake_deliver(job_arg, content_arg, adapters=None, loop=None):
        captured["job_id"] = job_arg["id"]
        captured["content"] = content_arg
        return None  # success

    with patch.object(sched, "_deliver_result", side_effect=fake_deliver):
        processed = sched._drain_due_delivery_retries(verbose=False)

    assert processed == 1
    assert captured["job_id"] == job["id"]
    assert captured["content"] == "the cached agent output"

    stored = next(j for j in load_jobs() if j["id"] == job["id"])
    # Success cleared the envelope
    assert stored["last_delivery_retry"] is None
    assert stored["last_delivery_error"] is None


def test_drain_failure_advances_backoff_then_exhausts():
    from cron import scheduler as sched

    job = _make_job()
    output_file = save_job_output(job["id"], "payload")
    schedule_delivery_retry(job["id"], str(output_file), "initial failure")

    def make_due():
        jobs = load_jobs()
        for j in jobs:
            if j["id"] == job["id"] and j.get("last_delivery_retry"):
                j["last_delivery_retry"]["next_attempt_at"] = (
                    _hermes_now() - timedelta(seconds=5)
                ).isoformat()
        save_jobs(jobs)

    with patch.object(
        sched,
        "_deliver_result",
        side_effect=lambda *a, **kw: "still 502",
    ):
        # First retry attempt - moves attempts: 0 -> 1
        make_due()
        assert sched._drain_due_delivery_retries() == 1
        env = next(j for j in load_jobs() if j["id"] == job["id"])["last_delivery_retry"]
        assert env is not None and env["attempts"] == 1

        # Second
        make_due()
        assert sched._drain_due_delivery_retries() == 1
        env = next(j for j in load_jobs() if j["id"] == job["id"])["last_delivery_retry"]
        assert env is not None and env["attempts"] == 2

        # Third - budget exhausted, envelope cleared but last_delivery_error visible
        make_due()
        assert sched._drain_due_delivery_retries() == 1
        stored = next(j for j in load_jobs() if j["id"] == job["id"])
        assert stored["last_delivery_retry"] is None
        assert stored["last_delivery_error"] == "still 502"


def test_drain_handles_missing_output_file():
    """Cached file deleted out from under us -> give up, no infinite loop."""
    from cron import scheduler as sched

    job = _make_job()
    schedule_delivery_retry(job["id"], "/tmp/does-not-exist-xyz.md", "boom")

    jobs = load_jobs()
    for j in jobs:
        if j["id"] == job["id"]:
            j["last_delivery_retry"]["next_attempt_at"] = (
                _hermes_now() - timedelta(seconds=5)
            ).isoformat()
    save_jobs(jobs)

    with patch.object(sched, "_deliver_result") as mock_deliver:
        processed = sched._drain_due_delivery_retries()
        assert processed == 1
        mock_deliver.assert_not_called()

    stored = next(j for j in load_jobs() if j["id"] == job["id"])
    # Envelope cleared so we don't retry forever.
    assert stored["last_delivery_retry"] is None


def test_drain_returns_zero_when_no_retries_pending():
    from cron import scheduler as sched

    _make_job()  # job with no retry envelope
    assert sched._drain_due_delivery_retries() == 0


# ─── Job initialization ───────────────────────────────────────────────────────


def test_new_job_has_empty_retry_envelope():
    job = _make_job()
    assert job.get("last_delivery_retry") is None
