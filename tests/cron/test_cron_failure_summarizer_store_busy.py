"""A local state.db write-lock refusal is not a provider failure.

Field report: a monitor-gated cron job failed with ``WriterMutexTimeout: database is locked (in-process
state.db writer mutex exceeded write patience)`` — the profile's own routing save held the writer for
~49 s — and the notice told the operator "the AI model service did not respond in time … this job is
pinned to its own provider/model … `hermes cron edit <id> --unpin`". No model call was involved, and the
advice named the wrong subsystem. The persistence texts must be recognised before the classifier, whose
Timeout ladder reads the exception type ``WriterMutexTimeout`` as a provider timeout.
"""

from __future__ import annotations

from cron import scheduler
from cron.scheduler import _summarize_cron_failure_for_delivery

JOB = {"name": "Nightly update self-heal", "id": "d1a2b3c4e5f6", "provider": "custom:relay"}

MUTEX_TEXT = ("WriterMutexTimeout: database is locked (in-process state.db writer mutex exceeded "
              "write patience)")

FOREIGN_LOCK_TEXT = (
    "sqlite3.OperationalError: database is locked (another Hermes process held the state.db write lock "
    "for over 30s — likely a long maintenance operation such as VACUUM, a large WAL checkpoint, or an "
    "older pre-update process; the database itself is healthy)")

PROVIDER_TEXT = "httpx.ReadTimeout: The read operation timed out"


def _assert_store_busy(msg: str) -> None:
    assert "state.db write lock" in msg
    assert "did not respond in time" not in msg
    assert "--unpin" not in msg
    assert "backup provider" not in msg.lower()
    assert "hermes cron run d1a2b3c4e5f6" in msg


def test_writer_mutex_timeout_is_not_a_provider_timeout():
    _assert_store_busy(_summarize_cron_failure_for_delivery(JOB, MUTEX_TEXT))


def test_foreign_state_db_lock_is_not_a_provider_timeout():
    _assert_store_busy(_summarize_cron_failure_for_delivery(JOB, FOREIGN_LOCK_TEXT))


def test_a_real_provider_timeout_still_reads_as_one(monkeypatch):
    monkeypatch.setattr(scheduler, "load_config", lambda: {"fallback_providers": []})
    monkeypatch.setattr(scheduler, "get_fallback_chain", lambda cfg: [])
    msg = _summarize_cron_failure_for_delivery(JOB, PROVIDER_TEXT)
    assert "did not respond in time" in msg
    assert "state.db" not in msg


def test_an_unrelated_locked_database_message_is_left_alone():
    # A job's own tool output can mention a locked database too; only the holder's own wording is a
    # Hermes-state.db refusal.
    msg = _summarize_cron_failure_for_delivery(JOB, "OperationalError: table jobs is locked")
    assert "state.db write lock" not in msg
