"""Regression for #146855: the cron session-title lineage fallback must be computed
lazily inside its recovery block. Evaluated eagerly in the tuple, a raising
``get_next_title_in_lineage`` escaped the except and aborted ``run_job``'s
finally before the ephemeral agent was deferred for teardown."""
from types import SimpleNamespace

import pytest


def test_title_lineage_fallback_failure_stays_contained(monkeypatch):
    from cron import scheduler

    monkeypatch.setattr(
        scheduler, "_set_cron_session_title",
        lambda *a: (_ for _ in ()).throw(TimeoutError("write lock")))

    class _RaisingDB:
        def get_compression_tip(self, sid):
            raise RuntimeError("db dead")

        def get_next_title_in_lineage(self, base):
            raise RuntimeError("session finalization disabled")

        def session_lifecycle_statuses(self, ids):
            return {}

        def end_session(self, sid, reason):
            raise RuntimeError("db dead")

        def close(self):
            pass

    monkeypatch.setattr(
        "hermes_state_registry.release_or_close", lambda db: None)
    monkeypatch.setattr(
        scheduler, "_BoundedCronSessionDB", lambda db, job_id: db)

    # Must not raise: every failure in the title path is best-effort recovery,
    # never a reason to skip the rest of session finalization.
    scheduler._finalize_cron_session(
        _RaisingDB(), SimpleNamespace(session_id="cron-x", _end_session_on_close=True),
        "job-1", "nightly", "cron-x")
