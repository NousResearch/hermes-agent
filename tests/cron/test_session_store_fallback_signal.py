"""A run that continues without a session store must say so above DEBUG.

The canonical (owner) path fails closed on a store failure — see
``test_sessiondb_init_hang.py``. The legacy non-owner path may still run with no
store at all, and that is the run whose transcript is missing from ``state.db``:
finding out only after the conversation fails to appear is not a signal.
"""

import logging

from cron import scheduler
from gateway import session_cron


def test_non_owner_store_failure_is_reported_at_warning(caplog, monkeypatch):
    monkeypatch.setattr(session_cron, "current_execution", lambda: None)
    monkeypatch.setattr(scheduler, "_get_session_db_timeout", lambda: 0.0)
    monkeypatch.setattr("agent.runtime_session_store.is_worker_process", lambda: False)

    def store_unavailable(*_args, **_kwargs):
        raise OSError("unable to open database file")

    monkeypatch.setattr("hermes_state_registry.acquire", store_unavailable)

    with caplog.at_level(logging.DEBUG, logger="cron.scheduler"):
        assert scheduler._open_cron_session_db({"id": "job-no-store"}) is None

    records = [r for r in caplog.records if "session store not available" in r.getMessage()]
    assert records, "the run continues without a store, so the fallback must be logged"
    assert records[0].levelno == logging.WARNING
    assert "job-no-store" in records[0].getMessage()
