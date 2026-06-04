from unittest.mock import MagicMock, patch

import pytest


@pytest.fixture(autouse=True)
def isolate_tick_state(tmp_path, monkeypatch):
    import cron.scheduler as sched

    sched._shutdown_parallel_pool()
    sched._running_job_ids.clear()

    lock_dir = tmp_path / "cron"
    lock_dir.mkdir()
    lock_file = lock_dir / ".tick.lock"
    monkeypatch.setattr(sched, "_get_lock_paths", lambda: (lock_dir, lock_file))

    yield sched

    sched._running_job_ids.clear()
    sched._shutdown_parallel_pool()


def _run_tick_with_retention(monkeypatch, retention_days):
    import cron.scheduler as sched

    job = {"id": "retention-job", "name": "Retention", "deliver": "local"}

    monkeypatch.setattr(sched, "get_due_jobs", lambda: [job])
    monkeypatch.setattr(sched, "advance_next_run", lambda *_a, **_kw: None)
    monkeypatch.setattr(sched, "run_job", lambda _job: (True, "output", "response", None))
    monkeypatch.setattr(sched, "save_job_output", lambda *_a, **_kw: "/tmp/out.md")
    monkeypatch.setattr(sched, "mark_job_run", lambda *_a, **_kw: None)
    monkeypatch.setattr(sched, "_deliver_result", lambda *_a, **_kw: None)
    monkeypatch.setattr(
        sched,
        "load_config",
        lambda: {"cron": {"session_retention_days": retention_days}},
    )

    fake_db = MagicMock()
    fake_db.prune_sessions.return_value = 3
    with patch("hermes_state.SessionDB", return_value=fake_db) as session_db_cls:
        result = sched.tick(verbose=False)

    return result, session_db_cls, fake_db


def test_session_retention_prunes_old_cron_sessions(monkeypatch):
    result, session_db_cls, fake_db = _run_tick_with_retention(monkeypatch, 7)

    assert result == 1
    session_db_cls.assert_called_once_with()
    fake_db.prune_sessions.assert_called_once_with(
        older_than_days=7,
        source="cron",
    )
    fake_db.close.assert_called_once_with()


def test_session_retention_zero_skips_prune(monkeypatch):
    result, session_db_cls, fake_db = _run_tick_with_retention(monkeypatch, 0)

    assert result == 1
    session_db_cls.assert_not_called()
    fake_db.prune_sessions.assert_not_called()
    fake_db.close.assert_not_called()


def test_session_retention_only_affects_cron_sessions(monkeypatch):
    _result, _session_db_cls, fake_db = _run_tick_with_retention(monkeypatch, 14)

    assert fake_db.prune_sessions.call_args.kwargs["source"] == "cron"

