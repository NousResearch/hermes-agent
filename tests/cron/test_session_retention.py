import concurrent.futures
import time
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
    monkeypatch.setattr(
        sched,
        "run_job",
        lambda _job, *, defer_agent_teardown=None: (
            True,
            "output",
            "response",
            None,
        ),
    )
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
    import cron.scheduler as sched

    result, session_db_cls, fake_db = _run_tick_with_retention(monkeypatch, 7)

    assert result == 1
    session_db_cls.assert_called_once_with()
    fake_db.prune_sessions.assert_called_once_with(
        older_than_days=7,
        source="cron",
        sessions_dir=sched._get_hermes_home() / "sessions",
    )
    fake_db.close.assert_called_once_with()


def test_session_retention_zero_skips_prune(monkeypatch):
    result, session_db_cls, fake_db = _run_tick_with_retention(monkeypatch, 0)

    assert result == 1
    session_db_cls.assert_not_called()
    fake_db.prune_sessions.assert_not_called()
    fake_db.close.assert_not_called()


def test_session_retention_default_is_disabled():
    from hermes_cli.config import DEFAULT_CONFIG

    assert DEFAULT_CONFIG["cron"]["session_retention_days"] == 0


def test_session_retention_only_affects_cron_sessions(monkeypatch):
    _result, _session_db_cls, fake_db = _run_tick_with_retention(monkeypatch, 14)

    assert fake_db.prune_sessions.call_args.kwargs["source"] == "cron"


def test_session_retention_removes_cron_artifacts_but_preserves_users_and_active(tmp_path, monkeypatch):
    import hermes_state
    import cron.scheduler as sched

    home = tmp_path / "profile"
    sessions_dir = home / "sessions"
    sessions_dir.mkdir(parents=True)
    db_path = home / "state.db"
    db = hermes_state.SessionDB(db_path=db_path)
    old_cron = db.create_session("old-cron", source="cron")
    old_user = db.create_session("old-user", source="cli")
    active = db.create_session("active-cron", source="cron")
    db.end_session(old_cron, "completed")
    db.end_session(old_user, "completed")
    with db._lock:
        old = time.time() - 10 * 86400
        db._conn.execute("UPDATE sessions SET started_at = ? WHERE id IN (?, ?)", (old, old_cron, old_user))
        db._conn.commit()
    db.close()

    for suffix in (".json", ".jsonl"):
        (sessions_dir / f"{old_cron}{suffix}").write_text("artifact")
    (sessions_dir / f"request_dump_{old_cron}_1.json").write_text("artifact")

    monkeypatch.setattr(sched, "_get_hermes_home", lambda: home)
    monkeypatch.setattr(sched, "load_config", lambda: {"cron": {"session_retention_days": 7}})
    monkeypatch.setattr(hermes_state, "DEFAULT_DB_PATH", db_path)
    sched._prune_cron_sessions_from_config()

    db = hermes_state.SessionDB(db_path=db_path)
    assert db.get_session(old_cron) is None
    assert db.get_session(old_user) is not None
    assert db.get_session(active) is not None
    db.close()
    assert not list(sessions_dir.glob(f"{old_cron}*"))
    assert not list(sessions_dir.glob(f"request_dump_{old_cron}_*.json"))


def test_tick_runs_post_run_maintenance_once_for_empty_batch(monkeypatch):
    import cron.scheduler as sched

    calls = []
    monkeypatch.setattr(sched, "get_due_jobs", lambda: [])
    monkeypatch.setattr(sched, "_run_post_run_maintenance", lambda: calls.append(1))

    assert sched.tick(verbose=True) == 0
    assert calls == [1]


def test_sync_tick_runs_post_run_maintenance_once_for_batch(monkeypatch):
    import cron.scheduler as sched

    calls = []
    monkeypatch.setattr(sched, "get_due_jobs", lambda: [{"id": "a"}, {"id": "b"}])
    monkeypatch.setattr(sched, "advance_next_run", lambda *_a, **_kw: None)
    monkeypatch.setattr(sched, "run_one_job", lambda *_a, **_kw: True)
    monkeypatch.setattr(sched, "_run_post_run_maintenance", lambda: calls.append(1))

    assert sched.tick(verbose=False) == 2
    assert calls == [1]


def test_async_tick_runs_post_run_maintenance_after_last_future(monkeypatch):
    import cron.scheduler as sched

    futures = []
    calls = []

    class Pool:
        def submit(self, fn):
            future = concurrent.futures.Future()
            futures.append((future, fn))
            return future

    monkeypatch.setattr(sched, "get_due_jobs", lambda: [{"id": "a"}, {"id": "b"}])
    monkeypatch.setattr(sched, "advance_next_run", lambda *_a, **_kw: None)
    monkeypatch.setattr(sched, "_get_parallel_pool", lambda *_a, **_kw: Pool())
    monkeypatch.setattr(sched, "run_one_job", lambda *_a, **_kw: True)
    monkeypatch.setattr(sched, "_run_post_run_maintenance", lambda: calls.append(1))

    assert sched.tick(verbose=False, sync=False) == 2
    assert calls == []
    futures[0][0].set_result(True)
    assert calls == []
    futures[1][0].set_result(True)
    assert calls == [1]

