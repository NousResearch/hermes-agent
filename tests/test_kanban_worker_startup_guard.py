import pytest


def test_worker_startup_guard_accepts_its_live_claim(monkeypatch, tmp_path):
    from hermes_cli import kanban_db as kb
    from hermes_cli import kanban_db_connect as kbc
    from cli import _kanban_worker_startup_guard

    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "home"))
    monkeypatch.setenv("HERMES_PROFILE", "worker")
    monkeypatch.setenv("HERMES_KANBAN_TASK", "task-1")
    kb._INITIALIZED_PATHS.clear()
    kb.init_db()
    conn = kbc.connect()
    try:
        task_id = kb.create_task(conn, title="live", assignee="worker")
        claimed = kb.claim_task(conn, task_id, claimer="lock-1")
        assert claimed is not None
        monkeypatch.setenv("HERMES_KANBAN_TASK", task_id)
        monkeypatch.setenv("HERMES_KANBAN_RUN_ID", str(claimed.current_run_id))
        monkeypatch.setenv("HERMES_KANBAN_CLAIM_LOCK", "lock-1")
    finally:
        conn.close()

    assert _kanban_worker_startup_guard() is True


def test_worker_startup_guard_rejects_reclaimed_task_without_failure(monkeypatch, tmp_path):
    from hermes_cli import kanban_db as kb
    from hermes_cli import kanban_db_connect as kbc
    from cli import _kanban_worker_startup_guard

    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "home"))
    monkeypatch.setenv("HERMES_PROFILE", "worker")
    kb._INITIALIZED_PATHS.clear()
    kb.init_db()
    conn = kbc.connect()
    try:
        task_id = kb.create_task(conn, title="reclaimed", assignee="worker")
        first = kb.claim_task(conn, task_id, claimer="old-lock")
        assert first is not None
        # Reclaim through the public lifecycle: the old run/lock no longer owns it.
        kb.reclaim_task(conn, task_id, reason="startup race")
        replacement = kb.claim_task(conn, task_id, claimer="new-lock")
        assert replacement is not None
    finally:
        conn.close()

    monkeypatch.setenv("HERMES_KANBAN_TASK", task_id)
    monkeypatch.setenv("HERMES_KANBAN_RUN_ID", str(first.current_run_id))
    monkeypatch.setenv("HERMES_KANBAN_CLAIM_LOCK", "old-lock")
    assert _kanban_worker_startup_guard() is False


def test_quiet_worker_exits_zero_before_session_claim_when_not_dispatchable(monkeypatch):
    import cli as cli_module

    monkeypatch.setenv("HERMES_KANBAN_TASK", "missing")
    monkeypatch.setattr(cli_module, "_kanban_worker_startup_guard", lambda: False)
    claimed = False

    class FakeCli:
        def _claim_active_session(self, *_args, **_kwargs):
            nonlocal claimed
            claimed = True
            return True

    with pytest.raises(SystemExit) as exc:
        cli_module._run_single_query_mode(FakeCli(), "work", None, True, False)
    assert exc.value.code == 0
    assert claimed is False
