import json

import hermes_cli.kanban_db as kb


class _UnreadablePausePath:
    def exists(self):
        raise OSError("pause state unavailable")


def _pause(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_KANBAN_HOME", str(tmp_path))
    path = tmp_path / "state" / "dispatch_pause.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"reason": "test pause"}), encoding="utf-8")
    return path


def test_pause_blocks_ready_and_review_claims_without_run_rows(tmp_path, monkeypatch):
    db_path = tmp_path / "kanban.db"
    kb.init_db(db_path=db_path)
    conn = kb.connect(db_path=db_path)
    ready_id = kb.create_task(conn, title="ready", assignee="default")
    review_id = kb.create_task(conn, title="review", assignee="default")
    conn.execute("UPDATE tasks SET status = 'review' WHERE id = ?", (review_id,))
    conn.commit()
    pause = _pause(tmp_path, monkeypatch)

    assert kb.claim_task(conn, ready_id) is None
    assert kb.claim_review_task(conn, review_id) is None
    assert conn.execute("SELECT COUNT(*) FROM task_runs").fetchone()[0] == 0

    pause.unlink()
    assert kb.claim_task(conn, ready_id) is not None
    assert kb.claim_review_task(conn, review_id) is not None


def test_pause_lookup_error_fails_closed(monkeypatch):
    monkeypatch.setattr(kb, "dispatch_pause_path", lambda: _UnreadablePausePath())
    assert kb.dispatch_is_paused() is True


def test_pause_blocks_dispatch_and_final_spawn_edge(tmp_path, monkeypatch):
    db_path = tmp_path / "kanban.db"
    kb.init_db(db_path=db_path)
    conn = kb.connect(db_path=db_path)
    task_id = kb.create_task(conn, title="ready", assignee="default")
    pause = _pause(tmp_path, monkeypatch)
    spawned = []

    result = kb.dispatch_once(
        conn,
        spawn_fn=lambda *args, **kwargs: spawned.append((args, kwargs)),
        max_spawn=1,
    )
    assert result.spawned == []
    assert spawned == []
    assert conn.execute("SELECT COUNT(*) FROM task_runs").fetchone()[0] == 0

    task = kb.get_task(conn, task_id)
    try:
        kb._default_spawn(task, str(tmp_path))
    except RuntimeError as exc:
        assert str(pause) in str(exc)
    else:
        raise AssertionError("paused final spawn edge did not fail closed")

    pause.unlink()
    resumed = kb.dispatch_once(
        conn,
        spawn_fn=lambda *args, **kwargs: 4242,
        max_spawn=1,
    )
    assert [item[0] for item in resumed.spawned] == [task_id]


def test_pause_arriving_after_ready_claim_requeues_without_failure(tmp_path, monkeypatch):
    db_path = tmp_path / "kanban.db"
    kb.init_db(db_path=db_path)
    conn = kb.connect(db_path=db_path)
    task_id = kb.create_task(conn, title="ready", assignee="default")

    def pause_at_spawn(*_args, **_kwargs):
        _pause(tmp_path, monkeypatch)
        raise kb.DispatchPausedError("paused at final edge")

    result = kb.dispatch_once(conn, spawn_fn=pause_at_spawn, max_spawn=1)
    task = kb.get_task(conn, task_id)
    assert result.spawned == []
    assert task.status == "ready"
    assert task.consecutive_failures == 0
    run = conn.execute(
        "SELECT status, outcome, ended_at FROM task_runs WHERE task_id = ?",
        (task_id,),
    ).fetchone()
    assert (run["status"], run["outcome"]) == ("reclaimed", "reclaimed")
    assert run["ended_at"] is not None


def test_pause_arriving_after_review_claim_returns_to_review(tmp_path, monkeypatch):
    db_path = tmp_path / "kanban.db"
    kb.init_db(db_path=db_path)
    conn = kb.connect(db_path=db_path)
    task_id = kb.create_task(conn, title="review", assignee="default")
    conn.execute("UPDATE tasks SET status = 'review' WHERE id = ?", (task_id,))
    conn.commit()
    monkeypatch.setattr(kb, "review_dispatch_enabled", lambda: True)

    def pause_at_spawn(*_args, **_kwargs):
        _pause(tmp_path, monkeypatch)
        raise kb.DispatchPausedError("paused at final edge")

    result = kb.dispatch_once(conn, spawn_fn=pause_at_spawn, max_spawn=1)
    task = kb.get_task(conn, task_id)
    assert result.spawned == []
    assert task.status == "review"
    assert task.consecutive_failures == 0
