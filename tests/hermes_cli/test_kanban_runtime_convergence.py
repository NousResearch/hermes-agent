from __future__ import annotations

import sqlite3

import pytest

from hermes_cli import kanban_db as kb


INTENT = "a" * 64


@pytest.fixture
def conn(tmp_path):
    connection = kb.connect(tmp_path / "kanban.db")
    yield connection
    connection.close()


def _steps():
    return [
        {"step": "build", "title": "Build", "assignee": "worker"},
        {
            "step": "deliver",
            "title": "Deliver",
            "assignee": "operator",
            "parents": ["build"],
        },
    ]


def test_batch_import_is_atomic_and_reimport_is_stable(conn):
    first = kb.import_card_batch(
        conn,
        project="kalibrio-aios",
        card_id="root-m1b",
        intent_hash=INTENT,
        steps=_steps(),
    )
    second = kb.import_card_batch(
        conn,
        project="kalibrio-aios",
        card_id="root-m1b",
        intent_hash=INTENT,
        steps=_steps(),
    )

    assert second["tasks"] == first["tasks"]
    assert second["created"] == []
    assert set(second["reused"]) == set(first["tasks"].values())
    build = kb.get_task(conn, first["tasks"]["build"])
    deliver = kb.get_task(conn, first["tasks"]["deliver"])
    assert build.idempotency_key == (
        f"v1:kalibrio-aios:root-m1b:{INTENT}:build"
    )
    assert build.status == "ready"
    assert deliver.status == "todo"
    assert conn.execute(
        "SELECT COUNT(*) FROM task_links WHERE parent_id = ? AND child_id = ?",
        (build.id, deliver.id),
    ).fetchone()[0] == 1
    with pytest.raises(ValueError, match="reserved"):
        kb.create_task(
            conn,
            title="worker-chosen canonical key",
            idempotency_key=f"v1:kalibrio-aios:root-m1b:{INTENT}:other",
        )


def test_batch_failure_rolls_back_every_task_and_link(conn, monkeypatch):
    real_append = kb._append_event
    calls = 0

    def fail_second_event(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("injected import failure")
        return real_append(*args, **kwargs)

    monkeypatch.setattr(kb, "_append_event", fail_second_event)
    with pytest.raises(RuntimeError, match="injected import failure"):
        kb.import_card_batch(
            conn,
            project="kalibrio-aios",
            card_id="root-rollback",
            intent_hash=INTENT,
            steps=_steps(),
        )
    assert conn.execute(
        "SELECT COUNT(*) FROM tasks WHERE card_id = 'root-rollback'"
    ).fetchone()[0] == 0
    assert conn.execute("SELECT COUNT(*) FROM task_links").fetchone()[0] == 0


def test_historical_rekey_keeps_deterministic_delivered_survivor(tmp_path):
    db_path = tmp_path / "legacy.db"
    conn = kb.connect(db_path)
    conn.execute("DROP INDEX ux_tasks_idempotency_survivor")
    old_key = f"card:root-history:{INTENT}:build"
    with kb.write_txn(conn):
        conn.execute(
            "INSERT INTO tasks (id, title, status, created_at, workspace_kind, "
            "idempotency_key, result) VALUES "
            "('t_ready', 'ready copy', 'ready', 1, 'scratch', ?, NULL), "
            "('t_done', 'delivered copy', 'done', 2, 'scratch', ?, 'commit abc')",
            (old_key, old_key),
        )
    conn.close()

    kb.init_db(db_path)
    conn = kb.connect(db_path)
    try:
        survivor = conn.execute(
            "SELECT id, idempotency_key FROM tasks WHERE superseded_by IS NULL "
            "AND card_id = 'root-history'"
        ).fetchone()
        loser = conn.execute(
            "SELECT superseded_by, status FROM tasks WHERE id = 't_ready'"
        ).fetchone()
        assert survivor["id"] == "t_done"
        assert survivor["idempotency_key"] == (
            f"v1:legacy:root-history:{INTENT}:build"
        )
        assert dict(loser) == {"superseded_by": "t_done", "status": "archived"}
        relation = conn.execute(
            "SELECT survivor_id FROM task_supersessions WHERE loser_id = 't_ready'"
        ).fetchone()
        assert relation["survivor_id"] == "t_done"
        with pytest.raises(sqlite3.IntegrityError):
            conn.execute(
                "INSERT INTO tasks (id, title, status, created_at, workspace_kind, "
                "idempotency_key) VALUES ('t_duplicate', 'dup', 'ready', 3, 'scratch', ?)",
                (survivor["idempotency_key"],),
            )
    finally:
        conn.close()


def test_three_blocked_runs_freeze_until_an_observable_changes(conn):
    task_id = kb.create_task(
        conn, title="repeat blocker", assignee="worker"
    )
    for attempt in range(3):
        claimed = kb.claim_task(conn, task_id)
        assert claimed is not None
        assert kb.block_task(conn, task_id, reason="same wall", kind="needs_input")
        if attempt == 0:
            assert kb.unblock_task(conn, task_id)
        elif attempt == 1:
            # The legacy loop breaker escalates at two. A triager explicitly
            # sends the task back through machinery; the third blocked run is
            # what activates the stronger M1B fingerprint quarantine.
            assert kb.specify_triage_task(conn, task_id)
            assert kb.get_task(conn, task_id).status == "ready"

    frozen = kb.get_task(conn, task_id)
    assert frozen.status == "triage"
    assert frozen.quarantine_fingerprint
    fingerprint = frozen.quarantine_fingerprint
    assert conn.execute(
        "SELECT COUNT(*) FROM tasks WHERE idempotency_key = ?",
        (f"quarantine:{task_id}:{fingerprint}",),
    ).fetchone()[0] == 1

    assert kb._refresh_quarantines(conn) == 0
    still_frozen = kb.get_task(conn, task_id)
    assert still_frozen.quarantine_fingerprint == fingerprint
    same = kb.set_task_observables(conn, task_id, generation=0)
    assert same["quarantine_released"] is False

    changed = kb.set_task_observables(conn, task_id, generation=1)
    assert changed["quarantine_released"] is True
    released = kb.get_task(conn, task_id)
    assert released.status == "ready"
    assert released.quarantine_fingerprint is None


def test_step_back_suffix_requires_completed_linked_retrospective(conn):
    imported = kb.import_card_batch(
        conn,
        project="kalibrio-aios",
        card_id="root-runtime-v2",
        intent_hash=INTENT,
        steps=[{"step": "build", "title": "Recovery build", "assignee": "worker"}],
    )
    task_id = imported["tasks"]["build"]
    assert kb._step_back_gate_reason(conn, task_id) == "retrospective_required"

    retro = kb.create_task(conn, title="Retrospective", assignee="reviewer")
    with kb.write_txn(conn):
        conn.execute(
            "UPDATE tasks SET status = 'done', completed_at = 1 WHERE id = ?",
            (retro,),
        )
    kb.import_card_batch(
        conn,
        project="kalibrio-aios",
        card_id="root-runtime-v2",
        intent_hash=INTENT,
        steps=[{"step": "build", "title": "Recovery build", "assignee": "worker"}],
        retrospective_task_id=retro,
    )
    assert kb._step_back_gate_reason(conn, task_id) is None
