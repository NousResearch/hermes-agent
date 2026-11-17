"""Disposable qualification for the native board-custody recovery path.

This is deliberately one end-to-end fixture.  It is not a live-board probe:
all databases, blobs, pointers, leases, and reports live below ``tmp_path``.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import sqlite3
from pathlib import Path

import pytest

from hermes_cli.kanban_generation import (
    GenerationError,
    acquire_fence,
    activate_first_write,
    attachment_path,
    build_ready,
    gc_generations,
    manifest_digest,
    open_generation,
    publish_prepared,
    read_pointer,
    release_fence,
    resolve_attachment_blob,
    rollback_pre_activation,
    validate_generation,
    initialize_generation_db,
)
from hermes_cli.kanban_reconcile import reconcile


SEEDS = tuple(f"qualification-seed-{index:02d}" for index in range(1, 17))
CHILD_TABLES = ("task_comments", "task_runs", "task_events", "task_attachments")


def _create_fixture(path: Path, *, tasks: list[str], blobs: Path, target: bool) -> None:
    with sqlite3.connect(path) as conn:
        from hermes_cli.kanban_db import SCHEMA_SQL
        conn.executescript(SCHEMA_SQL)
        conn.execute("ALTER TABLE tasks ADD COLUMN completion_contract TEXT")
        for sql in (
            "CREATE INDEX idx_tasks_assignee_status ON tasks(assignee, status)",
            "CREATE INDEX idx_tasks_tenant ON tasks(tenant)",
            "CREATE INDEX idx_tasks_idempotency ON tasks(idempotency_key)",
            "CREATE INDEX idx_tasks_session_id ON tasks(session_id)",
            "CREATE INDEX idx_events_run ON task_events(run_id, id)",
        ):
            conn.execute(sql)

    def add_task(task_id: str, *, status: str, target_row: bool = False) -> None:
        conn.execute(
            """INSERT INTO tasks
               (id, title, body, assignee, status, created_at, workspace_kind,
                workspace_path, branch_name, project_id, claim_lock, claim_expires,
                tenant, idempotency_key, block_kind, block_recurrences,
                current_run_id, session_id, worker_pid, worker_started_at,
                last_heartbeat_at)
               VALUES (?, ?, '', ?, ?, 1, 'worktree', ?, ?, NULL, ?, ?,
                       ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                task_id,
                "target title" if target_row else f"title {task_id}",
                "william" if target_row else None,
                status,
                "/absolute/workspace" if target_row else f"/source/{task_id}",
                "main" if target_row else None,
                "claim-token" if target_row else None,
                99 if target_row else None,
                None,
                f"key-{task_id}",
                "incident",
                1,
                10 if target_row else None,
                "session-1" if target_row else None,
                1234 if target_row else None,
                1 if target_row else None,
                98 if target_row else None,
            ),
        )

    for task_id in tasks:
        is_target = target and task_id == "target-running"
        add_task(task_id, status="running" if is_target else "blocked", target_row=is_target)
    if target:
        conn.execute("INSERT INTO kanban_notify_subs(task_id, platform, chat_id, created_at) VALUES ('target-running', 'test', 'qualification', 1)")
        conn.execute("INSERT INTO task_runs(id, task_id, status, started_at) VALUES (10, 'target-running', 'running', 1)")
        conn.execute("INSERT INTO task_events(id, task_id, kind, run_id, created_at) VALUES (10, 'target-running', 'started', 10, 1)")
    else:
        for index, task_id in enumerate(tasks, 1):
            conn.execute("INSERT INTO task_comments(id, task_id, author, body, created_at) VALUES (?, ?, 'fixture', ?, ?)", (index, task_id, f"comment {task_id}", index))
            conn.execute("INSERT INTO task_runs(id, task_id, status, started_at) VALUES (?, ?, ?, ?)", (index, task_id, "running" if index == 1 else "released", index))
            conn.execute("INSERT INTO task_events(id, task_id, kind, run_id, created_at) VALUES (?, ?, 'observed', ?, ?)", (index, task_id, index, index))
        for index, task_id in enumerate(tasks, 1):
            if index < len(tasks):
                conn.execute("INSERT INTO task_links VALUES (?, ?)", (task_id, tasks[index]))
        conn.execute("INSERT INTO task_links VALUES (?, 'qualification-context')", (tasks[0],))
        add_task("qualification-context", status="blocked")
        # The unrelated row makes one-hop closure observable.
        add_task("unrelated", status="blocked")
    if target:
        for index in range(1, 8):
            blob = blobs / f"target-{index}.bin"
            blob.parent.mkdir(parents=True, exist_ok=True)
            blob.write_bytes(f"target blob {index}".encode())
            conn.execute("INSERT INTO task_attachments(id, task_id, filename, stored_path, size, created_at) VALUES (?, 'target-running', ?, ?, ?, 1)", (index, f"target-{index}.bin", str(blob), blob.stat().st_size))
    else:
        blobs.mkdir(parents=True, exist_ok=True)
        for index, task_id in enumerate(tasks, 1):
            blob = blobs / f"source-{index}.bin"
            blob.write_bytes(f"source blob {index}".encode())
            conn.execute("INSERT INTO task_attachments(id, task_id, filename, stored_path, size, created_at) VALUES (?, ?, ?, ?, ?, ?)", (index, task_id, f"source-{index}.bin", str(blob), blob.stat().st_size, index))
    conn.commit()


def _logical_manifest(path: Path) -> dict:
    with sqlite3.connect(path / "kanban.db") as conn:
        conn.row_factory = sqlite3.Row
        manifest = json.loads(json.dumps(__import__("hermes_cli.kanban_generation", fromlist=["row_manifest"]).row_manifest(conn)))
    def normalize(value):
        if isinstance(value, list):
            return [normalize(item) for item in value]
        if isinstance(value, dict):
            return {key: normalize(item) for key, item in value.items()}
        if value in {"candidate-one", "candidate-two"}:
            return "generation"
        return value

    manifest = normalize(manifest)
    if "kanban_generation_meta" in manifest["tables"]:
        for row in manifest["tables"]["kanban_generation_meta"]["rows"]:
            row[2] = "nonce"
    return manifest


@pytest.fixture
def qualification_fixture(tmp_path: Path):
    source = tmp_path / "source.db"
    target = tmp_path / "target.db"
    source_blobs = tmp_path / "source-blobs"
    target_blobs = tmp_path / "target-blobs"
    _create_fixture(source, tasks=list(SEEDS), blobs=source_blobs, target=False)
    _create_fixture(target, tasks=["target-running"], blobs=target_blobs, target=True)
    seed_file = tmp_path / "seeds.json"
    seed_file.write_text(json.dumps({"task_ids": list(SEEDS)}), encoding="utf-8")
    board = tmp_path / "board"
    (board / "generations").mkdir(parents=True)
    return source, target, seed_file, board


def test_full_disposable_native_recovery_qualification(qualification_fixture, tmp_path):
    source, target, seed_file, board = qualification_fixture
    candidate = board / "generations" / "candidate-one"

    dry = reconcile(source, target, seed_file, candidate, apply=False)
    assert dry["dry_run"] is True
    assert set(SEEDS).issubset(dry["closure"])
    assert "qualification-context" in dry["closure"]
    assert not candidate.exists()

    report = reconcile(source, target, seed_file, candidate, apply=True)
    ready = validate_generation(candidate)
    assert ready["manifest_digest"] == report["manifest_digest"]
    assert len(report["seeds"]) == 16
    target_floors = {"task_comments": 0, "task_runs": 10, "task_events": 10, "task_attachments": 7}
    assert all(report["remaps"][table]["1"] > target_floors[table] for table in CHILD_TABLES)

    with sqlite3.connect(candidate / "kanban.db") as conn:
        conn.row_factory = sqlite3.Row
        recovered = conn.execute("SELECT * FROM tasks WHERE id LIKE 'qualification-seed-%'").fetchall()
        assert len(recovered) == 16
        assert all(row["status"] == "triage" and row["claim_lock"] is None for row in recovered)
        assert conn.execute("SELECT COUNT(*) FROM kanban_notify_subs").fetchone()[0] == 1
        assert conn.execute("SELECT COUNT(*) FROM recovery_row_provenance").fetchone()[0] >= 16
        for table in CHILD_TABLES:
            maximum = conn.execute(f"SELECT MAX(id) FROM {table}").fetchone()[0]
            sequence = conn.execute("SELECT seq FROM sqlite_sequence WHERE name=?", (table,)).fetchone()[0]
            assert sequence == maximum
            conn.execute(f"INSERT INTO {table}(task_id, status, started_at) VALUES ('future', 'released', 1)" if table == "task_runs" else f"INSERT INTO {table}(task_id, author, body, created_at) VALUES ('future', 'fixture', 'future', 1)" if table == "task_comments" else f"INSERT INTO {table}(task_id, kind, run_id, created_at) VALUES ('future', 'future', NULL, 1)" if table == "task_events" else f"INSERT INTO {table}(task_id, filename, stored_path, size, created_at) VALUES ('future', 'future', 'future', 0, 1)")
            assert conn.execute("SELECT last_insert_rowid()").fetchone()[0] > maximum
            conn.rollback()

    assert validate_generation(candidate)["manifest_digest"] == ready["manifest_digest"]
    with sqlite3.connect(candidate / "kanban.db") as conn:
        conn.row_factory = sqlite3.Row
        target_attachment = conn.execute("SELECT * FROM task_attachments WHERE task_id='target-running' ORDER BY id LIMIT 1").fetchone()
        resolved = attachment_path(conn, target_attachment["id"], generation=candidate)
        assert resolved.read_bytes() == b"target blob 1"
        assert resolve_attachment_blob(conn, target_attachment, board=candidate) == resolved

    fence = acquire_fence(board, generation=ready["generation_id"], owner="qualification")
    published = publish_prepared(candidate, expected_manifest=ready["manifest_digest"], fence_token=fence["token"])
    assert published["published"] is True
    release_fence(board, token=fence["token"])

    old_pointer = read_pointer(board)
    second = board / "generations" / "candidate-two"
    second_report = reconcile(source, target, seed_file, second, apply=True)
    second_ready = validate_generation(second)
    fence = acquire_fence(board, generation=second_ready["generation_id"], owner="qualification-republish")
    publish_prepared(second, expected_manifest=second_report["manifest_digest"], fence_token=fence["token"])
    rollback_pre_activation(board, expected_active_manifest=second_report["manifest_digest"], fence_token=fence["token"])
    assert read_pointer(board) == old_pointer
    release_fence(board, token=fence["token"])

    fence = acquire_fence(board, generation=second_ready["generation_id"], owner="qualification-activate")
    publish_prepared(second, expected_manifest=second_report["manifest_digest"], fence_token=fence["token"])
    activate_first_write(second, fence_epoch=fence["epoch"])
    release_fence(board, token=fence["token"])
    assert open_generation(board, require_activation=True)[0] == second / "kanban.db"
    with pytest.raises(GenerationError, match="ROLLBACK_CLOSED"):
        fence = acquire_fence(board, generation=second_ready["generation_id"], owner="qualification-refusal")
        try:
            rollback_pre_activation(board, expected_active_manifest=second_report["manifest_digest"], fence_token=fence["token"])
        finally:
            release_fence(board, token=fence["token"])

    normalized_one = _logical_manifest(candidate)
    normalized_two = _logical_manifest(second)
    assert manifest_digest(normalized_one) == manifest_digest(normalized_two)

    stale = board / "generations" / "stale"
    shutil.copytree(candidate, stale)
    gc = gc_generations(board, dry_run=True)
    assert "stale" in gc["marked"]
    qualification = {
        "source_db_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "target_db_sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
        "candidate_manifest_digest": second_ready["manifest_digest"],
        "logical_manifest_digest": manifest_digest(normalized_two),
        "seeds": 16,
        "target_absolute_attachments": 7,
        "live_gates_unexecuted": ["live board", "Office consumer", "production activation"],
    }
    (tmp_path / "qualification.json").write_text(json.dumps(qualification, sort_keys=True, indent=2), encoding="utf-8")
