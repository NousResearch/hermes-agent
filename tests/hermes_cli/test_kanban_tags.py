"""Tests for kanban task tags: create/list round-trip, tag filtering,
and the legacy-DB column migration."""

from __future__ import annotations

import argparse
from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc


@pytest.fixture
def kanban_home(tmp_path, monkeypatch):
    """Isolated HERMES_HOME with an empty kanban DB."""
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    return home


def test_create_with_tags_roundtrip(kanban_home):
    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="t", tags=["writing", "experiments"])
        assert kb.get_task(conn, tid).tags == ["writing", "experiments"]


def test_create_without_tags_is_untagged(kanban_home):
    with kbc.connect() as conn:
        task = kb.get_task(conn, kb.create_task(conn, title="t"))
        assert task.tags is None


def test_tags_are_stripped_deduped_and_empties_dropped(kanban_home):
    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="t", tags=["  a  ", "", "a", "b"])
        assert kb.get_task(conn, tid).tags == ["a", "b"]


def test_tag_with_comma_is_rejected(kanban_home):
    with kbc.connect() as conn:
        with pytest.raises(ValueError, match="comma"):
            kb.create_task(conn, title="t", tags=["a,b"])


def test_all_empty_tags_stores_null(kanban_home):
    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="t", tags=["", "   "])
        assert kb.get_task(conn, tid).tags is None


def test_list_filter_by_tag(kanban_home):
    with kbc.connect() as conn:
        writing_id = kb.create_task(conn, title="post", tags=["writing"])
        kb.create_task(conn, title="bench", tags=["experiments"])
        untagged_id = kb.create_task(conn, title="plain")

        hits = {t.id for t in kb.list_tasks(conn, tag="writing")}
        assert hits == {writing_id}
        assert untagged_id not in hits
        assert kb.list_tasks(conn, tag="nope") == []


def test_list_filter_tag_is_exact_not_substring(kanban_home):
    with kbc.connect() as conn:
        kb.create_task(conn, title="a", tags=["writing"])
        kb.create_task(conn, title="b", tags=["writing-team"])
        hits = {t.id for t in kb.list_tasks(conn, tag="writing")}
        assert len(hits) == 1


def test_tags_recorded_in_created_event(kanban_home):
    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="t", tags=["writing"])
        created = next(e for e in kb.list_events(conn, tid) if e.kind == "created")
        assert created.payload["tags"] == ["writing"]


def test_legacy_db_without_tags_column_is_migrated(kanban_home):
    """A DB created before ``tags`` existed re-gains the column on open."""
    db_path = kanban_home / "kanban.db"
    with kbc.connect() as conn:
        # Rebuild tasks as it looked before the tags column existed (DROP
        # COLUMN chokes on this schema; table-rebuild is what SQLite docs
        # recommend anyway).
        keep = ", ".join(
            c for c in (
                "id", "title", "body", "assignee", "status", "priority",
                "created_by", "created_at", "started_at", "completed_at",
                "workspace_kind", "workspace_path", "claim_lock", "claim_expires",
                "tenant", "branch_name", "project_id", "result", "idempotency_key",
                "consecutive_failures", "worker_pid", "last_failure_error",
                "max_runtime_seconds", "last_heartbeat_at", "current_run_id",
                "workflow_template_id", "current_step_key", "skills", "max_retries",
                "model_override", "provider_override", "reasoning_effort",
                "goal_mode", "goal_max_turns", "session_id", "block_kind",
                "block_recurrences",
            )
        )
        conn.executescript(
            f"CREATE TABLE tasks_pretags AS SELECT {keep} FROM tasks;"
            "DROP TABLE tasks;"
            "ALTER TABLE tasks_pretags RENAME TO tasks;"
        )
        conn.execute(
            "INSERT INTO tasks (id, title, status, created_at, workspace_kind) "
            "VALUES ('legacy-1', 'legacy', 'todo', 0, 'scratch')"
        )
    # Force the next connect() to re-run init_db's additive migration.
    kb._INITIALIZED_PATHS.discard(str(db_path.resolve()))

    with kbc.connect() as conn:
        cols = {r["name"] for r in conn.execute("PRAGMA table_info(tasks)")}
        assert "tags" in cols
        assert kb.get_task(conn, "legacy-1").tags is None
        tid = kb.create_task(conn, title="fresh", tags=["writing"])
        assert kb.get_task(conn, tid).tags == ["writing"]


def test_cli_create_comma_tag_exits_cleanly(kanban_home, capsys):
    from hermes_cli import kanban as kc
    parser = argparse.ArgumentParser(prog="hermes")
    sub = parser.add_subparsers(dest="sub")
    kc.build_parser(sub)
    args = parser.parse_args(["kanban", "create", "task with comma", "--tag", "a,b"])
    rc = kc.kanban_command(args)
    assert rc == 2
    captured = capsys.readouterr()
    assert "tag cannot contain comma" in captured.err
