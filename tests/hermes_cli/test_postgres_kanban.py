"""Kanban keeps its SQLite behavior when the shared board store is PostgreSQL."""

from __future__ import annotations

import concurrent.futures
import json
import os
import sqlite3
import subprocess
import sys
from contextlib import closing
from pathlib import Path

import pytest


@pytest.fixture
def postgres_kanban_home(postgres_home, monkeypatch):
    import psycopg
    from psycopg import sql
    from hermes_cli import kanban_db as kb
    from hermes_cli import kanban_db_postgres as pg

    for name in (
        "HERMES_KANBAN_DB",
        "HERMES_KANBAN_HOME",
        "HERMES_KANBAN_BOARD",
        "HERMES_KANBAN_WORKSPACES_ROOT",
        "HERMES_KANBAN_ATTACHMENTS_ROOT",
    ):
        monkeypatch.delenv(name, raising=False)
    kb._INITIALIZED_PATHS.clear()
    database_url = pg.database_settings().database_url
    with psycopg.connect(database_url, autocommit=True) as connection:
        before = {
            row[0]
            for row in connection.execute(
                "SELECT nspname FROM pg_namespace WHERE nspname LIKE '%_kanban_%'"
            )
        }
    try:
        yield postgres_home
    finally:
        pg.close_pools()
        with psycopg.connect(database_url, autocommit=True) as connection:
            after = {
                row[0]
                for row in connection.execute(
                    "SELECT nspname FROM pg_namespace WHERE nspname LIKE '%_kanban_%'"
                )
            }
            for schema in after - before:
                connection.execute(
                    sql.SQL("DROP SCHEMA IF EXISTS {} CASCADE").format(sql.Identifier(schema))
                )


def test_task_lifecycle_notifications_and_board_isolation(postgres_kanban_home):
    from hermes_cli import kanban_db as kb
    from hermes_cli import kanban_db_connect as kbc
    from hermes_cli import kanban_db_notify as notify

    kb.create_board("alpha")
    kb.create_board("beta")
    with kbc.connect_closing(board="alpha") as connection:
        task_id = kb.create_task(connection, title="PostgreSQL task", assignee="coder")
        comment_id = kb.add_comment(connection, task_id, "tester", "persist this")
        attachment_id = kb.store_attachment_bytes(
            connection, task_id, "proof.txt", b"postgres proof", board="alpha"
        )
        notify.add_notify_sub(
            connection, task_id=task_id, platform="telegram", chat_id="chat-1"
        )
        claimed = kb.claim_task(connection, task_id, claimer="host:worker")
        assert claimed is not None
        assert kb.heartbeat_claim(connection, task_id, claimer="host:worker") is True
        assert kb.complete_task(connection, task_id, result="done", summary="verified") is True

        assert kb.get_task(connection, task_id).status == "done"
        assert kb.list_comments(connection, task_id)[0].id == comment_id
        attachment = kb.get_attachment(connection, attachment_id)
        assert Path(attachment.stored_path).read_bytes() == b"postgres proof"
        old_cursor, new_cursor, events = notify.claim_unseen_events_for_sub(
            connection,
            task_id=task_id,
            platform="telegram",
            chat_id="chat-1",
            kinds={"completed"},
        )
        assert new_cursor > old_cursor
        assert [event.kind for event in events] == ["completed"]

    with kbc.connect_closing(board="beta") as connection:
        assert kb.list_tasks(connection) == []
    assert not (postgres_kanban_home / "kanban.db").exists()
    assert not (kb.board_dir("alpha") / "kanban.db").exists()


def test_concurrent_claim_and_notify_cursor_each_have_one_winner(postgres_kanban_home):
    from hermes_cli import kanban_db as kb
    from hermes_cli import kanban_db_connect as kbc
    from hermes_cli import kanban_db_notify as notify

    kb.create_board("race")
    with kbc.connect_closing(board="race") as connection:
        task_id = kb.create_task(connection, title="claim once", assignee="coder")
        notify.add_notify_sub(
            connection, task_id=task_id, platform="telegram", chat_id="chat-1"
        )
        kb.add_comment(connection, task_id, "tester", "new event")

    def claim_task(index: int) -> bool:
        with kbc.connect_closing(board="race") as connection:
            return kb.claim_task(connection, task_id, claimer=f"worker:{index}") is not None

    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        assert sorted(executor.map(claim_task, (1, 2))) == [False, True]

    def claim_events() -> int:
        with kbc.connect_closing(board="race") as connection:
            return len(notify.claim_unseen_events_for_sub(
                connection,
                task_id=task_id,
                platform="telegram",
                chat_id="chat-1",
                kinds={"commented", "claimed"},
            )[2])

    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        claimed_counts = sorted(executor.map(lambda _: claim_events(), (1, 2)))
    assert claimed_counts[0] == 0
    assert claimed_counts[1] > 0


def test_claim_is_atomic_across_processes_and_profiles(postgres_kanban_home):
    from hermes_cli import kanban_db as kb
    from hermes_cli import kanban_db_connect as kbc

    kb.create_board("shared")
    with kbc.connect_closing(board="shared") as connection:
        task_id = kb.create_task(connection, title="cross process", assignee="coder")

    profile_home = postgres_kanban_home / "profiles" / "coder"
    profile_home.mkdir(parents=True)
    (profile_home / "config.yaml").write_text("{}\n", encoding="utf-8")
    environment = os.environ.copy()
    environment["HERMES_HOME"] = str(profile_home)
    environment["HERMES_KANBAN_BOARD"] = "shared"
    script = (
        "import json; "
        "from hermes_cli.kanban_db_connect import connect_closing; "
        "from hermes_cli.kanban_db import claim_task; "
        f"tid={task_id!r}; "
        "cm=connect_closing(); c=cm.__enter__(); "
        "print(json.dumps(claim_task(c, tid) is not None)); cm.__exit__(None,None,None)"
    )
    processes = [
        subprocess.Popen(
            [sys.executable, "-c", script],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=environment,
        )
        for _ in range(2)
    ]
    outcomes = []
    for process in processes:
        stdout, stderr = process.communicate(timeout=30)
        assert process.returncode == 0, stderr
        outcomes.append(json.loads(stdout))
    assert sorted(outcomes) == [False, True]


def test_dispatch_lock_and_cross_board_running_count(postgres_kanban_home):
    from hermes_cli import kanban_db as kb
    from hermes_cli import kanban_db_connect as kbc
    from hermes_cli import kanban_db_dispatch as dispatch

    kb.create_board("one")
    kb.create_board("two")
    with kbc.connect_closing(board="one") as first, kbc.connect_closing(board="one") as second:
        with kbc._dispatch_tick_lock(kb.kanban_db_path("one"), conn=first) as first_held:
            with kbc._dispatch_tick_lock(kb.kanban_db_path("one"), conn=second) as second_held:
                assert first_held is True
                assert second_held is False

    with kbc.connect_closing(board="two") as connection:
        task_id = kb.create_task(connection, title="running elsewhere", assignee="coder")
        assert kb.claim_task(connection, task_id) is not None
    assert dispatch.count_running_tasks_other_boards("one") == 1


def test_dependencies_review_handoff_blocking_and_health_check(postgres_kanban_home):
    from hermes_cli import kanban_db as kb
    from hermes_cli import kanban_db_connect as kbc

    missing = kbc.repair_db(board="workflow")
    assert (missing.backend, missing.status) == ("postgres", "missing")
    kb.create_board("workflow")
    healthy = kbc.repair_db(board="workflow")
    assert (healthy.backend, healthy.status) == ("postgres", "ok")

    with kbc.connect_closing(board="workflow") as connection:
        parent_id = kb.create_task(connection, title="parent", assignee="coder")
        child_id = kb.create_task(
            connection, title="child", assignee="coder", parents=[parent_id]
        )
        assert kb.get_task(connection, child_id).status == "todo"
        assert kb.complete_task(connection, parent_id, result="parent complete") is True
        assert kb.get_task(connection, child_id).status == "ready"

        implementation = kb.claim_task(connection, child_id, claimer="impl:1")
        assert implementation is not None
        assert kb.request_review(
            connection,
            child_id,
            summary="ready for review",
            reviewer="reviewer",
            expected_run_id=implementation.current_run_id,
        ) is True
        review = kb.claim_review_task(connection, child_id, claimer="review:1")
        assert review is not None
        changed, implementer = kb.request_changes(
            connection,
            child_id,
            reason="add evidence",
            expected_run_id=review.current_run_id,
        )
        assert (changed, implementer) == (True, "coder")

        retry = kb.claim_task(connection, child_id, claimer="impl:2")
        assert retry is not None
        assert kb.block_task(
            connection,
            child_id,
            reason="need input",
            kind="needs_input",
            expected_run_id=retry.current_run_id,
        ) is True
        assert kb.get_task(connection, child_id).status == "blocked"
        assert kb.unblock_task(connection, child_id) is True
        assert kb.get_task(connection, child_id).status == "ready"


def test_archived_board_and_reused_slug_do_not_share_schema(postgres_kanban_home):
    from hermes_cli import kanban_db as kb
    from hermes_cli import kanban_db_connect as kbc

    kb.create_board("reused")
    old_database_id = kb.board_database_id("reused")
    with kbc.connect_closing(board="reused") as connection:
        kb.create_task(connection, title="archived data", assignee="coder")
    kb.remove_board("reused", archive=True)

    kb.create_board("reused")
    assert kb.board_database_id("reused") != old_database_id
    with kbc.connect_closing(board="reused") as connection:
        assert kb.list_tasks(connection) == []


def test_sqlite_migration_is_atomic_and_preserves_source(postgres_kanban_home, tmp_path):
    from hermes_cli import kanban_db as kb
    from hermes_cli import kanban_db_connect as kbc
    from hermes_cli import kanban_db_notify as notify
    from scripts.migrate_kanban_sqlite_to_postgres import migrate_kanban_sqlite_to_postgres

    source = tmp_path / "kanban.db"
    with closing(sqlite3.connect(source, isolation_level=None)) as connection:
        connection.row_factory = sqlite3.Row
        connection.executescript(kb.SCHEMA_SQL)
        kbc._migrate_add_optional_columns(connection)
        task_id = kb.create_task(connection, title="migrated", assignee="coder")
        kb.add_comment(connection, task_id, "tester", "from sqlite")
        notify.add_notify_sub(
            connection, task_id=task_id, platform="telegram", chat_id="chat-1"
        )
    source_bytes = source.read_bytes()

    counts = migrate_kanban_sqlite_to_postgres(source, board="default", batch_size=1)
    assert counts["tasks"] == 1
    assert counts["task_comments"] == 1
    assert counts["kanban_notify_subs"] == 1
    assert source.read_bytes() == source_bytes
    with kbc.connect_closing(board="default") as connection:
        assert kb.get_task(connection, task_id).title == "migrated"
        assert kb.list_comments(connection, task_id)[0].body == "from sqlite"

    with pytest.raises(ValueError, match="is not empty"):
        migrate_kanban_sqlite_to_postgres(source, board="default")
    assert source.read_bytes() == source_bytes


def test_postgres_board_export_import_round_trip(postgres_kanban_home, tmp_path):
    from hermes_cli import kanban_db as kb
    from hermes_cli import kanban_db_connect as kbc
    from hermes_cli import kanban_transfer

    kb.create_board("portable", name="Portable")
    with kbc.connect_closing(board="portable") as connection:
        task_id = kb.create_task(connection, title="round trip", assignee="coder")
        kb.add_comment(connection, task_id, "tester", "preserved")
        kb.store_attachment_bytes(
            connection, task_id, "proof.txt", b"portable bytes", board="portable"
        )

    archive = kanban_transfer.export_board(
        "portable", str(tmp_path / "portable-export")
    )["archive"]
    result = kanban_transfer.import_board(archive)
    assert result["board"] == "portable-2"
    with kbc.connect_closing(board=result["board"]) as connection:
        task = kb.list_tasks(connection)[0]
        assert task.title == "round trip"
        assert kb.list_comments(connection, task.id)[0].body == "preserved"
        attachment = kb.list_attachments(connection, task.id)[0]
        assert Path(attachment.stored_path).read_bytes() == b"portable bytes"
