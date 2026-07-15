"""Regression tests for legacy per-topic active-index receipt cleanup."""

from __future__ import annotations

import sqlite3
import time
from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb


@pytest.fixture
def kanban_home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    return home


def test_init_db_preserves_threaded_index_receipts_across_boards(kanban_home):
    """Startup keeps durable per-topic and shared active-index projections.

    The durable empty-thread receipt is the per-chat overview identity. Task
    subscriptions, task-card status surfaces, and terminal receipts share no
    table/key with old threaded index rows and must survive unchanged.
    """
    kb.create_board("alpha")
    kb.create_board("beta")
    expected_subscriptions: dict[str, list[dict[str, str]]] = {}
    expected_status_surfaces: dict[str, list[dict[str, str]]] = {}
    expected_terminal_receipts: dict[str, list[dict[str, str]]] = {}

    for board, chat_id, topics in (
        ("alpha", "chat-alpha", ("topic-a1", "topic-a2")),
        ("beta", "chat-beta", ("topic-b1", "topic-b2", "topic-b3")),
    ):
        with kb.connect(board=board) as conn:
            task_id = kb.create_task(conn, title=f"{board} task", assignee="heavy", board=board)
            kb.add_notify_sub(
                conn, task_id=task_id, platform="telegram", chat_id=chat_id, thread_id=topics[0],
            )
            now = int(time.time())
            conn.execute(
                "INSERT INTO kanban_status_surfaces "
                "(task_id, platform, chat_id, thread_id, message_id, created_at, updated_at) "
                "VALUES (?, 'telegram', ?, ?, ?, ?, ?)",
                (task_id, chat_id, topics[0], f"card-{board}", now, now),
            )
            conn.execute(
                "INSERT INTO kanban_terminal_notifications "
                "(task_id, platform, chat_id, thread_id, event_id, title, message_id, created_at, updated_at) "
                "VALUES (?, 'telegram', ?, ?, 1, 'terminal', ?, ?, ?)",
                (task_id, chat_id, topics[0], f"terminal-{board}", now, now),
            )
            for topic in topics:
                conn.execute(
                    "INSERT INTO kanban_active_task_indexes "
                    "(platform, chat_id, thread_id, notifier_profile, message_id, created_at, updated_at) "
                    "VALUES ('telegram', ?, ?, '', ?, ?, ?)",
                    (chat_id, topic, f"legacy-{topic}", now, now),
                )
            conn.execute(
                "INSERT INTO kanban_active_task_indexes "
                "(platform, chat_id, thread_id, notifier_profile, message_id, created_at, updated_at) "
                "VALUES ('telegram', ?, '', '', ?, ?, ?)",
                (chat_id, f"overview-{board}", now, now),
            )
            expected_subscriptions[board] = [dict(row) for row in conn.execute(
                "SELECT task_id, platform, chat_id, thread_id FROM kanban_notify_subs"
            )]
            expected_status_surfaces[board] = [dict(row) for row in conn.execute(
                "SELECT task_id, platform, chat_id, thread_id, message_id FROM kanban_status_surfaces"
            )]
            expected_terminal_receipts[board] = [dict(row) for row in conn.execute(
                "SELECT task_id, platform, chat_id, thread_id, message_id FROM kanban_terminal_notifications"
            )]

    for board in ("alpha", "beta"):
        kb.init_db(board=board)
        kb.init_db(board=board)  # restart-safe and idempotent
        with kb.connect(board=board) as conn:
            assert [dict(row) for row in conn.execute(
                "SELECT thread_id, message_id FROM kanban_active_task_indexes ORDER BY thread_id"
            )] == [
                {"thread_id": "", "message_id": f"overview-{board}"},
                *[{"thread_id": topic, "message_id": f"legacy-{topic}"} for topic in {
                    "alpha": ("topic-a1", "topic-a2"),
                    "beta": ("topic-b1", "topic-b2", "topic-b3"),
                }[board]],
            ]
            assert [dict(row) for row in conn.execute(
                "SELECT task_id, platform, chat_id, thread_id FROM kanban_notify_subs"
            )] == expected_subscriptions[board]
            assert [dict(row) for row in conn.execute(
                "SELECT task_id, platform, chat_id, thread_id, message_id FROM kanban_status_surfaces"
            )] == expected_status_surfaces[board]
            assert [dict(row) for row in conn.execute(
                "SELECT task_id, platform, chat_id, thread_id, message_id FROM kanban_terminal_notifications"
            )] == expected_terminal_receipts[board]


def test_init_db_does_not_delete_threaded_index_receipts(kanban_home):
    """Topic receipts survive startup without invoking destructive cleanup."""
    kb.create_board("atomic-cleanup")
    with kb.connect(board="atomic-cleanup") as conn:
        now = int(time.time())
        for topic in ("topic-1", "topic-2"):
            conn.execute(
                "INSERT INTO kanban_active_task_indexes "
                "(platform, chat_id, thread_id, notifier_profile, created_at, updated_at) "
                "VALUES ('telegram', 'atomic-chat', ?, '', ?, ?)",
                (topic, now, now),
            )
        conn.execute(
            "CREATE TRIGGER abort_legacy_index_cleanup BEFORE DELETE ON kanban_active_task_indexes "
            "WHEN OLD.thread_id != '' BEGIN SELECT RAISE(ABORT, 'migration stop'); END"
        )

    kb.init_db(board="atomic-cleanup")

    raw = sqlite3.connect(kb.kanban_db_path("atomic-cleanup"))
    try:
        assert raw.execute(
            "SELECT thread_id FROM kanban_active_task_indexes ORDER BY thread_id"
        ).fetchall() == [("topic-1",), ("topic-2",)]
    finally:
        raw.close()


def test_init_db_threaded_index_receipts_are_restart_safe(kanban_home):
    """A large set of topic receipts remains intact across restarts."""
    kb.create_board("bounded-cleanup")
    total = int(getattr(kb, "_LEGACY_THREADED_ACTIVE_INDEX_CLEANUP_LIMIT")) + 1
    with kb.connect(board="bounded-cleanup") as conn:
        now = int(time.time())
        conn.executemany(
            "INSERT INTO kanban_active_task_indexes "
            "(platform, chat_id, thread_id, notifier_profile, created_at, updated_at) "
            "VALUES ('telegram', 'bounded-chat', ?, '', ?, ?)",
            [(f"topic-{index}", now, now) for index in range(total)],
        )

    kb.init_db(board="bounded-cleanup")
    with kb.connect(board="bounded-cleanup") as conn:
        assert conn.execute("SELECT COUNT(*) FROM kanban_active_task_indexes").fetchone()[0] == total

    kb.init_db(board="bounded-cleanup")
    with kb.connect(board="bounded-cleanup") as conn:
        assert conn.execute("SELECT COUNT(*) FROM kanban_active_task_indexes").fetchone()[0] == total
