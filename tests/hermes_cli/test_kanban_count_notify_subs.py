"""Tests for ``kanban_db.count_notify_subs`` — the read-only subscription probe.

The gateway notifier uses it to skip boards with zero subscriptions BEFORE
any writable ``connect()``: the probe must never create the DB file, never
run schema init/migration, and never write — that first-open cost on every
tick is exactly what the zero-sub early exit avoids. It must also never
UNDER-count: rows sitting in a not-yet-checkpointed WAL still count, or the
notifier would skip a board that has a live subscription.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb


@pytest.fixture
def kanban_home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("HERMES_KANBAN_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    return home


def test_missing_db_counts_zero_and_creates_nothing(kanban_home):
    db_path = kb.kanban_db_path(board="default")
    assert not db_path.exists()
    assert kb.count_notify_subs(board="default") == 0
    assert kb.count_notify_subs(
        board="default", platform="tui", chat_id="session-1"
    ) == 0
    assert not db_path.exists(), "read-only probe must not create the DB"


def test_optional_filters_narrow_count_without_changing_unfiltered_count(kanban_home):
    conn = kb.connect(board="default")
    try:
        tid = kb.create_task(conn, title="t", assignee="w")
        kb.add_notify_sub(conn, task_id=tid, platform="tui", chat_id="session-1")
        kb.add_notify_sub(
            conn,
            task_id=tid,
            platform="TUI",
            chat_id="session-2",
            thread_id="thread-2",
        )
        kb.add_notify_sub(
            conn, task_id=tid, platform="telegram", chat_id="session-1"
        )
    finally:
        conn.close()

    assert kb.count_notify_subs(board="default") == 3
    assert kb.count_notify_subs(board="default", platform="tui") == 2
    assert kb.count_notify_subs(board="default", chat_id="session-1") == 2
    assert kb.count_notify_subs(board="default", thread_id="thread-2") == 1
    assert kb.count_notify_subs(
        board="default", platform="tui", chat_id="session-1"
    ) == 1
    assert kb.count_notify_subs(
        board="default", platform="tui", chat_id="session-1", thread_id=""
    ) == 1
    assert kb.count_notify_subs(
        board="default",
        platform="tui",
        chat_id="session-2",
        thread_id="thread-2",
    ) == 1
    assert kb.count_notify_subs(
        board="default", platform="tui", chat_id="other-session"
    ) == 0


def test_legacy_db_without_subs_table_counts_zero_and_stays_unmigrated(tmp_path):
    legacy = tmp_path / "legacy.db"
    conn = sqlite3.connect(legacy)
    try:
        conn.execute("CREATE TABLE something_else (id INTEGER)")
        conn.commit()
    finally:
        conn.close()
    assert kb.count_notify_subs(db_path=legacy) == 0
    assert kb.count_notify_subs(
        db_path=legacy, platform="tui", chat_id="session-1"
    ) == 0
    # The probe must not have run schema init on the foreign/legacy DB.
    conn = sqlite3.connect(legacy)
    try:
        tables = {
            r[0] for r in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            )
        }
    finally:
        conn.close()
    assert "kanban_notify_subs" not in tables, (
        "read-only probe must never create schema"
    )


def test_count_notify_subs_filters_profile_owners(tmp_path):
    db_path = tmp_path / "owners.db"
    kb.init_db(db_path)
    conn = kb.connect(db_path)
    try:
        for profile in ("default", "writer", None):
            task_id = kb.create_task(conn, title=f"owned by {profile}")
            kb.add_notify_sub(
                conn,
                task_id=task_id,
                platform="telegram",
                chat_id=f"chat-{profile}",
                notifier_profile=profile,
            )
    finally:
        conn.close()

    assert kb.count_notify_subs(
        db_path=db_path, notifier_profiles={"writer"},
    ) == 1
    assert kb.count_notify_subs(
        db_path=db_path,
        notifier_profiles={"default"},
        include_unowned=True,
    ) == 2
