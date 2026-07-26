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
    assert not db_path.exists(), "read-only probe must not create the DB"


def test_counts_rows_via_board_resolution(kanban_home):
    conn = kb.connect(board="default")
    try:
        tid = kb.create_task(conn, title="t", assignee="w")
        kb.add_notify_sub(conn, task_id=tid, platform="telegram", chat_id="c1")
        kb.add_notify_sub(conn, task_id=tid, platform="telegram", chat_id="c2")
    finally:
        conn.close()
    assert kb.count_notify_subs(board="default") == 2


def test_probe_is_read_only_and_sees_uncheckpointed_wal_rows(kanban_home):
    """A sub committed by a still-open writer (rows only in the WAL, not yet
    checkpointed into the main DB file) must be counted — under-counting
    would make the notifier skip a board that has a live subscription. And
    the probe itself must be read-only: the writer's connection stays the
    only writer."""
    conn = kb.connect(board="default")
    try:
        tid = kb.create_task(conn, title="t", assignee="w")
        kb.add_notify_sub(conn, task_id=tid, platform="telegram", chat_id="c1")
        # Writer still open: the row lives in the -wal, not the main file.
        assert kb.count_notify_subs(board="default") == 1
    finally:
        conn.close()


def test_legacy_db_without_subs_table_counts_zero_and_stays_unmigrated(tmp_path):
    legacy = tmp_path / "legacy.db"
    conn = sqlite3.connect(legacy)
    try:
        conn.execute("CREATE TABLE something_else (id INTEGER)")
        conn.commit()
    finally:
        conn.close()
    assert kb.count_notify_subs(db_path=legacy) == 0
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


def test_explicit_db_path_overrides_board(kanban_home, tmp_path):
    pinned = tmp_path / "pinned.db"
    conn = kb.connect(db_path=pinned)
    try:
        tid = kb.create_task(conn, title="t", assignee="w")
        kb.add_notify_sub(conn, task_id=tid, platform="telegram", chat_id="c1")
    finally:
        conn.close()
    assert kb.count_notify_subs(pinned) == 1
    assert kb.count_notify_subs(board="default") == 0
