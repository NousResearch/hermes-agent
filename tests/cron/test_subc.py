"""Focused contracts for the read-only subc idle analysis."""

from __future__ import annotations

import sqlite3
from pathlib import Path


def _state_db(path: Path, rows: list[tuple[str, str, str]]) -> None:
    conn = sqlite3.connect(path)
    conn.executescript(
        """
        CREATE TABLE sessions (id TEXT PRIMARY KEY, profile_name TEXT);
        CREATE TABLE messages (
            id INTEGER PRIMARY KEY, session_id TEXT, role TEXT, finish_reason TEXT,
            active INTEGER DEFAULT 1
        );
        """
    )
    for index, (session_id, profile, reason) in enumerate(rows, start=1):
        conn.execute("INSERT INTO sessions VALUES (?, ?)", (session_id, profile))
        conn.execute(
            "INSERT INTO messages (id, session_id, role, finish_reason) VALUES (?, ?, 'assistant', ?)",
            (index, session_id, reason),
        )
    conn.commit()
    conn.close()


def test_subc_emits_one_deduplicated_draft_for_repeated_failure(tmp_path):
    from cron.subc import analyze_idle_time

    state = tmp_path / "state.db"
    _state_db(state, [("a", "research", "error"), ("b", "research", "error"), ("c", "research", "error")])

    drafts = analyze_idle_time(state, profile_name="research", min_occurrences=3)

    assert len(drafts) == 1
    assert drafts[0].kind == "nightmare"
    assert drafts[0].occurrences == 3
    assert drafts[0].dedup_key == "subc:failure:error"


def test_subc_fails_open_for_missing_or_unreadable_data(tmp_path):
    from cron.subc import analyze_idle_time

    assert analyze_idle_time(tmp_path / "missing.db", profile_name="research") == []
    broken = tmp_path / "broken.db"
    broken.write_text("not sqlite", encoding="utf-8")
    assert analyze_idle_time(broken, profile_name="research") == []


def test_subc_never_crosses_profile_boundary(tmp_path):
    from cron.subc import analyze_idle_time

    state = tmp_path / "state.db"
    _state_db(
        state,
        [("a", "research", "error"), ("b", "research", "error"), ("c", "research", "error"),
         ("d", "other", "error"), ("e", "other", "error"), ("f", "other", "error")],
    )

    assert analyze_idle_time(state, profile_name="missing", min_occurrences=3) == []
    assert analyze_idle_time(state, profile_name="research", min_occurrences=4) == []


def test_subc_suppresses_a_draft_already_on_the_board(tmp_path):
    from cron.subc import analyze_profile_idle_time

    state = tmp_path / "state.db"
    _state_db(state, [("a", "research", "error"), ("b", "research", "error"), ("c", "research", "error")])
    board = sqlite3.connect(tmp_path / "kanban.db")
    board.execute("CREATE TABLE tasks (title TEXT, body TEXT)")
    board.execute("INSERT INTO tasks VALUES ('Existing nightmare', 'subc:failure:error')")
    board.commit()
    board.close()

    assert analyze_profile_idle_time(
        profile_home=tmp_path, profile_name="research", min_occurrences=3, kanban_db=tmp_path / "kanban.db",
    ) == []
