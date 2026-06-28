"""Tests for unified social memory sync orchestrator."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest

from sync_memory import _latest_started_at, _watermark_for_sources, load_index, run_sync, save_index


@pytest.fixture
def temp_state_db(tmp_path: Path) -> Path:
    db = tmp_path / "state.db"
    con = sqlite3.connect(db)
    con.executescript(
        """
        CREATE TABLE sessions (
            id TEXT PRIMARY KEY,
            source TEXT NOT NULL,
            title TEXT,
            started_at REAL NOT NULL
        );
        CREATE TABLE messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT NOT NULL,
            role TEXT NOT NULL,
            content TEXT,
            timestamp REAL,
            active INTEGER DEFAULT 1
        );
        """
    )
    con.execute(
        "INSERT INTO sessions VALUES (?, ?, ?, ?)",
        ("s-old", "line", "old chat", 100.0),
    )
    con.execute(
        "INSERT INTO sessions VALUES (?, ?, ?, ?)",
        ("s-new", "discord", "new chat", 200.0),
    )
    con.execute(
        "INSERT INTO messages (session_id, role, content, timestamp, active) VALUES (?, ?, ?, ?, 1)",
        ("s-new", "user", "Remember that I prefer concise Japanese updates.", 201.0),
    )
    con.commit()
    con.close()
    return db


def test_load_index_legacy_text(tmp_path: Path) -> None:
    path = tmp_path / "last_index.txt"
    path.write_text("150.5\n", encoding="utf-8")
    data = load_index(path)
    assert data["sources"]["*"]["last_started_at"] == 150.5


def test_watermark_min_across_sources() -> None:
    index = {
        "sources": {
            "line": {"last_started_at": 120.0},
            "discord": {"last_started_at": 180.0},
        }
    }
    assert _watermark_for_sources(index, ("line", "discord", "telegram")) == 120.0


def test_run_sync_incremental_writes_index(tmp_path: Path, temp_state_db: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    index_path = tmp_path / "memory_sync_last_index.json"
    memory_db = tmp_path / "ebbinghaus_memory.db"
    monkeypatch.setattr(
        "sync_memory._export_obsidian",
        lambda **_: {"success": True, "skipped": True},
    )

    first = run_sync(
        sources=("line", "discord"),
        max_sessions=10,
        max_x_events=0,
        sleep=False,
        incremental=False,
        index_path=index_path,
        export_obsidian=False,
        obsidian_dry_run=False,
        max_ebbinghaus_export=20,
        state_db=temp_state_db,
        memory_db=memory_db,
    )
    assert first["social"]["memories_written"] >= 1

    save_index(
        index_path,
        {
            "version": 1,
            "sources": {"line": {"last_started_at": 150.0}, "discord": {"last_started_at": 150.0}},
            "last_run_at": 150.0,
        },
    )

    second = run_sync(
        sources=("line", "discord"),
        max_sessions=10,
        max_x_events=0,
        sleep=False,
        incremental=True,
        index_path=index_path,
        export_obsidian=False,
        obsidian_dry_run=False,
        max_ebbinghaus_export=20,
        state_db=temp_state_db,
        memory_db=memory_db,
    )
    assert second["social"]["sessions_seen"] == 1
    assert second["social"]["min_started_at"] == 150.0

    saved = json.loads(index_path.read_text(encoding="utf-8"))
    assert saved["sources"]["discord"]["last_started_at"] == 200.0


def test_latest_started_at(temp_state_db: Path) -> None:
    latest = _latest_started_at(temp_state_db, ("line", "discord"))
    assert latest == {"line": 100.0, "discord": 200.0}
