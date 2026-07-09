"""Tests for kanban-metrics.py."""
import sqlite3
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

SCRIPT = Path(r"C:\Users\bbask\AppData\Local\hermes\scripts\kanban-metrics.py")
sys.path.insert(0, str(SCRIPT.parent))
import importlib.util
spec = importlib.util.spec_from_file_location("km", SCRIPT)
km = importlib.util.module_from_spec(spec)
spec.loader.exec_module(km)


# --- get_kanban_stats ---

def test_get_kanban_stats_missing_db(tmp_path, monkeypatch):
    monkeypatch.setattr(km, "KANBAN_DB", tmp_path / "missing.db")
    assert km.get_kanban_stats() == {}


def test_get_kanban_stats_queries_db(tmp_path, monkeypatch):
    """When db exists, returns counts and lists."""
    db = tmp_path / "kanban.db"
    conn = sqlite3.connect(str(db))
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE tasks (
            id TEXT, title TEXT, status TEXT, assignee TEXT,
            created_at INTEGER, completed_at INTEGER
        )
    """)
    import time
    now = int(time.time())
    week_ago = now - (7 * 86400)
    two_weeks_ago = now - (14 * 86400)
    tasks = [
        ("t1", "Task 1", "done", "alice", now - 86400, now - 80000),  # completed last week
        ("t2", "Task 2", "done", "bob", now - 86400, now - 70000),    # completed last week
        ("t3", "Task 3", "ready", None, now, None),                    # ready, not claimed
        ("t4", "Old unclaimed", "ready", "", two_weeks_ago, None),      # oldest unclaimed
        ("t5", "Blocked task", "blocked", "alice", now, None),          # blocked
        ("t6", "Old done", "done", "alice", two_weeks_ago, two_weeks_ago),  # done > 7 days ago
    ]
    for t in tasks:
        cur.execute("INSERT INTO tasks VALUES (?, ?, ?, ?, ?, ?)", t)
    conn.commit()
    conn.close()

    monkeypatch.setattr(km, "KANBAN_DB", db)
    stats = km.get_kanban_stats()

    assert stats["total"] == 6
    assert stats["by_status"]["done"] == 3
    assert stats["by_status"]["ready"] == 2
    assert stats["by_status"]["blocked"] == 1
    # Throughput: 2 completed last week, 1 done > 7 days ago
    assert stats["completed_last_7d"] == 2
    # Oldest unclaimed (ready + no assignee), ordered by created_at ASC
    assert len(stats["oldest_unclaimed"]) == 2
    assert stats["oldest_unclaimed"][0]["id"] == "t4"  # oldest first
    # Blocked
    assert len(stats["blocked"]) == 1
    assert stats["blocked"][0]["id"] == "t5"


# --- compose_report ---

def test_compose_report_includes_total():
    stats = {"total": 42, "by_status": {"done": 40, "ready": 2}}
    report = km.compose_report(stats)
    assert "42" in report


def test_compose_report_includes_by_status():
    stats = {"total": 10, "by_status": {"done": 5, "ready": 3, "blocked": 2}}
    report = km.compose_report(stats)
    assert "done" in report
    assert "5" in report
    assert "ready" in report


def test_compose_report_includes_throughput():
    stats = {"total": 10, "by_status": {}, "completed_last_7d": 5}
    report = km.compose_report(stats)
    assert "completed last 7 days" in report
    assert "5" in report


def test_compose_report_includes_blocked():
    stats = {"total": 5, "by_status": {"blocked": 1}, "blocked": [
        {"id": "t1", "title": "Stuck task"}
    ]}
    report = km.compose_report(stats)
    assert "blocked" in report.lower()
    assert "t1" in report


def test_compose_report_includes_oldest_unclaimed():
    stats = {"total": 5, "by_status": {}, "oldest_unclaimed": [
        {"id": "t9", "title": "Old task", "created_at": "2026-06-01T00:00:00"}
    ]}
    report = km.compose_report(stats)
    assert "oldest unclaimed" in report.lower()
    assert "t9" in report


def test_compose_report_handles_empty():
    report = km.compose_report({})
    assert "No kanban data available" in report
    assert "kanban weekly metrics" in report


# --- main flow ---

def test_main_no_db_exits_0(tmp_path, monkeypatch):
    monkeypatch.setattr(km, "LOG_FILE", tmp_path / "log.txt")
    monkeypatch.setattr(km, "KANBAN_DB", tmp_path / "missing.db")
    with patch.object(km, "send_telegram", return_value=True) as mock_tg:
        r = km.main()
    assert r == 0
    # telegram still called even with no data (it's the report)
    assert mock_tg.called


def test_main_with_db_exits_0(tmp_path, monkeypatch):
    db = tmp_path / "kanban.db"
    conn = sqlite3.connect(str(db))
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE tasks (
            id TEXT, title TEXT, status TEXT, assignee TEXT,
            created_at INTEGER, completed_at INTEGER
        )
    """)
    import time
    now = int(time.time())
    cur.execute("INSERT INTO tasks VALUES ('t1', 'Test', 'done', 'alice', ?, ?)", (now, now))
    conn.commit()
    conn.close()

    monkeypatch.setattr(km, "LOG_FILE", tmp_path / "log.txt")
    monkeypatch.setattr(km, "KANBAN_DB", db)
    with patch.object(km, "send_telegram", return_value=True) as mock_tg:
        r = km.main()
    assert r == 0
    assert mock_tg.called