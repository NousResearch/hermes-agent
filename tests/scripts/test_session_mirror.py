"""Tests for session-mirror.py."""
import json
import os
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

SCRIPT = Path(r"C:\Users\bbask\AppData\Local\hermes\scripts\session-mirror.py")
sys.path.insert(0, str(SCRIPT.parent))
import importlib.util
spec = importlib.util.spec_from_file_location("sm", SCRIPT)
sm = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sm)


# --- check_gateway ---

def test_check_gateway_200():
    mock_resp = MagicMock()
    mock_resp.status = 200
    mock_resp.__enter__ = MagicMock(return_value=mock_resp)
    mock_resp.__exit__ = MagicMock(return_value=False)
    with patch.object(sm.urllib.request, "urlopen", return_value=mock_resp):
        result = sm.check_gateway()
    assert "200" in result
    assert "🟢" in result


def test_check_gateway_down():
    with patch.object(sm.urllib.request, "urlopen", side_effect=ConnectionError("refused")):
        result = sm.check_gateway()
    assert "🔴" in result
    assert "down" in result


# --- check_memory ---

def test_check_memory_normal(tmp_path, monkeypatch):
    monkeypatch.setattr(sm, "HERMES_HOME", tmp_path)
    (tmp_path / "memories").mkdir()
    (tmp_path / "memories" / "MEMORY.md").write_text("a" * 3000)
    result = sm.check_memory()
    assert "3000" in result
    assert "50%" in result
    assert "🟢" in result


def test_check_memory_warning(tmp_path, monkeypatch):
    monkeypatch.setattr(sm, "HERMES_HOME", tmp_path)
    (tmp_path / "memories").mkdir()
    (tmp_path / "memories" / "MEMORY.md").write_text("a" * 5000)
    result = sm.check_memory()
    assert "83%" in result or "84%" in result
    assert "🟡" in result


def test_check_memory_critical(tmp_path, monkeypatch):
    monkeypatch.setattr(sm, "HERMES_HOME", tmp_path)
    (tmp_path / "memories").mkdir()
    (tmp_path / "memories" / "MEMORY.md").write_text("a" * 5900)
    result = sm.check_memory()
    assert "🔴" in result


def test_check_memory_missing_file(tmp_path, monkeypatch):
    monkeypatch.setattr(sm, "HERMES_HOME", tmp_path)
    result = sm.check_memory()
    assert "no MEMORY.md" in result


# --- list_fixlogs ---

def test_list_fixlogs_returns_recent(tmp_path, monkeypatch):
    monkeypatch.setattr(sm, "OBSIDIAN_FIXLOG_DIR", tmp_path)
    (tmp_path / "old.md").write_text("old")
    (tmp_path / "newer.md").write_text("newer")
    # newer.md has higher mtime
    import time
    (tmp_path / "newer.md").touch()
    time.sleep(0.01)
    (tmp_path / "newer.md").touch()
    files = sm.list_fixlogs(3)
    assert "newer" in files


def test_list_fixlogs_no_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(sm, "OBSIDIAN_FIXLOG_DIR", tmp_path / "missing")
    assert sm.list_fixlogs(3) == []


# --- count_pending ---

def test_count_pending_counts_decision_items(tmp_path, monkeypatch):
    monkeypatch.setattr(sm, "OBSIDIAN_PENDING", tmp_path / "PENDING.md")
    (tmp_path / "PENDING.md").write_text("""# PENDING
### 1. First item
### 2. Second item
### 3. Third item
""")
    assert sm.count_pending() == 3


def test_count_pending_no_file(tmp_path, monkeypatch):
    monkeypatch.setattr(sm, "OBSIDIAN_PENDING", tmp_path / "missing.md")
    assert sm.count_pending() == 0


# --- list_recent_prs ---

def test_list_recent_prs_parses_json():
    prs = [
        {"number": 1, "title": "Fix bug", "repository": {"nameWithOwner": "foo/bar"}},
    ]
    with patch.object(sm.subprocess, "run") as mock_run:
        mock_run.return_value = MagicMock(returncode=0, stdout=json.dumps(prs), stderr="")
        result = sm.list_recent_prs(5)
    assert len(result) == 1
    assert result[0]["number"] == 1


def test_list_recent_prs_handles_failure():
    with patch.object(sm.subprocess, "run") as mock_run:
        mock_run.return_value = MagicMock(returncode=1, stdout="", stderr="auth failed")
        result = sm.list_recent_prs(5)
    assert result == []


# --- compose_summary ---

def test_compose_summary_includes_session_info():
    meta = {"title": "Test session", "source": "cli", "message_count": 5}
    summary = sm.compose_summary(meta)
    assert "Test session" in summary
    assert "cli" in summary


def test_compose_summary_handles_empty_meta():
    summary = sm.compose_summary({})
    assert "install health" in summary
    assert "gateway" in summary


def test_compose_summary_truncates_long_title():
    meta = {"title": "a" * 100, "source": "cli", "message_count": 5}
    summary = sm.compose_summary(meta)
    # 60 char limit
    assert "a" * 60 in summary
    assert "a" * 61 not in summary


# --- get_session_metadata ---

def test_get_session_metadata_returns_latest(tmp_path, monkeypatch):
    monkeypatch.setattr(sm, "STATE_DB", tmp_path / "state.db")
    import sqlite3
    conn = sqlite3.connect(str(tmp_path / "state.db"))
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE sessions (
            id TEXT, title TEXT, source TEXT, started_at REAL,
            message_count INTEGER, model TEXT
        )
    """)
    cur.execute(
        "INSERT INTO sessions VALUES (?, ?, ?, ?, ?, ?)",
        ("s1", "Latest", "cli", 100.0, 5, "MiniMax-M3")
    )
    conn.commit()
    conn.close()
    meta = sm.get_session_metadata()
    assert meta["id"] == "s1"
    assert meta["title"] == "Latest"


# --- send_telegram ---

def test_send_telegram_no_token():
    """Returns False when token/chat_id are missing."""
    with patch.dict(os.environ, {}, clear=True):
        # also clear .env file lookup
        with patch.object(sm, "_read_env_value", return_value=None):
            assert sm.send_telegram("hello") is False