"""Tests for honcho-self-evolve.py."""
import json
import sqlite3
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

SCRIPT = Path(r"C:\Users\bbask\AppData\Local\hermes\scripts\honcho-self-evolve.py")
sys.path.insert(0, str(SCRIPT.parent))
import importlib.util
spec = importlib.util.spec_from_file_location("hse", SCRIPT)
hse = importlib.util.module_from_spec(spec)
spec.loader.exec_module(hse)


# --- get_last_session ---

def test_get_last_session_returns_latest(tmp_path, monkeypatch):
    monkeypatch.setattr(hse, "STATE_DB", tmp_path / "state.db")
    conn = sqlite3.connect(str(tmp_path / "state.db"))
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE sessions (
            id TEXT, title TEXT, source TEXT, started_at REAL,
            ended_at REAL, message_count INTEGER, tool_call_count INTEGER,
            model TEXT
        )
    """)
    cur.execute(
        "INSERT INTO sessions VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        ("s1", "Test session", "cli", 100.0, 200.0, 50, 30, "MiniMax-M3")
    )
    conn.commit()
    conn.close()
    result = hse.get_last_session()
    assert result["id"] == "s1"
    assert result["title"] == "Test session"
    assert result["message_count"] == 50


def test_get_last_session_no_db(tmp_path, monkeypatch):
    monkeypatch.setattr(hse, "STATE_DB", tmp_path / "missing.db")
    assert hse.get_last_session() is None


def test_get_last_session_empty_db(tmp_path, monkeypatch):
    monkeypatch.setattr(hse, "STATE_DB", tmp_path / "state.db")
    conn = sqlite3.connect(str(tmp_path / "state.db"))
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE sessions (
            id TEXT, title TEXT, source TEXT, started_at REAL,
            ended_at REAL, message_count INTEGER, tool_call_count INTEGER,
            model TEXT
        )
    """)
    conn.commit()
    conn.close()
    assert hse.get_last_session() is None


# --- extract_session_signals ---

def test_extract_signals_test_category():
    s = {"title": "Running tests for memory tool"}
    signals = hse.extract_session_signals(s)
    assert signals["category"] == "test"


def test_extract_signals_fix_category():
    s = {"title": "Fix NSSM false positive"}
    signals = hse.extract_session_signals(s)
    assert signals["category"] == "fix"


def test_extract_signals_build_category():
    s = {"title": "Build new tool scaffold"}
    signals = hse.extract_session_signals(s)
    assert signals["category"] == "build"


def test_extract_signals_audit_category():
    s = {"title": "Audit security vulnerabilities"}
    signals = hse.extract_session_signals(s)
    assert signals["category"] == "audit"


def test_extract_signals_deploy_category():
    s = {"title": "Deploy to vercel production"}
    signals = hse.extract_session_signals(s)
    assert signals["category"] == "deploy"


def test_extract_signals_general_fallback():
    s = {"title": "Some random task"}
    signals = hse.extract_session_signals(s)
    assert signals["category"] == "general"


def test_extract_signals_includes_counts():
    s = {"title": "Test", "message_count": 50, "tool_call_count": 30,
         "started_at": 100, "ended_at": 250}
    signals = hse.extract_session_signals(s)
    assert signals["msg_count"] == 50
    assert signals["tool_count"] == 30
    assert signals["duration_s"] == 150


# --- compose_conclusion ---

def test_compose_conclusion_includes_title():
    s = {"title": "Pimping hermes"}
    sig = {"category": "build", "duration_s": 600, "msg_count": 50, "tool_count": 30}
    conclusion = hse.compose_conclusion(s, sig)
    assert "Pimping hermes" in conclusion
    assert "build" in conclusion


def test_compose_conclusion_truncates_long_title():
    s = {"title": "a" * 100}
    sig = {"category": "general", "duration_s": 60, "msg_count": 10, "tool_count": 5}
    conclusion = hse.compose_conclusion(s, sig)
    assert "a" * 80 in conclusion
    assert "a" * 81 not in conclusion


# --- post_conclusion ---

def test_post_conclusion_success():
    mock_resp = MagicMock()
    mock_resp.read = MagicMock(return_value=b'{"id": "c_123"}')
    mock_resp.__enter__ = MagicMock(return_value=mock_resp)
    mock_resp.__exit__ = MagicMock(return_value=False)
    with patch.object(hse.urllib.request, "urlopen", return_value=mock_resp):
        ok, response = hse.post_conclusion("test content")
    assert ok is True


def test_post_conclusion_http_error():
    err = hse.urllib.error.HTTPError("url", 500, "Server Error", {}, None)
    err.read = MagicMock(return_value=b"internal error")
    with patch.object(hse.urllib.request, "urlopen", side_effect=err):
        ok, response = hse.post_conclusion("test")
    assert ok is False
    assert "500" in response


def test_post_conclusion_url_error():
    err = hse.urllib.error.URLError("connection refused")
    with patch.object(hse.urllib.request, "urlopen", side_effect=err):
        ok, response = hse.post_conclusion("test")
    assert ok is False
    assert "connection refused" in response


# --- main flow ---

def test_main_no_session_exits_0(tmp_path, monkeypatch):
    """No session in state.db → exit 0 cleanly."""
    monkeypatch.setattr(hse, "LOG_FILE", tmp_path / "log.txt")
    monkeypatch.setattr(hse, "STATE_DB", tmp_path / "missing.db")
    r = hse.main()
    assert r == 0


def test_main_with_session_posts(tmp_path, monkeypatch):
    """End-to-end with mocked session + post."""
    monkeypatch.setattr(hse, "LOG_FILE", tmp_path / "log.txt")
    monkeypatch.setattr(hse, "STATE_DB", tmp_path / "state.db")

    conn = sqlite3.connect(str(tmp_path / "state.db"))
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE sessions (
            id TEXT, title TEXT, source TEXT, started_at REAL,
            ended_at REAL, message_count INTEGER, tool_call_count INTEGER,
            model TEXT
        )
    """)
    cur.execute(
        "INSERT INTO sessions VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        ("s1", "Test", "cli", 100.0, 200.0, 50, 30, "MiniMax-M3")
    )
    conn.commit()
    conn.close()

    mock_resp = MagicMock()
    mock_resp.read = MagicMock(return_value=b'{"id": "c_1"}')
    mock_resp.__enter__ = MagicMock(return_value=mock_resp)
    mock_resp.__exit__ = MagicMock(return_value=False)
    with patch.object(hse.urllib.request, "urlopen", return_value=mock_resp):
        r = hse.main()
    assert r == 0
    log_text = (tmp_path / "log.txt").read_text()
    assert "posted" in log_text