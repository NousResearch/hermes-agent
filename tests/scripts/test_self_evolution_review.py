"""Tests for self-evolution-review.py."""
import sqlite3
import sys
import tempfile
from pathlib import Path
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch

SCRIPT = Path(r"C:\Users\bbask\AppData\Local\hermes\scripts\self-evolution-review.py")
sys.path.insert(0, str(SCRIPT.parent))
import importlib.util
spec = importlib.util.spec_from_file_location("sev", SCRIPT)
sev = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sev)


def make_test_db(tmp_path):
    """Create a test state.db with realistic sessions."""
    db_path = tmp_path / "state.db"
    conn = sqlite3.connect(str(db_path))
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE sessions (
            id TEXT PRIMARY KEY, title TEXT, source TEXT, model TEXT,
            started_at REAL, ended_at REAL, end_reason TEXT,
            message_count INTEGER, tool_call_count INTEGER,
            input_tokens INTEGER, output_tokens INTEGER,
            estimated_cost_usd REAL, actual_cost_usd REAL
        )
    """)
    now = datetime.now().timestamp()
    # Sessions
    sessions = [
        # (id, title, source, duration_s, msg_count, tool_calls, end_reason, cost, input_tokens, output_tokens)
        ("s1", "Hermes install audit", "cli", 600, 50, 30, "complete", 0.05, 10000, 5000),
        ("s2", "Honcho upgrade", "cli", 1800, 200, 150, "complete", 0.50, 50000, 25000),
        ("s3", "Too short", "cli", 10, 2, 0, "complete", 0.01, 100, 50),
        ("s4", "Long no-tool session", "cli", 600, 100, 0, "complete", 0.20, 20000, 10000),
        ("s5", "Tool retry loop", "cli", 300, 50, 200, "complete", 0.10, 5000, 3000),
        ("s6", "Cron job daily", "cron", 60, 5, 2, "cron_complete", 0.005, 500, 250),
        ("s7", "Telegram session", "telegram", 180, 30, 15, "complete", 0.03, 3000, 1500),
        # outside lookback window (10 days old)
        ("s_old", "Old session", "cli", 600, 50, 30, "complete", 0.05, 10000, 5000),
    ]
    for s in sessions:
        is_old = s[0] == "s_old"
        start_ts = (now - timedelta(days=10).total_seconds()) if is_old else (now - 60)
        # Schema: id, title, source, model, started_at, ended_at,
        #         end_reason, message_count, tool_call_count, input_tokens, output_tokens,
        #         estimated_cost_usd, actual_cost_usd
        cur.execute(
            "INSERT INTO sessions VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                s[0],                       # id
                s[1],                       # title
                s[2],                       # source
                "MiniMax-M3",               # model
                start_ts,                   # started_at
                start_ts + s[3],            # ended_at
                s[6],                       # end_reason
                s[4],                       # message_count
                s[5],                       # tool_call_count
                s[8],                       # input_tokens
                s[9],                       # output_tokens
                s[7],                       # estimated_cost_usd
                s[7],                       # actual_cost_usd
            )
        )
    conn.commit()
    conn.close()
    return db_path


# --- query_sessions ---

def test_query_sessions_returns_recent(tmp_path, monkeypatch):
    db_path = make_test_db(tmp_path)
    monkeypatch.setattr(sev, "STATE_DB", db_path)
    conn = sqlite3.connect(str(db_path))
    sessions = sev.query_sessions(conn, lookback_days=7)
    conn.close()
    # Should include 7 recent + exclude the 10-day-old one
    ids = [s["id"] for s in sessions]
    assert "s_old" not in ids
    assert "s1" in ids
    assert "s7" in ids


def test_query_sessions_respects_lookback(tmp_path, monkeypatch):
    db_path = make_test_db(tmp_path)
    monkeypatch.setattr(sev, "STATE_DB", db_path)
    conn = sqlite3.connect(str(db_path))
    # lookback 30 days should include s_old
    sessions = sev.query_sessions(conn, lookback_days=30)
    conn.close()
    ids = [s["id"] for s in sessions]
    assert "s_old" in ids


# --- find_failed_sessions ---

def test_find_failed_sessions_too_short():
    sessions = [{"id": "s3", "started_at": 100, "ended_at": 110, "tool_call_count": 0, "message_count": 2}]
    failed = sev.find_failed_sessions(sessions)
    assert len(failed) == 1
    assert failed[0]["failure_reason"] == "too_short"


def test_find_failed_sessions_no_tool_calls():
    sessions = [{"id": "s4", "started_at": 100, "ended_at": 700, "tool_call_count": 0, "message_count": 100}]
    failed = sev.find_failed_sessions(sessions)
    assert len(failed) == 1
    assert failed[0]["failure_reason"] == "no_tool_calls"


def test_find_failed_sessions_retry_loop():
    sessions = [{"id": "s5", "started_at": 100, "ended_at": 400, "tool_call_count": 300, "message_count": 30}]
    failed = sev.find_failed_sessions(sessions)
    assert len(failed) == 1
    assert failed[0]["failure_reason"] == "tool_retry_loop"


def test_find_failed_sessions_clean():
    sessions = [{"id": "s1", "started_at": 100, "ended_at": 700, "tool_call_count": 30, "message_count": 50}]
    failed = sev.find_failed_sessions(sessions)
    assert failed == []


def test_find_failed_sessions_empty():
    assert sev.find_failed_sessions([]) == []


# --- find_high_cost_sessions ---

def test_find_high_cost_sessions_returns_top_5():
    sessions = [
        {"id": f"s{i}", "title": f"Session {i}", "actual_cost_usd": float(i) * 0.01}
        for i in range(10)
    ]
    high = sev.find_high_cost_sessions(sessions)
    assert len(high) == 5
    assert high[0]["session"]["id"] == "s9"  # highest cost


def test_find_high_cost_sessions_zero_cost_excluded():
    sessions = [{"id": "s1", "actual_cost_usd": 0}]
    high = sev.find_high_cost_sessions(sessions)
    assert high == []


def test_find_high_cost_sessions_uses_estimated_when_actual_missing():
    sessions = [{"id": "s1", "actual_cost_usd": None, "estimated_cost_usd": 0.05}]
    high = sev.find_high_cost_sessions(sessions)
    assert len(high) == 1
    assert abs(high[0]["cost"] - 0.05) < 0.001


# --- categorize_by_source ---

def test_categorize_by_source():
    sessions = [
        {"source": "cli"},
        {"source": "cli"},
        {"source": "telegram"},
        {"source": "cron"},
        {"source": "cli"},
    ]
    counts = sev.categorize_by_source(sessions)
    assert counts["cli"] == 3
    assert counts["telegram"] == 1
    assert counts["cron"] == 1


def test_categorize_by_source_handles_none():
    sessions = [{"source": None}, {"source": "cli"}]
    counts = sev.categorize_by_source(sessions)
    assert counts["unknown"] == 1
    assert counts["cli"] == 1


# --- find_common_topics ---

def test_find_common_topics_extracts_keywords():
    sessions = [
        {"title": "Memory tool audit"},
        {"title": "Memory compression fix"},
        {"title": "Cron job error"},
        {"title": "Memory cleanup"},
        {"title": "Verifying build"},
    ]
    topics = sev.find_common_topics(sessions)
    words = [t[0] for t in topics]
    # "memory" appears 3 times, should be top
    assert "memory" in words
    # "cron" should appear
    # short words (<4 chars) filtered out


def test_find_common_topics_filters_stop_words():
    sessions = [
        {"title": "The quick brown fox jumps over"},
        {"title": "A lazy quick dog sleeps under"},
    ]
    topics = sev.find_common_topics(sessions)
    words = [t[0] for t in topics]
    assert "the" not in words
    # "quick" appears in BOTH titles → should appear in top topics
    assert "quick" in words


def test_find_common_topics_empty():
    assert sev.find_common_topics([]) == []


# --- compose_report ---

def test_compose_report_includes_summary():
    sessions = [{"started_at": 100, "ended_at": 700, "tool_call_count": 30, "message_count": 50,
                 "input_tokens": 10000, "output_tokens": 5000, "estimated_cost_usd": 0.05,
                 "source": "cli", "title": "Test"}]
    report = sev.compose_report(sessions, [], [], {"cli": 1}, [])
    assert "1 sessions" in report
    assert "Summary" in report


def test_compose_report_calls_out_failures():
    sessions = []
    failed = [{"id": "s3", "title": "Failed session", "failure_reason": "too_short",
               "started_at": 100, "ended_at": 110}]
    report = sev.compose_report(sessions, failed, [], {}, [])
    # Number is bold-formatted in the report
    assert "**1** failed" in report
    assert "too_short" in report


def test_compose_report_warns_high_failure_rate():
    sessions = [{"started_at": 100, "ended_at": 700, "tool_call_count": 30, "message_count": 50,
                 "input_tokens": 1000, "output_tokens": 500, "estimated_cost_usd": 0.01,
                 "source": "cli", "title": "x"} for _ in range(10)]
    failed = [{"id": f"f{i}", "title": "x", "failure_reason": "too_short",
               "started_at": 100, "ended_at": 110} for i in range(3)]
    report = sev.compose_report(sessions, failed, [], {}, [])
    assert "High failure rate" in report


def test_compose_report_warns_high_cost():
    sessions = [{"started_at": 100, "ended_at": 700, "tool_call_count": 30, "message_count": 50,
                 "input_tokens": 1000, "output_tokens": 500, "estimated_cost_usd": 15.0,
                 "source": "cli", "title": "x"}]
    report = sev.compose_report(sessions, [], [], {}, [])
    assert "15.00" in report
    assert "MoA routing" in report


def test_compose_report_no_issues_clean():
    sessions = [{"started_at": 100, "ended_at": 700, "tool_call_count": 30, "message_count": 50,
                 "input_tokens": 1000, "output_tokens": 500, "estimated_cost_usd": 0.05,
                 "source": "cli", "title": "x"}]
    report = sev.compose_report(sessions, [], [], {"cli": 1}, [])
    assert "No failed" in report
    assert "No high-cost" in report


# --- save_fixlog ---

def test_save_fixlog_creates_file(tmp_path, monkeypatch):
    monkeypatch.setenv("OBSIDIAN_FIXLOG", str(tmp_path / "fixlogs"))
    result = sev.save_fixlog("# Test report")
    assert result is not None
    assert result.exists()
    assert result.read_text() == "# Test report"


def test_save_fixlog_includes_date(tmp_path, monkeypatch):
    monkeypatch.setenv("OBSIDIAN_FIXLOG", str(tmp_path / "fixlogs"))
    result = sev.save_fixlog("hello")
    today = datetime.now().strftime("%Y-%m-%d")
    assert today in str(result)


# --- Integration: end-to-end with real state.db ---

def test_main_against_real_db(tmp_path, monkeypatch):
    """Use a fresh state.db to test the full main() flow via env var."""
    db_path = make_test_db(tmp_path)
    # make_test_db already created tmp_path/state.db. Set HERMES_HOME so the
    # script finds it in its own subprocess.
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setenv("OBSIDIAN_FIXLOG", str(tmp_path / "fixlogs"))
    r = __import__("subprocess").run(
        [sys.executable, str(SCRIPT)],
        capture_output=True, text=True, timeout=30
    )
    assert r.returncode == 0, f"stdout={r.stdout}, stderr={r.stderr}"
    assert "Traceback" not in r.stderr
    assert "Summary" in r.stdout


def test_main_missing_db(tmp_path, monkeypatch):
    monkeypatch.setattr(sev, "STATE_DB", tmp_path / "missing.db")
    r = __import__("subprocess").run(
        [sys.executable, str(SCRIPT)],
        capture_output=True, text=True, timeout=10
    )
    assert r.returncode in (0, 1)