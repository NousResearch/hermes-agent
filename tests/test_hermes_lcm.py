import importlib
import json
import os
import sqlite3
from pathlib import Path

import pytest

hermes_lcm = importlib.import_module("scripts.hermes_lcm")


def _seed_state_db(home: Path) -> Path:
    home.mkdir(parents=True, exist_ok=True)
    db = home / "state.db"
    conn = sqlite3.connect(db)
    conn.executescript(
        """
        CREATE TABLE sessions (
            id TEXT PRIMARY KEY,
            source TEXT NOT NULL,
            user_id TEXT,
            model TEXT,
            model_config TEXT,
            system_prompt TEXT,
            parent_session_id TEXT,
            started_at REAL NOT NULL,
            ended_at REAL,
            end_reason TEXT,
            message_count INTEGER DEFAULT 0,
            tool_call_count INTEGER DEFAULT 0,
            input_tokens INTEGER DEFAULT 0,
            output_tokens INTEGER DEFAULT 0,
            cache_read_tokens INTEGER DEFAULT 0,
            cache_write_tokens INTEGER DEFAULT 0,
            reasoning_tokens INTEGER DEFAULT 0,
            billing_provider TEXT,
            billing_base_url TEXT,
            billing_mode TEXT,
            estimated_cost_usd REAL,
            actual_cost_usd REAL,
            cost_status TEXT,
            cost_source TEXT,
            pricing_version TEXT,
            title TEXT,
            api_call_count INTEGER DEFAULT 0
        );
        CREATE TABLE messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT NOT NULL REFERENCES sessions(id),
            role TEXT NOT NULL,
            content TEXT,
            tool_call_id TEXT,
            tool_calls TEXT,
            tool_name TEXT,
            timestamp REAL NOT NULL,
            token_count INTEGER,
            finish_reason TEXT,
            reasoning TEXT,
            reasoning_details TEXT,
            codex_reasoning_items TEXT,
            reasoning_content TEXT,
            codex_message_items TEXT
        );
        CREATE VIRTUAL TABLE messages_fts USING fts5(content);
        CREATE VIRTUAL TABLE messages_fts_trigram USING fts5(content, tokenize='trigram');
        """
    )
    conn.execute(
        "INSERT INTO sessions (id, source, title, started_at, message_count) VALUES (?, ?, ?, ?, ?)",
        ("s1", "cli", "LCM test session api_key=sk-testSECRET1234567890", 1000.0, 4),
    )
    rows = [
        ("s1", "user", "Please inspect PER-2929 and remember command alpha", None, 1001.0),
        ("s1", "assistant", "I will search exact recall evidence", None, 1002.0),
        ("s1", "tool", "secret api_key=sk-testSECRET1234567890 and long " + "x" * 300, "terminal", 1003.0),
        ("s1", "assistant", "Final PASS for PER-2929", None, 1004.0),
    ]
    for session_id, role, content, tool_name, ts in rows:
        cur = conn.execute(
            "INSERT INTO messages (session_id, role, content, tool_name, timestamp) VALUES (?, ?, ?, ?, ?)",
            (session_id, role, content, tool_name, ts),
        )
        content_for_fts = f"{content or ''} {tool_name or ''}"
        conn.execute("INSERT INTO messages_fts(rowid, content) VALUES (?, ?)", (cur.lastrowid, content_for_fts))
        conn.execute("INSERT INTO messages_fts_trigram(rowid, content) VALUES (?, ?)", (cur.lastrowid, content_for_fts))
    conn.commit()
    conn.close()
    return db


@pytest.fixture()
def lcm_home(tmp_path, monkeypatch):
    home = tmp_path / "hermes-home"
    _seed_state_db(home)
    monkeypatch.setenv("HERMES_HOME", str(home))
    return home


def test_status_reports_readable_counts_and_fts(lcm_home):
    result = hermes_lcm.main(["status", "--json"])
    assert result == 0

    data = hermes_lcm.status(object())
    assert data["state_db_readable"] is True
    assert data["sessions_count"] == 1
    assert data["messages_count"] == 4
    assert data["has_messages_fts"] is True
    assert data["has_messages_fts_trigram"] is True
    assert "[REDACTED]" in data["latest_session"]["title"]


def test_grep_finds_seeded_message_and_bounds_output(lcm_home):
    args = type("Args", (), {
        "query": "PER-2929",
        "session": None,
        "session_id": None,
        "role": None,
        "tool_name": None,
        "since": None,
        "before": None,
        "sort": "rank",
        "limit": 2,
        "max_chars": 80,
    })()

    data = hermes_lcm.grep(args)

    assert data["count"] >= 1
    assert data["matches"][0]["session_id"] == "s1"
    assert "PER-2929" in data["matches"][0]["snippet"]
    assert len(data["matches"][0]["snippet"]) <= 120
    assert "sk-testSECRET" not in data["matches"][0]["title"]


def test_describe_returns_window_around_message(lcm_home):
    args = type("Args", (), {
        "message_id": 2,
        "session_id": None,
        "tail": None,
        "around": None,
        "window": 1,
        "max_chars": 200,
    })()

    data = hermes_lcm.describe(args)

    assert data["count"] == 3
    assert [m["message_id"] for m in data["messages"]] == [1, 2, 3]


def test_describe_invalid_message_id_returns_error(lcm_home):
    args = type("Args", (), {
        "message_id": 999,
        "session_id": None,
        "tail": None,
        "around": None,
        "window": 1,
        "max_chars": 200,
    })()

    data = hermes_lcm.describe(args)

    assert data["error"] == "message_id not found"


def test_redacts_secret_like_content_in_search_results(lcm_home):
    args = type("Args", (), {
        "query": "secret",
        "session": None,
        "session_id": None,
        "role": "tool",
        "tool_name": None,
        "since": None,
        "before": None,
        "sort": "time",
        "limit": 1,
        "max_chars": 1000,
    })()

    data = hermes_lcm.grep(args)
    snippet = data["matches"][0]["snippet"]

    assert "***" not in snippet
    assert "[REDACTED]" in snippet


def test_direct_calls_clamp_untrusted_bounds(lcm_home):
    args = type("Args", (), {
        "query": "secret",
        "session": None,
        "session_id": None,
        "role": "tool",
        "tool_name": None,
        "since": None,
        "before": None,
        "sort": "time",
        "limit": 9999,
        "max_chars": 0,
    })()

    data = hermes_lcm.grep(args)
    snippet = data["matches"][0]["snippet"]

    assert data["count"] <= 50
    assert len(snippet) < 900
    assert "[truncated by hermes_lcm]" in snippet


def test_lcm_toolset_is_opt_in_not_default(lcm_home):
    from hermes_cli.tools_config import _get_platform_tools

    assert "lcm" not in _get_platform_tools({}, "cli")
    assert "lcm" not in _get_platform_tools({}, "telegram")
    assert "lcm" in _get_platform_tools({"platform_toolsets": {"cli": ["hermes-cli", "lcm"]}}, "cli")


def test_sqlite_connection_is_read_only(lcm_home):
    conn = hermes_lcm._connect()
    try:
        with pytest.raises(sqlite3.OperationalError):
            conn.execute("INSERT INTO sessions (id, source, started_at) VALUES ('write-test', 'cli', 1)")
    finally:
        conn.close()


def test_missing_state_db_status_is_safe(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "missing-home"))

    data = hermes_lcm.status(object())

    assert data["state_db_readable"] is False
    assert data["error"] == "missing state.db"


def test_lcm_toolset_registration(lcm_home):
    import tools.lcm_tool  # noqa: F401 - registers tools at import time
    from tools.registry import registry
    from toolsets import resolve_toolset, validate_toolset

    assert validate_toolset("lcm") is True
    assert set(resolve_toolset("lcm")) == {
        "lcm_status",
        "lcm_grep",
        "lcm_describe",
        "lcm_recall",
        "lcm_expand_query",
    }
    for name in resolve_toolset("lcm"):
        assert registry.get_entry(name) is not None
