"""Durable lost-result repair: orphan tool-result backfill on resume.

Background (2026-08-30 incident): a wedged pre-exec guard thread held the
tool call open until the whole service was restarted. The synthetic
[Orphan recovery: ...] result existed only in the in-memory replay — the
durable transcript kept the dangling assistant(tool_calls) forever, so the
TUI showed an eternally in-flight call and every restart re-synthesized.
"""
import sqlite3
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from agent.replay_cleanup import (
    backfill_orphan_tool_results,
    collect_dangling_tool_call_ids,
    sanitize_replay_history,
)


def _assistant(calls):
    return {
        "role": "assistant",
        "tool_calls": [
            {"id": cid, "function": {"name": name, "arguments": "{}"}}
            for cid, name in calls
        ],
        "content": "",
    }


def _tool(cid, content="ok"):
    return {"role": "tool", "tool_call_id": cid, "content": content}


class _FakeDB:
    """Minimal SessionDB stand-in: tracks has_tool_result + appended rows."""

    def __init__(self, answered=()):
        self._answered = set(answered)
        self.appended = []

    def has_tool_result(self, session_id, tool_call_id):
        return tool_call_id in self._answered

    def append_message(self, session_id, role, *, content, tool_name,
                       tool_call_id, effect_disposition, timestamp=None):
        self.appended.append({
            "session_id": session_id, "role": role, "content": content,
            "tool_name": tool_name, "tool_call_id": tool_call_id,
            "effect_disposition": effect_disposition, "timestamp": timestamp,
        })
        self._answered.add(tool_call_id)
        return 1


def test_collect_dangling_ids_finds_unanswered_calls():
    hist = [
        {"role": "user", "content": "go"},
        _assistant([("call_A", "read_file"), ("call_B", "terminal")]),
        _tool("call_A"),
        # call_B never answered
    ]
    assert collect_dangling_tool_call_ids(hist) == ["call_B"]


def test_collect_dangling_ids_handles_codex_call_id_key():
    hist = [
        _assistant([]),
        {"role": "assistant", "tool_calls": [{"call_id": "call_C", "function": {"name": "terminal"}}]},
    ]
    assert collect_dangling_tool_call_ids(hist) == ["call_C"]


def test_backfill_writes_synthetic_rows(tmp_path):
    hist = [
        {"role": "user", "content": "go"},
        _assistant([("call_A", "read_file"), ("call_B", "terminal")]),
        _tool("call_A"),
    ]
    db = _FakeDB(answered={"call_A"})
    n = backfill_orphan_tool_results("sess1", hist, db)
    assert n == 1
    row = db.appended[0]
    assert row["role"] == "tool"
    assert row["tool_call_id"] == "call_B"
    assert "Orphan recovery" in row["content"]
    # side-effecting tool -> UNKNOWN disposition
    assert row["effect_disposition"] == "unknown"


def test_backfill_is_idempotent(tmp_path):
    hist = [
        _assistant([("call_A", "read_file")]),
        _tool("call_A"),
    ]
    db = _FakeDB(answered={"call_A"})
    assert backfill_orphan_tool_results("sess1", hist, db) == 0
    assert db.appended == []


def test_backfill_never_raises_on_db_failure():
    class _BrokenDB:
        def has_tool_result(self, *_a, **_k):
            raise RuntimeError("db gone")

    hist = [_assistant([("call_A", "read_file")])]
    # must not raise; returns 0
    assert backfill_orphan_tool_results("sess1", hist, _BrokenDB()) == 0


def test_backfill_skips_when_db_none_or_history_empty():
    assert backfill_orphan_tool_results("s", [], object()) == 0
    assert backfill_orphan_tool_results("s", [_assistant([("a", "t")])], None) == 0


def test_sanitize_replay_history_still_answers_tail_for_model():
    """End-to-end resume semantics: dangling tail gets orphan-recovery answers
    in the model-fed history (existing behavior, now also persisted by backfill)."""
    hist = [
        {"role": "user", "content": "hi"},
        _assistant([("call_X", "terminal")]),
    ]
    out = sanitize_replay_history(hist)
    tool_rows = [m for m in out if m.get("role") == "tool"]
    assert len(tool_rows) == 1
    assert "Orphan recovery" in str(tool_rows[0]["content"])


def test_has_tool_result_real_schema(tmp_path):
    """The real SessionDB.has_tool_result against the actual messages schema."""
    db = sqlite3.connect(tmp_path / "state.db")
    db.execute(
        """CREATE TABLE messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT, role TEXT, content TEXT,
            tool_name TEXT, tool_call_id TEXT, active INTEGER DEFAULT 1,
            timestamp REAL)"""
    )
    db.execute(
        "INSERT INTO messages (session_id, role, content, tool_call_id, active) "
        "VALUES ('s1', 'tool', 'x', 'call_1', 1)"
    )
    db.commit()

    from hermes_state import SessionDB  # noqa: F401  (import path sanity)

    # Bind the method to a lightweight object carrying _conn
    class _Shim:
        pass

    shim = _Shim()
    shim._conn = db
    import logging
    shim_logger = logging.getLogger("test")
    SessionDB.has_tool_result(shim, "s1", "call_1")  # would AttributeError if broken
    assert SessionDB.has_tool_result(shim, "s1", "call_1") is True
    assert SessionDB.has_tool_result(shim, "s1", "call_missing") is False


def test_has_tool_result_ignores_soft_deleted(tmp_path):
    db = sqlite3.connect(":memory:")
    db.execute(
        """CREATE TABLE messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT, role TEXT, content TEXT,
            tool_name TEXT, tool_call_id TEXT, active INTEGER DEFAULT 1,
            timestamp REAL)"""
    )
    db.execute(
        "INSERT INTO messages (session_id, role, content, tool_call_id, active) "
        "VALUES ('s1', 'tool', 'x', 'call_1', 0)"
    )
    db.commit()

    class _Shim:
        _conn = db

    from hermes_state import SessionDB
    # soft-deleted row must NOT count as an answer
    assert SessionDB.has_tool_result(_Shim(), "s1", "call_1") is False
