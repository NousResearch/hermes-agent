"""Regression tests for 2026-11-14 03:25-03:26 RCA:
1. Unscoped search_messages with 'tool' in role_filter queries messages_fts first
   and only runs _search_messages_like_fallback on zero FTS hits (or session-scoped top-up),
   logging path=like_tool_fallback instead of path=fts5 when fallback executes.
2. tui_gateway.ws._STREAMING_EVENT_TYPES includes 'tool.generating' so concurrent
   streaming tool-call generation coalesces on the WS buffer instead of blocking
   worker threads on fut.result().
3. Observation-only activity writes (touch_session_activity / clear_session_activity_labels)
   suppress log_write_lock_holders WARNING spam on sub-second lock timeouts, and
   _execute_write formats fractional patience_s as '0.5s' rather than '0s'.
"""

import logging
import sqlite3
import threading
import time

import pytest

import hermes_state_lockowners
from hermes_state import SessionDB
from hermes_state_common import FTS_TOOL_CONTENT_PREFIX_CHARS
from tui_gateway.ws import _STREAMING_EVENT_TYPES


def _long_tool_body(prefix: str, tail: str) -> str:
    padding = "padding " * (FTS_TOOL_CONTENT_PREFIX_CHARS // len("padding ") + 16)
    return f"{prefix} {padding} {tail}"


@pytest.fixture
def db(tmp_path):
    session_db = SessionDB(db_path=tmp_path / "state.db")
    if not session_db._fts_enabled:
        session_db.close()
        pytest.skip("SQLite FTS5 unavailable")
    session_db.create_session("s1", source="cli")
    try:
        yield session_db
    finally:
        session_db.close()


def test_unscoped_tool_role_filter_uses_fts_first_and_skips_like_scan_on_hit(
    db, monkeypatch, caplog
):
    """When messages_fts already has hits for an unscoped search that includes 'tool'
    in role_filter, search_messages must return the FTS hits without running a full-table
    LIKE scan, and log path=fts5."""
    msg_id = db.append_message(
        "s1",
        role="tool",
        content="Google Chat app not found in hangouts-chat prefix",
        tool_name="terminal",
    )
    like_calls = []
    orig_like = db._search_messages_like_fallback

    def _spy_like(*args, **kwargs):
        like_calls.append((args, kwargs))
        return orig_like(*args, **kwargs)

    monkeypatch.setattr(db, "_search_messages_like_fallback", _spy_like)
    monkeypatch.setenv("HERMES_SEARCH_SLOW_MS", "0")

    with caplog.at_level(logging.INFO, logger="hermes_state"):
        rows = db.search_messages(
            '"Google Chat app not found" OR "hangouts-chat"',
            role_filter=["user", "assistant", "tool"],
            limit=150,
        )

    assert [r["id"] for r in rows] == [msg_id]
    assert like_calls == [], "Unscoped search with FTS hits must not run full-table LIKE fallback"
    assert "path=fts5" in caplog.text


def test_tool_tail_miss_falls_back_to_like_and_logs_like_tool_fallback(
    db, monkeypatch, caplog
):
    """When a token exists only past FTS_TOOL_CONTENT_PREFIX_CHARS (>8192 chars),
    messages_fts returns 0 hits and search_messages falls back to LIKE, logging
    path=like_tool_fallback instead of path=fts5."""
    msg_id = db.append_message(
        "s1",
        role="tool",
        content=_long_tool_body("indexed-head-token", "unindexed-deep-tail-needle"),
        tool_name="terminal",
    )
    monkeypatch.setenv("HERMES_SEARCH_SLOW_MS", "0")

    with caplog.at_level(logging.INFO, logger="hermes_state"):
        rows = db.search_messages(
            "unindexed-deep-tail-needle",
            role_filter=["user", "assistant", "tool"],
            limit=10,
        )

    assert [r["id"] for r in rows] == [msg_id]
    assert "path=like_tool_fallback" in caplog.text


def test_session_scoped_search_tops_up_tool_tail_even_when_fts_hits_prefix(db):
    """When session_ids is provided, search_messages must return both the FTS prefix hit
    and any >8KB tool-tail hit in that session."""
    id_prefix = db.append_message(
        "s1",
        role="tool",
        content="scoped-shared-needle in short tool row",
        tool_name="terminal",
    )
    id_tail = db.append_message(
        "s1",
        role="tool",
        content=_long_tool_body("other-prefix", "scoped-shared-needle"),
        tool_name="terminal",
    )
    rows = db.search_messages(
        "scoped-shared-needle",
        role_filter=["user", "assistant", "tool"],
        session_ids=["s1"],
        limit=10,
    )
    assert {r["id"] for r in rows} == {id_prefix, id_tail}


def test_tool_generating_is_coalesced_as_streaming_ws_event():
    """tool.generating must be in _STREAMING_EVENT_TYPES so worker threads buffer it
    instead of blocking synchronously on fut.result() during concurrent completions."""
    assert "tool.generating" in _STREAMING_EVENT_TYPES


def test_activity_writes_suppress_lockowner_warning_and_preserve_fractional_patience(
    db, monkeypatch
):
    """Observation-only activity writes (touch_session_activity / clear_session_activity_labels)
    must not trigger log_write_lock_holders WARNING scans when yielding on their 0.5s budget,
    and operational error text must format 0.5s as '0.5s' rather than '0s'."""
    called = []
    monkeypatch.setattr(
        hermes_state_lockowners,
        "log_write_lock_holders",
        lambda db_path, patience_s: called.append((db_path, patience_s)),
    )
    monkeypatch.setattr(
        "hermes_state.log_write_lock_holders",
        lambda db_path, patience_s: called.append((db_path, patience_s)),
    )
    monkeypatch.setattr(SessionDB, "_ACTIVITY_WRITE_PATIENCE_S", 0.05)

    started = threading.Event()
    release = threading.Event()

    def _hold():
        conn = sqlite3.connect(str(db.db_path), timeout=1.0, isolation_level=None)
        try:
            conn.execute("BEGIN IMMEDIATE")
            started.set()
            release.wait(5.0)
            conn.execute("COMMIT")
        finally:
            conn.close()

    holder = threading.Thread(target=_hold)
    holder.start()
    try:
        assert started.wait(5.0)
        with pytest.raises(sqlite3.OperationalError) as exc_info:
            db.touch_session_activity("s1", description="executing tool", provenance="tool")
        assert "over 0.05s" in str(exc_info.value)
        assert "over 0s" not in str(exc_info.value)
        assert called == [], "Observation-only activity writes must not call log_write_lock_holders"
    finally:
        release.set()
        holder.join(timeout=5.0)
