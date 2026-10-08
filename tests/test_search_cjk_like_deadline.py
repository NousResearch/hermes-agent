"""Short CJK searches must not ride an unbounded LIKE full-table scan (#134779).

The CJK router (_search_cjk) answers lone-1-char-CJK runs — and any query when
the trigram route is unavailable — via _like_rows, a canonical-table LIKE scan.
That call site is the only one left without a cooperative SQLite VM deadline:
#129839 bounds the canonical LIKE fallback (the FTS-stale/disabled route), but
the CJK leg keeps scanning for minutes on long-lived stores until the tool
timeout kills the whole search (420s tool timeout, ``like_scan`` 720s in the
report's logs). This pins the CJK LIKE leg to the same deadline pattern
already used by recent-session browse (hermes_state_sessions).

Runs against a real SessionDB in a temp HERMES_HOME, on the LIKE route only —
no cjk_unicode61 tokenizer toolchain is required. A lone 1-char CJK run always
routes to LIKE regardless of index availability, so no monkeypatched index
flags are needed to reach the vulnerable path.
"""

import sqlite3

import pytest

import hermes_state_search
from hermes_state import SessionDB

LONE_CJK = "桂"  # single CJK char — a lone run, always routed to the LIKE scan


def make_db(tmp_path):
    database = SessionDB(db_path=tmp_path / "state.db")
    database.create_session("20260820_000001_cjk001", "cli")
    database.append_message("20260820_000001_cjk001", "user", "桂林山水甲天下")
    return database


def test_cjk_like_scan_honours_deadline_and_raises(tmp_path, monkeypatch):
    """A short CJK query over a large canonical table is cut by the deadline
    (cooperative interrupt translated into TimeoutError), never left scanning."""
    database = make_db(tmp_path)
    try:
        database._conn.executemany(
            "INSERT INTO messages (session_id, role, content, timestamp) VALUES ('20260820_000001_cjk001', 'user', ?, ?)",
            [(f"filler row {n}", 1.0 + n) for n in range(20000)],
        )
        database._conn.commit()
        # raising=False: pre-fix tree has no dial yet — setting it must be a no-op,
        # so the RED run proves the scan ignores the deadline rather than the dial missing.
        monkeypatch.setattr(
            hermes_state_search, "_CJK_LIKE_SEARCH_TIMEOUT_SECONDS", 0.0, raising=False
        )
        with pytest.raises(TimeoutError):
            database.search_messages(LONE_CJK, limit=3)
    finally:
        database.close()


def test_cjk_like_scan_returns_rows_within_deadline(tmp_path):
    """A deadline that is not exhausted changes nothing: the CJK LIKE leg still
    returns the matching rows (raise instead of a manufactured empty set only
    applies when the deadline is actually hit)."""
    database = make_db(tmp_path)
    try:
        matches = database.search_messages(
            LONE_CJK, limit=3, fields=("session_id", "snippet")
        )
        assert matches, "lone-char CJK query should match the seeded message via LIKE"
    finally:
        database.close()
