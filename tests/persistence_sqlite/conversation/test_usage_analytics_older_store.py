"""The usage analytics and the creating-surface probe on an older store opened read-only, against the legacy reads.

The store is written by the legacy ``SessionDB``, then regressed the way an older writer left it: no pinned
``idx_messages_assistant_calls_by_session``, no ``created_source`` column and no ``session_model_usage`` table. The
backend opens it read-only and must answer what the insights engine, the dashboard's auxiliary read and the session
listing's column probe answer on a read-only legacy open of the same file.
"""

from __future__ import annotations

import json
import sqlite3
from typing import Optional

import pytest

from persistence.sqlite import SQLiteBackend
from tests.persistence_sqlite._legacy_store import legacy_store
from tests.persistence_sqlite.conftest import profile_home, store_for

__all__ = ["legacy_store"]

_CALLS = [{"id": "k1", "type": "function", "function": {"name": "skill_view", "arguments": json.dumps({"name": "alpha"})}}]
_INSIGHTS = (
    ("insight_sessions", "_GET_SESSIONS"), ("insight_tool_result_counts", "_GET_TOOL_NAMES"),
    ("insight_tool_call_payloads", "_GET_TOOL_CALLS"), ("insight_skill_call_payloads", "_GET_SKILL_CALLS"),
)


# The legacy reads: the insights engine's queries on a legacy connection (this tree's engine reads the backend).
# A source filter also matches the gateway's ``local`` rows that surface created: interactive local
# sessions are stored as ``local`` with the surface kept in ``created_source``
# (gateway/session_a2a.storage_source), so ``--source cli`` still finds the classic CLI's sessions.
_LOCAL_SURFACE = " OR ({a}source = 'local' AND {a}created_source = ?)"


def _scoped(before: str, after: str = "", *, alias: str = "s.") -> tuple[str, str]:
    """(unfiltered, source-filtered) query pair sharing one body. Built once at class definition,
    so no runtime value can alter query structure."""
    src = f" AND ({alias}source = ?{_LOCAL_SURFACE.format(a=alias)})"
    return before + after, before + src + after


class _M2Insights:
    """Analyzes session history from a SessionDB (or raw sqlite3 connection)."""

    _SESSION_COLS = ("id, source, model, started_at, ended_at, "
                     "message_count, tool_call_count, input_tokens, output_tokens, "
                     "cache_read_tokens, cache_write_tokens, billing_provider, "
                     "billing_base_url, billing_mode, estimated_cost_usd, "
                     "actual_cost_usd, cost_status, cost_source, api_call_count")

    _GET_SESSIONS_ALL, _GET_SESSIONS_WITH_SOURCE = _scoped(
        f"SELECT {_SESSION_COLS} FROM sessions WHERE started_at >= ?",
        " ORDER BY started_at DESC",
        alias="",
    )

    # ``INDEXED BY`` pins the partial index so the plan is deterministic on a
    # fresh state.db (before ANALYZE) for both branches; without it the
    # source-filtered probe falls back to idx_messages_session_active and scans
    # each session's non-tool-call rows. The pin is a HARD dependency (SQLite
    # raises ``no such index``): read-only opens skip ``_init_schema``, so an
    # older writer's DB may lack it — ``__init__`` probes once and falls back
    # to the unpinned variants (identical rows, optimizer-chosen plan).
    _MESSAGES_ASSISTANT_CALLS_INDEX = "idx_messages_assistant_calls_by_session"
    _ASSISTANT_CALLS = (
        f" FROM messages m INDEXED BY {_MESSAGES_ASSISTANT_CALLS_INDEX}"
        " JOIN sessions s ON s.id = m.session_id"
        " WHERE s.started_at >= ?"
    )
    _GET_TOOL_CALLS_ALL, _GET_TOOL_CALLS_WITH_SOURCE = _scoped(
        "SELECT m.session_id, m.tool_calls" + _ASSISTANT_CALLS,
        " AND m.role = 'assistant' AND m.tool_calls IS NOT NULL",
    )
    _GET_SKILL_CALLS_ALL, _GET_SKILL_CALLS_WITH_SOURCE = _scoped(
        "SELECT m.tool_calls, m.timestamp" + _ASSISTANT_CALLS,
        " AND m.role = 'assistant' AND m.tool_calls IS NOT NULL"
        " AND (instr(m.tool_calls, 'skill_view') > 0"
        " OR instr(m.tool_calls, 'skill_manage') > 0)",
    )
    _GET_TOOL_NAMES_ALL, _GET_TOOL_NAMES_WITH_SOURCE = _scoped(
        """SELECT m.session_id, m.tool_name, COUNT(*) as count
                   FROM messages m
                   JOIN sessions s ON s.id = m.session_id
                   WHERE s.started_at >= ?""",
        """
                     AND m.role = 'tool' AND m.tool_name IS NOT NULL
                   GROUP BY m.session_id, m.tool_name""",
    )
    _GET_MESSAGE_STATS_ALL, _GET_MESSAGE_STATS_WITH_SOURCE = _scoped(
        """SELECT
                     COUNT(*) as total_messages,
                     SUM(CASE WHEN m.role = 'user' THEN 1 ELSE 0 END) as user_messages,
                     SUM(CASE WHEN m.role = 'assistant' THEN 1 ELSE 0 END) as assistant_messages,
                     SUM(CASE WHEN m.role = 'tool' THEN 1 ELSE 0 END) as tool_messages
                   FROM messages m
                   JOIN sessions s ON s.id = m.session_id
                   WHERE s.started_at >= ?""",
    )
    _GET_MODEL_USAGE_ALL, _GET_MODEL_USAGE_WITH_SOURCE = _scoped(
        "SELECT u.session_id, u.model, u.billing_provider, u.billing_base_url,"
        " u.api_call_count, u.input_tokens, u.output_tokens,"
        " u.cache_read_tokens, u.cache_write_tokens, u.reasoning_tokens,"
        " u.estimated_cost_usd, u.actual_cost_usd, u.cost_status,"
        " u.cost_source, u.billing_mode"
        " FROM session_model_usage u"
        " JOIN sessions s ON s.id = u.session_id"
        " WHERE s.started_at >= ?",
    )
    _PINNED = ("_GET_TOOL_CALLS", "_GET_SKILL_CALLS")
    _SOURCE_SCOPED = ("_GET_SESSIONS", "_GET_TOOL_CALLS", "_GET_SKILL_CALLS", "_GET_TOOL_NAMES",
                      "_GET_MESSAGE_STATS", "_GET_MODEL_USAGE")

    def __init__(self, db):
        self.db = db
        self._conn = db._conn
        try:
            self._has_assistant_calls_index = bool(self._conn.execute(
                "SELECT 1 FROM sqlite_master WHERE type='index' AND name=?", (self._MESSAGES_ASSISTANT_CALLS_INDEX,)).fetchone())
        except sqlite3.Error:
            self._has_assistant_calls_index = False
        if not self._has_assistant_calls_index:
            strip = f" INDEXED BY {self._MESSAGES_ASSISTANT_CALLS_INDEX}"
            for base in self._PINNED:
                for suffix in ("_ALL", "_WITH_SOURCE"):
                    setattr(self, base + suffix, getattr(self, base + suffix).replace(strip, ""))
        # Same read-only caveat: an older writer's DB has no ``created_source`` (and no ``local`` rows).
        try:
            self._local_surfaces = any(row[1] == "created_source" for row in self._conn.execute("PRAGMA table_info(sessions)"))
        except sqlite3.Error:
            self._local_surfaces = False
        if not self._local_surfaces:
            for base in self._SOURCE_SCOPED:
                sql = getattr(self, base + "_WITH_SOURCE")
                for alias in ("", "s."):
                    sql = sql.replace(_LOCAL_SURFACE.format(a=alias), "")
                setattr(self, base + "_WITH_SOURCE", sql)

    def _query(self, base: str, cutoff: float, source: Optional[str]) -> list:
        """Rows of ``<base>_WITH_SOURCE`` or ``<base>_ALL`` (instance attrs, so the unpinned fallback applies)."""
        if not source:
            return self._conn.execute(getattr(self, base + "_ALL"), (cutoff,)).fetchall()
        params = (cutoff, source, source) if self._local_surfaces else (cutoff, source)
        return self._conn.execute(getattr(self, base + "_WITH_SOURCE"), params).fetchall()

    def _get_message_stats(self, cutoff: float, source: str | None = None) -> dict:
        rows = self._query("_GET_MESSAGE_STATS", cutoff, source)
        return dict(rows[0]) if rows else {"total_messages": 0, "user_messages": 0, "assistant_messages": 0, "tool_messages": 0}

    def _get_model_usage(self, cutoff: float, source: str | None = None) -> list[dict]:
        """Per-model usage rows; [] when the table is missing (older DB) so the caller falls back to the per-session aggregate."""
        try:
            return [dict(row) for row in self._query("_GET_MODEL_USAGE", cutoff, source)]
        except sqlite3.OperationalError:
            return []


def _aux_usage_rows(db, cutoff: float) -> list[dict]:
    """The dashboard's auxiliary usage read on a legacy connection."""
    try:
        cur = db._conn.execute("""
            SELECT u.model,
                   u.task,
                   u.billing_provider,
                   SUM(u.input_tokens) as input_tokens,
                   SUM(u.output_tokens) as output_tokens,
                   SUM(u.cache_read_tokens) as cache_read_tokens,
                   SUM(u.reasoning_tokens) as reasoning_tokens,
                   COALESCE(SUM(u.estimated_cost_usd), 0) as estimated_cost,
                   COUNT(DISTINCT u.session_id) as sessions,
                   SUM(COALESCE(u.api_call_count, 0)) as api_calls,
                   MAX(u.last_seen) as last_used_at
            FROM session_model_usage u
            JOIN sessions s ON s.id = u.session_id
            WHERE s.started_at > ? AND u.task != ''
            GROUP BY u.model, u.task, u.billing_provider
            ORDER BY SUM(u.input_tokens) + SUM(u.output_tokens) DESC
        """, (cutoff,))
        return [dict(r) for r in cur.fetchall()]
    except Exception:
        return []


def _seed(path) -> None:
    from hermes_state import SessionDB

    db = SessionDB(db_path=path)
    try:
        for sid, source in (("cli-1", "cli"), ("loc-1", "local"), ("tg-1", "telegram")):
            db.create_session(sid, source, model="m-a")
            db.append_message(sid, "user", content=f"hello from {sid}")
            db.append_message(sid, "assistant", content="", tool_calls=_CALLS)
            db.append_message(sid, "tool", content="done", tool_name="skill_view", tool_call_id="k1")
        db.record_auxiliary_usage("cli-1", "vision", model="m-v", billing_provider="p-v", input_tokens=5)
    finally:
        db.close()
    with sqlite3.connect(path) as raw:
        raw.execute("UPDATE sessions SET created_source = 'cli' WHERE id = 'loc-1'")


def _regress(path) -> None:
    """Drop what an older writer never created; objects naming ``created_source`` go first so the column can."""
    with sqlite3.connect(path) as raw:
        raw.execute("DROP INDEX idx_messages_assistant_calls_by_session")
        raw.execute("DROP TABLE session_model_usage")
        for kind, name in raw.execute(
                "SELECT type, name FROM sqlite_master WHERE sql LIKE '%created_source%' AND name != 'sessions'").fetchall():
            raw.execute(f"DROP {kind.upper()} {name}")
        raw.execute("ALTER TABLE sessions DROP COLUMN created_source")


@pytest.mark.parametrize("older", [False, True], ids=["current", "older"])
def test_reads_equal_the_legacy_reads(scope_a, tmp_path, legacy_store, older):
    from hermes_state import SessionDB

    profile_home(scope_a.name)
    path = store_for(scope_a)
    _seed(path)
    if older:
        _regress(path)
    oracle_path = tmp_path / "oracle.db"
    with sqlite3.connect(path) as source, sqlite3.connect(oracle_path) as target:
        source.backup(target)
    reader = SQLiteBackend({scope_a.hermes_home_key: path}, read_only=True)
    oracle = SessionDB(db_path=oracle_path, read_only=True)
    try:
        store = reader.open(scope_a)
        analytics = store.usage_analytics
        engine = _M2Insights(oracle)
        assert engine._has_assistant_calls_index is (not older)
        assert engine._local_surfaces is (not older)
        # Main's probe of the column, on the legacy reader.
        main_has_column = any(row[1] == "created_source" for row in oracle._read_all("PRAGMA table_info(sessions)"))
        assert store.sessions.records_created_surface() is main_has_column
        for source in (None, "cli", "telegram"):
            for verb, base in _INSIGHTS:
                expected = [dict(row) for row in engine._query(base, 0.0, source)]
                assert getattr(analytics, verb)(started_since=0.0, source=source) == expected, (verb, source)
            assert analytics.insight_message_stats(started_since=0.0, source=source) == engine._get_message_stats(0.0, source)
            assert analytics.insight_model_usage(started_since=0.0, source=source) == engine._get_model_usage(0.0, source)
        assert analytics.auxiliary_usage_by_model_task(started_after=0.0) == _aux_usage_rows(oracle, 0.0)
        if older:
            assert analytics.call_usage_by_route(started_after=0.0) == []
    finally:
        oracle.close()
        reader.close()
