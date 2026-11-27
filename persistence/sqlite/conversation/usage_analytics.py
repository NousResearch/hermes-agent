"""``UsageAnalytics`` on SQLite: the dashboard's usage and model reads (hermes_cli/web_routers/analytics.py,
hermes_cli/web_server_profiles.py) and the insights reads (agent/insights.py), one statement each, rows as fresh
dicts."""

from __future__ import annotations

import sqlite3
from typing import Any, Optional

from persistence.core.identity import JsonValue
from sqlite_support import schema_not_built

from .._backend import SQLiteStore, register_repository
from ._errors import refusals

__all__ = ["SQLiteUsageAnalytics"]


# A source filter also matches the gateway's ``local`` rows that surface created: interactive local
# sessions are stored as ``local`` with the surface kept in ``created_source``.
_LOCAL_SURFACE = " OR ({a}source = 'local' AND {a}created_source = ?)"


def _scoped(before: str, after: str = "", *, alias: str = "s.") -> tuple[str, str]:
    """(unfiltered, source-filtered) query pair sharing one body."""
    src = f" AND ({alias}source = ?{_LOCAL_SURFACE.format(a=alias)})"
    return before + after, before + src + after


class _InsightQueries:
    """The insights statements of one store. ``INDEXED BY`` pins the partial index so the plan is deterministic on a
    fresh store; a store without the index (an older writer's file opened read-only) runs the unpinned variants, and
    one without ``created_source`` filters on the raw source only."""

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

    def __init__(self, conn: sqlite3.Connection) -> None:
        try:
            self._has_assistant_calls_index = bool(conn.execute(
                "SELECT 1 FROM sqlite_master WHERE type='index' AND name=?", (self._MESSAGES_ASSISTANT_CALLS_INDEX,)).fetchone())
        except sqlite3.Error:
            self._has_assistant_calls_index = False
        if not self._has_assistant_calls_index:
            strip = f" INDEXED BY {self._MESSAGES_ASSISTANT_CALLS_INDEX}"
            for base in self._PINNED:
                for suffix in ("_ALL", "_WITH_SOURCE"):
                    setattr(self, base + suffix, getattr(self, base + suffix).replace(strip, ""))
        try:
            self._local_surfaces = any(row[1] == "created_source" for row in conn.execute("PRAGMA table_info(sessions)"))
        except sqlite3.Error:
            self._local_surfaces = False
        if not self._local_surfaces:
            for base in self._SOURCE_SCOPED:
                sql = getattr(self, base + "_WITH_SOURCE")
                for alias in ("", "s."):
                    sql = sql.replace(_LOCAL_SURFACE.format(a=alias), "")
                setattr(self, base + "_WITH_SOURCE", sql)

    def _query(self, conn: sqlite3.Connection, base: str, cutoff: float, source: Optional[str]) -> list:
        """Rows of ``<base>_WITH_SOURCE`` or ``<base>_ALL`` (instance attributes, so the unpinned fallback applies)."""
        if not source:
            return conn.execute(getattr(self, base + "_ALL"), (cutoff,)).fetchall()
        params = (cutoff, source, source) if self._local_surfaces else (cutoff, source)
        return conn.execute(getattr(self, base + "_WITH_SOURCE"), params).fetchall()


class SQLiteUsageAnalytics:
    """Usage analytics of the bound profile's ``state.db``."""

    def __init__(self, store: SQLiteStore) -> None:
        self._store = store
        self._queries: Optional[_InsightQueries] = None

    def _insights(self, conn: sqlite3.Connection) -> _InsightQueries:
        """The insights statements; a store found without the pinned index or ``created_source`` is probed again on
        the next read, since another connection may add either."""
        queries = self._queries
        if queries is None or not (queries._has_assistant_calls_index and queries._local_surfaces):
            queries = self._queries = _InsightQueries(conn)
        return queries

    @staticmethod
    def _insight_wait(holder) -> float:
        """Seconds an insights read waits on a lock: the writer connection's 1 s on a writer store, the read-only
        connection's own budget on a read-only one."""
        return holder._read_busy_timeout_s if holder.read_only else 1.0

    def _insight_rows(self, base: str, started_since: float, source: Optional[str]) -> list[dict[str, Any]]:
        holder = self._store.holder()
        with refusals(holder), holder._read_ctx_with_timeout(self._insight_wait(holder)) as conn:
            return [dict(row) for row in self._insights(conn)._query(conn, base, started_since, source)]

    # ── dashboard reads ───────────────────────────────────────────────────────

    def daily_usage(self, *, started_after: float) -> list[dict[str, JsonValue]]:
        holder = self._store.holder()
        with refusals(holder), holder._read_ctx() as conn:
            return [dict(r) for r in conn.execute("""
            SELECT date(started_at, 'unixepoch', 'localtime') as day,
                   SUM(input_tokens) as input_tokens,
                   SUM(output_tokens) as output_tokens,
                   SUM(cache_read_tokens) as cache_read_tokens,
                   SUM(reasoning_tokens) as reasoning_tokens,
                   COALESCE(SUM(estimated_cost_usd), 0) as estimated_cost,
                   COALESCE(SUM(actual_cost_usd), 0) as actual_cost,
                   COUNT(*) as sessions,
                   SUM(COALESCE(api_call_count, 0)) as api_calls
            FROM sessions WHERE started_at > ?
            GROUP BY day ORDER BY day
        """, (started_after,)).fetchall()]

    def session_usage_by_model(self, *, started_after: float) -> list[dict[str, JsonValue]]:
        holder = self._store.holder()
        with refusals(holder), holder._read_ctx() as conn:
            return [dict(r) for r in conn.execute("""
            SELECT model,
                   SUM(input_tokens) as input_tokens,
                   SUM(output_tokens) as output_tokens,
                   COALESCE(SUM(estimated_cost_usd), 0) as estimated_cost,
                   COUNT(*) as sessions,
                   SUM(COALESCE(api_call_count, 0)) as api_calls
            FROM sessions WHERE started_at > ? AND model IS NOT NULL
            GROUP BY model ORDER BY SUM(input_tokens) + SUM(output_tokens) DESC
        """, (started_after,)).fetchall()]

    def session_usage_totals(self, *, started_after: float, model_rows_only: bool = False) -> dict[str, JsonValue]:
        holder = self._store.holder()
        with refusals(holder), holder._read_ctx() as conn:
            if not model_rows_only:
                return dict(conn.execute("""
            SELECT SUM(input_tokens) as total_input,
                   SUM(output_tokens) as total_output,
                   SUM(cache_read_tokens) as total_cache_read,
                   SUM(reasoning_tokens) as total_reasoning,
                   COALESCE(SUM(estimated_cost_usd), 0) as total_estimated_cost,
                   COALESCE(SUM(actual_cost_usd), 0) as total_actual_cost,
                   COUNT(*) as total_sessions,
                   SUM(COALESCE(api_call_count, 0)) as total_api_calls
            FROM sessions WHERE started_at > ?
        """, (started_after,)).fetchone())
            return dict(conn.execute("""
            SELECT SUM(input_tokens) as total_input,
                   SUM(output_tokens) as total_output,
                   SUM(cache_read_tokens) as total_cache_read,
                   SUM(reasoning_tokens) as total_reasoning,
                   COALESCE(SUM(estimated_cost_usd), 0) as total_estimated_cost,
                   COALESCE(SUM(actual_cost_usd), 0) as total_actual_cost,
                   COUNT(*) as total_sessions,
                   SUM(COALESCE(api_call_count, 0)) as total_api_calls
            FROM sessions WHERE started_at > ? AND model IS NOT NULL AND model != ''
        """, (started_after,)).fetchone())

    def session_usage_by_route(self, *, started_after: float) -> list[dict[str, JsonValue]]:
        holder = self._store.holder()
        with refusals(holder), holder._read_ctx() as conn:
            return [dict(r) for r in conn.execute("""
                SELECT model,
                       billing_provider,
                       SUM(input_tokens) as input_tokens,
                       SUM(output_tokens) as output_tokens,
                       SUM(cache_read_tokens) as cache_read_tokens,
                       SUM(reasoning_tokens) as reasoning_tokens,
                       COALESCE(SUM(estimated_cost_usd), 0) as estimated_cost,
                       COALESCE(SUM(actual_cost_usd), 0) as actual_cost,
                       COUNT(*) as sessions,
                       SUM(COALESCE(api_call_count, 0)) as api_calls,
                       SUM(tool_call_count) as tool_calls,
                       MAX(started_at) as last_used_at,
                       AVG(input_tokens + output_tokens) as avg_tokens_per_session
                FROM sessions WHERE started_at > ? AND model IS NOT NULL AND model != ''
                GROUP BY model, billing_provider
                ORDER BY SUM(input_tokens) + SUM(output_tokens) DESC
            """, (started_after,)).fetchall()]

    def session_tool_calls_by_route(self, *, started_after: float) -> list[dict[str, JsonValue]]:
        holder = self._store.holder()
        with refusals(holder), holder._read_ctx() as conn:
            return [dict(r) for r in conn.execute("""
        SELECT model, billing_provider, SUM(COALESCE(tool_call_count, 0)) as tool_calls
        FROM sessions WHERE started_at > ? AND model IS NOT NULL AND model != ''
        GROUP BY model, billing_provider
    """, (started_after,)).fetchall()]

    def call_usage_by_route(self, *, started_after: float) -> list[dict[str, JsonValue]]:
        holder = self._store.holder()
        with refusals(holder), holder._read_ctx() as conn:
            try:
                cur = conn.execute("""
                SELECT u.model,
                       u.billing_provider,
                       SUM(u.input_tokens) as input_tokens,
                       SUM(u.output_tokens) as output_tokens,
                       SUM(u.cache_read_tokens) as cache_read_tokens,
                       SUM(u.reasoning_tokens) as reasoning_tokens,
                       COALESCE(SUM(u.estimated_cost_usd), 0) as estimated_cost,
                       COALESCE(SUM(u.actual_cost_usd), 0) as actual_cost,
                       COUNT(DISTINCT u.session_id) as sessions,
                       SUM(COALESCE(u.api_call_count, 0)) as api_calls,
                       MAX(u.last_seen) as last_used_at,
                       AVG(u.input_tokens + u.output_tokens) as avg_tokens_per_session
                FROM session_model_usage u
                JOIN sessions s ON s.id = u.session_id
                WHERE s.started_at > ? AND u.model IS NOT NULL AND u.model != ''
                      AND u.task = ''
                GROUP BY u.model, u.billing_provider
                ORDER BY SUM(u.input_tokens) + SUM(u.output_tokens) DESC
            """, (started_after,))
                return [dict(r) for r in cur.fetchall()]
            except sqlite3.OperationalError as exc:
                if not schema_not_built(exc):
                    raise
                return []  # a store from before the per-call usage rows

    def auxiliary_usage_by_model_task(self, *, started_after: float) -> list[dict[str, JsonValue]]:
        holder = self._store.holder()
        with refusals(holder), holder._read_ctx() as conn:
            try:
                cur = conn.execute("""
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
        """, (started_after,))
                return [dict(r) for r in cur.fetchall()]
            except sqlite3.OperationalError as exc:
                if not schema_not_built(exc):
                    raise
                return []  # a store from before the task dimension

    # ── insights reads ────────────────────────────────────────────────────────

    def insight_sessions(self, *, started_since: float, source: Optional[str] = None) -> list[dict[str, JsonValue]]:
        return self._insight_rows("_GET_SESSIONS", started_since, source)

    def insight_tool_result_counts(self, *, started_since: float, source: Optional[str] = None) -> list[dict[str, JsonValue]]:
        return self._insight_rows("_GET_TOOL_NAMES", started_since, source)

    def insight_tool_call_payloads(self, *, started_since: float, source: Optional[str] = None) -> list[dict[str, JsonValue]]:
        return self._insight_rows("_GET_TOOL_CALLS", started_since, source)

    def insight_skill_call_payloads(self, *, started_since: float, source: Optional[str] = None) -> list[dict[str, JsonValue]]:
        return self._insight_rows("_GET_SKILL_CALLS", started_since, source)

    def insight_message_stats(self, *, started_since: float, source: Optional[str] = None) -> dict[str, JsonValue]:
        return self._insight_rows("_GET_MESSAGE_STATS", started_since, source)[0]

    def insight_model_usage(self, *, started_since: float, source: Optional[str] = None) -> list[dict[str, JsonValue]]:
        holder = self._store.holder()
        with refusals(holder), holder._read_ctx_with_timeout(self._insight_wait(holder)) as conn:
            try:
                return [dict(row) for row in self._insights(conn)._query(conn, "_GET_MODEL_USAGE", started_since, source)]
            except sqlite3.OperationalError as exc:
                if not schema_not_built(exc):
                    raise
                return []  # a store from before the per-call usage rows


register_repository("usage_analytics", SQLiteUsageAnalytics)
