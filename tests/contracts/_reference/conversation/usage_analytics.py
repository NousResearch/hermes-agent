"""In-memory reference ``UsageAnalytics`` (test fixture).

The reads aggregate the other references' tables the way the store's SQL aggregates its own: session rows through the sessions reference's ``_record`` (which composes the usage reference's counters and the transcript counters), transcript rows of every visibility from the messages reference, and the usage reference's attribution rows (``routes``, ``route_usage``, ``route_costs`` and ``route_last_write``). A ``message_count`` or ``tool_call_count`` cell written on a session row directly (the row-dict seam: a damaged or legacy cell, NULL included) reads as stored.
"""

from __future__ import annotations

import json
import time
from typing import Any, Callable, Hashable, Iterable, Iterator, Optional

from persistence.conversation.usage import TokenCounts

from ..backend import InMemoryBackend, register_repository
from .transcript_store import _StoredRow, transcripts_of

_Row = dict[str, Any]

_SESSION_COLUMNS = (
    "id", "source", "model", "started_at", "ended_at", "message_count", "tool_call_count", "input_tokens", "output_tokens",
    "cache_read_tokens", "cache_write_tokens", "billing_provider", "billing_base_url", "billing_mode", "estimated_cost_usd",
    "actual_cost_usd", "cost_status", "cost_source", "api_call_count",
)
# The transcript counters a cell written on the session row overrides.
_ROW_COUNTERS = ("message_count", "tool_call_count")
_TOKENS = ("input_tokens", "output_tokens", "cache_read_tokens", "cache_write_tokens", "reasoning_tokens")


def _sum(values: Iterable[Any]) -> Any:
    """``SUM``: ``None`` when no value is set."""
    present = [value for value in values if value is not None]
    return sum(present) if present else None


def _sum_or_zero(values: Iterable[Any]) -> Any:
    """``COALESCE(SUM(...), 0)``."""
    total = _sum(values)
    return 0 if total is None else total


def _sum_unset_as_zero(values: Iterable[Any]) -> Any:
    """``SUM(COALESCE(..., 0))``: ``None`` over no rows only."""
    return _sum(0 if value is None else value for value in values)


def _max(values: Iterable[Any]) -> Any:
    present = [value for value in values if value is not None]
    return max(present) if present else None


def _avg(values: Iterable[Any]) -> Optional[float]:
    present = [value for value in values if value is not None]
    return sum(present) / len(present) if present else None


def _tokens_of(row: _Row) -> Any:
    """``input_tokens + output_tokens`` of one row; ``None`` when either is unset."""
    if row["input_tokens"] is None or row["output_tokens"] is None:
        return None
    return row["input_tokens"] + row["output_tokens"]


def _group(rows: Iterable[_Row], key: Callable[[_Row], Hashable]) -> dict[Hashable, list[_Row]]:
    groups: dict[Hashable, list[_Row]] = {}
    for row in rows:
        groups.setdefault(key(row), []).append(row)
    return groups


def _by_volume(rows: list[_Row]) -> list[_Row]:
    """``ORDER BY SUM(input_tokens) + SUM(output_tokens) DESC`` over grouped rows carrying those sums: an unset volume sorts last, ties in no defined order."""
    def volume(row: _Row) -> tuple[int, int]:
        total = _tokens_of(row)
        return (0, 0) if total is None else (1, total)

    return sorted(rows, key=volume, reverse=True)


class InMemoryUsageAnalytics:
    """Reference implementation of the ``UsageAnalytics`` protocol."""

    def __init__(self, backend: InMemoryBackend) -> None:
        self._backend = backend

    # ── stored rows ───────────────────────────────────────────────────────────

    def _sessions(self) -> list[_Row]:
        """Every session row with the columns the reads use, as stored."""
        scope = self._backend.scope
        rows: list[_Row] = []
        for stored in self._backend.scope_store(scope).get("sessions", {}).values():
            record = self._backend.sessions._record(scope, stored, resolve_tool_pin=False)
            row = {name: getattr(record, name) for name in _SESSION_COLUMNS[1:]}
            row.update(id=str(record.session_id), created_source=record.created_source, reasoning_tokens=record.reasoning_tokens)
            row.update({name: stored[name] for name in _ROW_COUNTERS if name in stored})
            rows.append(row)
        return rows

    def _started_after(self, started_after: float) -> list[_Row]:
        return [row for row in self._sessions() if row["started_at"] > started_after]

    def _started_since(self, started_since: float, source: Optional[str]) -> list[_Row]:
        """The insights window: started at or after ``started_since``; with a source, that source or a ``local`` row created by it."""
        return [
            row for row in self._sessions()
            if row["started_at"] >= started_since
            and (not source or row["source"] == source or (row["source"] == "local" and row["created_source"] == source))
        ]

    def _messages(self, sessions: list[_Row]) -> Iterator[tuple[str, _StoredRow]]:
        """Every stored transcript row of ``sessions``, any visibility."""
        transcripts = transcripts_of(self._backend, self._backend.scope)
        for session in sessions:
            transcript = transcripts.get(session["id"])
            for stored in transcript.rows if transcript is not None else ():
                yield session["id"], stored

    def _calls(self, sessions: list[_Row]) -> list[_Row]:
        """The attribution rows of ``sessions`` as per-call usage rows (unset route fields ``''``, ``last_seen`` the latest write)."""
        tables = self._backend.scope_store(self._backend.scope).get("usage", {})
        ids = {session["id"] for session in sessions}
        rows: list[_Row] = []
        for (session_id, key), route in tables.get("routes", {}).items():
            if session_id not in ids:
                continue
            task, model, provider, base_url, mode = key
            tokens, estimated = tables["route_usage"].get((session_id, key), (TokenCounts(), 0.0))
            rows.append({
                "session_id": session_id, "model": model, "billing_provider": provider or "", "billing_base_url": base_url or "",
                "billing_mode": mode or "", "task": task or "", "api_call_count": route.api_call_count,
                **{name: getattr(tokens, name) for name in _TOKENS},
                "estimated_cost_usd": estimated,
                **dict(zip(("actual_cost_usd", "cost_status", "cost_source"), tables.get("route_costs", {}).get((session_id, key), (0.0, None, None)))),
                "last_seen": tables["route_last_write"].get((session_id, key)),
            })
        return rows

    # ── dashboard reads ───────────────────────────────────────────────────────

    def daily_usage(self, *, started_after: float) -> list[dict[str, Any]]:
        days = _group(self._started_after(started_after), lambda row: time.strftime("%Y-%m-%d", time.localtime(row["started_at"])))
        return [
            {
                "day": day,
                **{name: _sum(row[name] for row in rows) for name in ("input_tokens", "output_tokens", "cache_read_tokens", "reasoning_tokens")},
                "estimated_cost": _sum_or_zero(row["estimated_cost_usd"] for row in rows),
                "actual_cost": _sum_or_zero(row["actual_cost_usd"] for row in rows),
                "sessions": len(rows),
                "api_calls": _sum_unset_as_zero(row["api_call_count"] for row in rows),
            }
            for day, rows in sorted(days.items())
        ]

    def session_usage_by_model(self, *, started_after: float) -> list[dict[str, Any]]:
        models = _group((row for row in self._started_after(started_after) if row["model"] is not None), lambda row: row["model"])
        return _by_volume([
            {
                "model": model,
                "input_tokens": _sum(row["input_tokens"] for row in rows),
                "output_tokens": _sum(row["output_tokens"] for row in rows),
                "estimated_cost": _sum_or_zero(row["estimated_cost_usd"] for row in rows),
                "sessions": len(rows),
                "api_calls": _sum_unset_as_zero(row["api_call_count"] for row in rows),
            }
            for model, rows in models.items()
        ])

    def session_usage_totals(self, *, started_after: float, model_rows_only: bool = False) -> dict[str, Any]:
        rows = [row for row in self._started_after(started_after) if row["model"] or not model_rows_only]
        return {
            "total_input": _sum(row["input_tokens"] for row in rows),
            "total_output": _sum(row["output_tokens"] for row in rows),
            "total_cache_read": _sum(row["cache_read_tokens"] for row in rows),
            "total_reasoning": _sum(row["reasoning_tokens"] for row in rows),
            "total_estimated_cost": _sum_or_zero(row["estimated_cost_usd"] for row in rows),
            "total_actual_cost": _sum_or_zero(row["actual_cost_usd"] for row in rows),
            "total_sessions": len(rows),
            "total_api_calls": _sum_unset_as_zero(row["api_call_count"] for row in rows),
        }

    def _session_routes(self, started_after: float) -> dict[Hashable, list[_Row]]:
        """The window's sessions with a non-empty model, grouped by (model, billing provider)."""
        return _group((row for row in self._started_after(started_after) if row["model"]), lambda row: (row["model"], row["billing_provider"]))

    def session_usage_by_route(self, *, started_after: float) -> list[dict[str, Any]]:
        return _by_volume([
            {
                "model": model,
                "billing_provider": provider,
                **{name: _sum(row[name] for row in rows) for name in ("input_tokens", "output_tokens", "cache_read_tokens", "reasoning_tokens")},
                "estimated_cost": _sum_or_zero(row["estimated_cost_usd"] for row in rows),
                "actual_cost": _sum_or_zero(row["actual_cost_usd"] for row in rows),
                "sessions": len(rows),
                "api_calls": _sum_unset_as_zero(row["api_call_count"] for row in rows),
                "tool_calls": _sum(row["tool_call_count"] for row in rows),
                "last_used_at": _max(row["started_at"] for row in rows),
                "avg_tokens_per_session": _avg(_tokens_of(row) for row in rows),
            }
            for (model, provider), rows in self._session_routes(started_after).items()
        ])

    def session_tool_calls_by_route(self, *, started_after: float) -> list[dict[str, Any]]:
        return [
            {"model": model, "billing_provider": provider, "tool_calls": _sum_unset_as_zero(row["tool_call_count"] for row in rows)}
            for (model, provider), rows in self._session_routes(started_after).items()
        ]

    def call_usage_by_route(self, *, started_after: float) -> list[dict[str, Any]]:
        calls = [row for row in self._calls(self._started_after(started_after)) if not row["task"]]
        return _by_volume([
            {
                "model": model,
                "billing_provider": provider,
                **{name: _sum(row[name] for row in rows) for name in ("input_tokens", "output_tokens", "cache_read_tokens", "reasoning_tokens")},
                "estimated_cost": _sum_or_zero(row["estimated_cost_usd"] for row in rows),
                "actual_cost": _sum_or_zero(row["actual_cost_usd"] for row in rows),
                "sessions": len({row["session_id"] for row in rows}),
                "api_calls": _sum_unset_as_zero(row["api_call_count"] for row in rows),
                "last_used_at": _max(row["last_seen"] for row in rows),
                "avg_tokens_per_session": _avg(_tokens_of(row) for row in rows),
            }
            for (model, provider), rows in _group(calls, lambda row: (row["model"], row["billing_provider"])).items()
        ])

    def auxiliary_usage_by_model_task(self, *, started_after: float) -> list[dict[str, Any]]:
        calls = [row for row in self._calls(self._started_after(started_after)) if row["task"]]
        return _by_volume([
            {
                "model": model,
                "task": task,
                "billing_provider": provider,
                **{name: _sum(row[name] for row in rows) for name in ("input_tokens", "output_tokens", "cache_read_tokens", "reasoning_tokens")},
                "estimated_cost": _sum_or_zero(row["estimated_cost_usd"] for row in rows),
                "sessions": len({row["session_id"] for row in rows}),
                "api_calls": _sum_unset_as_zero(row["api_call_count"] for row in rows),
                "last_used_at": _max(row["last_seen"] for row in rows),
            }
            for (model, task, provider), rows in _group(calls, lambda row: (row["model"], row["task"], row["billing_provider"])).items()
        ])

    # ── insights reads ────────────────────────────────────────────────────────

    def insight_sessions(self, *, started_since: float, source: Optional[str] = None) -> list[dict[str, Any]]:
        rows = sorted(self._started_since(started_since, source), key=lambda row: row["started_at"], reverse=True)
        return [{name: row[name] for name in _SESSION_COLUMNS} for row in rows]

    def insight_tool_result_counts(self, *, started_since: float, source: Optional[str] = None) -> list[dict[str, Any]]:
        counts: dict[tuple[str, str], int] = {}
        for session_id, stored in self._messages(self._started_since(started_since, source)):
            if stored.message.role == "tool" and stored.message.tool_name is not None:
                key = (session_id, stored.message.tool_name)
                counts[key] = counts.get(key, 0) + 1
        return [{"session_id": session_id, "tool_name": tool_name, "count": count} for (session_id, tool_name), count in counts.items()]

    def _tool_call_rows(self, started_since: float, source: Optional[str]) -> Iterator[tuple[str, _StoredRow, str]]:
        """Assistant rows with tool calls and their stored ``tool_calls`` text (the JSON array as the store encodes it)."""
        for session_id, stored in self._messages(self._started_since(started_since, source)):
            if stored.message.role == "assistant" and stored.message.tool_calls:
                yield session_id, stored, json.dumps(list(stored.message.tool_calls))

    def insight_tool_call_payloads(self, *, started_since: float, source: Optional[str] = None) -> list[dict[str, Any]]:
        return [{"session_id": session_id, "tool_calls": text} for session_id, _, text in self._tool_call_rows(started_since, source)]

    def insight_skill_call_payloads(self, *, started_since: float, source: Optional[str] = None) -> list[dict[str, Any]]:
        return [
            {"tool_calls": text, "timestamp": stored.epoch}
            for _, stored, text in self._tool_call_rows(started_since, source)
            if "skill_view" in text or "skill_manage" in text
        ]

    def insight_message_stats(self, *, started_since: float, source: Optional[str] = None) -> dict[str, Any]:
        roles = [stored.message.role for _, stored in self._messages(self._started_since(started_since, source))]
        return {
            "total_messages": len(roles),
            "user_messages": _sum(int(role == "user") for role in roles),
            "assistant_messages": _sum(int(role == "assistant") for role in roles),
            "tool_messages": _sum(int(role == "tool") for role in roles),
        }

    def insight_model_usage(self, *, started_since: float, source: Optional[str] = None) -> list[dict[str, Any]]:
        columns = (
            "session_id", "model", "billing_provider", "billing_base_url", "api_call_count", *_TOKENS,
            "estimated_cost_usd", "actual_cost_usd", "cost_status", "cost_source", "billing_mode",
        )
        return [{name: row[name] for name in columns} for row in self._calls(self._started_since(started_since, source))]


register_repository("usage_analytics", InMemoryUsageAnalytics)
