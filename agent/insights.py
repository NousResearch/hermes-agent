"""
Session Insights Engine for Hermes Agent.

Analyzes historical session data from the SQLite state database to produce
comprehensive usage insights — token consumption, cost estimates, tool usage
patterns, activity trends, model/platform breakdowns, and session metrics.

Inspired by Claude Code's /insights command, adapted for Hermes Agent's
multi-platform architecture with additional cost estimation and platform
breakdown capabilities.

Data sources:
1. Session DB (SQLite, hermes_state.py): per-session token, tool, message,
   and model metadata.
2. Skill-usage telemetry JSONL (SIE AC#8): the two-phase dispatch log at
   ~/hermes-workspace/Lex-Workspace/sie/analytics/skill-usage.jsonl. Init
   and complete rows are reconciled by dispatch_id; the resulting
   skill_usage section is independent from and additive to the session-DB
   path. Missing or unreadable telemetry degrades gracefully (the
   skill_usage section reports available=False with a reason; the rest of
   the report still renders).

Usage:
    from agent.insights import InsightsEngine
    engine = InsightsEngine(db)
    report = engine.generate(days=30)
    print(engine.format_terminal(report))
"""

import json
import os
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from agent.usage_pricing import (
    CanonicalUsage,
    DEFAULT_PRICING,
    estimate_usage_cost,
    format_duration_compact,
    has_known_pricing,
)

_DEFAULT_PRICING = DEFAULT_PRICING

_DEFAULT_SKILL_USAGE_PATH = "~/hermes-workspace/Lex-Workspace/sie/analytics/skill-usage.jsonl"
_UNJOINED_INFLIGHT_THRESHOLD_S = 30 * 60  # 30 minutes


def _parse_iso8601(value: Any) -> Optional[float]:
    """Parse an ISO-8601 timestamp string into a Unix epoch seconds float.

    Accepts trailing 'Z' (UTC) and standard offsets. Returns None on any
    malformed input (caller decides whether the row is keepable).
    """
    if not isinstance(value, str) or not value:
        return None
    raw = value.strip()
    if raw.endswith("Z"):
        raw = raw[:-1] + "+00:00"
    try:
        dt = datetime.fromisoformat(raw)
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.timestamp()


class _SkillUsageAggregator:
    """Reads the SIE two-phase skill-usage JSONL log and aggregates it.

    The log is append-only with two row phases per dispatch:
      - phase=init     written at dispatch start
      - phase=complete written at dispatch end, carries outcome + duration_s

    Rows are reconciled by dispatch_id. A complete row with no matching init
    is still counted (init may have been written before the window). An init
    row with no matching complete row, older than the unjoined threshold
    (default 30 min), counts as unjoined_in_flight (stuck dispatch signal).

    The aggregator never raises on malformed input: missing files, decode
    errors, and schema drift all degrade gracefully and surface via the
    available/reason fields.
    """

    def __init__(self, path: str):
        self.path = path

    def aggregate(self, days: int, now_ts: Optional[float] = None) -> Dict[str, Any]:
        now_ts = now_ts if now_ts is not None else time.time()
        window_start = now_ts - (days * 86400)
        window_end = now_ts

        try:
            if not os.path.exists(self.path):
                return self._unavailable("file not found", days, window_start, window_end)
            with open(self.path, "r", encoding="utf-8") as fh:
                rows = self._parse_rows(fh)
        except OSError as exc:
            return self._unavailable(f"unreadable: {exc}", days, window_start, window_end)

        # Reconcile init + complete by dispatch_id.
        # Keep all complete rows in window. Track init rows separately for
        # unjoined detection.
        completes: List[Dict[str, Any]] = []
        inits_by_dispatch: Dict[str, Dict[str, Any]] = {}
        completed_dispatch_ids: set = set()

        for row in rows:
            phase = row.get("phase")
            dispatch_id = row.get("dispatch_id") or ""
            ts_epoch = _parse_iso8601(row.get("ts"))
            if ts_epoch is None:
                continue
            if phase == "complete":
                if ts_epoch < window_start or ts_epoch > window_end:
                    continue
                completes.append(row)
                if dispatch_id:
                    completed_dispatch_ids.add(dispatch_id)
            elif phase == "init":
                # We need init rows regardless of window for unjoined
                # detection (an old init with no complete is the signal),
                # but the unjoined window itself is bounded by `days`.
                if ts_epoch < window_start or ts_epoch > window_end:
                    continue
                if dispatch_id:
                    inits_by_dispatch[dispatch_id] = row

        unjoined = 0
        for dispatch_id, init_row in inits_by_dispatch.items():
            if dispatch_id in completed_dispatch_ids:
                continue
            started_ts = _parse_iso8601(init_row.get("started_at") or init_row.get("ts"))
            if started_ts is None:
                continue
            if (now_ts - started_ts) >= _UNJOINED_INFLIGHT_THRESHOLD_S:
                unjoined += 1

        by_skill = self._aggregate_by_skill(completes)
        by_agent = self._aggregate_by_agent(completes)
        top_failure_skills = self._top_failure_skills(by_skill, limit=5)

        return {
            "available": True,
            "total_dispatches": len(completes),
            "by_skill": by_skill,
            "by_agent": by_agent,
            "top_failure_skills": top_failure_skills,
            "unjoined_in_flight": unjoined,
            "window_start_iso": _epoch_to_iso(window_start),
            "window_end_iso": _epoch_to_iso(window_end),
            "source_path": self.path,
        }

    @staticmethod
    def _parse_rows(fh) -> List[Dict[str, Any]]:
        rows = []
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except (json.JSONDecodeError, ValueError):
                continue
            if isinstance(obj, dict):
                rows.append(obj)
        return rows

    @staticmethod
    def _aggregate_by_skill(completes: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        groups: Dict[str, Dict[str, Any]] = {}
        for row in completes:
            skill = row.get("skill") or "unknown"
            outcome = (row.get("outcome") or "").lower()
            duration = row.get("duration_s")
            g = groups.setdefault(skill, {
                "skill": skill, "total": 0, "success": 0, "failure": 0,
                "error": 0, "in_flight": 0, "_duration_sum": 0.0,
                "_duration_n": 0,
            })
            g["total"] += 1
            if outcome == "success":
                g["success"] += 1
            elif outcome == "failure":
                g["failure"] += 1
            elif outcome == "error":
                g["error"] += 1
            elif outcome == "in_flight":
                g["in_flight"] += 1
            if isinstance(duration, (int, float)) and duration >= 0:
                g["_duration_sum"] += float(duration)
                g["_duration_n"] += 1
        out = []
        for g in groups.values():
            total = g["total"]
            scored = g["success"] + g["failure"] + g["error"]
            success_rate = (g["success"] / scored) if scored else 0.0
            avg_duration = (g["_duration_sum"] / g["_duration_n"]) if g["_duration_n"] else 0.0
            out.append({
                "skill": g["skill"],
                "total": total,
                "success": g["success"],
                "failure": g["failure"],
                "error": g["error"],
                "in_flight": g["in_flight"],
                "success_rate": round(success_rate, 4),
                "avg_duration_s": round(avg_duration, 3),
            })
        out.sort(key=lambda r: r["total"], reverse=True)
        return out

    @staticmethod
    def _aggregate_by_agent(completes: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        groups: Dict[str, Dict[str, Any]] = {}
        for row in completes:
            agent = row.get("agent") or "unknown"
            outcome = (row.get("outcome") or "").lower()
            duration = row.get("duration_s")
            g = groups.setdefault(agent, {
                "agent": agent, "total": 0, "success": 0, "scored": 0,
                "_duration_sum": 0.0, "_duration_n": 0,
            })
            g["total"] += 1
            if outcome in ("success", "failure", "error"):
                g["scored"] += 1
                if outcome == "success":
                    g["success"] += 1
            if isinstance(duration, (int, float)) and duration >= 0:
                g["_duration_sum"] += float(duration)
                g["_duration_n"] += 1
        out = []
        for g in groups.values():
            success_rate = (g["success"] / g["scored"]) if g["scored"] else 0.0
            avg_duration = (g["_duration_sum"] / g["_duration_n"]) if g["_duration_n"] else 0.0
            out.append({
                "agent": g["agent"],
                "total": g["total"],
                "success_rate": round(success_rate, 4),
                "avg_duration_s": round(avg_duration, 3),
            })
        out.sort(key=lambda r: r["total"], reverse=True)
        return out

    @staticmethod
    def _top_failure_skills(by_skill: List[Dict[str, Any]], limit: int) -> List[Dict[str, Any]]:
        failing = [
            {"skill": r["skill"], "failure_count": r["failure"] + r["error"]}
            for r in by_skill
            if (r["failure"] + r["error"]) > 0
        ]
        failing.sort(key=lambda r: r["failure_count"], reverse=True)
        return failing[:limit]

    @staticmethod
    def _unavailable(reason: str, days: int, window_start: float, window_end: float) -> Dict[str, Any]:
        return {
            "available": False,
            "reason": reason,
            "total_dispatches": 0,
            "by_skill": [],
            "by_agent": [],
            "top_failure_skills": [],
            "unjoined_in_flight": 0,
            "window_start_iso": _epoch_to_iso(window_start),
            "window_end_iso": _epoch_to_iso(window_end),
        }


def _epoch_to_iso(epoch: float) -> str:
    return datetime.fromtimestamp(epoch, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _has_known_pricing(model_name: str, provider: str = None, base_url: str = None) -> bool:
    """Check if a model has known pricing (vs unknown/custom endpoint)."""
    return has_known_pricing(model_name, provider=provider, base_url=base_url)


def _estimate_cost(
    session_or_model: Dict[str, Any] | str,
    input_tokens: int = 0,
    output_tokens: int = 0,
    *,
    cache_read_tokens: int = 0,
    cache_write_tokens: int = 0,
    provider: str = None,
    base_url: str = None,
) -> tuple[float, str]:
    """Estimate the USD cost for a session row or a model/token tuple."""
    if isinstance(session_or_model, dict):
        session = session_or_model
        model = session.get("model") or ""
        usage = CanonicalUsage(
            input_tokens=session.get("input_tokens") or 0,
            output_tokens=session.get("output_tokens") or 0,
            cache_read_tokens=session.get("cache_read_tokens") or 0,
            cache_write_tokens=session.get("cache_write_tokens") or 0,
        )
        provider = session.get("billing_provider")
        base_url = session.get("billing_base_url")
    else:
        model = session_or_model or ""
        usage = CanonicalUsage(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cache_read_tokens=cache_read_tokens,
            cache_write_tokens=cache_write_tokens,
        )
    result = estimate_usage_cost(
        model,
        usage,
        provider=provider,
        base_url=base_url,
    )
    return float(result.amount_usd or 0.0), result.status


def _format_duration(seconds: float) -> str:
    """Format seconds into a human-readable duration string."""
    return format_duration_compact(seconds)


def _bar_chart(values: List[int], max_width: int = 20) -> List[str]:
    """Create simple horizontal bar chart strings from values."""
    peak = max(values) if values else 1
    if peak == 0:
        return ["" for _ in values]
    return ["█" * max(1, int(v / peak * max_width)) if v > 0 else "" for v in values]


class InsightsEngine:
    """
    Analyzes session history and produces usage insights.

    Works directly with a SessionDB instance (or raw sqlite3 connection)
    to query session and message data.
    """

    def __init__(self, db, skill_usage_path: Optional[str] = None):
        """
        Initialize with a SessionDB instance.

        Args:
            db: A SessionDB instance (from hermes_state.py)
            skill_usage_path: Optional override for the SIE skill-usage
                JSONL log location. Defaults to
                ~/hermes-workspace/Lex-Workspace/sie/analytics/skill-usage.jsonl
                expanded via os.path.expanduser. Missing file degrades
                gracefully (skill_usage section reports available=False).
        """
        self.db = db
        self._conn = db._conn
        resolved = skill_usage_path if skill_usage_path is not None else _DEFAULT_SKILL_USAGE_PATH
        self.skill_usage_path = os.path.expanduser(resolved)
        self._skill_usage_aggregator = _SkillUsageAggregator(self.skill_usage_path)

    def generate(self, days: int = 30, source: str = None) -> Dict[str, Any]:
        """
        Generate a complete insights report.

        Args:
            days: Number of days to look back (default: 30)
            source: Optional filter by source platform

        Returns:
            Dict with all computed insights
        """
        cutoff = time.time() - (days * 86400)

        # SIE telemetry aggregation is independent of session-DB content
        # and runs on its own data source, so compute it unconditionally
        # and attach to the report (even when no sessions exist).
        sie_skill_usage = self._skill_usage_aggregator.aggregate(days=days)

        # Gather raw data
        sessions = self._get_sessions(cutoff, source)
        tool_usage = self._get_tool_usage(cutoff, source)
        skill_usage = self._get_skill_usage(cutoff, source)
        message_stats = self._get_message_stats(cutoff, source)

        if not sessions:
            return {
                "days": days,
                "source_filter": source,
                "empty": True,
                "overview": {},
                "models": [],
                "platforms": [],
                "tools": [],
                "skills": {
                    "summary": {
                        "total_skill_loads": 0,
                        "total_skill_edits": 0,
                        "total_skill_actions": 0,
                        "distinct_skills_used": 0,
                    },
                    "top_skills": [],
                },
                "activity": {},
                "top_sessions": [],
                "skill_usage": sie_skill_usage,
            }

        # Compute insights
        overview = self._compute_overview(sessions, message_stats)
        models = self._compute_model_breakdown(sessions)
        platforms = self._compute_platform_breakdown(sessions)
        tools = self._compute_tool_breakdown(tool_usage)
        skills = self._compute_skill_breakdown(skill_usage)
        activity = self._compute_activity_patterns(sessions)
        top_sessions = self._compute_top_sessions(sessions)

        return {
            "days": days,
            "source_filter": source,
            "empty": False,
            "generated_at": time.time(),
            "overview": overview,
            "models": models,
            "platforms": platforms,
            "tools": tools,
            "skills": skills,
            "activity": activity,
            "top_sessions": top_sessions,
            "skill_usage": sie_skill_usage,
        }

    # =========================================================================
    # Data gathering (SQL queries)
    # =========================================================================

    # Columns we actually need (skip system_prompt, model_config blobs)
    _SESSION_COLS = ("id, source, model, started_at, ended_at, "
                     "message_count, tool_call_count, input_tokens, output_tokens, "
                     "cache_read_tokens, cache_write_tokens, billing_provider, "
                     "billing_base_url, billing_mode, estimated_cost_usd, "
                     "actual_cost_usd, cost_status, cost_source")

    # Pre-computed query strings — f-string evaluated once at class definition,
    # not at runtime, so no user-controlled value can alter the query structure.
    _GET_SESSIONS_WITH_SOURCE = (
        f"SELECT {_SESSION_COLS} FROM sessions"
        " WHERE started_at >= ? AND source = ?"
        " ORDER BY started_at DESC"
    )
    _GET_SESSIONS_ALL = (
        f"SELECT {_SESSION_COLS} FROM sessions"
        " WHERE started_at >= ?"
        " ORDER BY started_at DESC"
    )

    def _get_sessions(self, cutoff: float, source: str = None) -> List[Dict]:
        """Fetch sessions within the time window."""
        if source:
            cursor = self._conn.execute(self._GET_SESSIONS_WITH_SOURCE, (cutoff, source))
        else:
            cursor = self._conn.execute(self._GET_SESSIONS_ALL, (cutoff,))
        return [dict(row) for row in cursor.fetchall()]

    def _get_tool_usage(self, cutoff: float, source: str = None) -> List[Dict]:
        """Get tool call counts from messages.

        Uses two sources:
        1. tool_name column on 'tool' role messages (set by gateway)
        2. tool_calls JSON on 'assistant' role messages (covers CLI where
           tool_name is not populated on tool responses)
        """
        tool_counts = Counter()

        # Source 1: explicit tool_name on tool response messages
        if source:
            cursor = self._conn.execute(
                """SELECT m.tool_name, COUNT(*) as count
                   FROM messages m
                   JOIN sessions s ON s.id = m.session_id
                   WHERE s.started_at >= ? AND s.source = ?
                     AND m.role = 'tool' AND m.tool_name IS NOT NULL
                   GROUP BY m.tool_name
                   ORDER BY count DESC""",
                (cutoff, source),
            )
        else:
            cursor = self._conn.execute(
                """SELECT m.tool_name, COUNT(*) as count
                   FROM messages m
                   JOIN sessions s ON s.id = m.session_id
                   WHERE s.started_at >= ?
                     AND m.role = 'tool' AND m.tool_name IS NOT NULL
                   GROUP BY m.tool_name
                   ORDER BY count DESC""",
                (cutoff,),
            )
        for row in cursor.fetchall():
            tool_counts[row["tool_name"]] += row["count"]

        # Source 2: extract from tool_calls JSON on assistant messages
        # (covers CLI sessions where tool_name is NULL on tool responses)
        if source:
            cursor2 = self._conn.execute(
                """SELECT m.tool_calls
                   FROM messages m
                   JOIN sessions s ON s.id = m.session_id
                   WHERE s.started_at >= ? AND s.source = ?
                     AND m.role = 'assistant' AND m.tool_calls IS NOT NULL""",
                (cutoff, source),
            )
        else:
            cursor2 = self._conn.execute(
                """SELECT m.tool_calls
                   FROM messages m
                   JOIN sessions s ON s.id = m.session_id
                   WHERE s.started_at >= ?
                     AND m.role = 'assistant' AND m.tool_calls IS NOT NULL""",
                (cutoff,),
            )

        tool_calls_counts = Counter()
        for row in cursor2.fetchall():
            try:
                calls = row["tool_calls"]
                if isinstance(calls, str):
                    calls = json.loads(calls)
                if isinstance(calls, list):
                    for call in calls:
                        func = call.get("function", {}) if isinstance(call, dict) else {}
                        name = func.get("name")
                        if name:
                            tool_calls_counts[name] += 1
            except (json.JSONDecodeError, TypeError, AttributeError):
                continue

        # Merge: prefer tool_name source, supplement with tool_calls source
        # for tools not already counted
        if not tool_counts and tool_calls_counts:
            # No tool_name data at all — use tool_calls exclusively
            tool_counts = tool_calls_counts
        elif tool_counts and tool_calls_counts:
            # Both sources have data — use whichever has the higher count per tool
            # (they may overlap, so take the max to avoid double-counting)
            all_tools = set(tool_counts) | set(tool_calls_counts)
            merged = Counter()
            for tool in all_tools:
                merged[tool] = max(tool_counts.get(tool, 0), tool_calls_counts.get(tool, 0))
            tool_counts = merged

        # Convert to the expected format
        return [
            {"tool_name": name, "count": count}
            for name, count in tool_counts.most_common()
        ]

    def _get_skill_usage(self, cutoff: float, source: str = None) -> List[Dict]:
        """Extract per-skill usage from assistant tool calls."""
        skill_counts: Dict[str, Dict[str, Any]] = {}

        if source:
            cursor = self._conn.execute(
                """SELECT m.tool_calls, m.timestamp
                   FROM messages m
                   JOIN sessions s ON s.id = m.session_id
                   WHERE s.started_at >= ? AND s.source = ?
                     AND m.role = 'assistant' AND m.tool_calls IS NOT NULL""",
                (cutoff, source),
            )
        else:
            cursor = self._conn.execute(
                """SELECT m.tool_calls, m.timestamp
                   FROM messages m
                   JOIN sessions s ON s.id = m.session_id
                   WHERE s.started_at >= ?
                     AND m.role = 'assistant' AND m.tool_calls IS NOT NULL""",
                (cutoff,),
            )

        for row in cursor.fetchall():
            try:
                calls = row["tool_calls"]
                if isinstance(calls, str):
                    calls = json.loads(calls)
                if not isinstance(calls, list):
                    continue
            except (json.JSONDecodeError, TypeError):
                continue

            timestamp = row["timestamp"]
            for call in calls:
                if not isinstance(call, dict):
                    continue
                func = call.get("function", {})
                tool_name = func.get("name")
                if tool_name not in {"skill_view", "skill_manage"}:
                    continue

                args = func.get("arguments")
                if isinstance(args, str):
                    try:
                        args = json.loads(args)
                    except (json.JSONDecodeError, TypeError):
                        continue
                if not isinstance(args, dict):
                    continue

                skill_name = args.get("name")
                if not isinstance(skill_name, str) or not skill_name.strip():
                    continue

                entry = skill_counts.setdefault(
                    skill_name,
                    {
                        "skill": skill_name,
                        "view_count": 0,
                        "manage_count": 0,
                        "last_used_at": None,
                    },
                )
                if tool_name == "skill_view":
                    entry["view_count"] += 1
                else:
                    entry["manage_count"] += 1

                if timestamp is not None and (
                    entry["last_used_at"] is None or timestamp > entry["last_used_at"]
                ):
                    entry["last_used_at"] = timestamp

        return list(skill_counts.values())

    def _get_message_stats(self, cutoff: float, source: str = None) -> Dict:
        """Get aggregate message statistics."""
        if source:
            cursor = self._conn.execute(
                """SELECT
                     COUNT(*) as total_messages,
                     SUM(CASE WHEN m.role = 'user' THEN 1 ELSE 0 END) as user_messages,
                     SUM(CASE WHEN m.role = 'assistant' THEN 1 ELSE 0 END) as assistant_messages,
                     SUM(CASE WHEN m.role = 'tool' THEN 1 ELSE 0 END) as tool_messages
                   FROM messages m
                   JOIN sessions s ON s.id = m.session_id
                   WHERE s.started_at >= ? AND s.source = ?""",
                (cutoff, source),
            )
        else:
            cursor = self._conn.execute(
                """SELECT
                     COUNT(*) as total_messages,
                     SUM(CASE WHEN m.role = 'user' THEN 1 ELSE 0 END) as user_messages,
                     SUM(CASE WHEN m.role = 'assistant' THEN 1 ELSE 0 END) as assistant_messages,
                     SUM(CASE WHEN m.role = 'tool' THEN 1 ELSE 0 END) as tool_messages
                   FROM messages m
                   JOIN sessions s ON s.id = m.session_id
                   WHERE s.started_at >= ?""",
                (cutoff,),
            )
        row = cursor.fetchone()
        return dict(row) if row else {
            "total_messages": 0, "user_messages": 0,
            "assistant_messages": 0, "tool_messages": 0,
        }

    # =========================================================================
    # Computation
    # =========================================================================

    def _compute_overview(self, sessions: List[Dict], message_stats: Dict) -> Dict:
        """Compute high-level overview statistics."""
        total_input = sum(s.get("input_tokens") or 0 for s in sessions)
        total_output = sum(s.get("output_tokens") or 0 for s in sessions)
        total_cache_read = sum(s.get("cache_read_tokens") or 0 for s in sessions)
        total_cache_write = sum(s.get("cache_write_tokens") or 0 for s in sessions)
        total_tokens = total_input + total_output + total_cache_read + total_cache_write
        total_tool_calls = sum(s.get("tool_call_count") or 0 for s in sessions)
        total_messages = sum(s.get("message_count") or 0 for s in sessions)

        # Cost estimation (weighted by model)
        total_cost = 0.0
        actual_cost = 0.0
        models_with_pricing = set()
        models_without_pricing = set()
        unknown_cost_sessions = 0
        included_cost_sessions = 0
        for s in sessions:
            model = s.get("model") or ""
            estimated, status = _estimate_cost(s)
            total_cost += estimated
            actual_cost += s.get("actual_cost_usd") or 0.0
            display = model.split("/")[-1] if "/" in model else (model or "unknown")
            if status == "included":
                included_cost_sessions += 1
            elif status == "unknown":
                unknown_cost_sessions += 1
            if _has_known_pricing(model, s.get("billing_provider"), s.get("billing_base_url")):
                models_with_pricing.add(display)
            else:
                models_without_pricing.add(display)

        # Session duration stats (guard against negative durations from clock drift)
        durations = []
        for s in sessions:
            start = s.get("started_at")
            end = s.get("ended_at")
            if start and end and end > start:
                durations.append(end - start)

        total_hours = sum(durations) / 3600 if durations else 0
        avg_duration = sum(durations) / len(durations) if durations else 0

        # Earliest and latest session
        started_timestamps = [s["started_at"] for s in sessions if s.get("started_at")]
        date_range_start = min(started_timestamps) if started_timestamps else None
        date_range_end = max(started_timestamps) if started_timestamps else None

        return {
            "total_sessions": len(sessions),
            "total_messages": total_messages,
            "total_tool_calls": total_tool_calls,
            "total_input_tokens": total_input,
            "total_output_tokens": total_output,
            "total_cache_read_tokens": total_cache_read,
            "total_cache_write_tokens": total_cache_write,
            "total_tokens": total_tokens,
            "estimated_cost": total_cost,
            "actual_cost": actual_cost,
            "total_hours": total_hours,
            "avg_session_duration": avg_duration,
            "avg_messages_per_session": total_messages / len(sessions) if sessions else 0,
            "avg_tokens_per_session": total_tokens / len(sessions) if sessions else 0,
            "user_messages": message_stats.get("user_messages") or 0,
            "assistant_messages": message_stats.get("assistant_messages") or 0,
            "tool_messages": message_stats.get("tool_messages") or 0,
            "date_range_start": date_range_start,
            "date_range_end": date_range_end,
            "models_with_pricing": sorted(models_with_pricing),
            "models_without_pricing": sorted(models_without_pricing),
            "unknown_cost_sessions": unknown_cost_sessions,
            "included_cost_sessions": included_cost_sessions,
        }

    def _compute_model_breakdown(self, sessions: List[Dict]) -> List[Dict]:
        """Break down usage by model."""
        model_data = defaultdict(lambda: {
            "sessions": 0, "input_tokens": 0, "output_tokens": 0,
            "cache_read_tokens": 0, "cache_write_tokens": 0,
            "total_tokens": 0, "tool_calls": 0, "cost": 0.0,
        })

        for s in sessions:
            model = s.get("model") or "unknown"
            # Normalize: strip provider prefix for display
            display_model = model.split("/")[-1] if "/" in model else model
            d = model_data[display_model]
            d["sessions"] += 1
            inp = s.get("input_tokens") or 0
            out = s.get("output_tokens") or 0
            cache_read = s.get("cache_read_tokens") or 0
            cache_write = s.get("cache_write_tokens") or 0
            d["input_tokens"] += inp
            d["output_tokens"] += out
            d["cache_read_tokens"] += cache_read
            d["cache_write_tokens"] += cache_write
            d["total_tokens"] += inp + out + cache_read + cache_write
            d["tool_calls"] += s.get("tool_call_count") or 0
            estimate, status = _estimate_cost(s)
            d["cost"] += estimate
            d["has_pricing"] = _has_known_pricing(model, s.get("billing_provider"), s.get("billing_base_url"))
            d["cost_status"] = status

        result = [
            {"model": model, **data}
            for model, data in model_data.items()
        ]
        # Sort by tokens first, fall back to session count when tokens are 0
        result.sort(key=lambda x: (x["total_tokens"], x["sessions"]), reverse=True)
        return result

    def _compute_platform_breakdown(self, sessions: List[Dict]) -> List[Dict]:
        """Break down usage by platform/source."""
        platform_data = defaultdict(lambda: {
            "sessions": 0, "messages": 0, "input_tokens": 0,
            "output_tokens": 0, "cache_read_tokens": 0,
            "cache_write_tokens": 0, "total_tokens": 0, "tool_calls": 0,
        })

        for s in sessions:
            source = s.get("source") or "unknown"
            d = platform_data[source]
            d["sessions"] += 1
            d["messages"] += s.get("message_count") or 0
            inp = s.get("input_tokens") or 0
            out = s.get("output_tokens") or 0
            cache_read = s.get("cache_read_tokens") or 0
            cache_write = s.get("cache_write_tokens") or 0
            d["input_tokens"] += inp
            d["output_tokens"] += out
            d["cache_read_tokens"] += cache_read
            d["cache_write_tokens"] += cache_write
            d["total_tokens"] += inp + out + cache_read + cache_write
            d["tool_calls"] += s.get("tool_call_count") or 0

        result = [
            {"platform": platform, **data}
            for platform, data in platform_data.items()
        ]
        result.sort(key=lambda x: x["sessions"], reverse=True)
        return result

    def _compute_tool_breakdown(self, tool_usage: List[Dict]) -> List[Dict]:
        """Process tool usage data into a ranked list with percentages."""
        total_calls = sum(t["count"] for t in tool_usage) if tool_usage else 0
        result = []
        for t in tool_usage:
            pct = (t["count"] / total_calls * 100) if total_calls else 0
            result.append({
                "tool": t["tool_name"],
                "count": t["count"],
                "percentage": pct,
            })
        return result

    def _compute_skill_breakdown(self, skill_usage: List[Dict]) -> Dict[str, Any]:
        """Process per-skill usage into summary + ranked list."""
        total_skill_loads = sum(s["view_count"] for s in skill_usage) if skill_usage else 0
        total_skill_edits = sum(s["manage_count"] for s in skill_usage) if skill_usage else 0
        total_skill_actions = total_skill_loads + total_skill_edits

        top_skills = []
        for skill in skill_usage:
            total_count = skill["view_count"] + skill["manage_count"]
            percentage = (total_count / total_skill_actions * 100) if total_skill_actions else 0
            top_skills.append({
                "skill": skill["skill"],
                "view_count": skill["view_count"],
                "manage_count": skill["manage_count"],
                "total_count": total_count,
                "percentage": percentage,
                "last_used_at": skill.get("last_used_at"),
            })

        top_skills.sort(
            key=lambda s: (
                s["total_count"],
                s["view_count"],
                s["manage_count"],
                s["last_used_at"] or 0,
                s["skill"],
            ),
            reverse=True,
        )

        return {
            "summary": {
                "total_skill_loads": total_skill_loads,
                "total_skill_edits": total_skill_edits,
                "total_skill_actions": total_skill_actions,
                "distinct_skills_used": len(skill_usage),
            },
            "top_skills": top_skills,
        }

    def _compute_activity_patterns(self, sessions: List[Dict]) -> Dict:
        """Analyze activity patterns by day of week and hour."""
        day_counts = Counter()  # 0=Monday ... 6=Sunday
        hour_counts = Counter()
        daily_counts = Counter()  # date string -> count

        for s in sessions:
            ts = s.get("started_at")
            if not ts:
                continue
            dt = datetime.fromtimestamp(ts)
            day_counts[dt.weekday()] += 1
            hour_counts[dt.hour] += 1
            daily_counts[dt.strftime("%Y-%m-%d")] += 1

        day_names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
        day_breakdown = [
            {"day": day_names[i], "count": day_counts.get(i, 0)}
            for i in range(7)
        ]

        hour_breakdown = [
            {"hour": i, "count": hour_counts.get(i, 0)}
            for i in range(24)
        ]

        # Busiest day and hour
        busiest_day = max(day_breakdown, key=lambda x: x["count"]) if day_breakdown else None
        busiest_hour = max(hour_breakdown, key=lambda x: x["count"]) if hour_breakdown else None

        # Active days (days with at least one session)
        active_days = len(daily_counts)

        # Streak calculation
        if daily_counts:
            all_dates = sorted(daily_counts.keys())
            current_streak = 1
            max_streak = 1
            for i in range(1, len(all_dates)):
                d1 = datetime.strptime(all_dates[i - 1], "%Y-%m-%d")
                d2 = datetime.strptime(all_dates[i], "%Y-%m-%d")
                if (d2 - d1).days == 1:
                    current_streak += 1
                    max_streak = max(max_streak, current_streak)
                else:
                    current_streak = 1
        else:
            max_streak = 0

        return {
            "by_day": day_breakdown,
            "by_hour": hour_breakdown,
            "busiest_day": busiest_day,
            "busiest_hour": busiest_hour,
            "active_days": active_days,
            "max_streak": max_streak,
        }

    def _compute_top_sessions(self, sessions: List[Dict]) -> List[Dict]:
        """Find notable sessions (longest, most messages, most tokens)."""
        top = []

        # Longest by duration
        sessions_with_duration = [
            s for s in sessions
            if s.get("started_at") and s.get("ended_at")
        ]
        if sessions_with_duration:
            longest = max(
                sessions_with_duration,
                key=lambda s: (s["ended_at"] - s["started_at"]),
            )
            dur = longest["ended_at"] - longest["started_at"]
            top.append({
                "label": "Longest session",
                "session_id": longest["id"][:16],
                "value": _format_duration(dur),
                "date": datetime.fromtimestamp(longest["started_at"]).strftime("%b %d"),
            })

        # Most messages
        most_msgs = max(sessions, key=lambda s: s.get("message_count") or 0)
        if (most_msgs.get("message_count") or 0) > 0:
            top.append({
                "label": "Most messages",
                "session_id": most_msgs["id"][:16],
                "value": f"{most_msgs['message_count']} msgs",
                "date": datetime.fromtimestamp(most_msgs["started_at"]).strftime("%b %d") if most_msgs.get("started_at") else "?",
            })

        # Most tokens
        most_tokens = max(
            sessions,
            key=lambda s: (s.get("input_tokens") or 0) + (s.get("output_tokens") or 0),
        )
        token_total = (most_tokens.get("input_tokens") or 0) + (most_tokens.get("output_tokens") or 0)
        if token_total > 0:
            top.append({
                "label": "Most tokens",
                "session_id": most_tokens["id"][:16],
                "value": f"{token_total:,} tokens",
                "date": datetime.fromtimestamp(most_tokens["started_at"]).strftime("%b %d") if most_tokens.get("started_at") else "?",
            })

        # Most tool calls
        most_tools = max(sessions, key=lambda s: s.get("tool_call_count") or 0)
        if (most_tools.get("tool_call_count") or 0) > 0:
            top.append({
                "label": "Most tool calls",
                "session_id": most_tools["id"][:16],
                "value": f"{most_tools['tool_call_count']} calls",
                "date": datetime.fromtimestamp(most_tools["started_at"]).strftime("%b %d") if most_tools.get("started_at") else "?",
            })

        return top

    # =========================================================================
    # Formatting
    # =========================================================================

    def format_terminal(self, report: Dict) -> str:
        """Format the insights report for terminal display (CLI)."""
        if report.get("empty"):
            days = report.get("days", 30)
            src = f" (source: {report['source_filter']})" if report.get("source_filter") else ""
            header = f"  No sessions found in the last {days} days{src}."
            sie = report.get("skill_usage") or {}
            extra = self._format_skill_usage_section(sie)
            return "\n".join([header, ""] + extra) if extra else header

        lines = []
        o = report["overview"]
        days = report["days"]
        src_filter = report.get("source_filter")

        # Header
        lines.append("")
        lines.append("  ╔══════════════════════════════════════════════════════════╗")
        lines.append("  ║                    📊 Hermes Insights                    ║")
        period_label = f"Last {days} days"
        if src_filter:
            period_label += f" ({src_filter})"
        padding = 58 - len(period_label) - 2
        left_pad = padding // 2
        right_pad = padding - left_pad
        lines.append(f"  ║{' ' * left_pad} {period_label} {' ' * right_pad}║")
        lines.append("  ╚══════════════════════════════════════════════════════════╝")
        lines.append("")

        # Date range
        if o.get("date_range_start") and o.get("date_range_end"):
            start_str = datetime.fromtimestamp(o["date_range_start"]).strftime("%b %d, %Y")
            end_str = datetime.fromtimestamp(o["date_range_end"]).strftime("%b %d, %Y")
            lines.append(f"  Period: {start_str} — {end_str}")
            lines.append("")

        # Overview
        lines.append("  📋 Overview")
        lines.append("  " + "─" * 56)
        lines.append(f"  Sessions:          {o['total_sessions']:<12}  Messages:        {o['total_messages']:,}")
        lines.append(f"  Tool calls:        {o['total_tool_calls']:<12,}  User messages:   {o['user_messages']:,}")
        lines.append(f"  Input tokens:      {o['total_input_tokens']:<12,}  Output tokens:   {o['total_output_tokens']:,}")
        lines.append(f"  Total tokens:      {o['total_tokens']:,}")
        if o["total_hours"] > 0:
            lines.append(f"  Active time:       ~{_format_duration(o['total_hours'] * 3600):<11}  Avg session:     ~{_format_duration(o['avg_session_duration'])}")
        lines.append(f"  Avg msgs/session:  {o['avg_messages_per_session']:.1f}")
        lines.append("")

        # Model breakdown
        if report["models"]:
            lines.append("  🤖 Models Used")
            lines.append("  " + "─" * 56)
            lines.append(f"  {'Model':<30} {'Sessions':>8} {'Tokens':>12}")
            for m in report["models"]:
                model_name = m["model"][:28]
                lines.append(f"  {model_name:<30} {m['sessions']:>8} {m['total_tokens']:>12,}")
            lines.append("")

        # Platform breakdown
        if len(report["platforms"]) > 1 or (report["platforms"] and report["platforms"][0]["platform"] != "cli"):
            lines.append("  📱 Platforms")
            lines.append("  " + "─" * 56)
            lines.append(f"  {'Platform':<14} {'Sessions':>8} {'Messages':>10} {'Tokens':>14}")
            for p in report["platforms"]:
                lines.append(f"  {p['platform']:<14} {p['sessions']:>8} {p['messages']:>10,} {p['total_tokens']:>14,}")
            lines.append("")

        # Tool usage
        if report["tools"]:
            lines.append("  🔧 Top Tools")
            lines.append("  " + "─" * 56)
            lines.append(f"  {'Tool':<28} {'Calls':>8} {'%':>8}")
            for t in report["tools"][:15]:  # Top 15
                lines.append(f"  {t['tool']:<28} {t['count']:>8,} {t['percentage']:>7.1f}%")
            if len(report["tools"]) > 15:
                lines.append(f"  ... and {len(report['tools']) - 15} more tools")
            lines.append("")

        # Skill usage
        skills = report.get("skills", {})
        top_skills = skills.get("top_skills", [])
        if top_skills:
            lines.append("  🧠 Top Skills")
            lines.append("  " + "─" * 56)
            lines.append(f"  {'Skill':<28} {'Loads':>7} {'Edits':>7} {'Last used':>11}")
            for skill in top_skills[:10]:
                last_used = "—"
                if skill.get("last_used_at"):
                    last_used = datetime.fromtimestamp(skill["last_used_at"]).strftime("%b %d")
                lines.append(
                    f"  {skill['skill'][:28]:<28} {skill['view_count']:>7,} {skill['manage_count']:>7,} {last_used:>11}"
                )
            summary = skills.get("summary", {})
            lines.append(
                f"  Distinct skills: {summary.get('distinct_skills_used', 0)}  "
                f"Loads: {summary.get('total_skill_loads', 0):,}  "
                f"Edits: {summary.get('total_skill_edits', 0):,}"
            )
            lines.append("")

        # Activity patterns
        act = report.get("activity", {})
        if act.get("by_day"):
            lines.append("  📅 Activity Patterns")
            lines.append("  " + "─" * 56)

            # Day of week chart
            day_values = [d["count"] for d in act["by_day"]]
            bars = _bar_chart(day_values, max_width=15)
            for i, d in enumerate(act["by_day"]):
                bar = bars[i]
                lines.append(f"  {d['day']}  {bar:<15} {d['count']}")

            lines.append("")

            # Peak hours (show top 5 busiest hours)
            busy_hours = sorted(act["by_hour"], key=lambda x: x["count"], reverse=True)
            busy_hours = [h for h in busy_hours if h["count"] > 0][:5]
            if busy_hours:
                hour_strs = []
                for h in busy_hours:
                    hr = h["hour"]
                    ampm = "AM" if hr < 12 else "PM"
                    display_hr = hr % 12 or 12
                    hour_strs.append(f"{display_hr}{ampm} ({h['count']})")
                lines.append(f"  Peak hours: {', '.join(hour_strs)}")

            if act.get("active_days"):
                lines.append(f"  Active days: {act['active_days']}")
            if act.get("max_streak") and act["max_streak"] > 1:
                lines.append(f"  Best streak: {act['max_streak']} consecutive days")
            lines.append("")

        # Notable sessions
        if report.get("top_sessions"):
            lines.append("  🏆 Notable Sessions")
            lines.append("  " + "─" * 56)
            for ts in report["top_sessions"]:
                lines.append(f"  {ts['label']:<20} {ts['value']:<18} ({ts['date']}, {ts['session_id']})")
            lines.append("")

        # SIE Skill Usage (telemetry JSONL data source)
        sie = report.get("skill_usage") or {}
        lines.extend(self._format_skill_usage_section(sie))

        return "\n".join(lines)

    def _format_skill_usage_section(self, sie: Dict[str, Any]) -> List[str]:
        """Render the SIE skill-usage section. No em dashes (HR-8).

        Always emits a header so the report's structure is predictable.
        Degradation cases (no file, unreadable, zero dispatches) render a
        single explanatory line and skip the tables.
        """
        lines: List[str] = []
        lines.append("  ## Skill Usage")
        lines.append("  " + "─" * 56)
        if not sie:
            lines.append("  No skill-usage telemetry available.")
            lines.append("")
            return lines
        if not sie.get("available", False):
            reason = sie.get("reason", "unavailable")
            lines.append(f"  Telemetry unavailable: {reason}")
            lines.append("")
            return lines
        total = sie.get("total_dispatches", 0)
        unjoined = sie.get("unjoined_in_flight", 0)
        win_start = sie.get("window_start_iso", "")
        win_end = sie.get("window_end_iso", "")
        lines.append(f"  Window: {win_start} to {win_end}")
        lines.append(f"  Total dispatches: {total:,}   Unjoined in-flight: {unjoined}")
        if total == 0:
            lines.append("  No completed dispatches in window.")
            lines.append("")
            return lines

        by_skill = sie.get("by_skill") or []
        if by_skill:
            lines.append("")
            lines.append(f"  {'Skill':<28} {'Total':>6} {'Succ':>5} {'Fail':>5} {'Err':>4} {'Rate':>6} {'AvgDur':>8}")
            for r in by_skill[:10]:
                rate_pct = f"{r['success_rate'] * 100:.1f}%"
                dur = f"{r['avg_duration_s']:.2f}s"
                lines.append(
                    f"  {r['skill'][:28]:<28} {r['total']:>6} {r['success']:>5} "
                    f"{r['failure']:>5} {r['error']:>4} {rate_pct:>6} {dur:>8}"
                )
            if len(by_skill) > 10:
                lines.append(f"  ... and {len(by_skill) - 10} more skills")

        by_agent = sie.get("by_agent") or []
        if by_agent:
            lines.append("")
            lines.append(f"  {'Agent':<14} {'Total':>6} {'Rate':>7} {'AvgDur':>9}")
            for r in by_agent:
                rate_pct = f"{r['success_rate'] * 100:.1f}%"
                dur = f"{r['avg_duration_s']:.2f}s"
                lines.append(
                    f"  {r['agent'][:14]:<14} {r['total']:>6} {rate_pct:>7} {dur:>9}"
                )

        top_failures = sie.get("top_failure_skills") or []
        if top_failures:
            lines.append("")
            lines.append("  Top failure skills:")
            for r in top_failures:
                lines.append(f"    {r['skill']}: {r['failure_count']}")

        lines.append("")
        return lines

    def format_gateway(self, report: Dict) -> str:
        """Format the insights report for gateway/messaging (shorter)."""
        if report.get("empty"):
            days = report.get("days", 30)
            return f"No sessions found in the last {days} days."

        lines = []
        o = report["overview"]
        days = report["days"]

        lines.append(f"📊 **Hermes Insights** — Last {days} days\n")

        # Overview
        lines.append(f"**Sessions:** {o['total_sessions']} | **Messages:** {o['total_messages']:,} | **Tool calls:** {o['total_tool_calls']:,}")
        lines.append(f"**Tokens:** {o['total_tokens']:,} (in: {o['total_input_tokens']:,} / out: {o['total_output_tokens']:,})")
        if o["total_hours"] > 0:
            lines.append(f"**Active time:** ~{_format_duration(o['total_hours'] * 3600)} | **Avg session:** ~{_format_duration(o['avg_session_duration'])}")
        lines.append("")

        # Models (top 5)
        if report["models"]:
            lines.append("**🤖 Models:**")
            for m in report["models"][:5]:
                lines.append(f"  {m['model'][:25]} — {m['sessions']} sessions, {m['total_tokens']:,} tokens")
            lines.append("")

        # Platforms (if multi-platform)
        if len(report["platforms"]) > 1:
            lines.append("**📱 Platforms:**")
            for p in report["platforms"]:
                lines.append(f"  {p['platform']} — {p['sessions']} sessions, {p['messages']:,} msgs")
            lines.append("")

        # Tools (top 8)
        if report["tools"]:
            lines.append("**🔧 Top Tools:**")
            for t in report["tools"][:8]:
                lines.append(f"  {t['tool']} — {t['count']:,} calls ({t['percentage']:.1f}%)")
            lines.append("")

        skills = report.get("skills", {})
        if skills.get("top_skills"):
            lines.append("**🧠 Top Skills:**")
            for skill in skills["top_skills"][:5]:
                suffix = ""
                if skill.get("last_used_at"):
                    suffix = f", last used {datetime.fromtimestamp(skill['last_used_at']).strftime('%b %d')}"
                lines.append(
                    f"  {skill['skill']} — {skill['view_count']:,} loads, {skill['manage_count']:,} edits{suffix}"
                )
            lines.append("")

        # Activity summary
        act = report.get("activity", {})
        if act.get("busiest_day") and act.get("busiest_hour"):
            hr = act["busiest_hour"]["hour"]
            ampm = "AM" if hr < 12 else "PM"
            display_hr = hr % 12 or 12
            lines.append(f"**📅 Busiest:** {act['busiest_day']['day']}s ({act['busiest_day']['count']} sessions), {display_hr}{ampm} ({act['busiest_hour']['count']} sessions)")
            if act.get("active_days"):
                lines.append(f"**Active days:** {act['active_days']}", )
            if act.get("max_streak", 0) > 1:
                lines.append(f"**Best streak:** {act['max_streak']} consecutive days")

        return "\n".join(lines)
