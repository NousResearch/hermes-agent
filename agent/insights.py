"""
Session Insights Engine for Hermes Agent.

Analyzes historical session data from the SQLite state database to produce
comprehensive usage insights — token consumption, cost estimates, tool usage
patterns, activity trends, model/platform breakdowns, and session metrics.

Inspired by Claude Code's /insights command, adapted for Hermes Agent's
multi-platform architecture with additional cost estimation and platform
breakdown capabilities.

Usage:
    from agent.insights import InsightsEngine
    engine = InsightsEngine(db)
    report = engine.generate(days=30)
    print(engine.format_terminal(report))
"""

import html
import json
import re
import time
from collections import Counter, defaultdict
from datetime import datetime
from typing import Any, Dict, List

from agent.usage_pricing import (
    CanonicalUsage,
    DEFAULT_PRICING,
    estimate_usage_cost,
    format_duration_compact,
    has_known_pricing,
)

_DEFAULT_PRICING = DEFAULT_PRICING


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

    def __init__(self, db):
        """
        Initialize with a SessionDB instance.

        Args:
            db: A SessionDB instance (from hermes_state.py)
        """
        self.db = db
        self._conn = db._conn

    def generate(self, days: int = 30, source: str = None, workflow: bool = True) -> Dict[str, Any]:
        """
        Generate a complete insights report.

        Args:
            days: Number of days to look back (default: 30)
            source: Optional filter by source platform

        Returns:
            Dict with all computed insights
        """
        cutoff = time.time() - (days * 86400)

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
                "workflow_intelligence": self._empty_workflow_intelligence() if workflow else None,
            }

        # Compute insights
        overview = self._compute_overview(sessions, message_stats)
        models = self._compute_model_breakdown(sessions)
        platforms = self._compute_platform_breakdown(sessions)
        tools = self._compute_tool_breakdown(tool_usage)
        skills = self._compute_skill_breakdown(skill_usage)
        activity = self._compute_activity_patterns(sessions)
        top_sessions = self._compute_top_sessions(sessions)
        workflow_intelligence = self._compute_workflow_intelligence(cutoff, source, sessions) if workflow else None

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
            "workflow_intelligence": workflow_intelligence,
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
    # Workflow intelligence (heuristic qualitative layer)
    # =========================================================================

    def _empty_workflow_intelligence(self) -> Dict[str, Any]:
        return {
            "summary": {"sessions_analyzed": 0, "messages_analyzed": 0},
            "task_types": [],
            "outcomes": [],
            "what_worked": [],
            "friction_points": [],
            "quick_wins": [],
            "recommendations": [],
            "safety_or_privacy_flags": [],
            "approval_required": True,
            "method": "heuristic_transcript_scan",
        }

    def _get_session_messages(self, cutoff: float, source: str = None) -> Dict[str, List[Dict[str, Any]]]:
        """Fetch non-system messages grouped by session for workflow analysis."""
        if source:
            cursor = self._conn.execute(
                """SELECT m.session_id, m.role, m.content, m.tool_name, m.tool_calls, m.timestamp
                   FROM messages m
                   JOIN sessions s ON s.id = m.session_id
                   WHERE s.started_at >= ? AND s.source = ? AND m.role != 'system'
                   ORDER BY m.session_id, m.timestamp, m.id""",
                (cutoff, source),
            )
        else:
            cursor = self._conn.execute(
                """SELECT m.session_id, m.role, m.content, m.tool_name, m.tool_calls, m.timestamp
                   FROM messages m
                   JOIN sessions s ON s.id = m.session_id
                   WHERE s.started_at >= ? AND m.role != 'system'
                   ORDER BY m.session_id, m.timestamp, m.id""",
                (cutoff,),
            )
        grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        for row in cursor.fetchall():
            grouped[row["session_id"]].append(dict(row))
        return grouped

    def _extract_tool_names_from_messages(self, messages: List[Dict[str, Any]]) -> List[str]:
        tools: List[str] = []
        for msg in messages:
            if msg.get("tool_name"):
                tools.append(msg["tool_name"])
            calls = msg.get("tool_calls")
            if isinstance(calls, str):
                try:
                    calls = json.loads(calls)
                except (json.JSONDecodeError, TypeError):
                    calls = None
            if isinstance(calls, list):
                for call in calls:
                    if isinstance(call, dict):
                        name = (call.get("function") or {}).get("name")
                        if name:
                            tools.append(name)
        return tools

    def _classify_task_type(self, text: str, tools: List[str]) -> str:
        t = text.lower()
        if any(w in t for w in ("bug", "fix", "failing", "traceback", "error", "debug")):
            return "debugging"
        if any(w in t for w in ("build", "implement", "add", "create", "extend")) or {"write_file", "patch"} & set(tools):
            return "implementation"
        if any(w in t for w in ("research", "search", "compare", "investigate")) or {"web_search", "web_extract"} & set(tools):
            return "research"
        if any(w in t for w in ("plan", "roadmap", "design")):
            return "planning"
        if any(w in t for w in ("run", "deploy", "status", "health", "cron", "gateway")):
            return "ops"
        if any(w in t for w in ("docs", "document", "readme")):
            return "docs"
        if any(w in t for w in ("test", "verify", "qa")):
            return "qa"
        return "other"

    def _classify_outcome(self, text: str) -> str:
        t = text.lower()
        recent = t[-1200:]
        # Outcome should reflect where the session landed, not every obstacle
        # mentioned along the way. Explicit final blockers/negations win over
        # generic completion words such as "done" inside "not done".
        blocked_patterns = (
            r"\bnot done\b",
            r"\bincomplete\b",
            r"\bpartial(?:ly)?\b",
            r"\bblocked\b",
            r"\bcannot complete\b",
            r"\bcan't complete\b",
            r"\bpermission denied\b",
            r"\bmissing (?:credentials?|tokens?|context|permission|auth)\b",
            r"\bnot available\b",
            r"\bfailed to\b",
        )
        if any(re.search(pattern, recent) for pattern in blocked_patterns):
            return "blocked"
        if any(w in recent for w in ("abandoned", "stopped", "cancelled", "canceled")):
            return "abandoned"
        if any(w in recent for w in ("done", "complete", "completed", "fixed", "implemented", "passed", "success", "verified", "pushed", "opened pr")):
            return "completed"
        return "unclear"

    def _facet_for_session(self, session: Dict[str, Any], messages: List[Dict[str, Any]]) -> Dict[str, Any]:
        user_text = "\n".join((m.get("content") or "") for m in messages if m.get("role") == "user")
        assistant_text = "\n".join((m.get("content") or "") for m in messages if m.get("role") == "assistant")
        # Qualitative signals intentionally come from user/assistant text plus
        # structured diagnostic metadata. Raw tool output often contains copied
        # docs, code, or transcript snippets, so scanning it directly produces
        # false positives and risks surfacing private prompt fragments.
        all_text = "\n".join([user_text, assistant_text])
        tools = self._extract_tool_names_from_messages(messages)
        tool_set = set(tools)
        friction = []
        failure_candidates = []
        diagnostic_success_after_failure = False
        mutation_seen = False
        validation_after_mutation = False
        diagnostic_tool_names = {"terminal", "browser_console", "browser_snapshot", "browser_navigate", "process"}
        mutation_tool_names = {"patch", "write_file", "skill_manage"}
        validation_cue = re.compile(r"\b(pytest|tests? passed|test suite|lint|ruff|typecheck|smoke|verified|validation|build passed|passed)\b", re.I)
        pending_tool_names: List[str] = []
        for msg in messages:
            role = msg.get("role")
            content = msg.get("content") or ""
            if role == "assistant":
                calls = msg.get("tool_calls")
                if isinstance(calls, str):
                    try:
                        calls = json.loads(calls)
                    except (json.JSONDecodeError, TypeError):
                        calls = None
                if isinstance(calls, list):
                    for call in calls:
                        if isinstance(call, dict):
                            name = (call.get("function") or {}).get("name")
                            if name:
                                pending_tool_names.append(name)
            tool_name = msg.get("tool_name")
            if role == "tool" and not tool_name and pending_tool_names:
                # Some session stores associate the tool name with the prior
                # assistant tool_call but leave the tool response row's
                # tool_name NULL. Pair them in order so mutation/validation
                # sequencing does not produce false verification gaps.
                tool_name = pending_tool_names.pop(0)
            parsed = None
            if role == "tool":
                try:
                    parsed = json.loads(content)
                except (json.JSONDecodeError, TypeError):
                    parsed = None
                if tool_name in mutation_tool_names:
                    mutation_seen = True
                elif mutation_seen and tool_name in diagnostic_tool_names:
                    success = False
                    if isinstance(parsed, dict):
                        if tool_name == "terminal":
                            success = parsed.get("exit_code") == 0 and validation_cue.search(str(parsed.get("output") or "")) is not None
                        else:
                            success = parsed.get("success") is True and validation_cue.search(str(parsed)) is not None
                    elif tool_name == "terminal":
                        success = validation_cue.search(content) is not None and "exit_code: 1" not in content
                    if success:
                        validation_after_mutation = True
            elif role == "assistant" and mutation_seen and validation_cue.search(content):
                validation_after_mutation = True

            if role != "tool" or tool_name not in diagnostic_tool_names:
                continue
            lower_content = content.lower()
            failed = False
            if isinstance(parsed, dict):
                if tool_name == "terminal":
                    failed = parsed.get("exit_code") not in (None, 0)
                elif parsed.get("success") is False or parsed.get("error"):
                    failed = True
            else:
                failed = any(w in lower_content for w in ("traceback", "exception", "command failed", '"exit_code": 1', "exit_code: 1"))
            if failed:
                failure_candidates.append(tool_name or "tool")
            elif failure_candidates and isinstance(parsed, dict) and (
                parsed.get("exit_code") == 0 or parsed.get("success") is True
            ):
                diagnostic_success_after_failure = True
        mutated = bool(mutation_seen or ({"patch", "write_file", "skill_manage"} & tool_set))
        verified = validation_after_mutation or (not mutated and bool(re.search(r"\b(tests? passed|verified|passed)\b", assistant_text, re.I)))
        outcome = self._classify_outcome(assistant_text[-2000:])
        recovered_failure = bool(
            failure_candidates and (
                (verified and outcome == "completed") or diagnostic_success_after_failure
            )
        )
        if failure_candidates and not recovered_failure:
            tools_seen = ", ".join(sorted(set(failure_candidates)))
            friction.append({"type": "tool_failure", "evidence": f"Diagnostic tool output included unrecovered failure language ({tools_seen})."})
        if mutated and not verified:
            friction.append({"type": "verification_gap", "evidence": "Session changed files or skills without visible validation tool/output."})
        if re.search(r"\b(what do you mean|which repo|need.*clarif|missing context)\b", all_text, re.I):
            friction.append({"type": "missing_context", "evidence": "Transcript suggests missing or ambiguous context."})
        what_worked = []
        if "skill_view" in tool_set:
            what_worked.append("Loaded a relevant skill before or during execution.")
        if verified:
            what_worked.append("Included verification evidence instead of stopping at implementation intent.")
        if recovered_failure:
            what_worked.append("Recovered from an intermediate tool/test failure and finished with passing verification.")
        if {"search_files", "read_file"} <= tool_set:
            what_worked.append("Inspected existing code before editing.")
        return {
            "session_id": session.get("id"),
            "source": session.get("source") or "unknown",
            "task_type": self._classify_task_type(user_text + "\n" + assistant_text, tools),
            "outcome": outcome,
            "tool_count": len(tools),
            "message_count": len(messages),
            "what_worked": what_worked,
            "friction_points": friction,
        }

    def _compute_workflow_intelligence(self, cutoff: float, source: str, sessions: List[Dict[str, Any]]) -> Dict[str, Any]:
        grouped = self._get_session_messages(cutoff, source)
        facets = [self._facet_for_session(s, grouped.get(s.get("id"), [])) for s in sessions]
        task_counts = Counter(f["task_type"] for f in facets)
        outcome_counts = Counter(f["outcome"] for f in facets)
        friction_points = []
        what_worked_counts = Counter()
        for f in facets:
            for point in f["friction_points"]:
                item = {**point, "session_id": (f.get("session_id") or "")[:16], "task_type": f["task_type"]}
                friction_points.append(item)
            what_worked_counts.update(f["what_worked"])
        recommendations = []
        friction_type_counts = Counter(p["type"] for p in friction_points)
        if friction_type_counts.get("verification_gap"):
            recommendations.append({
                "priority": "high",
                "layer": "skills/AGENTS.md",
                "recommendation": "Strengthen done-gate verification language for sessions that write files or patch skills.",
                "reason": f"{friction_type_counts['verification_gap']} session(s) changed artifacts without visible validation evidence.",
                "action": "Add or refine a task-specific verification checklist; require a test/build/smoke command in final handoff for write operations.",
            })
        if friction_type_counts.get("tool_failure"):
            recommendations.append({
                "priority": "medium",
                "layer": "skills",
                "recommendation": "Capture recurring tool failure recovery patterns as skill pitfalls.",
                "reason": f"{friction_type_counts['tool_failure']} session(s) had unrecovered diagnostic tool failures.",
                "action": "Patch the relevant skill with the failing command, observed error, and verified recovery path after approval.",
            })
        if friction_type_counts.get("missing_context"):
            recommendations.append({
                "priority": "medium",
                "layer": "process",
                "recommendation": "Tighten kickoff context for ambiguous requests before execution begins.",
                "reason": f"{friction_type_counts['missing_context']} session(s) showed missing-context or clarification friction.",
                "action": "Add a short prerequisite/context checklist to the relevant workflow or project instructions after human approval.",
            })
        if outcome_counts.get("blocked"):
            recommendations.append({
                "priority": "medium",
                "layer": "ops",
                "recommendation": "Review blocked sessions for recurring missing credentials, permissions, or external dependencies.",
                "reason": f"{outcome_counts['blocked']} session(s) ended blocked in this window.",
                "action": "Create a follow-up issue or checklist only for blockers that recur across multiple sessions.",
            })
        if outcome_counts.get("unclear"):
            recommendations.append({
                "priority": "low",
                "layer": "process",
                "recommendation": "Make final handoffs more explicit when a session is neither completed nor blocked.",
                "reason": f"{outcome_counts['unclear']} session(s) had unclear outcome language.",
                "action": "End substantial tasks with a short status line: completed, blocked, or partial, plus evidence or blocker.",
            })
        quick_wins = []
        if task_counts:
            top_task, count = task_counts.most_common(1)[0]
            quick_wins.append(f"Create or refine one reusable workflow for the dominant task type: {top_task} ({count} sessions).")
            if count >= 3:
                recommendations.append({
                    "priority": "low",
                    "layer": "skills",
                    "recommendation": f"Consider a focused reusable skill for frequent {top_task} sessions.",
                    "reason": f"{top_task} was the most common task type ({count} sessions).",
                    "action": "Draft the skill from successful sessions only; keep it approval-gated and avoid importing raw transcript text.",
                })
        if friction_points:
            quick_wins.append("Review the friction list and approve only specific AGENTS.md/skill/memory updates; do not auto-apply transcript-derived advice.")
        else:
            quick_wins.append("Keep the current workflow shape; no recurring friction crossed the heuristic threshold.")
        if not recommendations:
            recommendations.append({
                "priority": "info",
                "layer": "process",
                "recommendation": "Keep using insights as an approval-gated review surface; no automatic instruction or memory writes were applied.",
                "reason": "No high-confidence recurring friction pattern crossed the heuristic threshold.",
                "action": "Review the HTML/Markdown report manually before changing skills, memory, or AGENTS.md.",
            })
        completed_sessions = outcome_counts.get("completed", 0)
        blocked_sessions = outcome_counts.get("blocked", 0)
        total_sessions = len(facets)
        completion_rate = round((completed_sessions / total_sessions) * 100, 1) if total_sessions else 0.0
        friction_rate = round((len(friction_points) / total_sessions) * 100, 1) if total_sessions else 0.0
        return {
            "summary": {
                "sessions_analyzed": total_sessions,
                "messages_analyzed": sum(f["message_count"] for f in facets),
                "friction_points": len(friction_points),
                "completed_sessions": completed_sessions,
                "blocked_sessions": blocked_sessions,
                "completion_rate": completion_rate,
                "friction_rate": friction_rate,
            },
            "task_types": [{"task_type": k, "count": v} for k, v in task_counts.most_common()],
            "outcomes": [{"outcome": k, "count": v} for k, v in outcome_counts.most_common()],
            "what_worked": [{"pattern": k, "count": v} for k, v in what_worked_counts.most_common(6)],
            "friction_points": friction_points[:12],
            "quick_wins": quick_wins,
            "recommendations": recommendations,
            "safety_or_privacy_flags": ["Transcript-derived recommendations are approval-gated and should not include secrets or raw private prompt dumps."],
            "approval_required": True,
            "method": "heuristic_transcript_scan",
        }

    # =========================================================================
    # Formatting
    # =========================================================================

    def format_terminal(self, report: Dict) -> str:
        """Format the insights report for terminal display (CLI)."""
        if report.get("empty"):
            days = report.get("days", 30)
            src = f" (source: {report['source_filter']})" if report.get("source_filter") else ""
            return f"  No sessions found in the last {days} days{src}."

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

        workflow = report.get("workflow_intelligence")
        if workflow:
            lines.append("  🧭 Workflow Intelligence")
            lines.append("  " + "─" * 56)
            summary = workflow.get("summary", {})
            lines.append(f"  Sessions analyzed: {summary.get('sessions_analyzed', 0)}  Completion: {summary.get('completion_rate', 0)}%  Friction: {summary.get('friction_points', 0)}")
            if workflow.get("task_types"):
                top_tasks = ", ".join(f"{t['task_type']} ({t['count']})" for t in workflow["task_types"][:4])
                lines.append(f"  Task mix: {top_tasks}")
            if workflow.get("outcomes"):
                outcomes = ", ".join(f"{o['outcome']} ({o['count']})" for o in workflow["outcomes"][:4])
                lines.append(f"  Outcomes: {outcomes}")
            if workflow.get("friction_points"):
                lines.append("  Friction:")
                for point in workflow["friction_points"][:4]:
                    lines.append(f"    - {point['type']}: {point['evidence']}")
            if workflow.get("recommendations"):
                lines.append("  Recommendations (approval-gated):")
                for rec in workflow["recommendations"][:4]:
                    priority = rec.get("priority", "info")
                    action = rec.get("action")
                    lines.append(f"    - [{priority} · {rec['layer']}] {rec['recommendation']}")
                    if action:
                        lines.append(f"      Action: {action}")
            lines.append("")

        # Notable sessions
        if report.get("top_sessions"):
            lines.append("  🏆 Notable Sessions")
            lines.append("  " + "─" * 56)
            for ts in report["top_sessions"]:
                lines.append(f"  {ts['label']:<20} {ts['value']:<18} ({ts['date']}, {ts['session_id']})")
            lines.append("")

        return "\n".join(lines)

    def format_markdown(self, report: Dict) -> str:
        """Format the insights report as Markdown for saved/shared reports."""
        if report.get("empty"):
            days = report.get("days", 30)
            return f"# Hermes Insights\n\nNo sessions found in the last {days} days."
        o = report["overview"]
        lines = [f"# Hermes Insights — Last {report['days']} days", ""]
        if report.get("source_filter"):
            lines.append(f"Source filter: `{report['source_filter']}`")
            lines.append("")
        lines.extend([
            "## Overview",
            f"- Sessions: {o['total_sessions']}",
            f"- Messages: {o['total_messages']:,}",
            f"- Tool calls: {o['total_tool_calls']:,}",
            f"- Total tokens: {o['total_tokens']:,}",
            "",
        ])
        if report.get("tools"):
            lines.extend(["## Top Tools", ""])
            for tool in report["tools"][:10]:
                lines.append(f"- `{tool['tool']}` — {tool['count']:,} calls ({tool['percentage']:.1f}%)")
            lines.append("")
        workflow = report.get("workflow_intelligence")
        if workflow:
            summary = workflow.get("summary", {})
            lines.extend([
                "## Workflow Intelligence",
                f"- Sessions analyzed: {summary.get('sessions_analyzed', 0)}",
                f"- Messages analyzed: {summary.get('messages_analyzed', 0)}",
                f"- Completion rate: {summary.get('completion_rate', 0)}%",
                f"- Friction points: {summary.get('friction_points', 0)}",
                f"- Friction rate: {summary.get('friction_rate', 0)}%",
                f"- Method: `{workflow.get('method', 'unknown')}`",
                "- Approval-gated: recommendations are not auto-applied.",
                "",
            ])
            if workflow.get("task_types"):
                lines.append("### Task Mix")
                for item in workflow["task_types"]:
                    lines.append(f"- {item['task_type']}: {item['count']}")
                lines.append("")
            if workflow.get("what_worked"):
                lines.append("### What Worked")
                for item in workflow["what_worked"]:
                    lines.append(f"- {item['pattern']} × {item['count']}")
                lines.append("")
            if workflow.get("quick_wins"):
                lines.append("### Quick Wins")
                for item in workflow["quick_wins"]:
                    lines.append(f"- {item}")
                lines.append("")
            if workflow.get("friction_points"):
                lines.append("### Friction Points")
                for point in workflow["friction_points"]:
                    lines.append(f"- `{point['type']}` ({point.get('session_id', '')}): {point['evidence']}")
                lines.append("")
            if workflow.get("recommendations"):
                lines.append("### Recommendations")
                for rec in workflow["recommendations"]:
                    lines.append(f"- **{rec.get('priority', 'info').upper()} · {rec['layer']}**: {rec['recommendation']} — {rec['reason']} Action: {rec.get('action', 'Review manually before applying.')}")
                lines.append("")
        return "\n".join(lines)

    def format_html(self, report: Dict) -> str:
        """Format the insights report as a self-contained HTML dashboard."""
        def esc(value: Any) -> str:
            return html.escape(str(value), quote=True)

        if report.get("empty"):
            days = report.get("days", 30)
            return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><title>Hermes Insights</title></head>
<body><h1>Hermes Insights</h1><p>No sessions found in the last {esc(days)} days.</p></body></html>"""

        o = report["overview"]
        workflow = report.get("workflow_intelligence") or {}
        summary = workflow.get("summary", {})
        source = report.get("source_filter") or "all sources"
        period = f"Last {report['days']} days · {source}"

        def metric(label: str, value: Any) -> str:
            return f"<div class='metric'><span>{esc(label)}</span><strong>{esc(value)}</strong></div>"

        def rows(items: List[Dict[str, Any]], key_name: str, value_name: str = "count") -> str:
            if not items:
                return "<p class='muted'>No data.</p>"
            body = "".join(
                f"<tr><td>{esc(item.get(key_name, ''))}</td><td>{esc(item.get(value_name, ''))}</td></tr>"
                for item in items
            )
            return f"<thead><tr><th>{esc(key_name.replace('_', ' ').title())}</th><th>{esc(value_name.replace('_', ' ').title())}</th></tr></thead><tbody>{body}</tbody>"

        def progress(label: str, value: float, klass: str = "") -> str:
            safe_value = max(0.0, min(100.0, float(value or 0)))
            return (
                "<div class='progress-row'>"
                f"<div><strong>{esc(label)}</strong><span>{safe_value:.1f}%</span></div>"
                f"<div class='bar'><i class='{esc(klass)}' style='width:{safe_value:.1f}%'></i></div>"
                "</div>"
            )

        recommendation_cards = []
        for rec in workflow.get("recommendations", []):
            priority = rec.get("priority", "info")
            recommendation_cards.append(
                "<article class='card rec'>"
                f"<div class='pill {esc(priority)}'>{esc(priority)}</div>"
                f"<h3>{esc(rec.get('recommendation', ''))}</h3>"
                f"<p><strong>Layer:</strong> {esc(rec.get('layer', ''))}</p>"
                f"<p><strong>Why:</strong> {esc(rec.get('reason', ''))}</p>"
                f"<p><strong>Suggested action:</strong> {esc(rec.get('action', 'Review manually before applying.'))}</p>"
                "</article>"
            )

        friction_items = "".join(
            f"<li><code>{esc(point.get('type', ''))}</code> <span class='muted'>({esc(point.get('task_type', ''))} · {esc(point.get('session_id', ''))})</span><br>{esc(point.get('evidence', ''))}</li>"
            for point in workflow.get("friction_points", [])
        ) or "<li class='muted'>No friction points detected.</li>"
        worked_items = "".join(
            f"<li>{esc(item.get('pattern', ''))} <span class='muted'>× {esc(item.get('count', 0))}</span></li>"
            for item in workflow.get("what_worked", [])
        ) or "<li class='muted'>No recurring success patterns detected.</li>"
        quick_win_items = "".join(f"<li>{esc(item)}</li>" for item in workflow.get("quick_wins", [])) or "<li class='muted'>No quick wins generated.</li>"
        safety_items = "".join(f"<li>{esc(item)}</li>" for item in workflow.get("safety_or_privacy_flags", []))

        return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Hermes Insights — {esc(period)}</title>
  <style>
    :root {{ color-scheme: dark; --bg:#090b10; --panel:#111827; --panel2:#0f172a; --text:#e5e7eb; --muted:#94a3b8; --line:#1f2937; --accent:#8b5cf6; --good:#22c55e; --warn:#f59e0b; --bad:#ef4444; }}
    * {{ box-sizing: border-box; }}
    body {{ margin:0; font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI, sans-serif; background: radial-gradient(circle at top left, #1e1b4b 0, var(--bg) 34rem); color:var(--text); }}
    main {{ max-width:1120px; margin:0 auto; padding:40px 20px 64px; }}
    h1 {{ font-size:42px; margin:0 0 8px; letter-spacing:-0.04em; }}
    h2 {{ margin:30px 0 14px; }}
    .muted {{ color:var(--muted); }}
    .grid {{ display:grid; gap:14px; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); }}
    .metric, .card, section {{ background:linear-gradient(180deg, rgba(255,255,255,.055), rgba(255,255,255,.025)); border:1px solid var(--line); border-radius:18px; padding:16px; box-shadow:0 14px 40px rgba(0,0,0,.28); }}
    .metric span {{ display:block; color:var(--muted); font-size:13px; }}
    .metric strong {{ display:block; font-size:28px; margin-top:6px; }}
    .two {{ display:grid; gap:16px; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); }}
    table {{ width:100%; border-collapse:collapse; }}
    td, th {{ padding:9px 0; border-bottom:1px solid var(--line); text-align:left; }}
    .pill {{ display:inline-block; padding:3px 9px; border-radius:999px; font-size:12px; text-transform:uppercase; letter-spacing:.08em; background:#334155; color:#dbeafe; }}
    .pill.high {{ background:rgba(239,68,68,.16); color:#fecaca; }} .pill.medium {{ background:rgba(245,158,11,.16); color:#fde68a; }} .pill.low {{ background:rgba(34,197,94,.14); color:#bbf7d0; }} .pill.info {{ background:rgba(139,92,246,.18); color:#ddd6fe; }}
    .rec h3 {{ margin:10px 0; }}
    .progress-row {{ display:grid; gap:8px; margin:12px 0; }} .progress-row div:first-child {{ display:flex; justify-content:space-between; color:var(--muted); }}
    .bar {{ height:10px; border-radius:999px; background:#1f2937; overflow:hidden; }} .bar i {{ display:block; height:100%; background:linear-gradient(90deg, var(--accent), #22d3ee); }} .bar i.good {{ background:linear-gradient(90deg, var(--good), #84cc16); }} .bar i.warn {{ background:linear-gradient(90deg, var(--warn), var(--bad)); }}
    li {{ margin:8px 0; }} code {{ color:#c4b5fd; }}
    footer {{ margin-top:28px; color:var(--muted); font-size:13px; }}
  </style>
</head>
<body><main>
  <header>
    <div class="pill info">approval-gated</div>
    <h1>Hermes Insights</h1>
    <p class="muted">{esc(period)} · heuristic workflow analysis; recommendations are not auto-applied.</p>
  </header>

  <section><h2>Overview</h2><div class="grid">
    {metric('Sessions', f"{o['total_sessions']:,}")}
    {metric('Messages', f"{o['total_messages']:,}")}
    {metric('Tool calls', f"{o['total_tool_calls']:,}")}
    {metric('Tokens', f"{o['total_tokens']:,}")}
    {metric('Estimated cost', f"${o.get('estimated_cost', 0):.4f}")}
    {metric('Active time', '~' + _format_duration(o['total_hours'] * 3600) if o.get('total_hours', 0) else '0s')}
  </div></section>

  <section><h2>Workflow Health</h2>
    {progress('Completion rate', summary.get('completion_rate', 0), 'good')}
    {progress('Friction rate', summary.get('friction_rate', 0), 'warn')}
    <p class="muted">{esc(summary.get('completed_sessions', 0))} completed, {esc(summary.get('blocked_sessions', 0))} blocked, {esc(summary.get('friction_points', 0))} friction point(s).</p>
  </section>

  <div class="two">
    <section><h2>Task Mix</h2><table>{rows(workflow.get('task_types', []), 'task_type')}</table></section>
    <section><h2>Outcomes</h2><table>{rows(workflow.get('outcomes', []), 'outcome')}</table></section>
  </div>

  <section><h2>Recommendations</h2><div class="grid">{''.join(recommendation_cards) or '<p class="muted">No recommendations generated.</p>'}</div></section>

  <div class="two">
    <section><h2>What Worked</h2><ul>{worked_items}</ul></section>
    <section><h2>Quick Wins</h2><ul>{quick_win_items}</ul></section>
  </div>

  <section><h2>Friction Points</h2><ul>{friction_items}</ul></section>
  <section><h2>Top Tools</h2><table>{rows(report.get('tools', [])[:10], 'tool')}</table></section>
  <section><h2>Safety & Privacy</h2><ul>{safety_items}</ul><p class="muted">Generated from aggregate metadata and static evidence labels; raw tool output and prompt dumps are not included.</p></section>
  <footer>Generated by Hermes Insights · method: {esc(workflow.get('method', 'standard_usage_report'))}</footer>
</main></body></html>"""

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

        workflow = report.get("workflow_intelligence")
        if workflow:
            summary = workflow.get("summary", {})
            lines.append("**🧭 Workflow Intelligence:**")
            lines.append(f"  {summary.get('sessions_analyzed', 0)} sessions analyzed, {summary.get('friction_points', 0)} friction points")
            for rec in workflow.get("recommendations", [])[:2]:
                lines.append(f"  Approval-gated: {rec['recommendation']}")
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
