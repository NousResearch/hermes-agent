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

import json
import re
import shlex
import time
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
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


def parse_insights_args(arg_string: str = "") -> Dict[str, Any]:
    """Parse shared CLI/gateway `/insights` arguments.

    Supports: `7`, `--days 7`, `--source cli`, `--qualitative`, and
    `qualitative`. Unknown tokens are ignored for backwards compatibility.
    """
    arg_string = (arg_string or "").strip()
    arg_string = re.sub(r'[\u2012\u2013\u2014\u2015](days|source|qualitative|no-write|no-report)', r'--\1', arg_string)
    try:
        parts = shlex.split(arg_string) if arg_string else []
    except ValueError as exc:
        return {"days": 30, "source": None, "qualitative": False, "write_report": True, "error": str(exc)}
    parsed = {"days": 30, "source": None, "qualitative": False, "write_report": True, "error": None}
    i = 0
    while i < len(parts):
        part = parts[i]
        if part in {"--qualitative", "qualitative", "coach", "retrospective"}:
            parsed["qualitative"] = True
            i += 1
        elif part in {"--no-write", "--no-report"}:
            parsed["write_report"] = False
            i += 1
        elif part == "--days" and i + 1 < len(parts):
            try:
                parsed["days"] = int(parts[i + 1])
            except ValueError:
                parsed["error"] = f"Invalid --days value: {parts[i + 1]}"
            i += 2
        elif part.startswith("--days="):
            value = part.split("=", 1)[1]
            try:
                parsed["days"] = int(value)
            except ValueError:
                parsed["error"] = f"Invalid --days value: {value}"
            i += 1
        elif part == "--source" and i + 1 < len(parts):
            parsed["source"] = parts[i + 1]
            i += 2
        elif part.startswith("--source="):
            parsed["source"] = part.split("=", 1)[1]
            i += 1
        elif part.isdigit():
            parsed["days"] = int(part)
            i += 1
        else:
            i += 1
    return parsed


_CORRECTION_RE = re.compile(
    r"\b(no|wrong|that's wrong|not what i asked|i asked|don't|do not|stop|instead|you forgot|you missed|you ignored|why did you|premature)\b",
    re.IGNORECASE,
)
_TOOL_FAILURE_RE = re.compile(
    r"\b(error|failed|failure|traceback|exception|timeout|timed out|permission denied|not found|command not found|old_string not found|exit_code\W*[1-9])\b",
    re.IGNORECASE,
)
_ACT_BEFORE_UNDERSTANDING_RE = re.compile(
    r"\b(just patch|patch .*now|fix it now|i'll (?:just )?(?:patch|change|edit|implement)|let me (?:just )?(?:patch|change|edit|implement))\b",
    re.IGNORECASE,
)
_MISSING_CONTEXT_RE = re.compile(
    r"\b(need context|missing context|not enough context|can't find|cannot find|unclear|ambiguous|which file|what path)\b",
    re.IGNORECASE,
)
_TOO_VERBOSE_RE = re.compile(
    r"\b(too much explanation|be concise|less explanation|stop explaining|just do it|low-volume|shorter)\b",
    re.IGNORECASE,
)
_REQUIRED_SKILLS = {
    "hermes-agent": ("hermes", "config", "gateway", "profile", "provider", "model", "tool", "skill"),
    "systematic-debugging": ("debug", "bug", "error", "failing", "failure", "root cause"),
    "test-driven-development": ("test", "tdd", "implement", "feature", "regression"),
    "writing-plans": ("plan", "spec", "implementation plan", "acceptance criteria"),
}


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
        }

    def generate_qualitative(self, days: int = 30, source: str = None) -> Dict[str, Any]:
        """Generate qualitative workflow coaching from local session transcripts.

        This is heuristic-only by default. It does not call an LLM and treats
        transcript text as untrusted data. Examples are short and forcibly
        secret-redacted before entering the report.
        """
        cutoff = time.time() - (days * 86400)
        sessions = self._get_qualitative_sessions(cutoff, source)
        if not sessions:
            return {
                "days": days,
                "source_filter": source,
                "empty": True,
                "summary": {},
                "friction": {},
                "recommendations": {},
                "projects": [],
                "safety": {
                    "local_only": True,
                    "raw_transcripts_included": False,
                    "redaction": "forced",
                },
            }

        session_ids = [s["id"] for s in sessions]
        messages = self._get_qualitative_messages(session_ids)
        messages_by_session: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        for msg in messages:
            messages_by_session[msg["session_id"]].append(msg)

        friction = self._compute_qualitative_friction(messages)
        projects = self._compute_project_breakdown(sessions, messages_by_session, friction)
        recommendations = self._build_qualitative_recommendations(friction, projects)
        summary = self._build_qualitative_summary(friction, recommendations, projects)

        return {
            "days": days,
            "source_filter": source,
            "empty": False,
            "generated_at": time.time(),
            "session_count": len(sessions),
            "message_count": len(messages),
            "summary": summary,
            "friction": friction,
            "recommendations": recommendations,
            "projects": projects,
            "safety": {
                "local_only": True,
                "raw_transcripts_included": False,
                "redaction": "forced",
            },
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

    def _get_qualitative_sessions(self, cutoff: float, source: str = None) -> List[Dict[str, Any]]:
        base = """SELECT id, source, model, started_at, ended_at, message_count,
                         tool_call_count, title
                  FROM sessions
                  WHERE started_at >= ?"""
        params: tuple[Any, ...]
        if source:
            base += " AND source = ?"
            params = (cutoff, source)
        else:
            params = (cutoff,)
        base += " ORDER BY started_at DESC LIMIT 200"
        cursor = self._conn.execute(base, params)
        return [dict(row) for row in cursor.fetchall()]

    def _get_qualitative_messages(self, session_ids: List[str]) -> List[Dict[str, Any]]:
        if not session_ids:
            return []
        placeholders = ",".join("?" for _ in session_ids)
        cursor = self._conn.execute(
            "SELECT session_id, role, content, tool_calls, tool_name, timestamp "
            f"FROM messages WHERE session_id IN ({placeholders}) "
            "ORDER BY timestamp, id",
            tuple(session_ids),
        )
        rows = []
        for row in cursor.fetchall():
            msg = dict(row)
            if msg.get("tool_calls"):
                try:
                    msg["tool_calls"] = json.loads(msg["tool_calls"])
                except (json.JSONDecodeError, TypeError):
                    msg["tool_calls"] = []
            rows.append(msg)
        return rows

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

    def _redact_excerpt(self, text: Any, limit: int = 180) -> str:
        if text is None:
            return ""
        if not isinstance(text, str):
            text = str(text)
        text = " ".join(text.split())
        if len(text) > limit:
            text = text[: limit - 1].rstrip() + "…"
        try:
            from agent.redact import redact_sensitive_text

            return redact_sensitive_text(text, force=True)
        except Exception:
            return re.sub(r"sk-[A-Za-z0-9_-]{10,}", "***", text)

    def _skill_names_from_message(self, msg: Dict[str, Any]) -> set[str]:
        skills: set[str] = set()
        calls = msg.get("tool_calls") or []
        if not isinstance(calls, list):
            return skills
        for call in calls:
            func = call.get("function", {}) if isinstance(call, dict) else {}
            if func.get("name") not in {"skill_view", "skill_manage"}:
                continue
            args = func.get("arguments")
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except (json.JSONDecodeError, TypeError):
                    args = {}
            if isinstance(args, dict) and args.get("name"):
                skills.add(str(args["name"]))
        return skills

    def _loaded_skill_names(self, messages: List[Dict[str, Any]]) -> set[str]:
        skills: set[str] = set()
        for msg in messages:
            skills.update(self._skill_names_from_message(msg))
        return skills

    def _event(self, msg: Dict[str, Any], reason: str) -> Dict[str, Any]:
        return {
            "session_id": str(msg.get("session_id", ""))[:16],
            "role": msg.get("role"),
            "tool": msg.get("tool_name"),
            "reason": reason,
            "excerpt": self._redact_excerpt(msg.get("content")),
        }

    def _tool_output_has_failure(self, content: Any) -> bool:
        """Return True for real tool failures, avoiding JSON fields like error=null."""
        if content is None:
            return False
        text = content if isinstance(content, str) else str(content)
        stripped = text.strip()
        if stripped.startswith("{"):
            try:
                data = json.loads(stripped)
                if isinstance(data, dict):
                    if data.get("success") is False:
                        return True
                    exit_code = data.get("exit_code")
                    if isinstance(exit_code, int) and exit_code != 0:
                        return True
                    # Successful tool envelopes often contain words like
                    # "error", "not found", or "fallback" in harmless data.
                    # Do not scan their payloads as failures.
                    if data.get("success") is True or exit_code == 0:
                        return False
                    error = data.get("error")
                    if error not in (None, "", False):
                        return True
                    results = data.get("results")
                    if isinstance(results, list):
                        saw_result = False
                        for item in results:
                            if not isinstance(item, dict):
                                continue
                            saw_result = True
                            if item.get("error") not in (None, "", False):
                                return True
                            status = str(item.get("status") or "").lower()
                            if status in {"failed", "error", "timeout"}:
                                return True
                            summary = item.get("summary")
                            if isinstance(summary, str) and _TOOL_FAILURE_RE.search(summary):
                                return True
                        if saw_result:
                            return False
                    output = data.get("output")
                    if isinstance(output, str):
                        return bool(_TOOL_FAILURE_RE.search(output))
            except (json.JSONDecodeError, TypeError):
                pass
        cleaned = re.sub(r'"error"\s*:\s*null', "", text, flags=re.IGNORECASE)
        cleaned = re.sub(r'"success"\s*:\s*true', "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r'"exit_code"\s*:\s*0', "", cleaned, flags=re.IGNORECASE)
        return bool(_TOOL_FAILURE_RE.search(cleaned))

    def _compute_qualitative_friction(self, messages: List[Dict[str, Any]]) -> Dict[str, Any]:
        buckets = {
            "user_corrections": {"count": 0, "examples": [], "description": "User corrections or reversals."},
            "acted_before_understanding": {"count": 0, "examples": [], "description": "Assistant appeared to edit/implement before inspection or planning."},
            "tool_failures": {"count": 0, "examples": [], "description": "Tool errors, failed commands, timeouts, or permission problems."},
            "missing_context": {"count": 0, "examples": [], "description": "Messages indicating missing paths, ambiguity, or insufficient context."},
            "missed_tool_or_skill": {"count": 0, "examples": [], "description": "Likely missed skill/session_search/memory/todo/verification opportunities."},
            "verbosity_mismatch": {"count": 0, "examples": [], "description": "User asked for less explanation or more concise execution."},
        }

        loaded_by_session: Dict[str, set[str]] = defaultdict(set)
        for msg in messages:
            role = msg.get("role")
            content = msg.get("content") or ""
            lower = content.lower() if isinstance(content, str) else ""
            session_id = str(msg.get("session_id") or "")
            loaded_so_far = loaded_by_session[session_id]
            if role == "user":
                if _CORRECTION_RE.search(content):
                    buckets["user_corrections"]["count"] += 1
                    if len(buckets["user_corrections"]["examples"]) < 5:
                        buckets["user_corrections"]["examples"].append(self._event(msg, "correction language"))
                if _TOO_VERBOSE_RE.search(content):
                    buckets["verbosity_mismatch"]["count"] += 1
                    if len(buckets["verbosity_mismatch"]["examples"]) < 5:
                        buckets["verbosity_mismatch"]["examples"].append(self._event(msg, "verbosity preference"))
                for skill, keywords in _REQUIRED_SKILLS.items():
                    if skill not in loaded_so_far and any(k in lower for k in keywords):
                        buckets["missed_tool_or_skill"]["count"] += 1
                        if len(buckets["missed_tool_or_skill"]["examples"]) < 5:
                            event = self._event(msg, f"consider loading {skill}")
                            buckets["missed_tool_or_skill"]["examples"].append(event)
                        break
                if any(k in lower for k in ("last time", "remember when", "we did this before", "as mentioned")):
                    buckets["missed_tool_or_skill"]["count"] += 1
                    if len(buckets["missed_tool_or_skill"]["examples"]) < 5:
                        buckets["missed_tool_or_skill"]["examples"].append(self._event(msg, "consider session_search earlier"))
            elif role == "assistant":
                if _ACT_BEFORE_UNDERSTANDING_RE.search(content):
                    buckets["acted_before_understanding"]["count"] += 1
                    if len(buckets["acted_before_understanding"]["examples"]) < 5:
                        buckets["acted_before_understanding"]["examples"].append(self._event(msg, "implementation language before evidence"))
            elif role == "tool":
                tool_failed = self._tool_output_has_failure(content)
                if tool_failed:
                    buckets["tool_failures"]["count"] += 1
                    if len(buckets["tool_failures"]["examples"]) < 5:
                        buckets["tool_failures"]["examples"].append(self._event(msg, "tool failure text"))
                if tool_failed and _MISSING_CONTEXT_RE.search(content):
                    buckets["missing_context"]["count"] += 1
                    if len(buckets["missing_context"]["examples"]) < 5:
                        buckets["missing_context"]["examples"].append(self._event(msg, "missing context text"))
            loaded_by_session[session_id].update(self._skill_names_from_message(msg))
        return buckets

    def _compute_project_breakdown(
        self,
        sessions: List[Dict[str, Any]],
        messages_by_session: Dict[str, List[Dict[str, Any]]],
        friction: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        project_data: Dict[str, Dict[str, Any]] = {}
        event_sessions: Dict[str, Counter] = defaultdict(Counter)
        for name, bucket in friction.items():
            for ex in bucket.get("examples", []):
                event_sessions[ex.get("session_id")][name] += 1

        for session in sessions:
            raw_label = (session.get("title") or session.get("source") or "unknown").strip()[:120]
            label = self._redact_excerpt(raw_label, limit=60) or "untitled"
            entry = project_data.setdefault(label, {
                "project": label,
                "source": self._redact_excerpt(session.get("source"), limit=40),
                "sessions": 0,
                "messages": 0,
                "recurring_tasks": Counter(),
                "failure_modes": Counter(),
            })
            entry["sessions"] += 1
            entry["messages"] += int(session.get("message_count") or 0)
            sid_short = str(session.get("id", ""))[:16]
            entry["failure_modes"].update(event_sessions.get(sid_short, {}))
            text = " ".join(str(m.get("content") or "").lower() for m in messages_by_session.get(session["id"], []) if m.get("role") == "user")
            for task, keys in {
                "coding/debugging": ("bug", "test", "fix", "implement", "code"),
                "Hermes configuration": ("hermes", "config", "provider", "gateway", "profile"),
                "research/analysis": ("research", "analyze", "summary", "report"),
                "planning/spec": ("plan", "spec", "acceptance criteria"),
            }.items():
                if any(k in text for k in keys):
                    entry["recurring_tasks"][task] += 1

        result = []
        for entry in project_data.values():
            result.append({
                **entry,
                "recurring_tasks": [name for name, _ in entry["recurring_tasks"].most_common(4)],
                "failure_modes": [name for name, _ in entry["failure_modes"].most_common(4)],
            })
        result.sort(key=lambda x: (x["sessions"], x["messages"]), reverse=True)
        return result[:12]

    def _build_qualitative_recommendations(self, friction: Dict[str, Any], projects: List[Dict[str, Any]]) -> Dict[str, List[str]]:
        recs = {
            "skill_updates": [],
            "candidate_new_skills": [],
            "memory_entries": [],
            "obsidian_notes": [],
            "config_tooling": [],
            "prompt_patterns": [],
        }
        if friction["user_corrections"]["count"]:
            recs["skill_updates"].append("Patch umbrella implementation/debugging skills with a rule: when the user says inspect/plan/load-skill first, do that before edits.")
            recs["prompt_patterns"].append("Start complex requests with: 'First inspect X/Y/Z, then write acceptance criteria, then implement only after tests fail.'")
        if friction["acted_before_understanding"]["count"]:
            recs["skill_updates"].append("Add to systematic-debugging: no patch/write actions until evidence has been gathered and the suspected root cause is stated.")
        if friction["tool_failures"]["count"]:
            recs["config_tooling"].append("For repeated tool failures, prefer read/search verification before patching and include exact failing command/output in the next attempt.")
        if friction["missed_tool_or_skill"]["count"]:
            recs["skill_updates"].append("Patch hermes-agent and class-level software-development skills with stronger trigger phrases for skill_view, session_search, todo, and verification.")
            recs["candidate_new_skills"].append("qualitative-retrospective: reusable workflow for auditing Hermes transcripts and converting findings into skill/memory/wiki updates.")
        if friction["verbosity_mismatch"]["count"]:
            recs["memory_entries"].append("User prefers concise, low-volume implementation/debugging updates; save only if not already present.")
        if any("Hermes" in p.get("project", "") for p in projects):
            recs["obsidian_notes"].append("Obsidian LLM Wiki note: Hermes qualitative-insights design, heuristics, and next iterations.")
        recs["prompt_patterns"].append("For retrospectives: 'Analyze local session history only; redact secrets; output recommendations as skill patches, memory candidates, wiki notes, and config changes.'")
        return recs

    def _build_qualitative_summary(
        self,
        friction: Dict[str, Any],
        recommendations: Dict[str, List[str]],
        projects: List[Dict[str, Any]],
    ) -> Dict[str, List[str]]:
        total_failures = friction["tool_failures"]["count"]
        total_corrections = friction["user_corrections"]["count"]
        return {
            "what_works": [
                "Hermes session history contains enough structured roles/tool metadata to infer workflow patterns locally.",
                "Skill and tool usage are visible, so missed-trigger recommendations can be grounded in actual transcript behavior.",
            ],
            "what_hinders": [
                f"Detected {total_corrections} user correction(s) and {total_failures} tool failure signal(s) in the selected window.",
                "Heuristics can identify likely friction, but they cannot fully infer intent without an optional LLM synthesis pass.",
            ],
            "quick_wins": (recommendations.get("skill_updates") or recommendations.get("prompt_patterns") or [])[:3],
            "ambitious_improvements": [
                "Add optional LLM synthesis over redacted event summaries, not raw transcripts.",
                "Track workdir/project metadata directly in session rows for stronger workspace-level coaching.",
            ],
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

        # Notable sessions
        if report.get("top_sessions"):
            lines.append("  🏆 Notable Sessions")
            lines.append("  " + "─" * 56)
            for ts in report["top_sessions"]:
                lines.append(f"  {ts['label']:<20} {ts['value']:<18} ({ts['date']}, {ts['session_id']})")
            lines.append("")

        return "\n".join(lines)

    def _append_bullets(self, lines: List[str], items: List[str], indent: str = "  ") -> None:
        if not items:
            lines.append(f"{indent}- None detected.")
            return
        for item in items:
            lines.append(f"{indent}- {item}")

    def format_qualitative_terminal(self, report: Dict[str, Any]) -> str:
        """Format qualitative insights for terminal display."""
        if report.get("empty"):
            days = report.get("days", 30)
            src = f" (source: {report['source_filter']})" if report.get("source_filter") else ""
            return f"  No sessions found in the last {days} days{src}."

        lines: List[str] = []
        days = report.get("days", 30)
        src_filter = report.get("source_filter")
        label = f"Last {days} days" + (f" ({src_filter})" if src_filter else "")
        lines.append("")
        lines.append("  Hermes Qualitative Insights")
        lines.append(f"  {label} · {report.get('session_count', 0)} sessions · local transcripts only")
        lines.append("  " + "=" * 72)
        lines.append("")

        summary = report.get("summary", {})
        lines.append("  At a glance")
        lines.append("  " + "-" * 72)
        for title, key in (("What's working", "what_works"), ("What's hindering", "what_hinders"), ("Quick wins", "quick_wins"), ("Ambitious improvements", "ambitious_improvements")):
            lines.append(f"  {title}:")
            self._append_bullets(lines, summary.get(key, []), indent="    ")
        lines.append("")

        lines.append("  Friction Analysis")
        lines.append("  " + "-" * 72)
        for key, bucket in (report.get("friction") or {}).items():
            count = bucket.get("count", 0)
            title = key.replace("_", " ").title()
            lines.append(f"  {title}: {count}")
            for ex in bucket.get("examples", [])[:3]:
                tool = f" tool={ex['tool']}" if ex.get("tool") else ""
                lines.append(f"    - {ex.get('reason')} ({ex.get('session_id')}{tool}): {ex.get('excerpt')}")
        lines.append("")

        recs = report.get("recommendations", {})
        lines.append("  Actionable Recommendations")
        lines.append("  " + "-" * 72)
        for title, key in (("Skill updates", "skill_updates"), ("Candidate new skills", "candidate_new_skills"), ("Memory candidates", "memory_entries"), ("Obsidian notes", "obsidian_notes"), ("Config/tooling", "config_tooling"), ("Prompt patterns", "prompt_patterns")):
            lines.append(f"  {title}:")
            self._append_bullets(lines, recs.get(key, []), indent="    ")
        lines.append("")

        projects = report.get("projects") or []
        if projects:
            lines.append("  Project / Workspace Breakdown")
            lines.append("  " + "-" * 72)
            for project in projects[:8]:
                tasks = ", ".join(project.get("recurring_tasks") or ["uncategorized"])
                failures = ", ".join(project.get("failure_modes") or ["none detected"])
                lines.append(f"  - {project['project']} ({project['sessions']} sessions, {project.get('source')})")
                lines.append(f"    tasks: {tasks}")
                lines.append(f"    friction: {failures}")
            lines.append("")

        lines.append("  Safety / Privacy")
        lines.append("  " + "-" * 72)
        lines.append("  - Analyzed local Hermes SQLite session history only.")
        lines.append("  - Raw transcripts are not included; examples are short excerpts.")
        lines.append("  - Report text uses forced secret redaction for likely keys/tokens.")
        if report.get("report_path"):
            lines.append(f"  - Markdown report: {report['report_path']}")
        return "\n".join(lines)

    def format_qualitative_markdown(self, report: Dict[str, Any]) -> str:
        """Format qualitative insights as Markdown."""
        if report.get("empty"):
            return f"# Hermes Qualitative Insights\n\nNo sessions found in the last {report.get('days', 30)} days.\n"

        lines = [
            "# Hermes Qualitative Insights",
            "",
            f"Window: last {report.get('days', 30)} days",
            f"Sessions analyzed: {report.get('session_count', 0)}",
            "Safety: local SQLite session history only; raw transcripts omitted; excerpts forcibly redacted.",
            "",
            "## At a glance",
        ]
        summary = report.get("summary", {})
        for title, key in (("What's working", "what_works"), ("What's hindering", "what_hinders"), ("Quick wins", "quick_wins"), ("Ambitious workflow improvements", "ambitious_improvements")):
            lines.extend([f"### {title}", ""])
            items = summary.get(key, []) or ["None detected."]
            lines.extend(f"- {item}" for item in items)
            lines.append("")

        lines.append("## Friction analysis")
        lines.append("")
        for key, bucket in (report.get("friction") or {}).items():
            lines.extend([f"### {key.replace('_', ' ').title()} ({bucket.get('count', 0)})", ""])
            lines.append(bucket.get("description", ""))
            lines.append("")
            for ex in bucket.get("examples", [])[:5]:
                lines.append(f"- `{ex.get('session_id')}` {ex.get('reason')}: {ex.get('excerpt')}")
            if not bucket.get("examples"):
                lines.append("- None detected.")
            lines.append("")

        lines.append("## Actionable recommendations")
        lines.append("")
        for title, key in (("Copy-pasteable skill updates", "skill_updates"), ("Candidate new skills", "candidate_new_skills"), ("Suggested memory entries", "memory_entries"), ("Suggested Obsidian LLM Wiki notes", "obsidian_notes"), ("Suggested config/tooling changes", "config_tooling"), ("Better prompt patterns", "prompt_patterns")):
            lines.extend([f"### {title}", ""])
            items = (report.get("recommendations") or {}).get(key, []) or ["None detected."]
            lines.extend(f"- {item}" for item in items)
            lines.append("")

        lines.append("## Project / workspace breakdown")
        lines.append("")
        for project in report.get("projects") or []:
            lines.append(f"- **{project['project']}**: {project['sessions']} session(s), source `{project.get('source')}`")
            lines.append(f"  - Recurring task types: {', '.join(project.get('recurring_tasks') or ['uncategorized'])}")
            lines.append(f"  - Recurring failure modes: {', '.join(project.get('failure_modes') or ['none detected'])}")
        lines.append("")
        return "\n".join(lines)

    def write_qualitative_markdown(self, report: Dict[str, Any], output_dir: Path | str = None) -> Path:
        """Write the qualitative Markdown report under HERMES_HOME/insights by default."""
        if output_dir is None:
            from hermes_constants import get_hermes_home

            output_dir = get_hermes_home() / "insights"
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        path = output_path / f"qualitative-report-{stamp}.md"
        path.write_text(self.format_qualitative_markdown(report), encoding="utf-8")
        report["report_path"] = str(path)
        return path

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
