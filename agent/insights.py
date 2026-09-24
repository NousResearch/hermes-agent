"""Session Insights Engine: aggregates the SQLite state DB into usage insights (tokens, cost estimates, tool/skill
usage, activity, model/platform breakdowns). ``InsightsEngine(db).generate(days=30)`` → ``format_terminal(report)``."""

import json
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from typing import Any, Dict, List, Mapping, Optional, Protocol, runtime_checkable

from agent.usage_pricing import CanonicalUsage, estimate_usage_cost, format_cost_label, format_duration_compact, has_known_pricing
from hermes_cli.timefmt import coerce_epoch
from hermes_time import safe_strftime

_TOKEN_KEYS = ("input_tokens", "output_tokens", "cache_read_tokens", "cache_write_tokens")
_SKILL_TOOLS = {"skill_view", "skill_manage"}
_SESSION_COLS = ("id, source, model, started_at, ended_at, message_count, tool_call_count, "
                 "input_tokens, output_tokens, cache_read_tokens, cache_write_tokens, billing_provider, "
                 "billing_base_url, billing_mode, estimated_cost_usd, actual_cost_usd, cost_status, "
                 "cost_source, api_call_count")


class InsightsReadError(RuntimeError):
    """Insights cannot produce a fresh, complete report from the selected store."""


@dataclass(frozen=True)
class InsightsSnapshot:
    """Canonical analytics rows. Backends must not expose cursors or connections here."""
    sessions: tuple[Mapping[str, Any], ...]
    tool_rows: tuple[Mapping[str, Any], ...]
    assistant_tool_call_rows: tuple[Mapping[str, Any], ...]
    skill_tool_call_rows: tuple[Mapping[str, Any], ...]
    message_stats: Mapping[str, Any]
    model_usage_rows: tuple[Mapping[str, Any], ...]


@runtime_checkable
class InsightsReadStore(Protocol):
    def flush_token_counts(self, timeout: float = 5.0) -> bool: ...
    def read_insights_snapshot(self, *, cutoff: float, source: str | None) -> InsightsSnapshot: ...


class SqliteInsightsReadStore:
    """Compatibility analytics adapter; SQLite details stay outside the engine."""
    _ASSISTANT_INDEX = "idx_messages_assistant_calls_by_session"

    def __init__(self, db: Any) -> None:
        self._db = db
        self._conn = db._conn
        try:
            self._has_assistant_index = bool(self._conn.execute(
                "SELECT 1 FROM sqlite_master WHERE type='index' AND name=?", (self._ASSISTANT_INDEX,)).fetchone())
        except Exception:
            self._has_assistant_index = False

    def flush_token_counts(self, timeout: float = 5.0) -> bool:
        flush = getattr(self._db, "flush_token_counts", None)
        return True if not callable(flush) else bool(flush(timeout))

    def read_insights_snapshot(self, *, cutoff: float, source: str | None) -> InsightsSnapshot:
        predicate, params = "s.started_at >= ?", [cutoff]
        if source is not None:
            predicate += " AND s.source = ?"
            params.append(source)
        assistant_from = "messages m"
        if self._has_assistant_index:
            assistant_from += f" INDEXED BY {self._ASSISTANT_INDEX}"
        def rows(sql: str, values: list[Any] = params) -> tuple[Mapping[str, Any], ...]:
            return tuple(dict(row) for row in self._conn.execute(sql, values).fetchall())
        sessions = rows(f"SELECT {_SESSION_COLS} FROM sessions s WHERE {predicate} ORDER BY s.started_at DESC")
        tool_rows = rows("SELECT m.session_id, m.tool_name, COUNT(*) AS count FROM messages m JOIN sessions s ON s.id=m.session_id "
                         f"WHERE {predicate} AND m.role='tool' AND m.tool_name IS NOT NULL GROUP BY m.session_id, m.tool_name")
        assistant_rows = rows(f"SELECT m.session_id, m.tool_calls FROM {assistant_from} JOIN sessions s ON s.id=m.session_id "
                              f"WHERE {predicate} AND m.role='assistant' AND m.tool_calls IS NOT NULL")
        skill_rows = rows(f"SELECT m.tool_calls, m.timestamp FROM {assistant_from} JOIN sessions s ON s.id=m.session_id "
                          f"WHERE {predicate} AND m.role='assistant' AND m.tool_calls IS NOT NULL "
                          "AND (instr(m.tool_calls, 'skill_view') > 0 OR instr(m.tool_calls, 'skill_manage') > 0)")
        stats_rows = rows("SELECT COUNT(*) AS total_messages, SUM(CASE WHEN m.role='user' THEN 1 ELSE 0 END) AS user_messages, "
                          "SUM(CASE WHEN m.role='assistant' THEN 1 ELSE 0 END) AS assistant_messages, "
                          "SUM(CASE WHEN m.role='tool' THEN 1 ELSE 0 END) AS tool_messages FROM messages m JOIN sessions s ON s.id=m.session_id "
                          f"WHERE {predicate}")
        try:
            usage_rows = rows("SELECT u.session_id, u.model, u.billing_provider, u.billing_base_url, u.api_call_count, "
                              "u.input_tokens, u.output_tokens, u.cache_read_tokens, u.cache_write_tokens, u.reasoning_tokens, "
                              "u.estimated_cost_usd, u.actual_cost_usd, u.cost_status, u.cost_source, u.billing_mode "
                              "FROM session_model_usage u JOIN sessions s ON s.id=u.session_id " + f"WHERE {predicate}")
        except Exception as exc:
            if "no such table" not in str(exc).lower():
                raise
            usage_rows = ()
        stats = dict(stats_rows[0]) if stats_rows else {}
        return InsightsSnapshot(sessions, tool_rows, assistant_rows, skill_rows, stats, usage_rows)


def _fmt_est_cost(est_cost: float) -> str:
    return format_cost_label(Decimal(str(est_cost)))


def _estimate_cost(session_or_model: Dict[str, Any] | str, input_tokens: int = 0, output_tokens: int = 0, *, cache_read_tokens: int = 0,
                   cache_write_tokens: int = 0, provider: Optional[str] = None, base_url: Optional[str] = None) -> tuple[float, str]:
    if isinstance(session_or_model, dict):
        s = session_or_model
        model = s.get("model") or ""
        usage = CanonicalUsage(**{k: s.get(k) or 0 for k in _TOKEN_KEYS})
        provider, base_url = s.get("billing_provider"), s.get("billing_base_url")
    else:
        model = session_or_model or ""
        usage = CanonicalUsage(input_tokens, output_tokens, cache_read_tokens, cache_write_tokens)
    result = estimate_usage_cost(model, usage, provider=provider, base_url=base_url)
    return float(result.amount_usd or 0.0), result.status


def _bar_chart(values: List[int], max_width: int = 20) -> List[str]:
    peak = max(values) if values else 1
    return ["" for _ in values] if peak == 0 else ["█" * max(1, int(v / peak * max_width)) if v > 0 else "" for v in values]


def _safe_float(val):
    try: return float(val) if val is not None else 0.0
    except (ValueError, TypeError): return 0.0


def _safe_int(val):
    try: return int(val) if val is not None else 0
    except (ValueError, TypeError): return 0


def _short_model(model: Optional[str]) -> str:
    return (model or "unknown").split("/")[-1]


def _parse_json(raw: Any, kind: type) -> Any:
    try:
        if isinstance(raw, str): raw = json.loads(raw)
    except (json.JSONDecodeError, TypeError): return None
    return raw if isinstance(raw, kind) else None


def _iter_functions(raw_calls: Any):
    for call in _parse_json(raw_calls, list) or []:
        if isinstance(call, dict): yield call.get("function", {})


def _hour12(hr: int) -> str:
    return f"{hr % 12 or 12}{'AM' if hr < 12 else 'PM'}"


def _day(ts: Any) -> str:
    return safe_strftime(datetime.fromtimestamp(ts), "%b %d") if ts and (ts := coerce_epoch(ts)) else "?"


class InsightsEngine:
    """Backend-neutral analysis and formatting over an :class:`InsightsReadStore`."""
    def __init__(self, store: Any):
        self.store: InsightsReadStore = store if isinstance(store, InsightsReadStore) else SqliteInsightsReadStore(store)
        self._snapshot: InsightsSnapshot | None = None

    def _fresh_snapshot(self, cutoff: float, source: str | None) -> InsightsSnapshot:
        if not self.store.flush_token_counts():
            raise InsightsReadError("token accounting flush did not complete; refusing stale insights")
        return self.store.read_insights_snapshot(cutoff=cutoff, source=source)

    def generate(self, days: int = 30, source: str = None) -> Dict[str, Any]:
        cutoff = time.time() - (days * 86400)
        self._snapshot = self._fresh_snapshot(cutoff, source)
        sessions = self._get_sessions(cutoff, source)
        tool_usage = self._get_tool_usage(cutoff, source)
        skill_usage = self._get_skill_usage(cutoff, source)
        message_stats = self._get_message_stats(cutoff, source)
        if not sessions:
            return {"days": days, "source_filter": source, "empty": True, "overview": {}, "models": [], "platforms": [], "tools": [], "skills": self._compute_skill_breakdown([]), "activity": {}, "top_sessions": []}
        models = self._compute_model_breakdown(sessions, cutoff, source)
        return {"days": days, "source_filter": source, "empty": False, "generated_at": time.time(),
                "overview": self._compute_overview(sessions, message_stats, models), "models": models,
                "platforms": self._compute_platform_breakdown(sessions), "tools": self._compute_tool_breakdown(tool_usage),
                "skills": self._compute_skill_breakdown(skill_usage), "activity": self._compute_activity_patterns(sessions),
                "top_sessions": self._compute_top_sessions(sessions)}

    def get_usage_breakdown(self, days: int = 30, source: str = None) -> Dict[str, Any]:
        cutoff = time.time() - (days * 86400)
        self._snapshot = self._fresh_snapshot(cutoff, source)
        return {"tools": self._compute_tool_breakdown(self._get_tool_usage(cutoff, source)), "skills": self._compute_skill_breakdown(self._get_skill_usage(cutoff, source))}

    def _require_snapshot(self) -> InsightsSnapshot:
        if self._snapshot is None:
            # Compatibility for callers of the formerly query-backed private helpers.
            self._snapshot = self._fresh_snapshot(0.0, None)
        return self._snapshot

    def _get_sessions(self, cutoff: float, source: str = None) -> List[Dict]:
        rows = [dict(row) for row in self._require_snapshot().sessions]
        for row in rows:
            for col in ("started_at", "ended_at"): row[col] = coerce_epoch(row.get(col), session_id=row.get("id"), field=col)
        return rows

    def _get_tool_usage(self, cutoff: float, source: str = None) -> List[Dict]:
        by_session_tool, calls_by_session_tool = Counter(), Counter()
        snapshot = self._require_snapshot()
        for row in snapshot.tool_rows: by_session_tool[(row["session_id"], row["tool_name"])] += row["count"]
        for row in snapshot.assistant_tool_call_rows:
            try: calls_by_session_tool.update((row["session_id"], name) for name in filter(None, (fn.get("name") for fn in _iter_functions(row["tool_calls"]))))
            except (TypeError, AttributeError): continue
        counts = Counter()
        for key in set(by_session_tool) | set(calls_by_session_tool): counts[key[1]] += max(by_session_tool.get(key, 0), calls_by_session_tool.get(key, 0))
        return [{"tool_name": name, "count": count} for name, count in counts.most_common()]

    def _get_skill_usage(self, cutoff: float, source: str = None) -> List[Dict]:
        skill_counts: Dict[str, Dict[str, Any]] = {}
        for row in self._require_snapshot().skill_tool_call_rows:
            for func in _iter_functions(row["tool_calls"]):
                tool_name = func.get("name")
                if tool_name not in _SKILL_TOOLS: continue
                skill_name = (_parse_json(func.get("arguments"), dict) or {}).get("name")
                if not isinstance(skill_name, str) or not skill_name.strip(): continue
                entry = skill_counts.setdefault(skill_name, {"skill": skill_name, "view_count": 0, "manage_count": 0, "last_used_at": None})
                entry["view_count" if tool_name == "skill_view" else "manage_count"] += 1
                timestamp = row.get("timestamp")
                if timestamp is not None and (entry["last_used_at"] is None or timestamp > entry["last_used_at"]): entry["last_used_at"] = timestamp
        return list(skill_counts.values())

    def _get_message_stats(self, cutoff: float, source: str = None) -> Dict:
        return {key: _safe_int(self._require_snapshot().message_stats.get(key)) for key in ("total_messages", "user_messages", "assistant_messages", "tool_messages")}

    def _get_model_usage(self, cutoff: float, source: str = None) -> List[Dict]:
        return [dict(row) for row in self._require_snapshot().model_usage_rows]

    # -------------------------------------------------------------- Compute

    def _compute_overview(self, sessions: List[Dict], message_stats: Dict, models: Optional[List[Dict]] = None) -> Dict:
        # Per-model breakdown includes auxiliary usage rows (vision/compression/
        # titles) plus reconciled residuals, while session counters carry
        # main-loop usage only — sum the breakdown when available so overview
        # totals match the per-model table and aux spend isn't undercounted.
        rows = models or sessions
        total_input, total_output, total_cache_read, total_cache_write = (sum(int(r.get(k) or 0) for r in rows) for k in _TOKEN_KEYS)
        total_tokens = total_input + total_output + total_cache_read + total_cache_write
        total_tool_calls = sum(s.get("tool_call_count") or 0 for s in sessions)
        total_messages = sum(s.get("message_count") or 0 for s in sessions)
        total_cost = actual_cost = 0.0
        models_with_pricing, models_without_pricing, status_counts = set(), set(), Counter()
        for s in sessions:
            model = s.get("model") or ""
            estimated, status = _estimate_cost(s)
            total_cost += estimated
            actual_cost += s.get("actual_cost_usd") or 0.0
            status_counts[status] += 1
            known = has_known_pricing(model, s.get("billing_provider"), s.get("billing_base_url"))
            (models_with_pricing if known else models_without_pricing).add(_short_model(model))
        if models:
            total_cost = sum(float(m.get("cost") or 0.0) for m in models)
        # Guard against negative durations from clock drift.
        durations = [s["ended_at"] - s["started_at"] for s in sessions
                     if s.get("started_at") and s.get("ended_at") and s["ended_at"] > s["started_at"]]
        started = [s["started_at"] for s in sessions if s.get("started_at")]
        n = len(sessions)
        return {
            "total_sessions": n, "total_messages": total_messages, "total_tool_calls": total_tool_calls,
            "total_input_tokens": total_input, "total_output_tokens": total_output,
            "total_cache_read_tokens": total_cache_read, "total_cache_write_tokens": total_cache_write,
            "total_tokens": total_tokens, "estimated_cost": total_cost, "actual_cost": actual_cost,
            "total_hours": sum(durations) / 3600 if durations else 0,
            "avg_session_duration": sum(durations) / len(durations) if durations else 0,
            "avg_messages_per_session": total_messages / n if sessions else 0,
            "avg_tokens_per_session": total_tokens / n if sessions else 0,
            "user_messages": message_stats.get("user_messages") or 0,
            "assistant_messages": message_stats.get("assistant_messages") or 0,
            "tool_messages": message_stats.get("tool_messages") or 0,
            "date_range_start": min(started) if started else None,
            "date_range_end": max(started) if started else None,
            "models_with_pricing": sorted(models_with_pricing),
            "models_without_pricing": sorted(models_without_pricing),
            "unknown_cost_sessions": status_counts["unknown"],
            "included_cost_sessions": status_counts["included"],
        }

    def _compute_model_breakdown(self, sessions: List[Dict], cutoff: float, source: str = None) -> List[Dict]:
        """Tokens/cost per model from session_model_usage, so a session that
        switched models via ``/model`` splits across every model it used.
        Sessions without per-model rows (pre-table data) fall back to their
        single recorded aggregate. Tool calls aren't tied to an API call, so
        they stay attributed to the session's recorded model."""
        count_keys = _TOKEN_KEYS + ("reasoning_tokens", "api_call_count")
        model_data = defaultdict(lambda: {"sessions": set(), **dict.fromkeys(_TOKEN_KEYS, 0), "reasoning_tokens": 0, "total_tokens": 0,
                                          "api_calls": 0, "tool_calls": 0, "cost": 0.0, "actual_cost": 0.0})

        def _accumulate(model, provider, base_url, session_id, counts: Dict[str, int], *,
                        stored_cost=None, actual_cost=None, cost_status=None):
            model = model or "unknown"
            d: Dict[str, Any] = model_data[_short_model(model)]
            d["sessions"].add(session_id)
            for key in _TOKEN_KEYS + ("reasoning_tokens",):
                d[key] += counts[key]
            d["total_tokens"] += sum(counts[k] for k in _TOKEN_KEYS)
            d["api_calls"] += counts["api_call_count"]
            if stored_cost is None:
                estimate, status = _estimate_cost(model, counts["input_tokens"], counts["output_tokens"], cache_read_tokens=counts["cache_read_tokens"],
                                                  cache_write_tokens=counts["cache_write_tokens"], provider=provider or None, base_url=base_url)
            else:
                estimate, status = float(stored_cost or 0.0), cost_status or "unknown"
            d["cost"] += estimate
            d["actual_cost"] += float(actual_cost or 0.0)
            d["cost_status"] = status
            d["has_pricing"] = has_known_pricing(model, provider or None, base_url) or d.get("has_pricing", False)
        usage_totals = defaultdict(lambda: dict.fromkeys(count_keys, 0) | {"estimated_cost_usd": 0.0, "actual_cost_usd": 0.0})
        for r in self._get_model_usage(cutoff, source):
            totals: Dict[str, Any] = usage_totals[r["session_id"]]
            counts = {key: r[key] or 0 for key in count_keys}
            for key in count_keys:
                totals[key] += counts[key]
            totals["estimated_cost_usd"] += r["estimated_cost_usd"] or 0.0
            totals["actual_cost_usd"] += r["actual_cost_usd"] or 0.0
            _accumulate(r["model"], r["billing_provider"], r.get("billing_base_url"), r["session_id"], counts,
                        stored_cost=r["estimated_cost_usd"] if r.get("cost_status") or r.get("cost_source") else None,
                        actual_cost=r["actual_cost_usd"], cost_status=r.get("cost_status"))
        # Reconcile against the aggregate row: covers legacy sessions,
        # interrupted migrations, and absolute cumulative updates without
        # double-counting already-attributed route deltas.
        for s in sessions:
            totals = usage_totals[s["id"]]
            residual = {k: max(0, (s.get(k) or 0) - totals[k]) for k in _TOKEN_KEYS + ("api_call_count",)}
            residual["reasoning_tokens"] = 0
            residual_cost = max(0.0, float(s.get("estimated_cost_usd") or 0.0) - totals["estimated_cost_usd"])
            residual_actual = max(0.0, float(s.get("actual_cost_usd") or 0.0) - totals["actual_cost_usd"])
            if any(residual.values()) or residual_cost or residual_actual:
                _accumulate(s.get("model"), s.get("billing_provider"), s.get("billing_base_url"), s["id"], residual,
                            stored_cost=residual_cost, actual_cost=residual_actual, cost_status=s.get("cost_status"))
        for s in sessions:
            if s.get("tool_call_count"):
                model_data[_short_model(s.get("model"))]["tool_calls"] += s["tool_call_count"]
        # Models seen only via tool-call attribution never hit _accumulate —
        # default has_pricing/cost_status so the output shape is uniform for JSON consumers.
        defaults = (("has_pricing", False), ("cost_status", "unknown"))
        result = [{"model": model, **data, "sessions": len(data["sessions"]), **{k: v for k, v in defaults if k not in data}}
                  for model, data in model_data.items()]
        return sorted(result, key=lambda x: (x["total_tokens"], x["sessions"]), reverse=True)

    def _compute_platform_breakdown(self, sessions: List[Dict]) -> List[Dict]:
        platform_data = defaultdict(lambda: {"sessions": 0, "messages": 0, **dict.fromkeys(_TOKEN_KEYS, 0), "total_tokens": 0, "tool_calls": 0})
        for s in sessions:
            d = platform_data[s.get("source") or "unknown"]
            d["sessions"] += 1
            d["messages"] += s.get("message_count") or 0
            for k in _TOKEN_KEYS:
                d[k] += s.get(k) or 0
                d["total_tokens"] += s.get(k) or 0
            d["tool_calls"] += s.get("tool_call_count") or 0
        return sorted(({"platform": platform, **data} for platform, data in platform_data.items()), key=lambda x: x["sessions"], reverse=True)

    def _compute_tool_breakdown(self, tool_usage: List[Dict]) -> List[Dict]:
        """Ranked tool list with percentages."""
        total_calls = sum(t["count"] for t in tool_usage)
        return [{"tool": t["tool_name"], "count": t["count"], "percentage": (t["count"] / total_calls * 100) if total_calls else 0} for t in tool_usage]

    def _compute_skill_breakdown(self, skill_usage: List[Dict]) -> Dict[str, Any]:
        """Per-skill usage → summary + ranked list."""
        total_skill_loads = sum(s["view_count"] for s in skill_usage)
        total_skill_edits = sum(s["manage_count"] for s in skill_usage)
        total_skill_actions = total_skill_loads + total_skill_edits
        top_skills = [{
            "skill": skill["skill"], "view_count": skill["view_count"], "manage_count": skill["manage_count"], "total_count": total_count,
            "percentage": (total_count / total_skill_actions * 100) if total_skill_actions else 0, "last_used_at": skill.get("last_used_at"),
        } for skill in skill_usage for total_count in (skill["view_count"] + skill["manage_count"],)]
        top_skills.sort(key=lambda s: (s["total_count"], s["view_count"], s["manage_count"], s["last_used_at"] or 0, s["skill"]), reverse=True)
        return {
            "summary": {"total_skill_loads": total_skill_loads, "total_skill_edits": total_skill_edits,
                        "total_skill_actions": total_skill_actions, "distinct_skills_used": len(skill_usage)},
            "top_skills": top_skills,
        }

    def _compute_activity_patterns(self, sessions: List[Dict]) -> Dict:
        """Activity by day of week, hour, and active-day streak."""
        day_counts, hour_counts, daily_counts = Counter(), Counter(), Counter()  # weekday (0=Monday), hour, "YYYY-MM-DD"
        for s in sessions:
            ts = s.get("started_at")
            if not ts:
                continue
            dt = datetime.fromtimestamp(ts)
            day_counts[dt.weekday()] += 1
            hour_counts[dt.hour] += 1
            daily_counts[dt.strftime("%Y-%m-%d")] += 1
        day_names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
        day_breakdown = [{"day": day_names[i], "count": day_counts.get(i, 0)} for i in range(7)]
        hour_breakdown = [{"hour": i, "count": hour_counts.get(i, 0)} for i in range(24)]
        max_streak = 0
        if daily_counts:
            dates = [datetime.strptime(d, "%Y-%m-%d") for d in sorted(daily_counts)]
            current_streak = max_streak = 1
            for prev, cur in zip(dates, dates[1:]):
                current_streak = current_streak + 1 if (cur - prev).days == 1 else 1
                max_streak = max(max_streak, current_streak)
        return {"by_day": day_breakdown, "by_hour": hour_breakdown, "busiest_day": max(day_breakdown, key=lambda x: x["count"]),
                "busiest_hour": max(hour_breakdown, key=lambda x: x["count"]), "active_days": len(daily_counts), "max_streak": max_streak}

    _TOP_METRICS = (
        ("Most messages", lambda s: s.get("message_count") or 0, "{} msgs"),
        ("Most tokens", lambda s: (s.get("input_tokens") or 0) + (s.get("output_tokens") or 0), "{:,} tokens"),
        ("Most tool calls", lambda s: s.get("tool_call_count") or 0, "{} calls"),
    )

    def _compute_top_sessions(self, sessions: List[Dict]) -> List[Dict]:
        """Notable sessions (longest, most messages, most tokens, most tool calls)."""
        top = []
        timed = [s for s in sessions if s.get("started_at") and s.get("ended_at")]
        if timed:
            longest = max(timed, key=lambda s: s["ended_at"] - s["started_at"])
            top.append({"label": "Longest session", "session_id": longest["id"][:16],
                        "value": format_duration_compact(longest["ended_at"] - longest["started_at"]), "date": _day(longest["started_at"])})
        for label, metric, fmt in self._TOP_METRICS:
            best = max(sessions, key=metric)
            value = metric(best)
            if value > 0:
                top.append({"label": label, "session_id": best["id"][:16], "value": fmt.format(value), "date": _day(best.get("started_at"))})
        return top

    # ------------------------------------------------------------- Formatting

    @staticmethod
    def _section(title: str) -> List[str]:
        return [f"  {title}", "  " + "─" * 56]

    @staticmethod
    def _cost_lines(o: Dict, templates: tuple) -> List[str]:
        """One formatted line per non-zero cost bucket (estimated, included, unknown)."""
        # Cost breakdown — surface the three buckets so subscription-included and unknown-cost sessions are
        # visible instead of silently collapsing to $0. See #77223.
        est_cost = o.get("estimated_cost", 0.0)
        values = (_fmt_est_cost(est_cost) if est_cost > 0 else "", o.get("included_cost_sessions", 0), o.get("unknown_cost_sessions", 0))
        return [tpl.format(v) for tpl, v in zip(templates, values) if v]

    def format_terminal(self, report: Dict) -> str:
        """Format the insights report for terminal display (CLI)."""
        if report.get("empty"):
            src = f" (source: {report['source_filter']})" if report.get("source_filter") else ""
            return f"  No sessions found in the last {report.get('days', 30)} days{src}."
        o = report["overview"]
        period_label = f"Last {report['days']} days"
        if report.get("source_filter"):
            period_label += f" ({report['source_filter']})"
        padding = 58 - len(period_label) - 2
        left_pad = padding // 2
        lines = [
            "",
            "  ╔══════════════════════════════════════════════════════════╗",
            "  ║                    📊 Hermes Insights                    ║",
            f"  ║{' ' * left_pad} {period_label} {' ' * (padding - left_pad)}║",
            "  ╚══════════════════════════════════════════════════════════╝",
            "",
        ]
        if (start := coerce_epoch(o.get("date_range_start"))) is not None and (end := coerce_epoch(o.get("date_range_end"))) is not None:
            start_str = safe_strftime(datetime.fromtimestamp(start), "%b %d, %Y")
            end_str = safe_strftime(datetime.fromtimestamp(end), "%b %d, %Y")
            lines += [f"  Period: {start_str} — {end_str}", ""]
        lines += self._section("📋 Overview") + [
            f"  Sessions:          {o['total_sessions']:<12}  Messages:        {o['total_messages']:,}",
            f"  Tool calls:        {o['total_tool_calls']:<12,}  User messages:   {o['user_messages']:,}",
            f"  Input tokens:      {o['total_input_tokens']:<12,}  Output tokens:   {o['total_output_tokens']:,}",
            f"  Total tokens:      {o['total_tokens']:,}",
        ]
        if o["total_hours"] > 0:
            lines.append(f"  Active time:       ~{format_duration_compact(o['total_hours'] * 3600):<11}  Avg session:     ~{format_duration_compact(o['avg_session_duration'])}")
        lines += [f"  Avg msgs/session:  {o['avg_messages_per_session']:.1f}", ""]
        # Cost buckets: show included/unknown sessions instead of collapsing to $0.
        cost_lines = self._cost_lines(o, ("  Estimated:          {}", "  Included:           {} session(s) (subscription — no provider invoice)",
                                          "  Unknown:            {} session(s) (no pricing data)"))
        if cost_lines:
            lines += self._section("💰 Cost") + cost_lines + [""]
        if report["models"]:
            lines += self._section("🤖 Models Used") + [f"  {'Model':<30} {'Sessions':>8} {'Tokens':>12}"]
            lines += [f"  {m['model'][:28]:<30} {m['sessions']:>8} {m['total_tokens']:>12,}" for m in report["models"]] + [""]
        platforms = report["platforms"]
        if len(platforms) > 1 or (platforms and platforms[0]["platform"] != "cli"):
            lines += self._section("📱 Platforms") + [f"  {'Platform':<14} {'Sessions':>8} {'Messages':>10} {'Tokens':>14}"]
            lines += [f"  {p['platform']:<14} {p['sessions']:>8} {p['messages']:>10,} {p['total_tokens']:>14,}" for p in platforms] + [""]
        if report["tools"]:
            lines += self._section("🔧 Top Tools") + [f"  {'Tool':<28} {'Calls':>8} {'%':>8}"]
            lines += [f"  {t['tool']:<28} {t['count']:>8,} {t['percentage']:>7.1f}%" for t in report["tools"][:15]]
            if len(report["tools"]) > 15:
                lines.append(f"  ... and {len(report['tools']) - 15} more tools")
            lines.append("")
        skills = report.get("skills", {})
        top_skills = skills.get("top_skills", [])
        if top_skills:
            lines += self._section("🧠 Top Skills") + [f"  {'Skill':<28} {'Loads':>7} {'Edits':>7} {'Last used':>11}"]
            for skill in top_skills[:10]:
                last_used = _day(skill.get("last_used_at")) if skill.get("last_used_at") else "—"
                lines.append(f"  {skill['skill'][:28]:<28} {skill['view_count']:>7,} {skill['manage_count']:>7,} {last_used:>11}")
            summary = skills.get("summary", {})
            lines += [f"  Distinct skills: {summary.get('distinct_skills_used', 0)}  Loads: {summary.get('total_skill_loads', 0):,}  "
                      f"Edits: {summary.get('total_skill_edits', 0):,}", ""]
        act = report.get("activity", {})
        if act.get("by_day"):
            lines += self._section("📅 Activity Patterns")
            bars = _bar_chart([d["count"] for d in act["by_day"]], max_width=15)
            lines += [f"  {d['day']}  {bar:<15} {d['count']}" for bar, d in zip(bars, act["by_day"])] + [""]
            busy_hours = [h for h in sorted(act["by_hour"], key=lambda x: x["count"], reverse=True) if h["count"] > 0][:5]
            if busy_hours:
                hour_strs = [f"{_hour12(h['hour'])} ({h['count']})" for h in busy_hours]
                lines.append(f"  Peak hours: {', '.join(hour_strs)}")
            if act.get("active_days"):
                lines.append(f"  Active days: {act['active_days']}")
            if act.get("max_streak") and act["max_streak"] > 1:
                lines.append(f"  Best streak: {act['max_streak']} consecutive days")
            lines.append("")
        if report.get("top_sessions"):
            lines += self._section("🏆 Notable Sessions")
            lines += [f"  {ts['label']:<20} {ts['value']:<18} ({ts['date']}, {ts['session_id']})" for ts in report["top_sessions"]] + [""]
        return "\n".join(lines)

    def format_gateway(self, report: Dict) -> str:
        """Format the insights report for gateway/messaging (shorter)."""
        if report.get("empty"):
            return f"No sessions found in the last {report.get('days', 30)} days."
        o = report["overview"]
        lines = [
            f"📊 **Hermes Insights** — Last {report['days']} days\n",
            f"**Sessions:** {o['total_sessions']} | **Messages:** {o['total_messages']:,} | **Tool calls:** {o['total_tool_calls']:,}",
            f"**Tokens:** {o['total_tokens']:,} (in: {o['total_input_tokens']:,} / out: {o['total_output_tokens']:,})",
        ]
        if o["total_hours"] > 0:
            lines.append(f"**Active time:** ~{format_duration_compact(o['total_hours'] * 3600)} | **Avg session:** ~{format_duration_compact(o['avg_session_duration'])}")
        lines.append("")
        cost_parts = self._cost_lines(o, ("{} estimated", "{} included (subscription)", "{} unknown"))
        if cost_parts:
            lines += [f"**Cost:** {' | '.join(cost_parts)}", ""]
        if report["models"]:
            lines += ["**🤖 Models:**"] + [f"  {m['model'][:25]} — {m['sessions']} sessions, {m['total_tokens']:,} tokens" for m in report["models"][:5]] + [""]
        if len(report["platforms"]) > 1:
            lines += ["**📱 Platforms:**"] + [f"  {p['platform']} — {p['sessions']} sessions, {p['messages']:,} msgs" for p in report["platforms"]] + [""]
        if report["tools"]:
            lines += ["**🔧 Top Tools:**"] + [f"  {t['tool']} — {t['count']:,} calls ({t['percentage']:.1f}%)" for t in report["tools"][:8]] + [""]
        skills = report.get("skills", {})
        if skills.get("top_skills"):
            lines.append("**🧠 Top Skills:**")
            for skill in skills["top_skills"][:5]:
                suffix = f", last used {_day(skill['last_used_at'])}" if skill.get("last_used_at") else ""
                lines.append(f"  {skill['skill']} — {skill['view_count']:,} loads, {skill['manage_count']:,} edits{suffix}")
            lines.append("")
        act = report.get("activity", {})
        if act.get("busiest_day") and act.get("busiest_hour"):
            lines.append(f"**📅 Busiest:** {act['busiest_day']['day']}s ({act['busiest_day']['count']} sessions), {_hour12(act['busiest_hour']['hour'])} ({act['busiest_hour']['count']} sessions)")
            if act.get("active_days"):
                lines.append(f"**Active days:** {act['active_days']}")
            if act.get("max_streak", 0) > 1:
                lines.append(f"**Best streak:** {act['max_streak']} consecutive days")
        return "\n".join(lines)
