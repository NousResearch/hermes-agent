"""Hermes Performance Benchmarking Report — Milestone I.

Generates a markdown performance report from structured.jsonl and state.db,
covering: task completion rates, slow tools, model usage, token costs,
loop detections, and error breakdown.

Usage:
    # Generate 7-day report, save to ~/.hermes/reports/, print to stdout
    python3 -m hermes_cli.report

    # Custom lookback window
    python3 -m hermes_cli.report --days 30

    # Via ops CLI
    python3 -m hermes_cli.ops report --days 7
    python3 -m hermes_cli.ops report --json
"""
from __future__ import annotations

import json
import os
import time
from collections import defaultdict
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

from utils import atomic_json_write, atomic_text_write

_HERMES_HOME = Path(os.getenv("HERMES_HOME", Path.home() / ".hermes"))
_LOG_PATH = _HERMES_HOME / "logs" / "structured.jsonl"
_DB_PATH = _HERMES_HOME / "state.db"
_REPORTS_DIR = _HERMES_HOME / "reports"

# Cost rates per million tokens (input, output) — same as sync.py
_COST_RATES: dict[str, tuple[float, float]] = {
    "claude-sonnet-4-6":              (3.00,  15.00),
    "claude-sonnet-4-5":              (3.00,  15.00),
    "claude-opus-4-6":                (15.00, 75.00),
    "claude-haiku-4-5-20251001":      (0.25,   1.25),
    "claude-haiku-4-5":               (0.25,   1.25),
    "gpt-4o":                         (2.50,  10.00),
    "gpt-4o-mini":                    (0.15,   0.60),
    "gpt-4.1":                        (2.00,   8.00),
    "gpt-4.1-mini":                   (0.40,   1.60),
    "gpt-4.1-nano":                   (0.10,   0.40),
    "openai/gpt-4o":                  (2.50,  10.00),
    "openai/gpt-4o-mini":             (0.15,   0.60),
    "anthropic/claude-sonnet-4-6":    (3.00,  15.00),
    "anthropic/claude-haiku-4-5-20251001": (0.25, 1.25),
    "google/gemini-flash-1.5":        (0.075,  0.30),
}


def _estimate_cost(model: str, tokens_in: int, tokens_out: int) -> float:
    ri, ro = _COST_RATES.get(model or "", (0.0, 0.0))
    return (tokens_in / 1_000_000 * ri) + (tokens_out / 1_000_000 * ro)


def _read_log_window(days: int) -> List[Dict[str, Any]]:
    """Read structured.jsonl events from the last N days."""
    if not _LOG_PATH.exists():
        return []
    cutoff = time.time() - days * 86400
    events = []
    with open(_LOG_PATH, encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                ev = json.loads(line)
                ts = ev.get("ts") or ev.get("timestamp") or 0
                # ts can be ISO string or float
                if isinstance(ts, str):
                    try:
                        ts = datetime.fromisoformat(ts.replace("Z", "+00:00")).timestamp()
                    except Exception:
                        ts = 0
                if ts >= cutoff:
                    events.append(ev)
            except Exception:
                continue
    return events


def _read_tasks_window(days: int) -> List[Dict[str, Any]]:
    """Read tasks from state.db started in the last N days."""
    if not _DB_PATH.exists():
        return []
    try:
        import sqlite3
        cutoff = time.time() - days * 86400
        conn = sqlite3.connect(str(_DB_PATH))
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            "SELECT * FROM tasks WHERE started_at >= ? ORDER BY started_at DESC",
            (cutoff,)
        ).fetchall()
        conn.close()
        return [dict(r) for r in rows]
    except Exception:
        return []


def _fmt_ms(ms: float) -> str:
    if ms >= 60_000:
        return f"{ms/60_000:.1f}m"
    if ms >= 1_000:
        return f"{ms/1_000:.1f}s"
    return f"{ms:.0f}ms"


def _fmt_tokens(n: int) -> str:
    if n >= 1_000_000:
        return f"{n/1_000_000:.1f}M"
    if n >= 1_000:
        return f"{n/1_000:.0f}k"
    return str(n)


def generate_report(days: int = 7, output_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Generate a performance benchmarking report from structured.jsonl + state.db.

    Returns a dict with:
        data      — structured stats (suitable for JSON output or further analysis)
        markdown  — formatted report string
        saved_to  — path where the file was saved (or None if save failed)
    """
    now = datetime.now(timezone.utc)
    period_start = now - timedelta(days=days)
    events = _read_log_window(days)
    tasks = _read_tasks_window(days)

    # ── Task stats ──────────────────────────────────────────────────────────
    status_counts: dict[str, int] = defaultdict(int)
    for t in tasks:
        status_counts[t.get("status", "unknown")] += 1
    total_tasks = len(tasks)
    completed = status_counts.get("completed", 0)
    failed = status_counts.get("failed", 0)
    interrupted = status_counts.get("interrupted", 0)
    completion_rate = round(completed / total_tasks * 100, 1) if total_tasks else 0

    error_slugs: dict[str, int] = defaultdict(int)
    for t in tasks:
        if t.get("error_info"):
            error_slugs[t["error_info"]] += 1

    # ── Tool performance ─────────────────────────────────────────────────────
    tool_stats: dict[str, dict] = defaultdict(lambda: {"calls": 0, "total_ms": 0, "errors": 0, "slowest_ms": 0})
    slow_events: list[dict] = []

    for ev in events:
        if ev.get("event") != "tool_result":
            continue
        tool = ev.get("tool_name", "unknown")
        dur = ev.get("duration_ms") or 0
        success = ev.get("success", True)
        s = tool_stats[tool]
        s["calls"] += 1
        s["total_ms"] += dur
        if not success:
            s["errors"] += 1
        if dur > s["slowest_ms"]:
            s["slowest_ms"] = dur
        slow_events.append(ev)

    # Sort by avg duration desc
    tool_rows = []
    for tool, s in tool_stats.items():
        avg = s["total_ms"] / s["calls"] if s["calls"] else 0
        tool_rows.append({
            "tool": tool,
            "calls": s["calls"],
            "avg_ms": round(avg, 1),
            "slowest_ms": round(s["slowest_ms"], 1),
            "errors": s["errors"],
            "error_rate": round(s["errors"] / s["calls"] * 100, 1) if s["calls"] else 0,
        })
    tool_rows.sort(key=lambda r: r["avg_ms"], reverse=True)

    # ── Model usage ──────────────────────────────────────────────────────────
    model_stats: dict[str, dict] = defaultdict(lambda: {
        "calls": 0, "tokens_in": 0, "tokens_out": 0, "cost_usd": 0.0,
    })
    for ev in events:
        if ev.get("event") != "model_call":
            continue
        model = ev.get("model", "unknown")
        ti = ev.get("input_tokens") or 0
        to = ev.get("output_tokens") or 0
        m = model_stats[model]
        m["calls"] += 1
        m["tokens_in"] += ti
        m["tokens_out"] += to
        m["cost_usd"] += _estimate_cost(model, ti, to)

    model_rows = []
    for model, m in model_stats.items():
        avg_in = m["tokens_in"] // m["calls"] if m["calls"] else 0
        avg_out = m["tokens_out"] // m["calls"] if m["calls"] else 0
        model_rows.append({
            "model": model,
            "calls": m["calls"],
            "total_tokens_in": m["tokens_in"],
            "total_tokens_out": m["tokens_out"],
            "avg_tokens_in": avg_in,
            "avg_tokens_out": avg_out,
            "est_cost_usd": round(m["cost_usd"], 4),
        })
    model_rows.sort(key=lambda r: r["calls"], reverse=True)
    total_cost = round(sum(r["est_cost_usd"] for r in model_rows), 4)

    # ── Loop detections ──────────────────────────────────────────────────────
    loops = [ev for ev in events if ev.get("event") == "loop_detected"]
    loop_tool_counts: dict[str, int] = defaultdict(int)
    for lp in loops:
        loop_tool_counts[lp.get("tool_name", "unknown")] += 1

    # ── Slow tool events ─────────────────────────────────────────────────────
    slow_tool_events = [ev for ev in events if ev.get("event") == "slow_tool"]

    # ── Build data dict ──────────────────────────────────────────────────────
    summary = {
        "period_days": days,
        "total_tasks": total_tasks,
        "completed_tasks": completed,
        "failed_tasks": failed,
        "interrupted_tasks": interrupted,
        "completion_rate_pct": completion_rate,
        "loop_detections": len(loops),
        "slow_tool_warnings": len(slow_tool_events),
        "total_cost_usd": total_cost,
        "top_tools": tool_rows[:5],
        "top_models": model_rows[:5],
    }

    data = {
        "period_days": days,
        "period_start": period_start.isoformat(),
        "period_end": now.isoformat(),
        "tasks": {
            "total": total_tasks,
            "completed": completed,
            "failed": failed,
            "interrupted": interrupted,
            "completion_rate_pct": completion_rate,
            "error_slugs": dict(error_slugs),
        },
        "tools": tool_rows,
        "models": model_rows,
        "total_cost_usd": total_cost,
        "loops": {
            "total": len(loops),
            "by_tool": dict(loop_tool_counts),
        },
        "slow_tool_warnings": len(slow_tool_events),
    }

    data["summary"] = summary

    # ── Render markdown ──────────────────────────────────────────────────────
    date_str = now.strftime("%Y-%m-%d")
    lines = [
        f"# Hermes Performance Report — {date_str}",
        f"**Period:** last {days} days ({period_start.strftime('%Y-%m-%d')} → {date_str})",
        "",
        "---",
        "",
        "## Task Summary",
        "",
        f"| Metric | Value |",
        f"|--------|-------|",
        f"| Total tasks | {total_tasks} |",
        f"| Completed | {completed} ({completion_rate}%) |",
        f"| Failed | {failed} |",
        f"| Interrupted | {interrupted} |",
        f"| Loop detections | {len(loops)} |",
        f"| Slow tool warnings | {len(slow_tool_events)} |",
    ]

    if error_slugs:
        lines += ["", "**Failure breakdown:**", ""]
        for slug, count in sorted(error_slugs.items(), key=lambda x: -x[1]):
            lines.append(f"- `{slug}`: {count}×")

    lines += [
        "",
        "---",
        "",
        "## Tool Performance",
        "",
        f"| Tool | Calls | Avg | Slowest | Errors |",
        f"|------|-------|-----|---------|--------|",
    ]
    for r in tool_rows[:15]:
        err_str = f"{r['errors']} ({r['error_rate']}%)" if r["errors"] else "—"
        lines.append(
            f"| `{r['tool']}` | {r['calls']} | {_fmt_ms(r['avg_ms'])} | {_fmt_ms(r['slowest_ms'])} | {err_str} |"
        )

    lines += [
        "",
        "---",
        "",
        "## Model Usage",
        "",
        f"| Model | Calls | Avg In | Avg Out | Est. Cost |",
        f"|-------|-------|--------|---------|-----------|",
    ]
    for r in model_rows:
        lines.append(
            f"| `{r['model']}` | {r['calls']} | {_fmt_tokens(r['avg_tokens_in'])} | {_fmt_tokens(r['avg_tokens_out'])} | ${r['est_cost_usd']:.4f} |"
        )
    lines.append(f"| **Total** | | | | **${total_cost:.4f}** |")

    if loop_tool_counts:
        lines += [
            "",
            "---",
            "",
            "## Loop Detections",
            "",
            f"| Tool | Count |",
            f"|------|-------|",
        ]
        for tool, count in sorted(loop_tool_counts.items(), key=lambda x: -x[1]):
            lines.append(f"| `{tool}` | {count} |")

    lines += [
        "",
        "---",
        "",
        f"*Generated by `hermes_cli.report` at {now.strftime('%Y-%m-%d %H:%M:%S')} UTC*",
    ]

    markdown = "\n".join(lines)

    # ── Save to file ─────────────────────────────────────────────────────────
    saved_to = None
    saved_json_to = None
    try:
        if output_path:
            out = Path(output_path)
        else:
            out = _REPORTS_DIR / f"perf-{date_str}.md"
        json_out = out.with_suffix(".json")
        atomic_text_write(out, markdown)
        sidecar = {
            "generated_at": now.isoformat(),
            "saved_to": str(out),
            "saved_json_to": str(json_out),
            "summary": summary,
            "data": data,
        }
        atomic_json_write(json_out, sidecar)
        saved_to = str(out)
        saved_json_to = str(json_out)
    except Exception:
        pass  # non-fatal — caller still gets markdown in memory

    return {
        "data": data,
        "summary": summary,
        "markdown": markdown,
        "saved_to": saved_to,
        "saved_json_to": saved_json_to,
    }


def main() -> None:
    import argparse
    p = argparse.ArgumentParser(description="Generate a Hermes performance report")
    p.add_argument("--days", type=int, default=7, help="Lookback window in days (default 7)")
    p.add_argument("--output", help="Output path (default: ~/.hermes/reports/perf-YYYY-MM-DD.md)")
    p.add_argument("--json", action="store_true", help="Print raw JSON data instead of markdown")
    args = p.parse_args()

    result = generate_report(days=args.days, output_path=args.output)
    if args.json:
        print(json.dumps(result["data"], indent=2, default=str))
    else:
        print(result["markdown"])
        if result.get("saved_to"):
            print(f"\nSaved to: {result['saved_to']}")


if __name__ == "__main__":
    main()
