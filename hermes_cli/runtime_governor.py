"""Deterministic observe-only Runtime Governor for Hermes.

The Runtime Governor reports runtime waste smells from the local session store.
It is intentionally backend-only, model-free, and side-effect free: no UI, no
policy enforcement, no cost claims when provider telemetry is missing.
"""

from __future__ import annotations

import argparse
import json
import re
import sqlite3
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

from hermes_constants import get_hermes_home

_RECEIPT_RE = re.compile(r"\b(receipts?|receipt-tagged|proof\s+notes?)\b", re.IGNORECASE)
_DOCS_RE = re.compile(r"\b(readme|handoff|current\.md|\.md\b|docs?|documentation)\b", re.IGNORECASE)
_WRITE_RE = re.compile(r"\b(write|wrote|save|saved|create|created|append|render|rendered)\b", re.IGNORECASE)
_READ_RE = re.compile(r"\b(read|opened|cat|full\s+read|read_file|readback|read\s+back)\b", re.IGNORECASE)
_SUMMARIZE_RE = re.compile(r"\b(summarize|summarise|summary|final\s+response|restate|recap)\b", re.IGNORECASE)
_DONE_RE = re.compile(r"\b(done|build passed|tests? passed|complete|completed|receipt)\b", re.IGNORECASE)

_DEFAULT_TOP_N = 10


@dataclass(frozen=True)
class RuntimeGovernorConfig:
    """Deterministic thresholds for observe-only runtime classification."""

    high_api_calls_per_session: int = 50
    high_non_cache_tokens_per_session: int = 250_000
    high_gross_tokens_per_session: int = 5_000_000
    high_receipt_readback_mentions: int = 3
    high_receipt_write_mentions: int = 3
    excessive_finalization_calls: int = 3
    high_same_file_read_count: int = 3
    long_session_api_calls: int = 80
    max_human_receipt_lines: int = 80


def generate_runtime_report(
    *,
    state_db_path: str | Path | None = None,
    days: int = 30,
    now: float | None = None,
    source: str | None = None,
    config: RuntimeGovernorConfig | None = None,
) -> dict[str, Any]:
    """Generate an observe-only Runtime Governor report from a Hermes state DB.

    The returned dict is structured for tests/automation. Use
    :func:`format_runtime_report` for a lean terminal rendering.
    """

    cfg = config or RuntimeGovernorConfig()
    timestamp = float(now if now is not None else time.time())
    db_path = Path(state_db_path) if state_db_path is not None else get_hermes_home() / "state.db"
    empty = _empty_report(days=days, generated_at=timestamp, db_path=db_path, config=cfg, source=source)
    if not db_path.exists():
        empty["warnings"].append(f"state DB not found: {db_path}")
        return empty

    try:
        conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True, timeout=1.0)
        conn.row_factory = sqlite3.Row
    except sqlite3.Error as exc:
        empty["warnings"].append(f"could not open state DB read-only: {type(exc).__name__}: {exc}")
        return empty

    try:
        sessions = _fetch_sessions(conn, since=timestamp - max(days, 1) * 86_400, source=source)
        messages_by_session = _fetch_messages_by_session(
            conn,
            [str(s.get("session_id")) for s in sessions],
            since=timestamp - max(days, 1) * 86_400,
        )
    finally:
        conn.close()

    report = _build_report(
        sessions=sessions,
        messages_by_session=messages_by_session,
        days=days,
        generated_at=timestamp,
        db_path=db_path,
        cfg=cfg,
        source=source,
    )
    return report


def format_runtime_report(report: dict[str, Any]) -> str:
    """Render a concise human-readable observe-only report."""

    totals = report.get("totals", {})
    cost = report.get("cost", {})
    receipt = report.get("receipt_overhead", {})
    lines = [
        f"Runtime Governor observe-only report ({report.get('days')}d)",
        f"DB: {report.get('db_path')}",
        "",
        "Totals:",
        f"- Sessions: {totals.get('sessions', 0)}",
        f"- API calls: {totals.get('api_calls', 0)}",
        f"- Gross context tokens: {totals.get('gross_context_tokens', 0)}",
        f"- Non-cache-ish tokens: {totals.get('non_cache_tokens', 0)}",
    ]
    if not cost.get("reliable", False):
        lines.append("- Cost telemetry unreliable/missing; no dollar spend claims made.")
    else:
        lines.append(f"- Recorded cost: ${cost.get('recorded_cost_usd', 0.0):.4f}")

    lines.extend(
        [
            "",
            "Receipt/docs overhead heuristics:",
            f"- Receipt-tagged sessions: {receipt.get('receipt_tagged_sessions', 0)} (upper bound, not marginal receipt spend)",
            f"- Docs-or-receipt-tagged sessions: {receipt.get('docs_or_receipt_tagged_sessions', 0)} (broad upper bound)",
            f"- Receipt write-ish mentions: {receipt.get('write_mentions', 0)}",
            f"- Receipt read-ish mentions: {receipt.get('read_mentions', 0)}",
            f"- Suspected write/read/summarize loops: {receipt.get('suspected_loops', 0)}",
        ]
    )

    estimates = receipt.get("avoidable_token_estimates", {})
    if estimates:
        lines.extend(
            [
                "- Avoidable-token estimate range:",
                f"  - conservative: {estimates.get('conservative', 0)}",
                f"  - plausible: {estimates.get('plausible', 0)}",
                f"  - aggressive: {estimates.get('aggressive', 0)}",
            ]
        )

    recommendations = report.get("recommendations") or []
    if recommendations:
        lines.extend(["", "Recommendations:"])
        for item in recommendations[:8]:
            lines.append(f"- [{item.get('confidence', 'unknown')}] {item.get('recommendation')}")

    warnings = report.get("warnings") or []
    if warnings:
        lines.extend(["", "Warnings:"])
        for warning in warnings:
            lines.append(f"- {warning}")

    return "\n".join(lines)


def _build_report(
    *,
    sessions: list[dict[str, Any]],
    messages_by_session: dict[str, list[dict[str, Any]]],
    days: int,
    generated_at: float,
    db_path: Path,
    cfg: RuntimeGovernorConfig,
    source: str | None,
) -> dict[str, Any]:
    totals = _compute_totals(sessions)
    cost = _compute_cost(sessions)
    per_session = [_analyze_session(session, messages_by_session.get(session["session_id"], []), cfg) for session in sessions]
    receipt = _compute_receipt_overhead(per_session, totals)
    outliers = _compute_outliers(per_session)
    recommendations = _compute_recommendations(per_session, receipt, cost, cfg)

    report = _empty_report(days=days, generated_at=generated_at, db_path=db_path, config=cfg, source=source)
    report.update(
        {
            "totals": totals,
            "cost": cost,
            "receipt_overhead": receipt,
            "outliers": outliers,
            "runtime_smells": _compute_runtime_smells(per_session, cfg),
            "recommendations": recommendations,
            "session_count_analyzed": len(per_session),
        }
    )
    if not cost["reliable"]:
        report["warnings"].append("cost telemetry unreliable/missing; cost fields are zero, null, or sparse")
    return report


def _empty_report(
    *,
    days: int,
    generated_at: float,
    db_path: Path,
    config: RuntimeGovernorConfig,
    source: str | None,
) -> dict[str, Any]:
    return {
        "mode": "observe",
        "days": days,
        "source_filter": source,
        "generated_at": generated_at,
        "db_path": str(db_path),
        "config": asdict(config),
        "totals": {
            "sessions": 0,
            "api_calls": 0,
            "gross_context_tokens": 0,
            "non_cache_tokens": 0,
            "output_tokens": 0,
            "cache_read_tokens": 0,
            "cache_write_tokens": 0,
            "reasoning_tokens": 0,
        },
        "cost": {"reliable": False, "recorded_cost_usd": 0.0, "sessions_with_cost": 0},
        "receipt_overhead": {
            "receipt_tagged_sessions": 0,
            "docs_or_receipt_tagged_sessions": 0,
            "write_mentions": 0,
            "read_mentions": 0,
            "suspected_loops": 0,
            "avoidable_token_estimates": {"conservative": 0, "plausible": 0, "aggressive": 0},
        },
        "outliers": {"by_api_calls": [], "by_non_cache_tokens": [], "by_gross_context_tokens": []},
        "runtime_smells": {},
        "recommendations": [],
        "warnings": [],
        "session_count_analyzed": 0,
    }


def _fetch_sessions(conn: sqlite3.Connection, *, since: float, source: str | None) -> list[dict[str, Any]]:
    if not _table_exists(conn, "sessions"):
        return []
    cols = _columns(conn, "sessions")
    required = {"id", "started_at"}
    if not required.issubset(cols):
        return []

    select_exprs = [
        _select_col(cols, "id", alias="session_id", text=True),
        _select_col(cols, "source", text=True),
        _select_col(cols, "title", text=True),
        _select_col(cols, "started_at"),
        _select_col(cols, "input_tokens"),
        _select_col(cols, "output_tokens"),
        _select_col(cols, "cache_read_tokens"),
        _select_col(cols, "cache_write_tokens"),
        _select_col(cols, "reasoning_tokens"),
        _select_col(cols, "api_call_count"),
        _select_col(cols, "estimated_cost_usd"),
        _select_col(cols, "actual_cost_usd"),
    ]
    sql = f"SELECT {', '.join(select_exprs)} FROM sessions WHERE started_at >= ?"
    params: list[Any] = [since]
    if source and "source" in cols:
        sql += " AND source = ?"
        params.append(source)
    sql += " ORDER BY started_at DESC"
    rows = conn.execute(sql, params).fetchall()
    return [_normalize_session(dict(row)) for row in rows]


def _fetch_messages_by_session(
    conn: sqlite3.Connection,
    session_ids: list[str],
    *,
    since: float,
) -> dict[str, list[dict[str, Any]]]:
    if not session_ids or not _table_exists(conn, "messages"):
        return {}
    cols = _columns(conn, "messages")
    if "session_id" not in cols:
        return {}
    select_exprs = [
        _select_col(cols, "session_id", text=True),
        _select_col(cols, "role", text=True),
        _select_col(cols, "content", text=True),
        _select_col(cols, "tool_name", text=True),
        _select_col(cols, "timestamp"),
    ]
    placeholders = ",".join("?" for _ in session_ids)
    sql = f"SELECT {', '.join(select_exprs)} FROM messages WHERE session_id IN ({placeholders})"
    params: list[Any] = list(session_ids)
    if "timestamp" in cols:
        sql += " AND timestamp >= ?"
        params.append(since)
    sql += " ORDER BY session_id, timestamp"
    rows = conn.execute(sql, params).fetchall()
    by_session: dict[str, list[dict[str, Any]]] = {sid: [] for sid in session_ids}
    for row in rows:
        data = dict(row)
        by_session.setdefault(str(data.get("session_id")), []).append(data)
    return by_session


def _normalize_session(row: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(row)
    for key in (
        "started_at",
        "input_tokens",
        "output_tokens",
        "cache_read_tokens",
        "cache_write_tokens",
        "reasoning_tokens",
        "api_call_count",
    ):
        normalized[key] = int(normalized.get(key) or 0)
    for key in ("estimated_cost_usd", "actual_cost_usd"):
        value = normalized.get(key)
        normalized[key] = float(value) if value not in (None, "") else None
    normalized["session_id"] = str(normalized.get("session_id") or "")
    return normalized


def _analyze_session(
    session: dict[str, Any],
    messages: list[dict[str, Any]],
    cfg: RuntimeGovernorConfig,
) -> dict[str, Any]:
    texts = [_message_text(m) for m in messages]
    receipt_matches = [text for text in texts if _RECEIPT_RE.search(text)]
    docs_matches = [text for text in texts if _DOCS_RE.search(text)]
    write_mentions = sum(1 for text in receipt_matches if _WRITE_RE.search(text))
    read_mentions = sum(1 for text in receipt_matches if _READ_RE.search(text))
    summarize_mentions = sum(1 for text in receipt_matches if _SUMMARIZE_RE.search(text))
    suspected_loop = _has_ordered_loop(texts)
    non_cache_tokens = _non_cache_tokens(session)
    gross_context_tokens = _gross_context_tokens(session)
    api_calls = int(session.get("api_call_count") or 0)
    finalization_churn = _finalization_churn(messages, cfg)

    return {
        "session_id": session.get("session_id"),
        "title": session.get("title"),
        "source": session.get("source"),
        "started_at": session.get("started_at"),
        "api_calls": api_calls,
        "gross_context_tokens": gross_context_tokens,
        "non_cache_tokens": non_cache_tokens,
        "receipt_tagged": bool(receipt_matches),
        "docs_or_receipt_tagged": bool(receipt_matches or docs_matches),
        "receipt_mentions": len(receipt_matches),
        "write_mentions": write_mentions,
        "read_mentions": read_mentions,
        "summarize_mentions": summarize_mentions,
        "suspected_receipt_loop": suspected_loop,
        "finalization_churn": finalization_churn,
        "high_api_calls": api_calls >= cfg.high_api_calls_per_session,
        "high_non_cache_tokens": non_cache_tokens >= cfg.high_non_cache_tokens_per_session,
        "high_gross_context_tokens": gross_context_tokens >= cfg.high_gross_tokens_per_session,
    }


def _compute_totals(sessions: list[dict[str, Any]]) -> dict[str, int]:
    totals = {
        "sessions": len(sessions),
        "api_calls": sum(int(s.get("api_call_count") or 0) for s in sessions),
        "gross_context_tokens": sum(_gross_context_tokens(s) for s in sessions),
        "non_cache_tokens": sum(_non_cache_tokens(s) for s in sessions),
        "output_tokens": sum(int(s.get("output_tokens") or 0) for s in sessions),
        "cache_read_tokens": sum(int(s.get("cache_read_tokens") or 0) for s in sessions),
        "cache_write_tokens": sum(int(s.get("cache_write_tokens") or 0) for s in sessions),
        "reasoning_tokens": sum(int(s.get("reasoning_tokens") or 0) for s in sessions),
    }
    return totals


def _compute_cost(sessions: list[dict[str, Any]]) -> dict[str, Any]:
    cost_values = [_session_recorded_cost(s) for s in sessions]
    sessions_with_cost = sum(1 for value in cost_values if value > 0)
    recorded_cost = sum(cost_values)
    reliable = bool(sessions and recorded_cost > 0 and sessions_with_cost / len(sessions) >= 0.5)
    return {
        "reliable": reliable,
        "recorded_cost_usd": recorded_cost,
        "sessions_with_cost": sessions_with_cost,
        "sessions_total": len(sessions),
    }


def _session_recorded_cost(session: dict[str, Any]) -> float:
    value = session.get("actual_cost_usd")
    if value is None:
        value = session.get("estimated_cost_usd")
    if value in (None, ""):
        return 0.0
    return float(value)



def _compute_receipt_overhead(per_session: list[dict[str, Any]], totals: dict[str, int]) -> dict[str, Any]:
    receipt_sessions = [s for s in per_session if s["receipt_tagged"]]
    loop_sessions = [s for s in per_session if s["suspected_receipt_loop"]]
    write_mentions = sum(int(s["write_mentions"]) for s in per_session)
    read_mentions = sum(int(s["read_mentions"]) for s in per_session)
    avg_non_cache_per_call = 0
    if totals["api_calls"] > 0:
        avg_non_cache_per_call = max(1, int(totals["non_cache_tokens"] / totals["api_calls"]))
    loop_basis = max(len(loop_sessions), 0)
    return {
        "receipt_tagged_sessions": len(receipt_sessions),
        "docs_or_receipt_tagged_sessions": sum(1 for s in per_session if s["docs_or_receipt_tagged"]),
        "write_mentions": write_mentions,
        "read_mentions": read_mentions,
        "summarize_mentions": sum(int(s["summarize_mentions"]) for s in per_session),
        "suspected_loops": loop_basis,
        "receipt_tagged_upper_bound": {
            "api_calls": sum(int(s["api_calls"]) for s in receipt_sessions),
            "gross_context_tokens": sum(int(s["gross_context_tokens"]) for s in receipt_sessions),
            "non_cache_tokens": sum(int(s["non_cache_tokens"]) for s in receipt_sessions),
        },
        "avoidable_token_estimates": {
            "conservative": loop_basis * avg_non_cache_per_call,
            "plausible": loop_basis * avg_non_cache_per_call * 2,
            "aggressive": loop_basis * avg_non_cache_per_call * 3,
        },
    }


def _compute_outliers(per_session: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    return {
        "by_api_calls": _top_sessions(per_session, "api_calls"),
        "by_non_cache_tokens": _top_sessions(per_session, "non_cache_tokens"),
        "by_gross_context_tokens": _top_sessions(per_session, "gross_context_tokens"),
    }


def _compute_runtime_smells(per_session: list[dict[str, Any]], cfg: RuntimeGovernorConfig) -> dict[str, Any]:
    return {
        "high_api_call_sessions": [s["session_id"] for s in per_session if s["high_api_calls"]],
        "high_non_cache_token_sessions": [s["session_id"] for s in per_session if s["high_non_cache_tokens"]],
        "high_gross_context_sessions": [s["session_id"] for s in per_session if s["high_gross_context_tokens"]],
        "receipt_readback_sessions": [s["session_id"] for s in per_session if s["write_mentions"] and s["read_mentions"]],
        "finalization_churn_sessions": [s["session_id"] for s in per_session if s["finalization_churn"]],
    }


def _compute_recommendations(
    per_session: list[dict[str, Any]],
    receipt: dict[str, Any],
    cost: dict[str, Any],
    cfg: RuntimeGovernorConfig,
) -> list[dict[str, str]]:
    recommendations: list[dict[str, str]] = []
    if not cost.get("reliable"):
        recommendations.append(
            {
                "confidence": "high",
                "recommendation": "Treat cost as unavailable until per-call provider telemetry is populated; report token/API-call bounds instead of dollars.",
            }
        )
    if receipt.get("suspected_loops", 0) > 0:
        recommendations.append(
            {
                "confidence": "high",
                "recommendation": "Replace receipt write/read/summarize loops with structured ledger rendering plus stat/checksum/excerpt verification.",
            }
        )
    elif receipt.get("write_mentions", 0) and receipt.get("read_mentions", 0):
        recommendations.append(
            {
                "confidence": "medium",
                "recommendation": "Receipt readback should use stat/checksum/excerpt instead of full markdown reads.",
            }
        )
    if any(s["high_api_calls"] and s["receipt_tagged"] for s in per_session):
        recommendations.append(
            {
                "confidence": "medium",
                "recommendation": "High-call receipt-tagged sessions should enter lean finalization: receipt path plus key proof only.",
            }
        )
    if any(s["high_non_cache_tokens"] and s["receipt_tagged"] for s in per_session):
        recommendations.append(
            {
                "confidence": "medium",
                "recommendation": "Under high non-cache token pressure, render receipts from structured evidence rather than asking the main session to synthesize prose.",
            }
        )
    return recommendations


def _top_sessions(per_session: list[dict[str, Any]], key: str) -> list[dict[str, Any]]:
    rows = sorted(per_session, key=lambda s: int(s.get(key) or 0), reverse=True)
    trimmed = []
    for row in rows[:_DEFAULT_TOP_N]:
        trimmed.append(
            {
                "session_id": row.get("session_id"),
                "title": row.get("title"),
                "api_calls": row.get("api_calls"),
                "gross_context_tokens": row.get("gross_context_tokens"),
                "non_cache_tokens": row.get("non_cache_tokens"),
                "receipt_tagged": row.get("receipt_tagged"),
            }
        )
    return trimmed


def _non_cache_tokens(session: dict[str, Any]) -> int:
    return (
        int(session.get("input_tokens") or 0)
        + int(session.get("output_tokens") or 0)
        + int(session.get("cache_write_tokens") or 0)
        + int(session.get("reasoning_tokens") or 0)
    )


def _gross_context_tokens(session: dict[str, Any]) -> int:
    return (
        int(session.get("input_tokens") or 0)
        + int(session.get("output_tokens") or 0)
        + int(session.get("cache_read_tokens") or 0)
        + int(session.get("cache_write_tokens") or 0)
        + int(session.get("reasoning_tokens") or 0)
    )


def _has_ordered_loop(texts: Iterable[str]) -> bool:
    state = 0
    for text in texts:
        if not _RECEIPT_RE.search(text):
            continue
        if state == 0 and _WRITE_RE.search(text):
            state = 1
        elif state == 1 and _READ_RE.search(text):
            state = 2
        elif state == 2 and _SUMMARIZE_RE.search(text):
            return True
    return False


def _finalization_churn(messages: list[dict[str, Any]], cfg: RuntimeGovernorConfig) -> bool:
    done_index = None
    for idx, msg in enumerate(messages):
        if _DONE_RE.search(_message_text(msg)):
            done_index = idx
            break
    if done_index is None:
        return False
    remaining_assistant = sum(1 for msg in messages[done_index + 1 :] if (msg.get("role") or "") == "assistant")
    return remaining_assistant >= cfg.excessive_finalization_calls


def _message_text(message: dict[str, Any]) -> str:
    pieces = [str(message.get("content") or "")]
    tool_name = message.get("tool_name")
    if tool_name:
        pieces.append(str(tool_name))
    return " ".join(pieces)


def _table_exists(conn: sqlite3.Connection, table: str) -> bool:
    row = conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name=?", (table,)).fetchone()
    return row is not None


def _columns(conn: sqlite3.Connection, table: str) -> set[str]:
    return {str(row[1]) for row in conn.execute(f"PRAGMA table_info({table})").fetchall()}


def _select_col(cols: set[str], name: str, *, alias: str | None = None, text: bool = False) -> str:
    out = alias or name
    if name in cols:
        return f"{name} AS {out}" if out != name else name
    default = "''" if text else "0"
    return f"{default} AS {out}"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Hermes Runtime Governor observe-only report")
    parser.add_argument("--days", type=int, default=30)
    parser.add_argument("--state-db", type=Path, default=None)
    parser.add_argument("--source", default=None)
    parser.add_argument("--json", action="store_true", help="Emit structured JSON instead of text")
    args = parser.parse_args(argv)

    report = generate_runtime_report(state_db_path=args.state_db, days=args.days, source=args.source)
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        print(format_runtime_report(report))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
