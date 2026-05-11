#!/usr/bin/env python3
"""Read-only Hermes session-history recall over ~/.hermes/state.db.

Phase-1 LCM dogfood surface: bounded status/search/context extraction only.
No migrations, no writes, no context-engine activation.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
import re
import sqlite3
import sys
from pathlib import Path
from typing import Any

_SECRET_PATTERNS = [
    re.compile(r"sk-[A-Za-z0-9_\-]{16,}"),
    re.compile(r"(?i)(api[_-]?key|token|secret|password)\s*[:=]\s*[^\s,'\"]{8,}"),
    re.compile(r"gh[pousr]_[A-Za-z0-9_]{20,}"),
    re.compile(r"xox[baprs]-[A-Za-z0-9\-]{20,}"),
]

_MAX_SEARCH_LIMIT = 50
_MAX_DESCRIBE_TAIL = 100
_MAX_DESCRIBE_WINDOW = 20
_MAX_RECALL_PER_SESSION = 10
_MAX_SEARCH_CHARS = 6000
_MAX_DESCRIBE_CHARS = 8000


def _home() -> Path:
    return Path(os.environ.get("HERMES_HOME") or (Path.home() / ".hermes"))


def _state_db_path() -> Path:
    return _home() / "state.db"


def _connect() -> sqlite3.Connection:
    path = _state_db_path()
    # uri mode=ro is the write guard; no migrations/triggers can fire from us.
    conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    return conn


def _redact(text: Any) -> str:
    if text is None:
        return ""
    s = str(text)
    for pat in _SECRET_PATTERNS:
        s = pat.sub("[REDACTED]", s)
    return s


def _clip(text: Any, max_chars: int) -> str:
    s = _redact(text)
    if max_chars <= 0:
        max_chars = 800
    if len(s) <= max_chars:
        return s
    half = max(1, (max_chars - 35) // 2)
    return s[:half] + "\n...[truncated by hermes_lcm]...\n" + s[-half:]


def _bounded_int(value: Any, *, default: int, minimum: int, maximum: int) -> int:
    try:
        n = int(value)
    except Exception:
        n = default
    return max(minimum, min(maximum, n))


def _normalize_search_args(args: argparse.Namespace) -> argparse.Namespace:
    args.limit = _bounded_int(getattr(args, "limit", None), default=5, minimum=1, maximum=_MAX_SEARCH_LIMIT)
    args.max_chars = _bounded_int(getattr(args, "max_chars", None), default=800, minimum=100, maximum=_MAX_SEARCH_CHARS)
    return args


def _normalize_describe_args(args: argparse.Namespace) -> argparse.Namespace:
    args.tail = None if getattr(args, "tail", None) is None else _bounded_int(args.tail, default=20, minimum=1, maximum=_MAX_DESCRIBE_TAIL)
    args.window = _bounded_int(getattr(args, "window", None), default=3, minimum=0, maximum=_MAX_DESCRIBE_WINDOW)
    args.max_chars = _bounded_int(getattr(args, "max_chars", None), default=1200, minimum=100, maximum=_MAX_DESCRIBE_CHARS)
    return args


def _normalize_recall_args(args: argparse.Namespace) -> argparse.Namespace:
    args.limit = _bounded_int(getattr(args, "limit", None), default=10, minimum=1, maximum=_MAX_SEARCH_LIMIT)
    args.per_session = _bounded_int(getattr(args, "per_session", None), default=3, minimum=1, maximum=_MAX_RECALL_PER_SESSION)
    args.window = _bounded_int(getattr(args, "window", None), default=1, minimum=0, maximum=5)
    args.max_chars = _bounded_int(getattr(args, "max_chars", None), default=800, minimum=100, maximum=_MAX_SEARCH_CHARS)
    return args


def _iso(ts: Any) -> str | None:
    if ts is None:
        return None
    try:
        return _dt.datetime.fromtimestamp(float(ts), tz=_dt.timezone.utc).isoformat().replace("+00:00", "Z")
    except Exception:
        return str(ts)


def _parse_time_expr(expr: str | None) -> float | None:
    if not expr:
        return None
    expr = expr.strip()
    m = re.fullmatch(r"(\d+)([smhdw])", expr)
    if m:
        n = int(m.group(1)); unit = m.group(2)
        seconds = {"s": 1, "m": 60, "h": 3600, "d": 86400, "w": 604800}[unit] * n
        return (_dt.datetime.now(tz=_dt.timezone.utc).timestamp() - seconds)
    # Support ISO-ish date/datetime.
    try:
        normalized = expr.replace("Z", "+00:00")
        return _dt.datetime.fromisoformat(normalized).timestamp()
    except Exception:
        try:
            return float(expr)
        except Exception as e:
            raise SystemExit(f"Invalid time expression {expr!r}: use e.g. 7d, 12h, ISO datetime, or unix timestamp") from e


def _schema_counts(conn: sqlite3.Connection) -> dict[str, Any]:
    tables = [r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")]
    out: dict[str, Any] = {
        "state_db": str(_state_db_path()),
        "state_db_readable": True,
        "tables": tables,
        "has_messages_fts": "messages_fts" in tables,
        "has_messages_fts_trigram": "messages_fts_trigram" in tables,
    }
    for table in ("sessions", "messages"):
        if table in tables:
            out[f"{table}_count"] = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
    latest = conn.execute("SELECT id, source, title, started_at FROM sessions ORDER BY started_at DESC LIMIT 1").fetchone()
    if latest:
        out["latest_session"] = {
            "id": latest["id"], "source": _redact(latest["source"]), "title": _redact(latest["title"]), "started_at": _iso(latest["started_at"])
        }
    return out


def status(args: argparse.Namespace) -> dict[str, Any]:
    path = _state_db_path()
    if not path.exists():
        return {"state_db": str(path), "state_db_readable": False, "error": "missing state.db"}
    with _connect() as conn:
        return _schema_counts(conn)


def _where_filters(args: argparse.Namespace, alias: str = "m") -> tuple[str, list[Any]]:
    clauses: list[str] = []
    vals: list[Any] = []
    if getattr(args, "session", None) or getattr(args, "session_id", None):
        clauses.append(f"{alias}.session_id = ?")
        vals.append(getattr(args, "session", None) or getattr(args, "session_id", None))
    if getattr(args, "role", None):
        clauses.append(f"{alias}.role = ?")
        vals.append(args.role)
    if getattr(args, "tool_name", None):
        clauses.append(f"{alias}.tool_name = ?")
        vals.append(args.tool_name)
    since = _parse_time_expr(getattr(args, "since", None))
    if since is not None:
        clauses.append(f"{alias}.timestamp >= ?")
        vals.append(since)
    before = _parse_time_expr(getattr(args, "before", None))
    if before is not None:
        clauses.append(f"{alias}.timestamp <= ?")
        vals.append(before)
    return (" AND ".join(clauses), vals)


def _fts_query(q: str) -> str:
    # Prefer exact phrase to avoid FTS syntax explosions on PER-123, paths, quotes.
    q = q.replace('"', '""')
    return f'"{q}"'


def grep(args: argparse.Namespace) -> dict[str, Any]:
    args = _normalize_search_args(args)
    with _connect() as conn:
        where, vals = _where_filters(args)
        base_where = (" AND " + where) if where else ""
        order = "m.timestamp DESC" if args.sort == "time" else "rank"
        # First try FTS phrase. Fallback to LIKE for punctuation-heavy queries.
        try:
            rows = conn.execute(
                f"""
                SELECT m.id, m.session_id, m.role, m.tool_name, m.timestamp, s.source, s.title, m.content,
                       bm25(messages_fts) AS rank
                FROM messages_fts
                JOIN messages m ON m.id = messages_fts.rowid
                JOIN sessions s ON s.id = m.session_id
                WHERE messages_fts MATCH ? {base_where}
                ORDER BY {order}
                LIMIT ?
                """,
                [_fts_query(args.query), *vals, args.limit],
            ).fetchall()
            mode = "fts_phrase"
        except sqlite3.Error:
            like = f"%{args.query}%"
            rows = conn.execute(
                f"""
                SELECT m.id, m.session_id, m.role, m.tool_name, m.timestamp, s.source, s.title, m.content,
                       0.0 AS rank
                FROM messages m
                JOIN sessions s ON s.id = m.session_id
                WHERE (m.content LIKE ? OR m.tool_name LIKE ? OR m.tool_calls LIKE ?) {base_where}
                ORDER BY m.timestamp DESC
                LIMIT ?
                """,
                [like, like, like, *vals, args.limit],
            ).fetchall()
            mode = "like_fallback"
    matches = []
    for r in rows:
        matches.append({
            "message_id": r["id"],
            "session_id": r["session_id"],
            "source": _redact(r["source"]),
            "title": _redact(r["title"]),
            "role": r["role"],
            "tool_name": r["tool_name"],
            "timestamp": _iso(r["timestamp"]),
            "rank": r["rank"],
            "snippet": _clip(r["content"], args.max_chars),
        })
    return {"query": args.query, "mode": mode, "count": len(matches), "matches": matches}


def _message_dict(r: sqlite3.Row, max_chars: int) -> dict[str, Any]:
    return {
        "message_id": r["id"],
        "session_id": r["session_id"],
        "role": r["role"],
        "tool_name": r["tool_name"],
        "timestamp": _iso(r["timestamp"]),
        "content": _clip(r["content"], max_chars),
    }


def describe(args: argparse.Namespace) -> dict[str, Any]:
    args = _normalize_describe_args(args)
    with _connect() as conn:
        if args.message_id is not None:
            center = conn.execute("SELECT id, session_id FROM messages WHERE id = ?", [args.message_id]).fetchone()
            if not center:
                return {"error": "message_id not found", "message_id": args.message_id}
            session_id = center["session_id"]
            before = conn.execute(
                "SELECT * FROM messages WHERE session_id=? AND id < ? ORDER BY id DESC LIMIT ?",
                [session_id, args.message_id, args.window],
            ).fetchall()
            mid = conn.execute("SELECT * FROM messages WHERE id=?", [args.message_id]).fetchall()
            after = conn.execute(
                "SELECT * FROM messages WHERE session_id=? AND id > ? ORDER BY id ASC LIMIT ?",
                [session_id, args.message_id, args.window],
            ).fetchall()
            rows = list(reversed(before)) + mid + after
        elif args.session_id and args.tail:
            session_id = args.session_id
            rows = conn.execute(
                "SELECT * FROM messages WHERE session_id=? ORDER BY id DESC LIMIT ?",
                [session_id, args.tail],
            ).fetchall()
            rows = list(reversed(rows))
        elif args.session_id and args.around:
            gargs = argparse.Namespace(query=args.around, session=args.session_id, session_id=None, role=None, tool_name=None, since=None, before=None, sort="time", limit=1, max_chars=args.max_chars)
            found = grep(gargs).get("matches", [])
            if not found:
                return {"session_id": args.session_id, "around": args.around, "count": 0, "messages": []}
            args.message_id = found[0]["message_id"]
            return describe(args)
        else:
            return {"error": "provide --message-id, or --session-id with --tail/--around"}
        session = conn.execute("SELECT id, source, title, started_at, ended_at FROM sessions WHERE id=?", [rows[0]["session_id"] if rows else args.session_id]).fetchone() if rows else None
    session_obj = None
    if session:
        session_obj = dict(session)
        session_obj["source"] = _redact(session_obj.get("source"))
        session_obj["title"] = _redact(session_obj.get("title"))
        session_obj |= {"started_at_iso": _iso(session["started_at"]), "ended_at_iso": _iso(session["ended_at"])}
    return {
        "session": session_obj,
        "count": len(rows),
        "messages": [_message_dict(r, args.max_chars) for r in rows],
    }


def recall(args: argparse.Namespace) -> dict[str, Any]:
    # Deterministic extractive recall: grouped search evidence, no LLM call.
    args = _normalize_recall_args(args)
    gargs = argparse.Namespace(query=args.query, session=None if args.all_sessions else args.session, session_id=None, role=args.role, tool_name=args.tool_name, since=args.since, before=args.before, sort=args.sort, limit=args.limit, max_chars=args.max_chars)
    results = grep(gargs)["matches"]
    grouped: dict[str, list[dict[str, Any]]] = {}
    for m in results:
        grouped.setdefault(m["session_id"], []).append(m)
    return {
        "query": args.query,
        "prompt": args.prompt,
        "format": args.format,
        "session_count": len(grouped),
        "evidence": [
            {"session_id": sid, "matches": ms[: args.per_session]} for sid, ms in grouped.items()
        ],
        "note": "extractive read-only recall; verify important claims against live source of truth",
    }


def _emit(obj: dict[str, Any], as_json: bool) -> None:
    if as_json:
        print(json.dumps(obj, ensure_ascii=False, indent=2))
    else:
        print(json.dumps(obj, ensure_ascii=False, indent=2))


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Hermes read-only LCM recall over state.db")
    sub = p.add_subparsers(dest="command", required=True)

    ps = sub.add_parser("status", help="Show state DB readability/counts")
    ps.add_argument("--json", action="store_true", help="Emit JSON (accepted here for compatibility)")
    ps.set_defaults(func=status)

    pg = sub.add_parser("grep", help="Search messages")
    pg.add_argument("query")
    pg.add_argument("--json", action="store_true", help="Emit JSON (accepted here for compatibility)")
    pg.add_argument("--all-sessions", action="store_true", help="Accepted for compatibility; search is global by default")
    pg.add_argument("--session", "--session-id", dest="session", default=None)
    pg.add_argument("--role", default=None)
    pg.add_argument("--tool-name", default=None)
    pg.add_argument("--since", default=None)
    pg.add_argument("--before", default=None)
    pg.add_argument("--sort", choices=["rank", "time"], default="rank")
    pg.add_argument("--limit", type=int, default=5)
    pg.add_argument("--max-chars", type=int, default=800)
    pg.set_defaults(func=grep)

    pd = sub.add_parser("describe", help="Describe bounded context around a message/session")
    pd.add_argument("--json", action="store_true", help="Emit JSON (accepted here for compatibility)")
    pd.add_argument("--message-id", type=int, default=None)
    pd.add_argument("--session-id", default=None)
    pd.add_argument("--tail", type=int, default=None)
    pd.add_argument("--around", default=None)
    pd.add_argument("--window", type=int, default=3)
    pd.add_argument("--max-chars", type=int, default=1200)
    pd.set_defaults(func=describe)

    pr = sub.add_parser("recall", help="Grouped extractive recall evidence")
    pr.add_argument("query")
    pr.add_argument("--json", action="store_true", help="Emit JSON (accepted here for compatibility)")
    pr.add_argument("--prompt", default="")
    pr.add_argument("--all-sessions", action="store_true")
    pr.add_argument("--session", default=None)
    pr.add_argument("--role", default=None)
    pr.add_argument("--tool-name", default=None)
    pr.add_argument("--since", default=None)
    pr.add_argument("--before", default=None)
    pr.add_argument("--sort", choices=["rank", "time"], default="rank")
    pr.add_argument("--limit", type=int, default=10)
    pr.add_argument("--per-session", type=int, default=3)
    pr.add_argument("--window", type=int, default=1, help="Compatibility flag; use describe for windows")
    pr.add_argument("--format", choices=["compact", "evidence", "full-json"], default="compact")
    pr.add_argument("--max-chars", type=int, default=800)
    pr.set_defaults(func=recall)

    pe = sub.add_parser("expand", help="Alias for recall")
    pe.add_argument("query")
    pe.add_argument("--json", action="store_true", help="Emit JSON (accepted here for compatibility)")
    pe.add_argument("--prompt", default="")
    pe.add_argument("--all-sessions", action="store_true")
    pe.add_argument("--session", default=None)
    pe.add_argument("--role", default=None)
    pe.add_argument("--tool-name", default=None)
    pe.add_argument("--since", default=None)
    pe.add_argument("--before", default=None)
    pe.add_argument("--sort", choices=["rank", "time"], default="rank")
    pe.add_argument("--limit", type=int, default=10)
    pe.add_argument("--per-session", type=int, default=3)
    pe.add_argument("--window", type=int, default=1)
    pe.add_argument("--format", choices=["compact", "evidence", "full-json"], default="compact")
    pe.add_argument("--max-chars", type=int, default=800)
    pe.set_defaults(func=recall)

    p.add_argument("--json", action="store_true", help="Emit JSON (default; retained for compatibility)")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        obj = args.func(args)
        _emit(obj, True)
        return 0 if "error" not in obj else 1
    except sqlite3.Error as e:
        print(json.dumps({"error": f"sqlite: {e}", "state_db": str(_state_db_path())}, ensure_ascii=False), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
