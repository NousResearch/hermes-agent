#!/usr/bin/env python3
"""Per-tool call/result metrics for one session, from the Hermes session store.

Reads the canonical SQLite session store (``state.db``) strictly READ-ONLY
(SQLite ``mode=ro`` URI, same pattern as ``hermes_state.SessionDB(read_only=True)``)
and reports, for a single session, only metrics the store can prove:

* ``calls``    — ``tool_calls`` entries on assistant rows
* ``results``  — tool-role rows correlated back to a call by ``tool_call_id``
* ``orphaned`` — calls with no matching result row
* ``errors``   — correlated results whose payload is a JSON object with a
  truthy ``"error"`` key. This is deliberately the same classification rule
  the runtime observer applies (``model_tools._tool_result_observer_fields``),
  so the audit never claims more than the runtime itself would.

Latency is deliberately NOT reported: per-call ``duration_ms`` is computed in
``model_tools.handle_function_call`` and exposed only to ``post_tool_call``
observer hooks — it is never persisted to ``state.db``, so no store-derived
latency figure would be honest.
"""

from __future__ import annotations

import argparse
import json
import os
import sqlite3
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Sentinel used by hermes_state.SessionDB._encode_content for structured
# (list/dict) message content. Kept as a literal so this script stays
# importable outside the Hermes process (system Python, CI).
CONTENT_JSON_PREFIX = "\x00json:"

# Sample error payloads included per tool in --json output (truncated).
MAX_SAMPLE_ERRORS = 3
MAX_SAMPLE_ERROR_CHARS = 200


def get_hermes_home() -> Path:
    """Resolve the active Hermes home, profile-aware when possible.

    Prefers ``hermes_constants.get_hermes_home()`` (honors profile overrides
    and future resolution changes). Falls back to the same core logic using
    only the stdlib: ``HERMES_HOME`` env var, else the platform default.
    """
    try:
        from hermes_constants import get_hermes_home as _real

        return _real()
    except (ModuleNotFoundError, ImportError):
        pass
    val = os.environ.get("HERMES_HOME", "").strip()
    if val:
        return Path(val)
    if sys.platform == "win32":
        local_appdata = os.environ.get("LOCALAPPDATA", "").strip()
        base = Path(local_appdata) if local_appdata else Path.home() / "AppData" / "Local"
        return base / "hermes"
    return Path.home() / ".hermes"


def open_db_readonly(db_path: Path) -> sqlite3.Connection:
    """Open ``state.db`` read-only (no write lock, no schema init)."""
    conn = sqlite3.connect(
        f"file:{db_path}?mode=ro",
        uri=True,
        timeout=1.0,
        isolation_level=None,
    )
    conn.row_factory = sqlite3.Row
    return conn


def resolve_session(conn: sqlite3.Connection, session_arg: Optional[str]) -> str:
    """Resolve the target session id.

    Order: explicit ``session_arg`` (exact id, then unique prefix) →
    ``HERMES_SESSION_ID`` env (set by the running agent) → most recently
    started session. Raises ``LookupError`` when nothing matches or a prefix
    is ambiguous.
    """
    candidate = (session_arg or "").strip() or os.environ.get("HERMES_SESSION_ID", "").strip()
    if candidate:
        row = conn.execute("SELECT id FROM sessions WHERE id = ?", (candidate,)).fetchone()
        if row:
            return row["id"]
        rows = conn.execute(
            "SELECT id FROM sessions WHERE id LIKE ? ORDER BY started_at DESC LIMIT 5",
            (candidate + "%",),
        ).fetchall()
        if len(rows) == 1:
            return rows[0]["id"]
        if not rows:
            raise LookupError(f"no session matches '{candidate}'")
        matches = ", ".join(r["id"] for r in rows)
        raise LookupError(f"ambiguous session prefix '{candidate}': {matches}")
    row = conn.execute("SELECT id FROM sessions ORDER BY started_at DESC LIMIT 1").fetchone()
    if not row:
        raise LookupError("session store contains no sessions")
    return row["id"]


def fetch_messages(conn: sqlite3.Connection, session_id: str) -> List[Dict[str, Any]]:
    """Load active messages in insertion order (mirrors SessionDB.get_messages)."""
    sql = "SELECT role, content, tool_call_id, tool_calls, tool_name FROM messages WHERE session_id = ?{} ORDER BY id"
    try:
        rows = conn.execute(sql.format(" AND active = 1"), (session_id,)).fetchall()
    except sqlite3.OperationalError:
        # Older schema without the soft-delete column.
        rows = conn.execute(sql.format(""), (session_id,)).fetchall()
    return [dict(r) for r in rows]


def decode_content(content: Any) -> Any:
    """Reverse SessionDB._encode_content's sentinel-prefixed JSON encoding."""
    if isinstance(content, str) and content.startswith(CONTENT_JSON_PREFIX):
        try:
            return json.loads(content[len(CONTENT_JSON_PREFIX):])
        except (json.JSONDecodeError, TypeError):
            return content
    return content


def result_error(content: Any) -> Optional[str]:
    """Return the error message when a tool result payload is a provable error.

    Same rule as ``model_tools._tool_result_observer_fields``: the payload
    (JSON-decoded when it is a string) must be a dict with a truthy
    ``"error"`` key. Anything else — including plain-text failure prose —
    is NOT counted; the audit reports a lower bound, not a guess.
    """
    payload = decode_content(content)
    if isinstance(payload, str):
        try:
            payload = json.loads(payload)
        except (json.JSONDecodeError, TypeError):
            return None
    if isinstance(payload, dict) and payload.get("error"):
        return str(payload.get("error"))
    return None


def audit_messages(messages: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Correlate tool calls with results by ``tool_call_id`` and aggregate."""
    call_id_to_tool: Dict[str, str] = {}
    stats: Dict[str, Dict[str, int]] = {}
    samples: Dict[str, List[str]] = {}
    uncorrelated_results = 0

    def _tool_stats(name: str) -> Dict[str, int]:
        return stats.setdefault(name, {"calls": 0, "results": 0, "orphaned": 0, "errors": 0})

    for msg in messages:
        if msg.get("role") != "assistant" or not msg.get("tool_calls"):
            continue
        try:
            tool_calls = json.loads(msg["tool_calls"])
        except (json.JSONDecodeError, TypeError):
            continue
        if not isinstance(tool_calls, list):
            continue
        for entry in tool_calls:
            if not isinstance(entry, dict):
                continue
            name = (entry.get("function") or {}).get("name") or entry.get("name") or "<unknown>"
            _tool_stats(name)["calls"] += 1
            call_id = entry.get("id")
            if call_id:
                call_id_to_tool[str(call_id)] = name

    matched_ids = set()
    for msg in messages:
        if msg.get("role") != "tool":
            continue
        call_id = str(msg.get("tool_call_id") or "")
        name = call_id_to_tool.get(call_id)
        if name is None:
            uncorrelated_results += 1
            continue
        matched_ids.add(call_id)
        tool = _tool_stats(name)
        tool["results"] += 1
        error = result_error(msg.get("content"))
        if error is not None:
            tool["errors"] += 1
            bucket = samples.setdefault(name, [])
            if len(bucket) < MAX_SAMPLE_ERRORS:
                bucket.append(error[:MAX_SAMPLE_ERROR_CHARS])

    for call_id, name in call_id_to_tool.items():
        if call_id not in matched_ids:
            _tool_stats(name)["orphaned"] += 1

    tools = [
        {"name": name, **counts, "sample_errors": samples.get(name, [])}
        for name, counts in sorted(stats.items(), key=lambda kv: (-kv[1]["calls"], kv[0]))
    ]
    totals = {
        key: sum(t[key] for t in tools) for key in ("calls", "results", "orphaned", "errors")
    }
    return {"tools": tools, "totals": totals, "uncorrelated_results": uncorrelated_results}


def _session_header(conn: sqlite3.Connection, session_id: str) -> Dict[str, Any]:
    row = conn.execute(
        "SELECT id, source, title, started_at, message_count, tool_call_count"
        " FROM sessions WHERE id = ?",
        (session_id,),
    ).fetchone()
    header = dict(row) if row else {"id": session_id}
    started = header.get("started_at")
    if isinstance(started, (int, float)):
        try:
            header["started_at"] = datetime.fromtimestamp(started).isoformat(sep=" ", timespec="seconds")
        except (OSError, OverflowError, ValueError):
            pass
    return header


def render_text(header: Dict[str, Any], report: Dict[str, Any]) -> str:
    lines = [
        f"Tool audit — session {header.get('id')}",
        "  source: {}  started: {}  messages: {}".format(
            header.get("source", "?"), header.get("started_at", "?"), header.get("message_count", "?")
        ),
        "",
        f"{'tool':<32} {'calls':>6} {'results':>8} {'orphaned':>9} {'errors':>7}",
    ]
    for tool in report["tools"]:
        lines.append(
            f"{tool['name']:<32} {tool['calls']:>6} {tool['results']:>8}"
            f" {tool['orphaned']:>9} {tool['errors']:>7}"
        )
    totals = report["totals"]
    lines.append(
        f"{'TOTAL':<32} {totals['calls']:>6} {totals['results']:>8}"
        f" {totals['orphaned']:>9} {totals['errors']:>7}"
    )
    if report["uncorrelated_results"]:
        lines.append(f"uncorrelated tool results (no surviving call): {report['uncorrelated_results']}")
    lines.append("errors = results whose JSON payload has a truthy 'error' key (lower bound).")
    lines.append("Latency is not persisted to state.db and is intentionally not reported.")
    return "\n".join(lines)


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Read-only per-tool audit of a Hermes session.")
    parser.add_argument("--session", help="session id or unique prefix (default: $HERMES_SESSION_ID, else latest)")
    parser.add_argument("--db", type=Path, help="path to state.db (default: <hermes home>/state.db)")
    parser.add_argument("--json", action="store_true", help="emit machine-readable JSON")
    args = parser.parse_args(argv)

    db_path = args.db or (get_hermes_home() / "state.db")
    if not db_path.exists():
        print(f"error: session store not found: {db_path}", file=sys.stderr)
        return 2

    try:
        conn = open_db_readonly(db_path)
    except sqlite3.Error as exc:
        print(f"error: cannot open {db_path} read-only: {exc}", file=sys.stderr)
        return 2
    try:
        try:
            session_id = resolve_session(conn, args.session)
        except LookupError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 2
        header = _session_header(conn, session_id)
        report = audit_messages(fetch_messages(conn, session_id))
    finally:
        conn.close()

    if args.json:
        print(json.dumps({"session": header, **report}, ensure_ascii=False, indent=2))
    else:
        print(render_text(header, report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
