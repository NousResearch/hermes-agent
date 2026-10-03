"""``hermes trace show|export``: read the Relay events recorded for one session."""

from __future__ import annotations

import json
import sys
from datetime import datetime
from typing import Any

_GLYPH = {"agent": "◆", "function": "▸", "llm": "✦", "tool": "⚙"}


def _epoch(timestamp: Any) -> float | None:
    try:
        return datetime.fromisoformat(str(timestamp)).timestamp()
    except ValueError:
        return None


def _label(start: dict[str, Any]) -> str:
    name, metadata = str(start.get("name") or ""), start.get("metadata") or {}
    if name == "hermes.session":
        role = "subagent" if metadata.get("hermes.parent_session_id") else "session"
        return f"{role} {metadata.get('hermes.session_id', '')}".strip()
    if start.get("category") == "llm":
        model = (start.get("category_profile") or {}).get("model_name")
        return f"{name} · {model}" if model else name
    return name


def _failed(end: dict[str, Any] | None) -> bool:
    if not end:
        return False
    data = end.get("data")
    outcome = data.get("outcome") if isinstance(data, dict) else None
    return (end.get("metadata") or {}).get("otel.status_code") == "ERROR" or outcome == "failed"


def format_tree(events: list[dict[str, Any]]) -> list[str]:
    """One line per span, indented under its parent, in start order."""
    starts: dict[str, dict[str, Any]] = {}
    ends: dict[str, dict[str, Any]] = {}
    for event in events:
        if event.get("kind") != "scope":
            continue
        uuid = str(event.get("uuid") or "")
        (starts if event.get("scope_category") == "start" else ends)[uuid] = event
    children: dict[str, list[str]] = {}
    for uuid, start in starts.items():
        parent = str(start.get("parent_uuid") or "")
        children.setdefault(parent if parent in starts else "", []).append(uuid)
    for siblings in children.values():
        siblings.sort(key=lambda u: str(starts[u].get("timestamp") or ""))

    lines: list[str] = []

    def walk(parent: str, depth: int) -> None:
        for uuid in children.get(parent, []):
            start, end = starts[uuid], ends.get(uuid)
            began, finished = _epoch(start.get("timestamp")), _epoch((end or {}).get("timestamp"))
            mark = "✗" if _failed(end) else _GLYPH.get(str(start.get("category")), "·")
            took = f"{finished - began:7.2f}s" if began is not None and finished is not None else "running"
            lines.append(f"{'  ' * depth}{mark} {_label(start)[:72]:<72} {took}")
            walk(uuid, depth + 1)

    walk("", 0)
    return lines


def run_trace_command(args: Any) -> int:
    from hermes_cli.observability.relay_traces import read_session_events
    from hermes_constants import get_hermes_home
    from hermes_state import SessionDB

    db = SessionDB()
    try:
        session_id = db.resolve_session_id(args.session) or args.session
    finally:
        db.close()
    events, truncated = read_session_events(get_hermes_home(), session_id)
    if not events:
        print(f"No trace recorded for session {session_id}", file=sys.stderr)
        return 1
    if args.trace_action == "show":
        print(f"trace {session_id} · {len(events)} events{' (newest kept)' if truncated else ''}")
        print("\n".join(format_tree(events)))
        return 0
    payload = "".join(json.dumps(event, ensure_ascii=False) + "\n" for event in events)
    output = getattr(args, "output", None)
    if not output or output == "-":
        sys.stdout.write(payload)
    else:
        with open(output, "w", encoding="utf-8") as handle:
            handle.write(payload)
        print(f"Wrote {len(events)} Relay events to {output}")
    return 0
