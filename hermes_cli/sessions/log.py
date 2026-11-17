"""``hermes sessions log``: print a session's stored transcript as text or JSON lines.

Reads this profile's ``state.db`` read-only (the gateway keeps writing it), across the session's
compression lineage, so a script can inspect what a session it drives with ``send`` said and did.
"""
from __future__ import annotations

from fnmatch import fnmatchcase
import json
import sys

from hermes_cli.sessions.target import resolve_local, session_argument


def _entries(row):
    """One stored message as log entries: its own row, plus one ``tool_call`` per call it made."""
    base = {"id": row.get("id"), "timestamp": row.get("timestamp")}
    content = row.get("content")
    if row.get("role") == "tool":
        yield {**base, "type": "tool", "name": row.get("tool_name"), "tool_call_id": row.get("tool_call_id"),
               "content": content}
        return
    if content:
        yield {**base, "type": row.get("role"), "content": content}
    for call in row.get("tool_calls") or ():
        if not isinstance(call, dict):
            continue
        function = call.get("function") or {}
        yield {**base, "type": "tool_call", "name": function.get("name"), "tool_call_id": call.get("id"),
               "arguments": function.get("arguments")}


def _render(entry) -> str:
    content = entry.get("content")
    text = content if isinstance(content, str) else json.dumps(content, ensure_ascii=False)
    renderers = {
        "tool_call": lambda: f"[tool_call] {entry.get('name')} {entry.get('arguments') or ''}".rstrip(),
        "tool": lambda: f"[tool] {entry.get('name') or ''}: {text}",
    }
    return renderers.get(entry["type"], lambda: f"{entry['type']}: {text}")()


def _wanted(patterns):
    globs = [p.strip() for p in (patterns or "").split(",") if p.strip()]
    return lambda entry: not globs or any(fnmatchcase(str(entry["type"]), glob) for glob in globs)


def cmd_log(db, args) -> int:
    session_id = resolve_local(args.session, db)
    if not session_id:
        print(f"Error: No session found matching '{args.session}'. Use 'hermes sessions list' to see available sessions.",
              file=sys.stderr)
        return 1
    for entry in _select(db, session_id, _wanted(args.types), args.limit):
        print(json.dumps({"session_id": session_id, **entry}, ensure_ascii=False, default=str)
              if args.json else _render(entry))
    return 0


_PAGE = 200


def _select(db, session_id, wanted, limit):
    """The matching entries oldest first; with *limit*, only the newest *limit*, read back from the
    end in pages so a short tail never loads a long session. Compacted rows are the display history
    (``include_compacted``), so an in-place compaction does not hide earlier turns."""
    def page(offset):
        return db.get_messages(session_id, include_ancestors=True, include_compacted=True,
                               latest=bool(limit), limit=_PAGE if limit else None, offset=offset)
    if not limit:
        return [entry for row in page(0) for entry in _entries(row) if wanted(entry)]
    tail, offset = [], 0
    while len(tail) < limit:
        rows = page(offset)
        tail = [entry for row in rows for entry in _entries(row) if wanted(entry)] + tail
        if len(rows) < _PAGE:
            break
        offset += _PAGE
    return tail[-limit:]


def add_log_parser(sessions_subparsers) -> None:
    parser = sessions_subparsers.add_parser(
        "log", help="Print a session's stored transcript (messages and tool calls)",
        description="Print the stored transcript of a session, oldest first, across its compression "
            "lineage. --json prints one JSON object per entry; --types keeps only entries whose type "
            "(user, assistant, tool_call, tool, system) matches a comma-separated glob list.")
    session_argument(parser)
    parser.add_argument("--json", action="store_true", help="One JSON object per entry")
    parser.add_argument("--types", metavar="PATTERN", help="Comma-separated type globs, e.g. 'assistant,tool*'")
    parser.add_argument("--limit", type=int, default=0, metavar="N", help="Only the last N entries")
    parser.set_defaults(sessions_action="log")
