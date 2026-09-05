"""Backend-local discovery and previews for the desktop session importer."""

import hashlib
import re
import socket
from pathlib import Path

from hermes_cli.foreign_sessions import _SOURCES, _SOURCE_LABELS, _SOURCE_DB_NAMES

MAX_LOG_BYTES = 32 * 1024 * 1024


def _display_title(parsed, source):
    # Codex attachment envelopes put the actual request after a file listing.
    # Keep the history intact; only the browser's display title skips that header.
    first_user = next((turn["content"] for turn in parsed["turns"] if turn["role"] == "user"), "")
    request = re.split(r"(?im)^#{1,6}\s*My request:\s*$", first_user, maxsplit=1)
    title = request[-1].strip().splitlines()[0] if len(request) > 1 and request[-1].strip() else parsed["title_guess"]
    return (title or _SOURCE_LABELS[source]).lstrip("# ")[:180]


def _candidates(source=None):
    if source is not None and source not in _SOURCES:
        raise ValueError("Unknown session source")
    rows = []
    for name, (parts, pattern, recursive, _) in _SOURCES.items():
        if source and source != name:
            continue
        root = Path.home().joinpath(*parts).resolve()
        if not root.is_dir():
            continue
        for path in root.rglob(pattern) if recursive else root.glob(pattern):
            resolved = path.resolve()
            if not resolved.is_relative_to(root):
                continue
            try:
                stat = resolved.stat()
            except FileNotFoundError:
                continue  # A running source may rotate a log during discovery.
            if not resolved.is_file():
                continue
            handle = hashlib.sha256(f"{name}:{resolved}".encode()).hexdigest()
            rows.append((stat.st_mtime, handle, name, resolved, stat.st_size))
    return sorted(rows, key=lambda row: (row[0], row[1]), reverse=True)


def _parse(candidate):
    _, _, source, path, size = candidate
    if size > MAX_LOG_BYTES:
        raise ValueError("This log exceeds the 32 MB preview and import limit")
    parsed = _SOURCES[source][3](path)
    if not parsed["turns"]:
        raise ValueError("This session has no readable conversation messages")
    return parsed


def resolve_foreign_session(handle):
    if not isinstance(handle, str) or len(handle) != 64:
        raise ValueError("Unknown session. Refresh the list and try again")
    for candidate in _candidates():
        if candidate[1] == handle:
            return candidate, _parse(candidate)
    raise ValueError("Session no longer available. Refresh the list and try again")


def list_foreign_sessions(source=None, offset=0, limit=25):
    if not isinstance(offset, int) or offset < 0 or not isinstance(limit, int) or not 1 <= limit <= 50:
        raise ValueError("Invalid session page")
    candidates = _candidates(source)
    rows, unreadable = [], 0
    # Page candidates before parsing. Empty or oversized logs cannot turn a
    # request for 25 rows into an unbounded transcript scan.
    for candidate in candidates[offset:offset + limit]:
        mtime, handle, name, _, _ = candidate
        try:
            parsed = _parse(candidate)
        except (ValueError, OSError):
            unreadable += 1
            continue
        rows.append({"id": handle, "source": name, "label": _SOURCE_LABELS[name],
                     "title": _display_title(parsed, name),
                     "cwd": parsed["cwd"], "mtime": mtime, "turn_count": len(parsed["turns"]),
                     "excerpt": parsed["turns"][0]["content"][:200]})
    next_offset = offset + limit
    return {"sessions": rows, "next_offset": next_offset if next_offset < len(candidates) else None,
            "host": socket.gethostname(), "unreadable": unreadable}


def foreign_origin(candidate, parsed):
    return {"tool": _SOURCE_DB_NAMES[candidate[2]], "path": str(candidate[3]),
            "foreign_session_id": parsed["session_id"]}


def preview_foreign_session(handle, db):
    candidate, parsed = resolve_foreign_session(handle)
    origin = foreign_origin(candidate, parsed)
    existing = db.find_foreign_import(origin)
    # A bounded preview avoids mounting thousands of messages in the renderer.
    messages = [{**turn, "content": turn["content"][:8000]} for turn in parsed["turns"][-40:]]
    return {"messages": messages, "total": len(parsed["turns"]),
            "truncated": len(parsed["turns"]) > 40 or any(len(turn["content"]) > 8000 for turn in parsed["turns"][-40:]),
            "already_imported": existing, "cwd": parsed["cwd"]}


def import_browser_session(handle, db, profile):
    candidate, parsed = resolve_foreign_session(handle)
    origin = foreign_origin(candidate, parsed)
    cwd = parsed["cwd"]
    return db.import_foreign_history(origin, parsed["turns"],
                                     title=_display_title(parsed, candidate[2]),
                                     cwd=cwd if cwd and Path(cwd).is_dir() else None, profile=profile)
