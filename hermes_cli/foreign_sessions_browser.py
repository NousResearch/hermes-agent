"""Backend-local discovery and previews for the desktop session importer."""

import base64
import hashlib
import json
import os
import re
import socket
from pathlib import Path
from stat import S_ISREG

from hermes_cli.foreign_sessions import _SOURCE_DB_NAMES, _SOURCE_LABELS, _SOURCES, _source_roots, _walk

MAX_LOG_BYTES = 32 * 1024 * 1024
MAX_COWORK_METADATA_BYTES = 8 * 1024 * 1024
MAX_METADATA_TEXT = 180
MAX_SNAPSHOT_BYTES = 5 * 1024 * 1024


def _display_title(parsed, source):
    # Codex attachment envelopes put the actual request after a file listing.
    # Keep the history intact; only the browser's display title skips that header.
    first_user = next((turn["content"] for turn in parsed["turns"] if turn["role"] == "user"), "")
    request = re.split(r"(?im)^#{1,6}\s*My request:\s*$", first_user, maxsplit=1)
    title = request[-1].strip().splitlines()[0] if len(request) > 1 and request[-1].strip() else parsed["title_guess"]
    return (title or _SOURCE_LABELS[source]).lstrip("# ")[:180]


def _cowork_metadata(path):
    claude_dir = next((parent for parent in path.parents if parent.name == ".claude"), None)
    session_dir = claude_dir.parent if claude_dir is not None else None
    if session_dir is None or not session_dir.name.startswith("local_"):
        return {}
    try:
        metadata_path = session_dir.with_suffix(".json").resolve()
        st = metadata_path.stat()
        if metadata_path.parent != session_dir.parent or not S_ISREG(st.st_mode) or st.st_size > MAX_COWORK_METADATA_BYTES:
            return {}
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    if not isinstance(metadata, dict):
        return {}
    folders = metadata.get("userSelectedFolders")
    names = [re.split(r"[/\\]", folder.rstrip("/\\"))[-1]
             for folder in folders if isinstance(folder, str) and folder.rstrip("/\\")] if isinstance(folders, list) else []
    title = metadata.get("title")
    return {"title": title.strip()[:MAX_METADATA_TEXT] if isinstance(title, str) and title.strip() else None,
            "project": ", ".join(dict.fromkeys(names))[:MAX_METADATA_TEXT] or None}


def _grok_bot_name(path, key):
    # ponytail: one exact account-scoped roster lookup, never scan other accounts.
    scope, separator, agent_id = (key or "").rpartition(".transcript.replicas.")
    if not separator or not scope.startswith("sand.client.slice.account.") or not agent_id:
        return None
    roster_key = scope + ".roster.last-roster"
    filename = base64.b32encode(roster_key.encode()).decode().rstrip("=").lower() + ".blob"
    try:
        parent = path.parent.resolve()
        if not any(parent.is_relative_to(root) for root in _source_roots("grok")):
            return None
        fd = os.open(parent / filename, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0))
        with os.fdopen(fd, "rb") as stream:
            if not S_ISREG(os.fstat(stream.fileno()).st_mode):
                return None
            raw = stream.read(1024 * 1024 + 1)
        if len(raw) > 1024 * 1024:
            return None
        data = json.loads(raw)
    except (OSError, ValueError):
        return None
    value = data.get("value") if isinstance(data, dict) and data.get("schemaVersion") == 4 else None
    rows = value.get("rows") if isinstance(value, dict) else None
    if not isinstance(rows, list):
        return None
    matches = [row for row in rows if isinstance(row, dict) and row.get("id") == agent_id]
    name = matches[0].get("name") if len(matches) == 1 else None
    return " ".join(name.split())[:MAX_METADATA_TEXT] or None if isinstance(name, str) else None


def _candidates(source=None):
    """``(mtime, handle, source, path, size)`` rows across sources, newest first. The handle is
    the only identifier handed to the client; a request can never name a path."""
    if source is not None and source not in _SOURCES:
        raise ValueError("Unknown session source")
    rows = [(st.st_mtime, hashlib.sha256(f"{name}:{path}".encode()).hexdigest(), name, path, st.st_size)
            for name in _SOURCES if source in (None, name) for path, st in _walk(name)]
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
        mtime, handle, name, path, _ = candidate
        try:
            parsed = _parse(candidate)
        except (ValueError, OSError):
            unreadable += 1
            continue
        metadata = _cowork_metadata(path) if name == "cowork" else {}
        cwd = parsed["cwd"]
        project = _grok_bot_name(path, parsed["session_id"]) if name == "grok" else metadata.get("project")
        if not project and isinstance(cwd, str) and cwd.rstrip("/\\"):
            project = re.split(r"[/\\]", cwd.rstrip("/\\"))[-1][:MAX_METADATA_TEXT] or None
        rows.append({"id": handle, "source": name, "label": _SOURCE_LABELS[name],
                     "title": metadata.get("title") or _display_title(parsed, name),
                     "project": project,
                     "cwd": cwd, "mtime": mtime, "turn_count": len(parsed["turns"]),
                     "excerpt": parsed["turns"][0]["content"][:200]})
    next_offset = offset + limit
    available = [name for name in _SOURCES
                 if (bool(_walk(name)) if name == "grok" else any(root.is_dir() for root in _source_roots(name)))]
    return {"sessions": rows, "sources": available,
            "next_offset": next_offset if next_offset < len(candidates) else None,
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


def export_browser_session(handle):
    """Portable transcript snapshot for importing into another gateway; local paths never leave the source."""
    candidate, parsed = resolve_foreign_session(handle)
    metadata = _cowork_metadata(candidate[3]) if candidate[2] == "cowork" else {}
    snapshot = {
        "origin": {"tool": _SOURCE_DB_NAMES[candidate[2]], "path": f"desktop:{handle}",
                   "foreign_session_id": parsed["session_id"]},
        "messages": parsed["turns"],
        "title": metadata.get("title") or _display_title(parsed, candidate[2]),
    }
    if len(json.dumps(snapshot, ensure_ascii=False, separators=(",", ":")).encode("utf-8")) > MAX_SNAPSHOT_BYTES:
        raise ValueError("This session exceeds the 5 MB cross-gateway import limit")
    return snapshot


def import_browser_snapshot(snapshot, db, profile):
    """Validate and adopt a portable snapshot received from another authenticated gateway."""
    if not isinstance(snapshot, dict):
        raise ValueError("Import snapshot must be an object")
    try:
        size = len(json.dumps(snapshot, ensure_ascii=False, separators=(",", ":")).encode("utf-8"))
    except (TypeError, ValueError):
        raise ValueError("Import snapshot must be JSON serializable") from None
    if size > MAX_SNAPSHOT_BYTES:
        raise ValueError("Import snapshot exceeds the 5 MB limit")
    origin = snapshot.get("origin")
    if not isinstance(origin, dict) or origin.get("tool") not in _SOURCE_DB_NAMES.values():
        raise ValueError("Import snapshot has an unknown source")
    path = origin.get("path")
    if not isinstance(path, str) or not re.fullmatch(r"desktop:[0-9a-f]{64}", path):
        raise ValueError("Import snapshot has an invalid source handle")
    foreign_id = origin.get("foreign_session_id")
    if foreign_id is not None and (not isinstance(foreign_id, str) or len(foreign_id) > 256):
        raise ValueError("Import snapshot has an invalid session id")
    title = snapshot.get("title")
    if not isinstance(title, str) or not title.strip() or len(title) > MAX_METADATA_TEXT:
        raise ValueError("Import snapshot has an invalid title")
    messages = snapshot.get("messages")
    if not isinstance(messages, list) or not messages:
        raise ValueError("Import snapshot has no conversation messages")
    normalized = []
    for index, message in enumerate(messages):
        if not isinstance(message, dict) or message.get("role") not in ("user", "assistant"):
            raise ValueError(f"Import snapshot message {index} has an invalid role")
        content = message.get("content")
        if not isinstance(content, str) or not content.strip():
            raise ValueError(f"Import snapshot message {index} has invalid content")
        if (index == 0 and message["role"] != "user") or (normalized and normalized[-1]["role"] == message["role"]):
            raise ValueError("Import snapshot messages must alternate from a user message")
        normalized.append({"role": message["role"], "content": content})
    return db.import_foreign_history(
        {"tool": origin["tool"], "path": path, "foreign_session_id": foreign_id},
        normalized, title=title.strip(), cwd=None, profile=profile)


def import_browser_session(handle, db, profile):
    candidate, parsed = resolve_foreign_session(handle)
    origin = foreign_origin(candidate, parsed)
    cwd = parsed["cwd"]
    return db.import_foreign_history(origin, parsed["turns"],
                                     title=_display_title(parsed, candidate[2]),
                                     cwd=cwd if cwd and Path(cwd).is_dir() else None, profile=profile)
