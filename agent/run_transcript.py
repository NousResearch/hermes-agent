"""Append-only JSONL history for named in-process agent runs.

Every named run appends to one file::

    <HERMES_HOME>/sessions/<agent>/<session_id>.jsonl

``<agent>`` is the name the caller gave the run, so two agents never share a file and a
continuing agent reads its own earlier turns back with :func:`read_history` (host
``conversation_history`` shape; the system prompt is excluded because the host owns it). The file
is the history, not a mirror of ``state.db``: SQLite keeps the main conversation, this keeps what
a peer agent said and did across runs of the same session.

One JSON object per line, append-only, never rewritten. Rows are ``{"kind": "run"|"msg", ...}``:
a ``run`` header per accepted run, then the run's messages (``role``, ``text``, ``tool_calls``,
``tool_call_id``/``name``/``result``, reasoning). Each message row carries ``fp`` — a content
fingerprint used to drop a history prefix the file already holds, because a continuing run is
handed its merged history and appends it again.

Writes are serialized per file. Read paths stay forgiving: a truncated final line is skipped,
never raised.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import threading
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional

from hermes_constants import get_hermes_home

logger = logging.getLogger(__name__)

#: Lines scanned for duplicate fingerprints when a run appends its merged history.
TAIL_SCAN_LINES = 800
_UNSAFE_NAME = re.compile(r"[^A-Za-z0-9._-]+")

_LOCKS: dict[str, threading.Lock] = {}
_LOCKS_GUARD = threading.Lock()


def sessions_dir(home: Optional[Path] = None) -> Path:
    """``<home>/sessions`` — the one root every run transcript lives under."""
    base = Path(home) if home is not None else get_hermes_home()
    return base / "sessions"


def _stem(value: Any, fallback: str) -> str:
    """One path component: separators and NUL replaced, ``.``/``..`` impossible, collisions hashed."""
    raw = str(value or "").strip()
    clean = _UNSAFE_NAME.sub("_", raw).strip("._")[:96]
    if not clean:
        return fallback
    if clean == raw:
        return clean
    digest = hashlib.sha256(raw.encode("utf-8", "surrogatepass")).hexdigest()[:12]
    return f"{clean}_{digest}"


def path_for(agent: str, session_id: str, *, home: Optional[Path] = None) -> Path:
    """The file one agent's run history for one session lands on (no file is created)."""
    return sessions_dir(home) / _stem(agent, "agent") / f"{_stem(session_id, 'session')}.jsonl"


def _lock_for(path: Path) -> threading.Lock:
    key = str(path)
    with _LOCKS_GUARD:
        return _LOCKS.setdefault(key, threading.Lock())


def _iso(ts: Optional[float] = None) -> str:
    return datetime.fromtimestamp(float(ts or time.time()), tz=UTC).isoformat()


def _text(content: Any) -> str:
    """Host message content -> text. A list of parts is joined, never flattened."""
    if isinstance(content, str):
        return content
    if content is None:
        return ""
    if isinstance(content, (list, tuple)):
        parts: list[str] = []
        for item in content:
            if isinstance(item, Mapping):
                part = item.get("text")
                parts.append(str(part) if part is not None
                             else json.dumps(item, ensure_ascii=False))
            else:
                parts.append(str(item))
        return "\n".join(parts)
    return str(content)


def _tool_calls(value: Any) -> list[dict[str, str]]:
    """Normalise both tool-call shapes (``{id,name,args}`` / ``{id,function:{...}}``)."""
    calls: list[dict[str, str]] = []
    for item in value or ():
        if not isinstance(item, Mapping):
            continue
        function = item.get("function") if isinstance(item.get("function"), Mapping) else {}
        args = item.get("args", function.get("arguments", ""))
        calls.append({"id": str(item.get("id") or ""),
                      "name": str(item.get("name") or function.get("name") or ""),
                      "args": args if isinstance(args, str)
                      else json.dumps(args, ensure_ascii=False)})
    return calls


def fingerprint(role: str, text: str, *, tool_call_id: str = "", name: str = "") -> str:
    """Content identity: what the dedupe step compares between a history prefix and the file."""
    digest = hashlib.sha256()
    for part in (role, tool_call_id, name, text):
        digest.update(str(part or "").encode("utf-8", "replace"))
        digest.update(b"\x00")
    return "v1:" + digest.hexdigest()[:32]


def _message_rows(message: Mapping[str, Any], run_id: str) -> list[dict[str, Any]]:
    """One host message -> the row(s) it becomes. Non-conversation roles are dropped."""
    role = str(message.get("role") or "")
    if role not in {"user", "assistant", "tool"}:
        return []
    text = _text(message.get("content"))
    tool_call_id = str(message.get("tool_call_id") or "")
    name = str(message.get("name") or "")
    row: dict[str, Any] = {"kind": "msg", "role": role, "text": text, "run_id": run_id,
                           "fp": fingerprint(role, text, tool_call_id=tool_call_id, name=name)}
    if role == "tool":
        row.update({"tool_call_id": tool_call_id, "name": name, "result": text, "text": ""})
        return [row]
    if role != "assistant":
        return [row]
    calls = _tool_calls(message.get("tool_calls"))
    if calls:
        row["tool_calls"] = calls
    if message.get("model"):
        row["model"] = str(message.get("model") or "")
    rows = [row]
    reasoning = message.get("reasoning_content") or message.get("reasoning")
    if str(reasoning or "").strip():
        body = str(reasoning)
        rows.append({"kind": "msg", "role": "reasoning", "text": body, "run_id": run_id,
                     "fp": fingerprint("reasoning", body)})
    return rows


def _tail_rows(path: Path, count: int = TAIL_SCAN_LINES) -> list[dict[str, Any]]:
    """The newest *count* parseable rows, without loading a long file into memory."""
    try:
        size = path.stat().st_size
    except OSError:
        return []
    if size <= 0:
        return []
    block = max(64 * 1024, size // 4)
    data = b""
    offset = size
    try:
        with open(path, "rb") as handle:
            while offset > 0 and data.count(b"\n") <= count:
                step = min(block, offset)
                offset -= step
                handle.seek(offset)
                data = handle.read(step) + data
    except OSError:
        return []
    rows: list[dict[str, Any]] = []
    for line in reversed(data.splitlines()):
        if not line.strip():
            continue
        try:
            row = json.loads(line.decode("utf-8", "replace"))
        except ValueError:
            continue
        if isinstance(row, dict):
            rows.append(row)
        if len(rows) >= count:
            break
    return rows


def _read_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    try:
        with open(path, "r", encoding="utf-8-sig") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except ValueError:
                    continue
                if isinstance(row, dict):
                    rows.append(row)
    except OSError:
        logger.debug("run transcript: cannot read %s", path, exc_info=True)
    return rows


def _system_row(system_message: str, run_id: str) -> Optional[dict[str, Any]]:
    """The run's system prompt as one row per distinct text, or ``None`` for no prompt."""
    text = str(system_message or "")
    if not text.strip():
        return None
    digest = hashlib.sha256(text.encode("utf-8", "replace")).hexdigest()[:32]
    return {"kind": "msg", "role": "system", "text": text, "run_id": run_id,
            "hash": digest, "fp": fingerprint("system", text)}


def _known_systems(rows: Iterable[Mapping[str, Any]]) -> set[str]:
    return {str(row.get("hash") or "") for row in rows
            if str(row.get("kind") or "") == "msg" and str(row.get("role") or "") == "system"}


def _append(path: Path, rows: Iterable[Mapping[str, Any]]) -> int:
    """Append rows as JSONL; the caller holds the file lock."""
    payload = list(rows)
    if not payload:
        return 0
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [json.dumps(dict(row), ensure_ascii=False, default=str) + "\n" for row in payload]
    with open(path, "a", encoding="utf-8", newline="\n") as handle:
        handle.writelines(lines)
        handle.flush()
    return len(payload)


def record(agent: str, session_id: str, messages: Iterable[Mapping[str, Any]], *,
           run_id: str = "", model: str = "", provider: str = "",
           system_message: str = "", usage: Optional[Mapping[str, Any]] = None,
           home: Optional[Path] = None) -> dict[str, Any]:
    """Append one run to its agent stream, dropping a history prefix the file already holds.

    A continuing run is handed the file's own earlier messages as ``history``; those come back in
    ``messages``. Matching fingerprints are removed once (multiset), so the file grows by what
    happened this run, not by the length of the prompt. Returns ``{"path", "written", "appended"}``.
    Raises ``OSError`` when the write itself fails: the caller decides whether that is fatal.
    """
    path = path_for(agent, session_id, home=home)
    run_id = str(run_id or f"run-{int(time.time() * 1000)}")
    incoming: list[dict[str, Any]] = []
    for message in messages or ():
        if isinstance(message, Mapping):
            incoming.extend(_message_rows(message, run_id))
    system = _system_row(system_message, run_id)
    with _lock_for(path):
        tail = _tail_rows(path)
        seen: dict[str, int] = {}
        for row in tail:
            if str(row.get("kind") or "") == "msg" and row.get("fp"):
                key = str(row["fp"])
                seen[key] = seen.get(key, 0) + 1
        kept: list[dict[str, Any]] = []
        for row in incoming:
            fp = str(row.get("fp") or "")
            if seen.get(fp, 0) > 0:
                seen[fp] -= 1
                continue
            kept.append(row)
        new_system = system is not None and system["hash"] not in _known_systems(tail)
        if not kept and not new_system:
            return {"path": str(path), "written": False, "appended": 0}
        header = {"kind": "run", "run_id": run_id, "agent": str(agent or ""),
                  "session_id": str(session_id or ""), "ts": _iso(),
                  "model": str(model or ""), "provider": str(provider or ""),
                  "usage": dict(usage or {}), "messages": len(kept)}
        payload = [header]
        if new_system and system is not None:
            payload.append(system)
        payload.extend(kept)
        appended = _append(path, payload)
    return {"path": str(path), "written": True, "appended": appended - 1 - int(new_system)}


def read_events(agent: str, session_id: str, *, home: Optional[Path] = None) -> list[dict[str, Any]]:
    """Every parsed row of one agent stream; an unreadable or missing file is an empty list."""
    return _read_rows(path_for(agent, session_id, home=home))


def read_history(agent: str, session_id: str, *, home: Optional[Path] = None) -> list[dict[str, Any]]:
    """One agent stream as host ``conversation_history`` messages (system prompt excluded)."""
    messages: list[dict[str, Any]] = []
    for row in read_events(agent, session_id, home=home):
        if str(row.get("kind") or "") != "msg":
            continue
        role = str(row.get("role") or "")
        if role == "system":
            continue  # the host owns the system prompt; it is not a conversation turn
        if role in {"user", "assistant"}:
            message: dict[str, Any] = {"role": role, "content": str(row.get("text") or "")}
            calls = row.get("tool_calls")
            if role == "assistant" and isinstance(calls, list) and calls:
                message["tool_calls"] = [
                    {"id": str(call.get("id") or ""), "type": "function",
                     "function": {"name": str(call.get("name") or ""),
                                  "arguments": str(call.get("args") or "")}}
                    for call in calls if isinstance(call, Mapping)]
            if row.get("model"):
                message["model"] = str(row.get("model") or "")
            messages.append(message)
        elif role == "tool":
            messages.append({"role": "tool", "tool_call_id": str(row.get("tool_call_id") or ""),
                             "name": str(row.get("name") or ""),
                             "content": str(row.get("result") or "")})
    return messages


__all__ = ["fingerprint", "path_for", "read_events", "read_history", "record", "sessions_dir"]
