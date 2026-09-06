"""Import sessions from foreign coding agents (Claude Code, Codex CLI). Foreign files are only ever read;
imported history must satisfy the provider role-alternation invariant (see ``_merge_turns``)."""

from __future__ import annotations

import contextlib
import json
import os
import re
import sys
import uuid
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# User-message texts that are really injected context wrappers, not typed input.
_WRAPPER_TAG_RE = re.compile(
    r"^<(?:user_instructions|environment_context|recommended_plugins|"
    r"skills_instructions|permissions[_-]instructions|turn_context|"
    r"command-name|command-message|local-command-stdout|system-reminder)\b", re.IGNORECASE)

_TITLE_MAX = 60
_SOURCE_LABELS = {"claude": "Claude Code", "codex": "Codex CLI"}
_SOURCE_DB_NAMES = {"claude": "claude-code", "codex": "codex-cli"}


@dataclass
class ForeignSession:
    """A discoverable session in another tool's on-disk store."""

    source: str  # "claude" | "codex"
    path: Path
    mtime: float
    cwd: Optional[str] = None
    title_guess: Optional[str] = None
    turn_count: int = 0
    session_id: Optional[str] = None  # the foreign tool's own id

    @property
    def label(self) -> str:
        title = (self.title_guess or "").strip() or self.path.stem
        return f"[{_SOURCE_LABELS.get(self.source, self.source)}] {title[:_TITLE_MAX]}"


def _read_json_lines(path: Path):
    """Yield parsed JSON objects, silently skipping unparseable lines."""
    with contextlib.suppress(OSError), open(path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            try:
                obj = json.loads(line)
            except ValueError:
                continue
            if isinstance(obj, dict):
                yield obj


def _block_text(block: Any) -> str:
    """Plain text of one content block; tool_result, thinking/reasoning and unknown types yield ''."""
    if isinstance(block, str):
        return block
    if not isinstance(block, dict):
        return ""
    btype = block.get("type")
    if btype in ("text", "input_text", "output_text"):
        return text if isinstance(text := block.get("text"), str) else ""
    if btype == "tool_use":  # Claude Code assistant block
        return f"[ran tool: {block.get('name') or 'tool'}]"
    return "[image]" if btype == "image" else ""


def _flatten_blocks(content: Any) -> str:
    """Flatten a message ``content`` (string or block list) to plain text."""
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return ""
    return "\n\n".join(p for p in (_block_text(b).strip() for b in content) if p)


def _merge_turns(raw_turns: List[Tuple[str, str]]) -> List[Dict[str, str]]:
    """Merge consecutive same-role turns; guarantee strict alternation.

    A leading assistant turn (session began before the log window) gets a minimal user stub so the
    first message is always ``user``; this is the only place a stub is ever inserted.
    """
    merged: List[Dict[str, str]] = []
    for role, text in raw_turns:
        if not (text := text.strip()):
            continue
        if merged and merged[-1]["role"] == role:
            merged[-1]["content"] += "\n\n" + text
        else:
            merged.append({"role": role, "content": text})
    if merged and merged[0]["role"] == "assistant":
        merged.insert(0, {"role": "user", "content": "(imported conversation begins with an assistant reply)"})
    return merged


def _message_turn(message: Any) -> Optional[Tuple[str, str]]:
    """Normalize one message dict into a ``(role, text)`` turn, or None when it is not importable."""
    role = message.get("role") if isinstance(message, dict) else None
    if role not in ("user", "assistant"):
        return None
    text = _flatten_blocks(message.get("content"))
    return None if not text or (role == "user" and _WRAPPER_TAG_RE.match(text.lstrip())) else (role, text)


def _first_user_line(turns: List[Tuple[str, str]]) -> Optional[str]:
    for role, text in turns:
        if role == "user" and (line := text.strip().splitlines()[0].strip()):
            return line[:_TITLE_MAX * 2]
    return None


def _parsed(turns: List[Tuple[str, str]], cwd: Optional[str], session_id: Optional[str],
            title: Optional[str] = None) -> Dict[str, Any]:
    return {"turns": _merge_turns(turns), "cwd": cwd, "title_guess": title or _first_user_line(turns),
            "session_id": session_id}


def parse_claude_session(path: Path) -> Dict[str, Any]:
    """Parse one Claude Code session JSONL into normalized turns + meta."""
    turns: List[Tuple[str, str]] = []
    cwd = summary = session_id = None
    for obj in _read_json_lines(path):
        otype = obj.get("type")
        if otype == "summary":
            if isinstance(s := obj.get("summary"), str) and s.strip():
                summary = s.strip()
        elif otype in ("user", "assistant") and not (obj.get("isSidechain") or obj.get("isMeta")):
            if cwd is None and isinstance(obj.get("cwd"), str):
                cwd = obj["cwd"]
            if session_id is None and isinstance(obj.get("sessionId"), str):
                session_id = obj["sessionId"]
            if turn := _message_turn(obj.get("message")):
                turns.append(turn)
    return _parsed(turns, cwd, session_id, summary)


def parse_codex_session(path: Path) -> Dict[str, Any]:
    """Parse one Codex CLI rollout JSONL into normalized turns + meta."""
    turns: List[Tuple[str, str]] = []
    cwd = session_id = None
    for obj in _read_json_lines(path):
        otype, payload = obj.get("type"), obj.get("payload")
        if not isinstance(payload, dict):
            continue
        if otype == "session_meta":
            if isinstance(payload.get("cwd"), str):
                cwd = payload["cwd"]
            if isinstance(sid := payload.get("session_id") or payload.get("id"), str):
                session_id = sid
        elif otype == "response_item":
            ptype = payload.get("type")
            if ptype == "message" and (turn := _message_turn(payload)):  # developer/system payloads skipped
                turns.append(turn)
            elif ptype in ("custom_tool_call", "function_call", "local_shell_call"):
                # Assistant activity; merged into neighbours later. Tool outputs / reasoning skipped.
                name = payload.get("name") or payload.get("tool") or "tool"
                turns.append(("assistant", f"[ran tool: {name}]"))
    return _parsed(turns, cwd, session_id)


# source -> (default root under ~, glob pattern, recursive, parser)
_SOURCES = {
    "claude": ((".claude", "projects"), "*/*.jsonl", False, parse_claude_session),
    "codex": ((".codex", "sessions"), "rollout-*.jsonl", True, parse_codex_session),
}


def _list_sessions(source: str, root: Optional[Path]) -> List[ForeignSession]:
    default_root, pattern, recursive, parse = _SOURCES[source]
    root = Path(root) if root else Path.home().joinpath(*default_root)
    results: List[ForeignSession] = []
    for jsonl in sorted((root.rglob(pattern) if recursive else root.glob(pattern)) if root.is_dir() else ()):
        try:
            mtime = jsonl.stat().st_mtime
        except OSError:
            continue
        parsed = parse(jsonl)
        if parsed["turns"]:
            results.append(ForeignSession(source, jsonl, mtime, parsed["cwd"], parsed["title_guess"],
                                          len(parsed["turns"]), parsed["session_id"]))
    results.sort(key=lambda s: s.mtime, reverse=True)
    return results


def import_foreign_session(source: str, path, db=None) -> str:
    """Import one foreign session into the Hermes SessionDB; returns the new Hermes session id.

    Raises ``ValueError`` on unknown source or a session with no usable conversation turns."""
    source = (source or "").strip().lower().lstrip("@")
    if source not in _SOURCE_LABELS:
        raise ValueError(f"Unknown foreign session source: {source!r}")
    path = Path(path).expanduser()
    if not path.is_file():
        raise ValueError(f"Session file not found: {path}")
    parsed = _SOURCES[source][3](path)
    turns = parsed["turns"]
    if not turns:
        raise ValueError(f"No user/assistant conversation turns found in {path}")
    first_user = _first_user_line([(t["role"], t["content"]) for t in turns]) or path.stem
    if len(first_user) > _TITLE_MAX:
        first_user = first_user[: _TITLE_MAX - 1] + "…"
    tool = _SOURCE_DB_NAMES[source]
    owns_db = db is None
    if owns_db:
        from hermes_state import SessionDB
        db = SessionDB()
    try:
        session_id = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"
        origin = {"imported_from": {"tool": tool, "path": str(path), "foreign_session_id": parsed.get("session_id")}}
        db.create_session(session_id, source=tool, cwd=parsed.get("cwd"), origin_json=json.dumps(origin))
        for turn in turns:
            db.append_message(session_id, turn["role"], turn["content"])
        with contextlib.suppress(Exception):  # title is cosmetic; the import itself succeeded
            db.set_session_title(session_id, f"Imported from {_SOURCE_LABELS[source]}: {first_user}")
        return session_id
    finally:
        if owns_db:
            with contextlib.suppress(Exception):
                db.close()


def gather_foreign_sessions(source: Optional[str] = None, *, claude_root: Optional[Path] = None,
                            codex_root: Optional[Path] = None, limit: int = 25) -> List[ForeignSession]:
    """List foreign sessions across sources, newest first."""
    sessions = [s for name, root in (("claude", claude_root), ("codex", codex_root)) if source in (None, name)
                for s in _list_sessions(name, root)]
    sessions.sort(key=lambda s: s.mtime, reverse=True)
    return sessions[:limit] if limit else sessions


def pick_foreign_session(source: Optional[str] = None, *, limit: int = 25) -> Optional[ForeignSession]:
    """Interactive numbered picker. Returns None when nothing was chosen."""
    sessions = gather_foreign_sessions(source, limit=limit)
    if not sessions:
        where = _SOURCE_LABELS.get(source or "", "Claude Code or Codex CLI")
        print(f"No {where} sessions found on this machine.")
        return None
    print("Foreign sessions (newest first):")
    for i, s in enumerate(sessions, 1):
        ws = f"  ({os.path.basename(s.cwd.rstrip('/')) or s.cwd})" if s.cwd else ""
        print(f"  {i:>2}. {datetime.fromtimestamp(s.mtime):%Y-%m-%d %H:%M}  {s.label}{ws}  [{s.turn_count} turns]")
    if not sys.stdin.isatty():
        print("Non-interactive terminal — pass the file path directly:\n"
              "  hermes sessions import --from claude|codex <path>")
        return None
    try:
        raw = input(f"Import which session? [1-{len(sessions)}, empty to cancel] ").strip()
        idx = int(raw) if raw else None
    except (EOFError, KeyboardInterrupt):
        return None
    except ValueError:
        print(f"Not a number: {raw}")
        return None
    if idx is not None and 1 <= idx <= len(sessions):
        return sessions[idx - 1]
    if idx is not None:
        print(f"Out of range: {idx}")
    return None


def run_sessions_import(args, db=None) -> Optional[str]:
    """`hermes sessions import` entry point. Returns new session id or None."""
    source = getattr(args, "from_source", None)
    path = getattr(args, "path", None)
    if path:
        # A missing file is reported as such, not as the misleading "cannot infer source".
        if not Path(path).exists():
            print(f"Error: file not found: {path}")
            return None
        if not source:  # guess from the path shape; a codex match wins over a claude match
            p = str(path)
            if "/.claude/" in p or p.endswith(".jsonl") and "claude" in p:
                source = "claude"
            if "/.codex/" in p or Path(p).name.startswith("rollout-"):
                source = "codex"
        if not source:
            print("Cannot infer source from path; pass --from claude|codex.")
            return None
        chosen_path = Path(path)
    else:
        if (picked := pick_foreign_session(source)) is None:
            return None
        source, chosen_path = picked.source, picked.path
    try:
        session_id = import_foreign_session(source, chosen_path, db=db)
    except ValueError as e:
        print(f"Error: {e}")
        return None
    print(f"✓ Imported {_SOURCE_LABELS.get(source, source)} session as {session_id}")
    print(f"  Continue it with:  hermes --resume {session_id}")
    return session_id


# ---- BEGIN PLUGIN-COMPAT (revert-scheduled; see COMPAT_MANIFEST.md) ----
# Names external plugins imported from this module before the Sep 2026 decomposition.
# Internal code MUST NOT use these (scripts/check_compat_pointers.py fails CI if it does).
# The whole block is removed by reverting the commit that added it.

def list_claude_sessions(root: Optional[Path] = None) -> List[ForeignSession]:
    """Discover Claude Code sessions under ``~/.claude/projects``."""
    root = Path(root) if root else Path.home() / ".claude" / "projects"
    results: List[ForeignSession] = []
    if not root.is_dir():
        return results
    for jsonl in sorted(root.glob("*/*.jsonl")):
        try:
            mtime = jsonl.stat().st_mtime
        except OSError:
            continue
        parsed = parse_claude_session(jsonl)
        if not parsed["turns"]:
            continue
        results.append(
            ForeignSession(
                source="claude",
                path=jsonl,
                mtime=mtime,
                cwd=parsed["cwd"],
                title_guess=parsed["title_guess"],
                turn_count=len(parsed["turns"]),
                session_id=parsed["session_id"],
            )
        )
    results.sort(key=lambda s: s.mtime, reverse=True)
    return results

def list_codex_sessions(root: Optional[Path] = None) -> List[ForeignSession]:
    """Discover Codex CLI rollouts under ``~/.codex/sessions``."""
    root = Path(root) if root else Path.home() / ".codex" / "sessions"
    results: List[ForeignSession] = []
    if not root.is_dir():
        return results
    for jsonl in sorted(root.rglob("rollout-*.jsonl")):
        try:
            mtime = jsonl.stat().st_mtime
        except OSError:
            continue
        parsed = parse_codex_session(jsonl)
        if not parsed["turns"]:
            continue
        results.append(
            ForeignSession(
                source="codex",
                path=jsonl,
                mtime=mtime,
                cwd=parsed["cwd"],
                title_guess=parsed["title_guess"],
                turn_count=len(parsed["turns"]),
                session_id=parsed["session_id"],
            )
        )
    results.sort(key=lambda s: s.mtime, reverse=True)
    return results
# ---- END PLUGIN-COMPAT ----
