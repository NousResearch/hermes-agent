"""Import sessions from foreign coding agents (Claude Code, Codex CLI).

``hermes sessions import`` (and ``--resume @claude`` / ``--resume @codex``)
let a user pull a conversation they started in another agent CLI into
Hermes and continue it here.

Sources (read-only — foreign files are never modified):

* **Claude Code** stores one JSONL file per session under
  ``~/.claude/projects/<encoded-cwd>/<uuid>.jsonl``.  Each line is a JSON
  object; ``type: "user"`` / ``type: "assistant"`` lines carry an
  Anthropic-format ``message`` payload whose ``content`` is either a string
  or a list of blocks (``text``, ``tool_use``, ``tool_result``, ...).
  ``type: "summary"`` lines carry a human title for the thread.

* **Codex CLI** stores rollout JSONL under
  ``~/.codex/sessions/YYYY/MM/DD/rollout-*.jsonl``.  The first line is a
  ``session_meta`` record (cwd, session id); conversation turns are
  ``response_item`` records whose payload is ``{"type": "message",
  "role": user|assistant|developer, "content": [{"type": "input_text"|
  "output_text", "text": ...}]}`` plus ``custom_tool_call`` /
  ``function_call`` payloads for tool activity.  (Schema verified against
  real rollout files, Codex CLI 0.147.)

Conversion contract — imported history must satisfy the provider
role-alternation invariant Hermes enforces everywhere else:

* only plain ``user`` / ``assistant`` text messages are produced (tool
  calls become short bracketed summaries inside the assistant text; we
  never fabricate ``tool_calls`` structures);
* consecutive same-role turns are merged rather than stubbed;
* system/developer payloads are never imported.
"""

from __future__ import annotations

import json
import re
import uuid
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# User-message texts that are really injected context wrappers, not typed
# input. Matched against the stripped start of the text.
_WRAPPER_TAG_RE = re.compile(
    r"^<(?:user_instructions|environment_context|recommended_plugins|"
    r"skills_instructions|permissions[_-]instructions|turn_context|"
    r"command-name|command-message|local-command-stdout|system-reminder)\b",
    re.IGNORECASE,
)

_TITLE_MAX = 60


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
        name = {"claude": "Claude Code", "codex": "Codex CLI"}.get(
            self.source, self.source
        )
        title = (self.title_guess or "").strip() or self.path.stem
        return f"[{name}] {title[:_TITLE_MAX]}"


def _read_json_lines(path: Path):
    """Yield parsed JSON objects, silently skipping unparseable lines."""
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except (json.JSONDecodeError, ValueError):
                    continue
                if isinstance(obj, dict):
                    yield obj
    except OSError:
        return


def _flatten_blocks(content: Any, *, source: str) -> str:
    """Flatten a message ``content`` (string or block list) to plain text.

    Tool activity becomes a short bracketed summary; unknown block types
    are skipped rather than guessed at.
    """
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return ""
    parts: List[str] = []
    for block in content:
        if not isinstance(block, dict):
            if isinstance(block, str):
                parts.append(block)
            continue
        btype = block.get("type")
        if btype in ("text", "input_text", "output_text"):
            text = block.get("text")
            if isinstance(text, str) and text:
                parts.append(text)
        elif btype == "tool_use":  # Claude Code assistant block
            name = block.get("name") or "tool"
            parts.append(f"[ran tool: {name}]")
        elif btype == "tool_result":
            # Tool output echoed into a user message — not typed input.
            continue
        elif btype in ("thinking", "redacted_thinking", "reasoning"):
            continue
        elif btype == "image":
            parts.append("[image]")
    return "\n\n".join(p for p in (s.strip() for s in parts) if p)


def _is_wrapper_text(text: str) -> bool:
    return bool(_WRAPPER_TAG_RE.match(text.lstrip()))


def _merge_turns(raw_turns: List[Tuple[str, str]]) -> List[Dict[str, str]]:
    """Merge consecutive same-role turns; guarantee strict alternation.

    A leading assistant turn (session began before the log window) gets a
    minimal user stub so the first message is always ``user``; this is the
    only place a stub is ever inserted.
    """
    merged: List[Dict[str, str]] = []
    for role, text in raw_turns:
        text = text.strip()
        if not text:
            continue
        if merged and merged[-1]["role"] == role:
            merged[-1]["content"] += "\n\n" + text
        else:
            merged.append({"role": role, "content": text})
    if merged and merged[0]["role"] == "assistant":
        merged.insert(
            0,
            {
                "role": "user",
                "content": "(imported conversation begins with an assistant reply)",
            },
        )
    return merged


# ── Claude Code ──────────────────────────────────────────────────────────


def parse_claude_session(path: Path) -> Dict[str, Any]:
    """Parse one Claude Code session JSONL into normalized turns + meta."""
    turns: List[Tuple[str, str]] = []
    cwd: Optional[str] = None
    summary: Optional[str] = None
    session_id: Optional[str] = None
    for obj in _read_json_lines(path):
        otype = obj.get("type")
        if otype == "summary":
            s = obj.get("summary")
            if isinstance(s, str) and s.strip():
                summary = s.strip()
            continue
        if otype not in ("user", "assistant"):
            continue
        if obj.get("isSidechain") or obj.get("isMeta"):
            continue
        if cwd is None and isinstance(obj.get("cwd"), str):
            cwd = obj["cwd"]
        if session_id is None and isinstance(obj.get("sessionId"), str):
            session_id = obj["sessionId"]
        message = obj.get("message")
        if not isinstance(message, dict):
            continue
        role = message.get("role")
        if role not in ("user", "assistant"):
            continue
        text = _flatten_blocks(message.get("content"), source="claude")
        if not text or (role == "user" and _is_wrapper_text(text)):
            continue
        turns.append((role, text))
    return {
        "turns": _merge_turns(turns),
        "cwd": cwd,
        "title_guess": summary or _first_user_line(turns),
        "session_id": session_id,
    }


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


# ── Codex CLI ────────────────────────────────────────────────────────────


def parse_codex_session(path: Path) -> Dict[str, Any]:
    """Parse one Codex CLI rollout JSONL into normalized turns + meta."""
    turns: List[Tuple[str, str]] = []
    cwd: Optional[str] = None
    session_id: Optional[str] = None
    for obj in _read_json_lines(path):
        otype = obj.get("type")
        payload = obj.get("payload")
        if not isinstance(payload, dict):
            continue
        if otype == "session_meta":
            if isinstance(payload.get("cwd"), str):
                cwd = payload["cwd"]
            sid = payload.get("session_id") or payload.get("id")
            if isinstance(sid, str):
                session_id = sid
            continue
        if otype != "response_item":
            continue
        ptype = payload.get("type")
        if ptype == "message":
            role = payload.get("role")
            if role not in ("user", "assistant"):
                continue  # developer/system payloads never imported
            text = _flatten_blocks(payload.get("content"), source="codex")
            if not text or (role == "user" and _is_wrapper_text(text)):
                continue
            turns.append((role, text))
        elif ptype in ("custom_tool_call", "function_call", "local_shell_call"):
            name = payload.get("name") or payload.get("tool") or "tool"
            # Attach as assistant activity; merged into neighbors later.
            turns.append(("assistant", f"[ran tool: {name}]"))
        # tool outputs / reasoning / web_search etc. are skipped
    return {
        "turns": _merge_turns(turns),
        "cwd": cwd,
        "title_guess": _first_user_line(turns),
        "session_id": session_id,
    }


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


def _first_user_line(turns: List[Tuple[str, str]]) -> Optional[str]:
    for role, text in turns:
        if role == "user":
            line = text.strip().splitlines()[0].strip()
            if line:
                return line[:_TITLE_MAX * 2]
    return None


# ── Import ───────────────────────────────────────────────────────────────

_SOURCE_LABELS = {"claude": "Claude Code", "codex": "Codex CLI"}
_SOURCE_DB_NAMES = {"claude": "claude-code", "codex": "codex-cli"}


def import_foreign_session(source: str, path, db=None) -> str:
    """Import one foreign session into the Hermes SessionDB.

    Returns the new Hermes session id.  The foreign file is only read.
    Raises ``ValueError`` on unknown source or a session with no usable
    conversation turns.
    """
    source = (source or "").strip().lower().lstrip("@")
    if source not in _SOURCE_LABELS:
        raise ValueError(f"Unknown foreign session source: {source!r}")
    path = Path(path).expanduser()
    if not path.is_file():
        raise ValueError(f"Session file not found: {path}")

    parsed = (
        parse_claude_session(path)
        if source == "claude"
        else parse_codex_session(path)
    )
    turns = parsed["turns"]
    if not turns:
        raise ValueError(
            f"No user/assistant conversation turns found in {path}"
        )

    label = _SOURCE_LABELS[source]
    first_user = _first_user_line(
        [(t["role"], t["content"]) for t in turns]
    ) or path.stem
    if len(first_user) > _TITLE_MAX:
        first_user = first_user[: _TITLE_MAX - 1] + "…"
    title = f"Imported from {label}: {first_user}"

    owns_db = db is None
    if owns_db:
        from hermes_state import SessionDB

        db = SessionDB()
    try:
        session_id = (
            f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"
        )
        origin = {
            "imported_from": {
                "tool": _SOURCE_DB_NAMES[source],
                "path": str(path),
                "foreign_session_id": parsed.get("session_id"),
            }
        }
        db.create_session(
            session_id,
            source=_SOURCE_DB_NAMES[source],
            cwd=parsed.get("cwd"),
            origin_json=json.dumps(origin),
        )
        for turn in turns:
            db.append_message(session_id, turn["role"], turn["content"])
        try:
            db.set_session_title(session_id, title)
        except Exception:
            pass  # title is cosmetic; the import itself succeeded
        return session_id
    finally:
        if owns_db:
            try:
                db.close()
            except Exception:
                pass


# ── Picker / CLI helpers ─────────────────────────────────────────────────


def gather_foreign_sessions(
    source: Optional[str] = None,
    *,
    claude_root: Optional[Path] = None,
    codex_root: Optional[Path] = None,
    limit: int = 25,
) -> List[ForeignSession]:
    """List foreign sessions across sources, newest first."""
    sessions: List[ForeignSession] = []
    if source in (None, "claude"):
        sessions.extend(list_claude_sessions(claude_root))
    if source in (None, "codex"):
        sessions.extend(list_codex_sessions(codex_root))
    sessions.sort(key=lambda s: s.mtime, reverse=True)
    return sessions[:limit] if limit else sessions


def pick_foreign_session(
    source: Optional[str] = None, *, limit: int = 25
) -> Optional[ForeignSession]:
    """Interactive numbered picker. Returns None when nothing was chosen."""
    import os
    import sys

    sessions = gather_foreign_sessions(source, limit=limit)
    if not sessions:
        where = _SOURCE_LABELS.get(source or "", "Claude Code or Codex CLI")
        print(f"No {where} sessions found on this machine.")
        return None
    print("Foreign sessions (newest first):")
    for i, s in enumerate(sessions, 1):
        when = datetime.fromtimestamp(s.mtime).strftime("%Y-%m-%d %H:%M")
        ws = ""
        if s.cwd:
            ws = f"  ({os.path.basename(s.cwd.rstrip('/')) or s.cwd})"
        print(f"  {i:>2}. {when}  {s.label}{ws}  [{s.turn_count} turns]")
    if not sys.stdin.isatty():
        print(
            "Non-interactive terminal — pass the file path directly:\n"
            "  hermes sessions import --from claude|codex <path>"
        )
        return None
    try:
        raw = input(f"Import which session? [1-{len(sessions)}, empty to cancel] ")
    except (EOFError, KeyboardInterrupt):
        return None
    raw = raw.strip()
    if not raw:
        return None
    try:
        idx = int(raw)
    except ValueError:
        print(f"Not a number: {raw}")
        return None
    if not 1 <= idx <= len(sessions):
        print(f"Out of range: {idx}")
        return None
    return sessions[idx - 1]


def run_sessions_import(args, db=None) -> Optional[str]:
    """`hermes sessions import` entry point. Returns new session id or None."""
    source = getattr(args, "from_source", None)
    path = getattr(args, "path", None)

    if path:
        if not source:
            # Guess from the path shape.
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
        picked = pick_foreign_session(source)
        if picked is None:
            return None
        source, chosen_path = picked.source, picked.path

    try:
        session_id = import_foreign_session(source, chosen_path, db=db)
    except ValueError as e:
        print(f"Error: {e}")
        return None
    label = _SOURCE_LABELS.get(source, source)
    print(f"✓ Imported {label} session as {session_id}")
    print(f"  Continue it with:  hermes --resume {session_id}")
    return session_id
