"""
Cross-session workspace awareness.

Writes a lightweight presence file after significant tool calls so that
other hermes sessions sharing the same working directory can discover each
other.  At turn start each session reads the directory and, when other live
sessions exist, builds a compact workspace context block that is injected
into the user message.

Presence is determined by *process liveness* (``os.kill(pid, 0)``) — no
time-based staleness heuristics.  Dead sessions are detected instantly and
their stale files are removed on the next read.

Design
------
* **Per-session files** — ``activity/sessions/<session_id>.json`` under
  ``$HERMES_HOME``.  Each session writes only its own file → no locking
  needed.  Atomic writes via temp-file + ``os.replace()``.
* **cwd filter** — only sessions sharing the same working directory are
  surfaced.  Sessions in different repos never see each other.
* **Write-on-action** — presence is updated after tool calls that have
  side-effects (writes, terminal, etc.).  Read-only tools are skipped.
* **Fail-open** — any error during update or read is silently swallowed;
  workspace awareness is a best-effort convenience, not a critical path.
"""

from __future__ import annotations

import json
import logging
import os
import signal
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from hermes_constants import get_hermes_home

logger = logging.getLogger(__name__)

# ── configuration ────────────────────────────────────────────────────────

# Tools that only read — updating presence for them is noise.
_SKIP_TOOLS: frozenset[str] = frozenset({
    "read_file",
    "search_files",
    "web_search",
    "web_extract",
    "session_search",
    "browser_snapshot",
    "vision_analyze",
    "video_analyze",
    "browser_console",
    "browser_get_images",
    "clarify",
    "memory",
    "send_message",
    "todo",
    "skills_list",
    "skill_view",
})

# Maximum actions to retain per session (ring buffer).
_MAX_ACTIONS = 2


# ── helpers ──────────────────────────────────────────────────────────────

def _sessions_dir() -> Path:
    """Return ``$HERMES_HOME/activity/sessions/``, creating if needed."""
    d = get_hermes_home() / "activity" / "sessions"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _detect_profile() -> str:
    """Best-effort profile detection.

    Checks (in order):
    1. ``HERMES_KANBAN_PROFILE`` — set by the kanban dispatcher
    2. ``HERMES_PROFILE`` — set by the gateway / CLI
    3. ``"default"`` — fallback
    """
    return os.environ.get(
        "HERMES_KANBAN_PROFILE",
        os.environ.get("HERMES_PROFILE", "default"),
    )


def _extract_target(tool_name: str, args: dict) -> str:
    """Extract a human-readable target string from tool arguments."""
    if tool_name in ("write_file", "read_file"):
        return str(args.get("path", ""))
    if tool_name == "patch":
        return str(args.get("path", args.get("file_path", "")))
    if tool_name == "terminal":
        cmd = str(args.get("command", ""))
        return cmd[:80]  # truncate long commands
    if tool_name == "web_search":
        return str(args.get("query", ""))[:80]
    if tool_name == "browser_navigate":
        return str(args.get("url", ""))
    if tool_name in ("browser_click", "browser_type"):
        return str(args.get("ref", ""))
    if tool_name == "kanban_create":
        return str(args.get("title", ""))
    return ""


def _pid_alive(pid: int) -> bool:
    """Return True if *pid* refers to a running process."""
    try:
        os.kill(pid, 0)
        return True
    except (OSError, ProcessLookupError):
        return False


# ── public API ───────────────────────────────────────────────────────────

def update_presence(
    session_id: str,
    tool_name: str,
    args: dict,
    cwd: Optional[str] = None,
    task_id: Optional[str] = None,
) -> None:
    """Update this session's presence file after a tool call.

    Called from ``model_tools.handle_function_call`` after every tool
    dispatch.  No-op for read-only tools and when the feature is disabled.

    Parameters
    ----------
    session_id:
        The hermes session ID (``agent.session_id``).
    tool_name:
        Name of the dispatched tool (e.g. ``"file_write"``).
    args:
        Raw tool arguments dict.
    cwd:
        Working directory at dispatch time.  Defaults to ``os.getcwd()``.
    task_id:
        Kanban task ID, if this session is a spawned worker.
    """
    # ── guard: disabled? ─────────────────────────────────────────────
    if os.environ.get("HERMES_WORKSPACE_AWARENESS") == "0":
        return

    # ── guard: read-only tool? ───────────────────────────────────────
    if tool_name in _SKIP_TOOLS:
        return

    if cwd is None:
        cwd = os.getcwd()

    target = _extract_target(tool_name, args)
    profile = _detect_profile()
    now_iso = datetime.now(timezone.utc).isoformat()

    # Load existing file to preserve action history.
    path = _sessions_dir() / f"{session_id}.json"
    existing: dict[str, Any] = {}
    try:
        if path.exists():
            existing = json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        pass  # corrupt or unreadable — start fresh

    # Build / update the record.
    last_actions: list[dict] = existing.get("last_actions", [])
    last_actions.append({"tool": tool_name, "target": target, "ts": now_iso})
    if len(last_actions) > _MAX_ACTIONS:
        last_actions = last_actions[-_MAX_ACTIONS:]

    record: dict[str, Any] = {
        "session_id": session_id,
        "pid": os.getpid(),
        "profile": profile,
        "cwd": cwd,
        "task_id": task_id or "",
        "ts": now_iso,
        "last_actions": last_actions,
    }

    # Atomic write: temp file + rename.
    tmp = path.with_suffix(".tmp")
    try:
        tmp.write_text(json.dumps(record, ensure_ascii=False))
        os.replace(tmp, path)
    except OSError:
        pass  # best-effort


def _read_all_sessions() -> list[dict]:
    """Return parsed presence records for all sessions (including stale ones)."""
    sessions: list[dict] = []
    sd = _sessions_dir()
    try:
        for entry in sd.iterdir():
            if not entry.is_file() or entry.suffix != ".json":
                continue
            try:
                data = json.loads(entry.read_text())
                if isinstance(data, dict) and data.get("session_id"):
                    sessions.append(data)
            except (json.JSONDecodeError, OSError):
                # Stale or corrupt — remove it.
                try:
                    entry.unlink(missing_ok=True)
                except OSError:
                    pass
    except OSError:
        pass
    return sessions


def get_coworkers(my_session_id: str, my_cwd: str) -> list[dict]:
    """Return active other sessions sharing *my_cwd*.

    Each returned dict has keys: ``session_id``, ``profile``, ``ts``,
    ``last_actions`` (list of ``{tool, target, ts}``), ``seconds_ago``.
    """
    all_sessions = _read_all_sessions()
    now = time.time()
    coworkers: list[dict] = []

    for s in all_sessions:
        sid = s.get("session_id", "")
        if sid == my_session_id:
            continue  # skip self

        pid = s.get("pid", 0)
        if not _pid_alive(pid):
            # Dead session — clean up stale file.
            _cleanup_stale(sid)
            continue

        s_cwd = s.get("cwd", "")
        if s_cwd != my_cwd:
            continue  # different directory — not relevant

        # Calculate seconds since last update.
        try:
            ts = datetime.fromisoformat(s.get("ts", ""))
            seconds_ago = int((now - ts.timestamp()))
        except (ValueError, OSError):
            seconds_ago = 999

        coworkers.append({
            "session_id": sid,
            "profile": s.get("profile", "?"),
            "ts": s.get("ts", ""),
            "last_actions": s.get("last_actions", []),
            "seconds_ago": max(seconds_ago, 0),
        })

    # Sort: most recently active first.
    coworkers.sort(key=lambda c: c["seconds_ago"])
    return coworkers


def _cleanup_stale(session_id: str) -> None:
    """Remove a stale presence file for a dead session."""
    path = _sessions_dir() / f"{session_id}.json"
    try:
        path.unlink(missing_ok=True)
    except OSError:
        pass


def build_context_block(session_id: str, cwd: Optional[str] = None) -> Optional[str]:
    """Build the workspace awareness context block for injection.

    Returns ``None`` when no other sessions are active in the same cwd
    → zero prompt tokens overhead.
    """
    if os.environ.get("HERMES_WORKSPACE_AWARENESS") == "0":
        return None

    if cwd is None:
        cwd = os.getcwd()

    coworkers = get_coworkers(session_id, cwd)
    if not coworkers:
        return None

    # Map cwd to ~/ if under HOME for readability.
    home = os.path.expanduser("~")
    cwd_display = cwd.replace(home, "~", 1) if cwd.startswith(home) else cwd

    profile = _detect_profile()
    lines = [
        "══ WORKSPACE ══",
        f"  you: session {session_id[:8]} ({profile}) in {cwd_display}",
    ]

    count = len(coworkers)
    s_label = "session" if count == 1 else "sessions"
    lines.append(f"  {count} other hermes {s_label} active here:")

    for c in coworkers:
        sid_short = c["session_id"][:8]
        prof = c["profile"]
        ago = c["seconds_ago"]
        ago_str = f"{ago}s ago" if ago < 60 else f"{ago // 60}m ago"
        lines.append(f"    {sid_short} ({prof}) — last seen {ago_str}")

        for action in c["last_actions"]:
            tool = action.get("tool", "?")
            target = action.get("target", "")
            if target:
                lines.append(f"      {tool} → {target}")
            else:
                lines.append(f"      {tool}")

    # Collect files modified by other sessions.
    modified: set[str] = set()
    for c in coworkers:
        for action in c.get("last_actions", []):
            tname = action.get("tool", "")
            target = action.get("target", "")
            if tname in ("write_file", "patch") and target:
                # Extract just the filename for brevity.
                modified.add(Path(target).name)
    if modified:
        files_str = ", ".join(sorted(modified))
        lines.append(f"  files modified by other sessions: {files_str}")

    return "\n".join(lines)
