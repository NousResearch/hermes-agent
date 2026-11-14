"""Best-effort terminal session bindings: breadcrumbs for ``hermes -c`` / ``--continue`` and
client-owned Herdr identity. Breadcrumbs require a stable terminal identity and are gated by
``session.terminal_continue`` (default true); Herdr reporting is independent of that setting."""

from __future__ import annotations

import json
import os
import re
import sys
import subprocess
import threading
import time
from pathlib import Path
from typing import Optional
from utils import atomic_json_write, unlink_files_older_than

# Multiplexer / terminal-emulator identity env vars, checked in order when no real tty path is
# available (e.g. stdin piped but stdout still a pty owned by a known terminal).
_TERMINAL_ENV_VARS = ("ZELLIJ_PANE_ID", "TMUX_PANE", "KITTY_WINDOW_ID", "WEZTERM_PANE", "TERM_SESSION_ID", "WT_SESSION")

# Breadcrumbs older than this are pruned opportunistically on each write — a pane id from a tmux
# server restarted last month means nothing today.
_STALE_AFTER_SECONDS = 30 * 24 * 60 * 60

_SANITIZE_RE = re.compile(r"[^A-Za-z0-9._-]")

_herdr_last_seq = 0
_herdr_seq_lock = threading.Lock()


def _herdr_resume_argv(session_id: str, profile: str | None) -> list[str]:
    """Resume command for a named profile; empty keeps Herdr's built-in ``hermes --resume <id>``,
    which opens the default profile and so cannot find a named profile's session. A home that is
    not a profile (``profile_name_for_home`` is None) cannot be named with ``-p``."""
    if not profile or profile == "default":
        return []
    return ["--", "hermes", "-p", profile, "--resume", session_id]


def report_agent_session(session_id: str) -> None:
    """Bind this frontend's durable identity to its Herdr pane, independently of breadcrumbs.

    Explicit gateway URLs (including loopback tunnels) may own a different session DB.
    Only automatically discovered local owners can safely bind to this host's Herdr.
    Herdr keeps the resume command only while it sees this Hermes process in the pane; a
    refused command (``resume_not_accepted``) still records the session identity.
    """
    global _herdr_last_seq
    try:
        pane = os.environ.get("HERDR_PANE_ID", "")
        if (os.environ.get("HERDR_ENV") != "1" or not pane or not session_id
                or os.environ.get("HERMES_TUI_GATEWAY_URL", "").strip()
                or os.environ.get("HERMES_TUI_SIDECAR_URL", "").strip()):
            return
        with _herdr_seq_lock:
            seq = max(time.time_ns(), _herdr_last_seq + 1)
            _herdr_last_seq = seq
        from hermes_constants import get_hermes_home, profile_name_for_home
        subprocess.run(
            [os.environ.get("HERDR_BIN_PATH") or "herdr", "pane", "report-agent-session", pane,
             "--source", "herdr:hermes", "--agent", "hermes", "--seq", str(seq),
             "--agent-session-id", session_id, "--session-start-source", "resume",
             *_herdr_resume_argv(session_id, profile_name_for_home(get_hermes_home()))],
            stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            timeout=1, check=False,
        )
    except Exception:
        pass  # Terminal identity reporting must never break chat or print binary/transport errors.


def _breadcrumbs_dir() -> Path:
    from hermes_constants import get_hermes_home
    return get_hermes_home() / "terminal-sessions"


def _sanitize(raw: str) -> str:
    """Make an id safe to use as a filename (``/dev/pts/3`` -> ``dev-pts-3``)."""
    return _SANITIZE_RE.sub("-", raw.strip().strip("/"))[:120]


def get_terminal_id() -> Optional[str]:
    """Stable identity for this terminal: the tty device path (stdin, then stdout), else the first
    present multiplexer/emulator env var; ``None`` when neither exists (callers skip breadcrumbs)."""
    for fd in (sys.stdin, sys.stdout):
        try:
            name = os.ttyname(fd.fileno())
        except Exception:
            continue
        if name:
            return f"tty-{_sanitize(name)}"
    for var in _TERMINAL_ENV_VARS:
        val = os.environ.get(var)
        if val:
            return f"{var.lower()}-{_sanitize(val)}"
    return None


def is_enabled() -> bool:
    """Config gate: ``session.terminal_continue`` (default true)."""
    try:
        from hermes_cli.config import load_config

        return bool((load_config().get("session") or {}).get("terminal_continue", True))
    except Exception:
        return True


def write_breadcrumb(session_id: str, cwd: Optional[str] = None) -> None:
    """Record that this terminal's live session is ``session_id``. Never raises; no-op when the
    feature is disabled, the session id is empty, or no terminal identity exists."""
    try:
        if not session_id or not is_enabled():
            return
        terminal_id = get_terminal_id()
        if not terminal_id:
            return
        directory = _breadcrumbs_dir()
        from hermes_constants import mkdir_under_hermes_home
        mkdir_under_hermes_home(directory)
        now = time.time()
        payload = {"session_id": session_id, "cwd": cwd or os.getcwd(), "ts": now}
        atomic_json_write(directory / terminal_id, payload, indent=None)
        unlink_files_older_than(directory, "*", _STALE_AFTER_SECONDS, now=now)
    except Exception:
        pass


def read_breadcrumb() -> Optional[dict]:
    """This terminal's breadcrumb payload, or ``None`` (missing, corrupt, or stale). Never raises."""
    try:
        terminal_id = get_terminal_id()
        if not terminal_id:
            return None
        path = _breadcrumbs_dir() / terminal_id
        data = json.loads(path.read_text(encoding="utf-8-sig"))
        if not isinstance(data, dict) or not str(data.get("session_id") or "").strip():
            return None
        ts = data.get("ts")
        if isinstance(ts, (int, float)) and time.time() - ts > _STALE_AFTER_SECONDS:
            return None
        return data
    except Exception:
        return None


def resolve_breadcrumb_session() -> Optional[str]:
    """Resolve a bare ``-c`` for this terminal, or ``None`` to fall back. The breadcrumb's session
    id counts only if it still exists in the DB, projected through the compression chain so the
    resume lands on the live tip (same projection as ``main._resolve_session_by_name_or_id``)."""
    if not is_enabled():
        return None
    crumb = read_breadcrumb()
    if not crumb:
        return None
    session_id = str(crumb.get("session_id") or "").strip()
    if not session_id:
        return None
    try:
        from hermes_state import SessionDB
        from hermes_constants import get_hermes_home

        db = SessionDB(db_path=get_hermes_home() / "state.db", read_only=True)
    except Exception:
        return None
    try:
        if not db.get_session(session_id):
            return None  # session was deleted — fall back to latest
        try:
            return db.get_compression_tip(session_id) or session_id
        except Exception:
            return session_id
    except Exception:
        return None
    finally:
        try:
            db.close()
        except Exception:
            pass
