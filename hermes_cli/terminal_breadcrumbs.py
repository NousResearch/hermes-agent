"""Per-terminal session breadcrumbs for ``hermes -c`` / ``--continue``.

Each CLI session writes a tiny breadcrumb file
``$HERMES_HOME/terminal-sessions/<terminal-id>`` containing
``{"session_id": ..., "cwd": ..., "ts": ...}``.  A bare ``hermes -c`` then
resumes the session that belongs to THIS terminal (tty / tmux pane / kitty
window / wezterm pane / ...) instead of the globally most-recent session —
so two terminals side by side each continue their own conversation.

Everything here is strictly best-effort: no function raises, and when no
stable terminal identity can be derived (no tty and no known multiplexer
env var) breadcrumbs are skipped entirely and ``-c`` falls back to the
existing latest-session behavior.  Gated by ``session.terminal_continue``
in config.yaml (default true).
"""

from __future__ import annotations

import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Optional

# Multiplexer / terminal-emulator identity env vars, checked in order when
# no real tty path is available (e.g. stdin piped but stdout still a pty
# owned by a known terminal).
_TERMINAL_ENV_VARS = (
    "ZELLIJ_PANE_ID",
    "TMUX_PANE",
    "KITTY_WINDOW_ID",
    "WEZTERM_PANE",
    "TERM_SESSION_ID",
    "WT_SESSION",
)

# Breadcrumbs older than this are pruned opportunistically on each write —
# a pane id from a tmux server restarted last month means nothing today.
_STALE_AFTER_SECONDS = 30 * 24 * 60 * 60

_SANITIZE_RE = re.compile(r"[^A-Za-z0-9._-]")


def _breadcrumbs_dir() -> Path:
    from hermes_constants import get_hermes_home

    return get_hermes_home() / "terminal-sessions"


def _sanitize(raw: str) -> str:
    """Make an id safe to use as a filename (``/dev/pts/3`` -> ``dev-pts-3``)."""
    return _SANITIZE_RE.sub("-", raw.strip().strip("/"))[:120]


def get_terminal_id() -> Optional[str]:
    """Derive a stable identity for the terminal this process runs in.

    Prefers the real tty device path (stdin, then stdout), else the first
    present multiplexer/emulator env var. Returns ``None`` when neither is
    available — callers must then skip breadcrumbs entirely.
    """
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


def _prune_stale(directory: Path, now: float) -> None:
    """Best-effort removal of breadcrumbs older than the staleness window."""
    try:
        for entry in directory.iterdir():
            try:
                if entry.is_file() and now - entry.stat().st_mtime > _STALE_AFTER_SECONDS:
                    entry.unlink()
            except OSError:
                continue
    except OSError:
        pass


def write_breadcrumb(session_id: str, cwd: Optional[str] = None) -> None:
    """Record that this terminal's live session is ``session_id``.

    Synchronous, best-effort, never raises. No-op when the feature is
    disabled, the session id is empty, or no terminal identity exists.
    """
    try:
        if not session_id or not is_enabled():
            return
        terminal_id = get_terminal_id()
        if not terminal_id:
            return
        directory = _breadcrumbs_dir()
        directory.mkdir(parents=True, exist_ok=True)
        now = time.time()
        payload = {
            "session_id": session_id,
            "cwd": cwd or os.getcwd(),
            "ts": now,
        }
        tmp = directory / f".{terminal_id}.tmp"
        tmp.write_text(json.dumps(payload), encoding="utf-8")
        os.replace(tmp, directory / terminal_id)
        _prune_stale(directory, now)
    except Exception:
        pass


def read_breadcrumb() -> Optional[dict]:
    """Return this terminal's breadcrumb payload, or ``None``.

    Ignores breadcrumbs older than the staleness window. Never raises.
    """
    try:
        terminal_id = get_terminal_id()
        if not terminal_id:
            return None
        path = _breadcrumbs_dir() / terminal_id
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, dict) or not str(data.get("session_id") or "").strip():
            return None
        ts = data.get("ts")
        if isinstance(ts, (int, float)) and time.time() - ts > _STALE_AFTER_SECONDS:
            return None
        return data
    except Exception:
        return None


def resolve_breadcrumb_session() -> Optional[str]:
    """Resolve a bare ``-c`` for this terminal, or ``None`` to fall back.

    Returns the breadcrumb's session id only when it still exists in the
    session DB, projected forward through the compression chain so the
    resume lands on the live tip rather than a dead compressed parent
    (same projection as ``main._resolve_session_by_name_or_id``).
    """
    if not is_enabled():
        return None
    crumb = read_breadcrumb()
    if not crumb:
        return None
    session_id = str(crumb.get("session_id") or "").strip()
    if not session_id:
        return None
    db = None
    try:
        from hermes_state import SessionDB

        db = SessionDB()
        if not db.get_session(session_id):
            return None  # session was deleted — fall back to latest
        try:
            session_id = db.get_compression_tip(session_id) or session_id
        except Exception:
            pass
        return session_id
    except Exception:
        return None
    finally:
        if db is not None:
            try:
                db.close()
            except Exception:
                pass
