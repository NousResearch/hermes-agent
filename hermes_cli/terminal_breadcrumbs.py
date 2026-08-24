"""Per-terminal session breadcrumbs for ``hermes -c`` / ``--continue``. Strictly best-effort: no
function raises; without a stable terminal identity (no tty, no known multiplexer env var) ``-c``
falls back to latest-session. Gated by ``session.terminal_continue`` (default true)."""

from __future__ import annotations

import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Optional

# Multiplexer / terminal-emulator identity env vars, checked in order when no real tty path is
# available (e.g. stdin piped but stdout still a pty owned by a known terminal).
_TERMINAL_ENV_VARS = ("ZELLIJ_PANE_ID", "TMUX_PANE", "KITTY_WINDOW_ID", "WEZTERM_PANE", "TERM_SESSION_ID", "WT_SESSION")

# Breadcrumbs older than this are pruned opportunistically on each write — a pane id from a tmux
# server restarted last month means nothing today.
_STALE_AFTER_SECONDS = 30 * 24 * 60 * 60

_SANITIZE_RE = re.compile(r"[^A-Za-z0-9._-]")


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
    """Record that this terminal's live session is ``session_id``. Never raises; no-op when the
    feature is disabled, the session id is empty, or no terminal identity exists."""
    try:
        if not session_id or not is_enabled():
            return
        terminal_id = get_terminal_id()
        if not terminal_id:
            return
        directory = _breadcrumbs_dir()
        directory.mkdir(parents=True, exist_ok=True)
        now = time.time()
        payload = {"session_id": session_id, "cwd": cwd or os.getcwd(), "ts": now}
        tmp = directory / f".{terminal_id}.tmp"
        tmp.write_text(json.dumps(payload), encoding="utf-8")
        os.replace(tmp, directory / terminal_id)
        _prune_stale(directory, now)
    except Exception:
        pass


def read_breadcrumb() -> Optional[dict]:
    """This terminal's breadcrumb payload, or ``None`` (missing, corrupt, or stale). Never raises."""
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

        db = SessionDB()
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
