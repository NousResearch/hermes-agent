"""User-command status-bar segment for the CLI/TUI.

``display.status_bar.custom_command`` names a shell command whose first output line renders
as an opt-in status-bar segment (field ``custom``). The command NEVER runs on the repaint
path: repaints return the last cached value instantly and a background thread refreshes it
when the TTL lapses, so a slow or hung command can only make the segment stale, not freeze
the UI. Hermes-managed secrets are filtered from the child environment (same policy as the
``!`` bang shell).
"""

from __future__ import annotations

import subprocess
import threading
import time
from typing import Optional

# Long enough that a per-repaint call is free; short enough that prompt-style commands
# (git status counts, kubectl context, todo counters) feel live.
_TTL_SECONDS = 10.0
_TIMEOUT_SECONDS = 5.0
_MAX_WIDTH = 40

_lock = threading.Lock()
# {"command": str, "at": monotonic, "value": str, "running": bool}
_state: dict = {"command": None, "at": 0.0, "value": "", "running": False}


def _shape_output(raw: Optional[str]) -> str:
    """First non-empty line, stripped and width-capped; ``""`` for no usable output."""
    for line in (raw or "").splitlines():
        line = line.strip()
        if line:
            return line if len(line) <= _MAX_WIDTH else f"{line[:_MAX_WIDTH - 1]}…"
    return ""


def _run_command(command: str) -> str:
    """Execute *command* once and shape its stdout; failures render as ``""``.

    A non-zero exit still uses whatever stdout was produced (prompt tools like
    ``git describe`` exit non-zero in edge cases while printing a usable value).
    """
    from hermes_cli.bang_shell import _bang_env, resolve_bang_cwd

    try:
        result = subprocess.run(
            command, shell=True, capture_output=True, text=True,
            encoding="utf-8", errors="replace", timeout=_TIMEOUT_SECONDS,
            cwd=resolve_bang_cwd(), env=_bang_env())
        return _shape_output(result.stdout)
    except Exception:
        return ""


def _refresh(command: str) -> None:
    value = _run_command(command)
    with _lock:
        _state.update(command=command, at=time.monotonic(), value=value, running=False)


def custom_segment(command: str) -> str:
    """Cached segment text for *command*; kicks off a background refresh when stale.

    The first call for a command returns ``""`` (the refresh hasn't landed yet) — the
    segment simply appears on a later repaint, which the status bar already tolerates
    for every other conditional field.
    """
    command = (command or "").strip()
    if not command:
        return ""
    now = time.monotonic()
    with _lock:
        same = _state["command"] == command
        fresh = same and now - _state["at"] < _TTL_SECONDS
        value = _state["value"] if same else ""
        if fresh or _state["running"]:
            return value
        _state["running"] = True
        if not same:
            _state.update(command=command, value="")
    threading.Thread(target=_refresh, args=(command,), daemon=True).start()
    return value
