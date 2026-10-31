"""Who holds a TCP port: ``(pid, command)`` for every LISTEN socket on it.

An EADDRINUSE that only names the port sends the operator hunting with lsof; naming the
holder tells them at once whether it is a stale Hermes backend, another profile's gateway
or an unrelated program (each has a different fix). Ported from cline/cline#14532.

Diagnostic only: nothing here kills or signals a process. Listeners, never clients — a bare
``lsof -i :PORT`` once killed a user's browser (WhatsApp bridge, #89614 class).
"""

from __future__ import annotations

import logging
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

_MAX_COMMAND_CHARS = 160
_MAX_OWNERS = 4


def _run(cmd: list[str], timeout: float, *, creationflags: int = 0) -> str:
    """stdout of ``cmd`` as text; raises ``FileNotFoundError`` when the tool is absent."""
    return subprocess.run(
        cmd, capture_output=True, text=True, encoding="utf-8", errors="replace",
        stdin=subprocess.DEVNULL, timeout=timeout, creationflags=creationflags,
    ).stdout


@dataclass(frozen=True)
class PortOwner:
    pid: int
    command: str  # redacted, capped; "" when the process was not readable


def listening_pids(port: int, *, timeout: float = 3.0) -> list[int]:
    """PIDs with a socket in LISTEN state on ``port``; ``[]`` when nothing could be determined.

    psutil first (no subprocess; sees this user's processes on Linux, everything as root), then the
    OS tools for the sockets psutil cannot attribute: ``lsof``/``ss`` on POSIX, ``netstat`` on Windows.
    """
    pids = _psutil_listening_pids(port)
    if pids:
        return pids
    try:
        if sys.platform == "win32":
            return _windows_listening_pids(port, timeout)
        return _posix_listening_pids(port, timeout)
    except (OSError, subprocess.SubprocessError):
        logger.debug("port owner scan failed for port %d", port, exc_info=True)
        return []


def port_owners(port: int, *, timeout: float = 3.0) -> list[PortOwner]:
    """Listening PIDs on ``port`` with their command lines (home paths redacted, capped)."""
    return [PortOwner(pid, _describe_process(pid)) for pid in listening_pids(port, timeout=timeout)[:_MAX_OWNERS]]


def describe_port_owners(port: int, *, timeout: float = 3.0) -> str:
    """Human fragment for a port-conflict message: ``" (held by PID 4242: hermes serve --port 9191)"``.

    ``""`` when no holder could be determined, so callers can append it unconditionally.
    """
    owners = port_owners(port, timeout=timeout)
    if not owners:
        return ""
    parts = [f"PID {o.pid}" + (f": {o.command}" if o.command else "") for o in owners]
    return f" (held by {'; '.join(parts)})"


def _psutil_listening_pids(port: int) -> list[int]:
    try:
        import psutil

        conns = psutil.net_connections(kind="tcp")
    except Exception:  # AccessDenied on macOS without root, psutil missing on exotic builds
        return []
    pids: list[int] = []
    for conn in conns:
        if conn.status == psutil.CONN_LISTEN and conn.laddr and conn.laddr.port == port and conn.pid:
            if conn.pid not in pids:
                pids.append(conn.pid)
    return pids


def _posix_listening_pids(port: int, timeout: float) -> list[int]:
    try:
        out = _run(["lsof", "-ti", f"tcp:{port}", "-sTCP:LISTEN"], timeout)
        pids = _safe_ints(out.split())
        if pids:
            return pids
    except FileNotFoundError:
        pass
    try:
        out = _run(["ss", "-ltnHp", f"sport = :{port}"], timeout)
    except FileNotFoundError:
        return []
    return list(dict.fromkeys(int(m.group(1)) for m in re.finditer(r"pid=(\d+)", out)))


def _windows_listening_pids(port: int, timeout: float) -> list[int]:
    from hermes_cli._subprocess_compat import windows_hide_flags

    out = _run(["netstat", "-ano", "-p", "TCP"], timeout, creationflags=windows_hide_flags())
    rows = (line.split() for line in out.splitlines())
    return list(dict.fromkeys(_safe_ints(
        row[4] for row in rows if len(row) >= 5 and row[3] == "LISTENING" and row[1].endswith(f":{port}")
    )))


def _safe_ints(tokens) -> list[int]:
    out: list[int] = []
    for tok in tokens:
        try:
            out.append(int(tok))
        except ValueError:
            pass
    return out


def _describe_process(pid: int) -> str:
    """``name args…`` for ``pid`` with the home directory redacted, capped; ``""`` when unreadable."""
    try:
        import psutil

        proc = psutil.Process(pid)
        try:
            text = " ".join(proc.cmdline() or [])
        except (psutil.AccessDenied, psutil.ZombieProcess):
            text = ""
        if not text:
            text = proc.name() or ""
    except Exception:
        return ""
    home = str(Path.home())
    if home and home != "/":
        text = text.replace(home, "~")
    text = " ".join(text.split())
    if len(text) > _MAX_COMMAND_CHARS:
        text = text[: _MAX_COMMAND_CHARS - 1] + "…"
    return text
