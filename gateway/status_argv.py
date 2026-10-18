"""Reading a live process's command line and rendering argv as one matcher-ready string.

Split from ``gateway/status.py`` (FILE_LINES offset, #140831): the reader and the argv renderer
live together because the renderer exists for the readers' output — every consumer re-tokenizes
the joined string with ``shlex.split``, so element boundaries must survive the join."""

import contextlib
import subprocess
import sys
from pathlib import Path
from typing import Iterable, Optional

_IS_WINDOWS = sys.platform == "win32"


def _argv_to_command_line(parts: Iterable[str]) -> str:
    """Argv elements as one command line whose element boundaries survive re-tokenization.

    The matchers re-tokenize this string with ``shlex.split``, so an element containing whitespace
    must be quoted or it splits into two tokens: an interpreter path like ``C:\\Users\\John
    Doe\\...python.exe`` made the strict identity mapper reject a healthy gateway and abort
    ``hermes update`` fail-closed (#140831). Whitespace-free elements join unchanged."""
    rendered = []
    for part in parts:
        text = str(part)
        if not text:
            text = '""'
        elif any(ch.isspace() for ch in text):
            text = '"' + text.replace('"', '\\"') + '"'
        rendered.append(text)
    return " ".join(rendered)


def _read_process_cmdline(pid: int) -> Optional[str]:
    """Process command line as one string: /proc, then psutil, then ``ps``.

    Order is by cost, and this runs per live gateway on every roster/status poll. ``psutil`` reads
    the process table in-process (a ``sysctl`` on macOS) where ``ps`` costs a fork+exec — measured
    0.02ms against 4.2ms on macOS for the same string. It cannot always answer: on macOS it raises
    ``AccessDenied`` for a process owned by another user, which ``ps`` still reports, so ``ps``
    stays as the fallback rather than being replaced."""
    with contextlib.suppress(OSError):
        raw = Path(f"/proc/{pid}/cmdline").read_bytes()
        if raw:
            parts = [p.decode("utf-8", errors="ignore") for p in raw.split(b"\x00") if p]
            if parts:
                return _argv_to_command_line(parts)
    with contextlib.suppress(Exception):
        import psutil  # type: ignore
        cmdline_parts = psutil.Process(pid).cmdline()
        if cmdline_parts:
            return _argv_to_command_line(cmdline_parts)
    if not _IS_WINDOWS:
        with contextlib.suppress(OSError, subprocess.TimeoutExpired):
            result = subprocess.run(
                ["ps", "-p", str(pid), "-o", "command="],
                capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=5,
                check=False,
            )
            if result.returncode == 0 and result.stdout.strip():
                return result.stdout.strip()
    return None
