"""Windows-safe stdio configuration.

On Windows, Python's ``sys.stdout``/``sys.stderr`` default to the console's
active code page (often ``cp1252``, sometimes ``cp437``, occasionally ``cp932``
on Japanese locales, etc.).  Hermes's banners, tool output feed, and slash
command listings all contain Unicode: box-drawing characters (``─┌┐└┘├┤``),
mathematical and geometric symbols (``◆ ◇ ◎ ▣ ⚔ ⚖ →``), and user-supplied
text in any language.  Printing those to a cp1252 console raises
``UnicodeEncodeError: 'charmap' codec can't encode character…`` and kills the
whole CLI before the REPL even opens.

The fix is to force UTF-8 on the Python side and also flip the console's
code page to UTF-8 (65001).  Both matter: Python-level only helps when
Python's stdout is a real TTY; code-page flipping lets subprocesses and
child Python ``print()`` calls agree on encoding.

This module is a no-op on every non-Windows platform, and idempotent.
Entry points (``cli.py`` ``main``, ``hermes_cli/main.py`` CLI dispatch,
``gateway/run.py`` startup) call :func:`configure_windows_stdio` exactly
once early in startup.

Patterns cribbed from Claude Code (``src/utils/platform.ts``), OpenCode
(``packages/opencode/src/pty/index.ts`` env injection), and OpenAI Codex
(``codex-rs/core/src/unified_exec/process_manager.rs``).  None of those
actually flip the console code page — they rely on their runtime (Node or
Rust) writing UTF-16 to the Win32 console API and letting the terminal
sort it out.  Python doesn't get that luxury.
"""

from __future__ import annotations

import os
import sys

__all__ = ["configure_windows_stdio", "is_windows"]


_CONFIGURED = False


def is_windows() -> bool:
    """Return True iff running on native Windows (not WSL)."""
    return sys.platform == "win32"


def _flip_console_code_page_to_utf8() -> None:
    """Set the attached console's input and output code pages to UTF-8.

    Uses ``SetConsoleCP`` / ``SetConsoleOutputCP`` via ``ctypes``.  Failure
    is silent — if there's no attached console (e.g. Hermes is running
    behind a redirected stdout, under a service, or inside a PTY-less CI
    runner) these calls simply return 0 and we move on.

    CP_UTF8 is 65001.
    """
    try:
        import ctypes

        kernel32 = ctypes.windll.kernel32  # type: ignore[attr-defined]
        # Best-effort; if there's no console attached these just fail silently.
        kernel32.SetConsoleCP(65001)
        kernel32.SetConsoleOutputCP(65001)
    except Exception:
        # ctypes import, missing kernel32, or non-Windows — any failure here
        # is non-fatal.  We've still reconfigured Python's own streams below.
        pass


def _reconfigure_stream(stream, *, encoding: str = "utf-8", errors: str = "replace") -> None:
    """Reconfigure a text stream to UTF-8 in place.

    Uses ``TextIOWrapper.reconfigure`` (Python 3.7+).  If the stream isn't
    a ``TextIOWrapper`` (e.g. it's been redirected to an ``io.StringIO``
    during tests), we skip rather than blow up.
    """
    try:
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is None:
            return
        reconfigure(encoding=encoding, errors=errors)
    except Exception:
        pass


def configure_windows_stdio() -> bool:
    """Force UTF-8 stdio on Windows.  No-op elsewhere.

    Idempotent — safe to call multiple times from different entry points.

    Returns ``True`` if anything was actually changed, ``False`` on
    non-Windows or on a repeat call.

    Set ``HERMES_DISABLE_WINDOWS_UTF8=1`` in the environment to opt out
    (for diagnosing encoding-related bugs by forcing the old cp1252 path).
    """
    global _CONFIGURED

    if _CONFIGURED:
        return False
    if not is_windows():
        # Mark configured so repeated calls on POSIX are true no-ops.
        _CONFIGURED = True
        return False

    if os.environ.get("HERMES_DISABLE_WINDOWS_UTF8") in ("1", "true", "True", "yes"):
        _CONFIGURED = True
        return False

    # Encourage every child Python process spawned by the agent to also use
    # UTF-8 for its stdio.  PYTHONIOENCODING wins over the locale-based
    # default in subprocesses.  Don't override an explicit user setting.
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")
    # PYTHONUTF8 = 1 enables UTF-8 Mode globally for any Python subprocess
    # (PEP 540).  Again, don't override an explicit setting.
    os.environ.setdefault("PYTHONUTF8", "1")

    # Flip the console code page first so that any subprocess that
    # inherits the console (e.g. a launched shell) also sees CP_UTF8.
    _flip_console_code_page_to_utf8()

    # Reconfigure Python's own stdio wrappers so ``print()`` calls from
    # this process round-trip emoji / box-drawing / non-Latin text.
    # ``errors="replace"`` means a genuinely unencodable byte sequence
    # gets a ``?`` rather than crashing the interpreter — we prefer
    # degraded output over a stack trace.
    _reconfigure_stream(sys.stdout)
    _reconfigure_stream(sys.stderr)
    # stdin is re-configured for completeness; Hermes's interactive
    # input path uses prompt_toolkit which manages its own encoding,
    # but batch/pipe input benefits from UTF-8 decoding on stdin too.
    _reconfigure_stream(sys.stdin)

    _CONFIGURED = True
    return True
