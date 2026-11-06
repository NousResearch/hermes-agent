"""Windows-safe stdio configuration.

Forces UTF-8 on the Python side and also flips the console's code page to UTF-8 (65001). Both
matter: Python-level only helps when Python's stdout is a real TTY; code-page flipping lets
subprocesses and child Python ``print()`` calls agree on encoding.
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
    """``SetConsoleCP``/``SetConsoleOutputCP`` to CP_UTF8 (65001). Silent on failure: without an
    attached console (redirected stdout, service, PTY-less CI) the calls return 0 and we move on."""
    try:
        import ctypes

        kernel32 = ctypes.windll.kernel32  # type: ignore[attr-defined]
        kernel32.SetConsoleCP(65001)
        kernel32.SetConsoleOutputCP(65001)
    except Exception:
        pass


def _reconfigure_stream(stream, *, encoding: str = "utf-8", errors: str = "replace") -> None:
    """Reconfigure a text stream to UTF-8 in place; skips streams without ``reconfigure`` (e.g. an
    ``io.StringIO`` substituted during tests)."""
    try:
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            reconfigure(encoding=encoding, errors=errors)
    except Exception:
        pass


def configure_windows_stdio() -> bool:
    """Force UTF-8 stdio on Windows. No-op elsewhere.

    Idempotent; returns ``True`` only when something actually changed. Set
    ``HERMES_DISABLE_WINDOWS_UTF8=1`` to opt out (forces the old cp1252 path for diagnosing
    encoding bugs). Also sets a default ``EDITOR`` on Windows if none is set.
    """
    global _CONFIGURED

    if _CONFIGURED:
        return False
    if not is_windows() or os.environ.get("HERMES_DISABLE_WINDOWS_UTF8") in {"1", "true", "True", "yes"}:
        _CONFIGURED = True  # repeated calls on POSIX / opted-out are true no-ops
        return False

    # Make child Python processes use UTF-8 stdio too (PYTHONIOENCODING wins over the locale
    # default; PYTHONUTF8=1 enables UTF-8 Mode, PEP 540). Never override an explicit user setting.
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")
    os.environ.setdefault("PYTHONUTF8", "1")

    # prompt_toolkit's ``open_in_editor`` falls back to POSIX-only paths (/usr/bin/nano, /usr/bin/vi)
    # that don't exist on Windows — Ctrl+X Ctrl+E and ``/edit`` silently do nothing there
    # otherwise, even with full Git for Windows installed.
    _default_editor = _default_windows_editor()
    if _default_editor and not os.environ.get("EDITOR") and not os.environ.get("VISUAL"):
        os.environ["EDITOR"] = _default_editor

    _augment_path_with_known_tools()
    # Flip the console code page first so any subprocess inheriting the console also sees CP_UTF8.
    _flip_console_code_page_to_utf8()
    # ``errors="replace"``: a genuinely unencodable sequence prints ``?`` rather than crashing the
    # interpreter. stdin is included for batch/pipe input (prompt_toolkit manages its own encoding).
    for stream in (sys.stdout, sys.stderr, sys.stdin):
        _reconfigure_stream(stream)
    _CONFIGURED = True
    return True


def _default_windows_editor() -> str:
    """Windows default for ``$EDITOR``: ``notepad`` (ships with every install, blocks until the
    window closes). The bare name keeps prompt_toolkit's shlex split away from paths with spaces;
    "" when even notepad is missing (WinPE, Nano Server) so prompt_toolkit's no-op applies."""
    import shutil
    return "notepad" if shutil.which("notepad") else ""


def _augment_path_with_known_tools() -> None:
    r"""Prepend Hermes-managed tool directories to ``PATH`` (no-op on POSIX / missing dirs).

    install.ps1 adds entries like ``%LOCALAPPDATA%\hermes\git\bin`` to the User PATH via
    ``SetEnvironmentVariable``, but already-running shells never see that broadcast, so a hermes
    launched from the install session would not find rg / bash / grep. Prepending the known dirs
    at startup closes that first-launch gap.
    """
    if not is_windows():
        return
    local_appdata = os.environ.get("LOCALAPPDATA", "")
    if not local_appdata:
        return

    # Kept in sync with the PATH entries scripts/install.ps1 adds to User scope. The venv Scripts
    # dir hosts hermes.exe + pip console scripts; WinGet\Links is where ``winget install`` drops
    # CLI shims (ripgrep lands there as rg.exe).
    candidate_dirs = [
        os.path.join(local_appdata, "hermes", "git", "cmd"),
        os.path.join(local_appdata, "hermes", "git", "bin"),
        os.path.join(local_appdata, "hermes", "git", "usr", "bin"),
        os.path.join(local_appdata, "hermes", "hermes-agent", "venv", "Scripts"),
        os.path.join(local_appdata, "Microsoft", "WinGet", "Links")]
    existing = os.environ.get("PATH", "")
    existing_lower = {p.lower() for p in existing.split(os.pathsep) if p}
    prepend = [d for d in candidate_dirs if os.path.isdir(d) and d.lower() not in existing_lower]
    if prepend:
        os.environ["PATH"] = os.pathsep.join([*prepend, existing])
