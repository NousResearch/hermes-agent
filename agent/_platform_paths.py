"""Platform-aware path display helpers for the agent system prompt.

On native Windows, Hermes profile data lives under ``%LOCALAPPDATA%\\hermes\\``,
but on WSL and Linux it lives under ``~/.hermes/``.  These helpers return the
correct display string for the current platform so the system prompt doesn't
mislead agents about where their profile data actually lives.

Detection
---------
* ``os.name == "nt"`` → native Windows (use ``%LOCALAPPDATA%\\hermes\\``)
* ``os.name == "posix"`` + ``/proc/version`` contains ``microsoft`` → WSL
  (use ``~/.hermes/`` — WSL sessions run the Linux path)
* Anything else (Linux, macOS) → ``~/.hermes/``

The WSL check follows the same pattern as ``cli.py``'s ``_is_wsl()`` but is
kept here as a standalone copy so ``agent/system_prompt.py`` doesn't need to
import the entire CLI module.
"""

from __future__ import annotations

import os
from pathlib import Path


def _is_wsl() -> bool:
    """Return ``True`` when running inside WSL (Windows Subsystem for Linux).

    ``os.name`` is ``"posix"`` inside WSL, so we must inspect ``/proc/version``
    or ``/proc/sys/kernel/osrelease`` for the ``"microsoft"`` marker.  This is
    the same logic used in ``hermes_cli/cli.py:3139-3149``.
    """
    for path in ("/proc/version", "/proc/sys/kernel/osrelease"):
        try:
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                if "microsoft" in f.read().lower():
                    return True
        except OSError:
            continue
    return False


def _is_native_windows() -> bool:
    """Return ``True`` on native Windows (not WSL)."""
    return os.name == "nt"


def _is_wsl_or_linux() -> bool:
    """Return ``True`` on non-Windows (WSL, Linux, macOS)."""
    return not _is_native_windows()


def _display_hermes_root() -> str:
    """Return the human-readable Hermes root path for the current platform.

    Results
    -------
    Native Windows
        ``%LOCALAPPDATA%\\hermes``
    WSL / Linux / macOS
        ``~/.hermes``
    """
    if _is_native_windows():
        return r"%LOCALAPPDATA%\hermes"
    return "~/.hermes"


def _display_profile_path(profile_name: str) -> str:
    """Return the display path for *profile_name* under the Hermes profiles dir.

    Results
    -------
    Native Windows
        ``%LOCALAPPDATA%\\hermes\\profiles\\<profile_name>``
    WSL / Linux / macOS
        ``~/.hermes/profiles/<profile_name>``
    """
    if _is_native_windows():
        return rf"%LOCALAPPDATA%\hermes\profiles\{profile_name}"
    return f"~/.hermes/profiles/{profile_name}"


def _display_default_root() -> str:
    """Return the display path for the default profile's sub-directories.

    Results
    -------
    Native Windows
        ``%LOCALAPPDATA%\\hermes``
    WSL / Linux / macOS
        ``~/.hermes``
    """
    return _display_hermes_root()


# ── Testability helper ──────────────────────────────────────────────────────

def _display_hermes_root_for(os_name: str) -> str:
    """Same as :func:`_display_hermes_root` but with an explicit *os_name*
    argument so tests can verify both branches without mocking ``os.name``."""
    if os_name == "nt":
        return r"%LOCALAPPDATA%\hermes"
    return "~/.hermes"


def _display_profile_path_for(os_name: str, profile_name: str) -> str:
    """Same as :func:`_display_profile_path` but with an explicit *os_name*
    argument so tests can verify both branches without mocking ``os.name``."""
    if os_name == "nt":
        return rf"%LOCALAPPDATA%\hermes\profiles\{profile_name}"
    return f"~/.hermes/profiles/{profile_name}"
