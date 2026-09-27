"""Directories where tools install outside PATH, grouped by the ecosystem that owns them.

Every table in Hermes lives here. A directory literal outside this module fails the ratchet in
`tests/test_managed_runtime_resolution.py`. Each table is empty on an OS where the ecosystem
does not install there, so callers compose tables without OS branches.
"""

from __future__ import annotations

import sys

_POSIX = sys.platform != "win32"
_WIN = sys.platform == "win32"


def homebrew_dirs() -> tuple[str, ...]:
    return ("/opt/homebrew/bin", "/usr/local/bin") if sys.platform == "darwin" else ()


def user_local_bin() -> tuple[str, ...]:
    return ("~/.local/bin",) if _POSIX else ("%USERPROFILE%/.local/bin",)


def rust_tool_dirs() -> tuple[str, ...]:
    return ("~/.cargo/bin",) if _POSIX else ("%USERPROFILE%/.cargo/bin",)


def uv_tool_dirs() -> tuple[str, ...]:
    """uv's install locations outside PATH, in uv's own install order: the per-user
    installer's ``~/.local/bin`` (every OS — that is where uv's docs put it), then
    Homebrew (Apple Silicon ``/opt``, Intel / from-source ``/usr/local``). The tilde
    form is deliberate: callers that only ``expanduser`` (the stdio launcher
    fallback) get the same result as ``locate_command``'s expandvars+expanduser."""
    if _POSIX:
        return ("~/.local/bin", "/opt/homebrew/bin", "/usr/local/bin")
    return ("~/.local/bin",)


def node_tool_dirs() -> tuple[str, ...]:
    return ("~/.npm-global/bin", "~/.bun/bin", "~/.volta/bin") if _POSIX else ("%APPDATA%/npm", "%USERPROFILE%/.bun/bin", "%LOCALAPPDATA%/Volta/bin")


def hermes_vendored_dirs() -> tuple[str, ...]:
    return ("~/.hermes/bin",) if _POSIX else ("%USERPROFILE%/.hermes/bin",)


def windows_user_program_dirs() -> tuple[str, ...]:
    if not _WIN:
        return ()
    return (
        "%LOCALAPPDATA%/Programs",
        "%USERPROFILE%/scoop/shims",
        "%LOCALAPPDATA%/Microsoft/WinGet/Links",
    )
