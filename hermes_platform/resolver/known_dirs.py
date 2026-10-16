"""Directories where tools install outside PATH, grouped by the ecosystem that owns them.

Every table in Hermes lives here. A directory literal outside this module fails the ratchet in
`tests/test_managed_runtime_resolution.py`. Each table is empty on an OS where the ecosystem
does not install there, so callers compose tables without OS branches.
"""

from __future__ import annotations

import os
import sys

_POSIX = sys.platform != "win32"
_WIN = sys.platform == "win32"


def homebrew_dirs() -> tuple[str, ...]:
    return ("/opt/homebrew/bin", "/usr/local/bin") if sys.platform == "darwin" else ()


def user_local_bin() -> tuple[str, ...]:
    return ("~/.local/bin",) if _POSIX else ("%USERPROFILE%/.local/bin",)


def rust_tool_dirs() -> tuple[str, ...]:
    return ("~/.cargo/bin",) if _POSIX else ("%USERPROFILE%/.cargo/bin",)


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


def windows_git_bash_bin_dirs(root: str) -> list[str]:
    """Select one native Git toolchain, then the shared MSYS coreutils dirs."""
    if not _WIN:
        return []
    from hermes_platform.host.facts import native_arch

    # Prefer the native layout and the current UCRT toolchain over leftovers
    # from older installs; Windows can still run an installed emulated layout.
    prefixes = {
        "arm64": ("clangarm64", "ucrt64", "mingw64", "mingw32"),
        "amd64": ("ucrt64", "mingw64", "mingw32"),
        "x86": ("mingw32",),
    }.get(native_arch(), ())
    toolchains = (os.path.join(root, prefix, "bin") for prefix in prefixes)
    active = next((path for path in toolchains if os.path.isdir(path)), None)
    shared = (os.path.join(root, *sub) for sub in (("usr", "local", "bin"), ("usr", "bin"), ("bin",)))
    return [path for path in (active, *shared) if path and os.path.isdir(path)]
