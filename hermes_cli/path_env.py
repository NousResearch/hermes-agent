"""Helpers for making Hermes subprocess PATH lookups resilient.

Hermes is often launched by non-interactive process managers (launchd,
systemd, cron, IDEs, dashboard actions). Those environments can miss user shell
initialisation, so common tools installed through Homebrew, mise, asdf, nvm, or
Termux may not be visible even though they work in an interactive shell.
"""

from __future__ import annotations

import os
from pathlib import Path


_STATIC_TOOL_PATH_DIRS: tuple[str, ...] = (
    # Android / Termux
    "/data/data/com.termux/files/usr/bin",
    "/data/data/com.termux/files/usr/sbin",
    # macOS package managers
    "/opt/homebrew/bin",
    "/opt/homebrew/sbin",
    "/usr/local/bin",
    "/usr/local/sbin",
    # Standard Unix fallbacks
    "/usr/bin",
    "/bin",
    "/usr/sbin",
    "/sbin",
)


def _split_path(path: str | None) -> list[str]:
    """Split a PATH string into non-empty entries."""
    return [part for part in (path or "").split(os.pathsep) if part]


def _dedupe(paths: list[str]) -> list[str]:
    """Return paths with duplicates removed while preserving order."""
    seen: set[str] = set()
    result: list[str] = []
    for path in paths:
        if not path or path in seen:
            continue
        seen.add(path)
        result.append(path)
    return result


def _resolve_home() -> Path | None:
    """Return the user's home directory, preferring ``$HOME`` over passwd lookups.

    ``Path.home()`` consults ``pwd.getpwuid(os.getuid()).pw_dir`` on POSIX, which
    can ignore the current process's ``HOME`` environment variable. That makes
    behaviour surprising for callers (and tests) that override ``HOME`` to point
    at a sandbox directory. We honour ``HOME`` when set so non-interactive
    launches and unit tests both see the directory the rest of the process is
    using.
    """
    home_str = os.environ.get("HOME") or os.path.expanduser("~")
    if not home_str or home_str == "~":
        return None
    return Path(home_str)


def _home_tool_path_dirs() -> list[str]:
    """Return user-level tool directories managed by common version managers."""
    home = _resolve_home()
    if home is None:
        return []

    candidates = [
        home / ".local" / "bin",
        home / ".local" / "share" / "mise" / "shims",
        home / ".asdf" / "shims",
        home / ".bun" / "bin",
    ]

    nvm_dir = Path(os.environ.get("NVM_DIR") or (home / ".nvm"))
    node_versions_dir = nvm_dir / "versions" / "node"
    try:
        candidates.extend(sorted(node_versions_dir.glob("*/bin"), reverse=True))
    except OSError:
        pass

    return [str(path) for path in candidates]


def common_tool_path_dirs(*, existing_only: bool = True) -> tuple[str, ...]:
    """Return ordered PATH fallback directories for external tools.

    Dynamic user-level directories come first so non-interactive launches can
    see version-manager shims. Static platform directories follow. By default
    only directories that exist on the current machine are returned.
    """
    candidates = _dedupe([*_home_tool_path_dirs(), *_STATIC_TOOL_PATH_DIRS])
    if not existing_only:
        return tuple(candidates)
    return tuple(path for path in candidates if os.path.isdir(path))


def merge_common_tool_path(existing_path: str | None = None, *, prepend: bool = False) -> str:
    """Merge common tool fallbacks into an existing PATH.

    Appending is the default so an explicitly configured PATH keeps priority.
    Callers that need fallback tools to win (for example browser command
    execution) can request ``prepend=True``.
    """
    path_parts = _split_path(existing_path if existing_path is not None else os.environ.get("PATH", ""))
    fallback_parts = [part for part in common_tool_path_dirs() if part not in path_parts]
    merged = [*fallback_parts, *path_parts] if prepend else [*path_parts, *fallback_parts]
    return os.pathsep.join(merged)


def ensure_common_tool_paths(*, prepend: bool = False) -> str:
    """Update ``os.environ['PATH']`` with common external-tool fallbacks."""
    merged = merge_common_tool_path(prepend=prepend)
    os.environ["PATH"] = merged
    return merged
