"""Workspace and project-root resolution for LSP.

1. **Workspace gate** — LSP only runs when the cwd (or the edited file) sits inside a git
   worktree, so gateway users on user-home cwd's never spawn daemons.
2. **nearest_root** — the per-server project-root walk: up from a start path looking for marker
   files (``pyproject.toml``, ``Cargo.toml``, ...), optionally bailing if an exclude marker
   shows up first.
"""
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Iterable, Iterator, Optional, Tuple

logger = logging.getLogger("agent.lsp.workspace")

# Cache: start dir → (worktree_root, is_git) so repeated calls don't re-stat.  Cleared on shutdown.
_workspace_cache: dict = {}

# Walk cap: the deepest reasonable monorepo is well under 64 levels; bounds a
# pathological cwd or symlink cycle even though parent-equality normally stops us.
_MAX_WALK = 64


def normalize_path(path: str) -> str:
    """Expand ``~``, make absolute, collapse ``.``/``..``.  Symlinks are deliberately NOT resolved —
    some servers (rust-analyzer's Cargo workspace identity) care, and we want the path the user typed."""
    return os.path.abspath(os.path.expanduser(path))


def _start_dir(start: str) -> Optional[Path]:
    """Normalized start directory (a file's parent), or ``None`` on pathological input."""
    try:
        start_path = Path(normalize_path(start))
        if start_path.is_file():
            start_path = start_path.parent
    except (OSError, RuntimeError, ValueError):
        # Symlink loop, encoding error, etc. — bail rather than crash the lint hook.
        return None
    return start_path


def _walk_up(start: Path) -> Iterator[Path]:
    """Yield ``start`` and its ancestors up to the filesystem root, bounded by ``_MAX_WALK``."""
    cur = start
    for _ in range(_MAX_WALK):
        yield cur
        parent = cur.parent
        if parent == cur:
            return
        cur = parent


def find_git_worktree(start: str) -> Optional[str]:
    """Return the nearest ancestor dir containing ``.git`` (file or dir — worktrees count), else ``None``."""
    start_path = _start_dir(start)
    if start_path is None:
        return None
    cached = _workspace_cache.get(str(start_path))
    if cached is not None:
        return cached[0]
    for cur in _walk_up(start_path):
        try:
            if (cur / ".git").exists():
                resolved = str(cur)
                _workspace_cache[str(start_path)] = (resolved, True)
                return resolved
        except OSError:
            break  # permission error on a parent dir — bail out cleanly
    _workspace_cache[str(start_path)] = (None, False)
    return None


def is_inside_workspace(path: str, workspace_root: str) -> bool:
    """True iff ``path`` is inside (or equal to) ``workspace_root``.  Symlinks are not resolved: a
    symlink pointing outside still counts as outside, matching servers that reject didOpen for
    unrelated files."""
    p = normalize_path(path)
    root = normalize_path(workspace_root)
    if p == root:
        return True
    # commonpath handles case-insensitive filesystems on macOS/Windows.
    try:
        return os.path.commonpath([p, root]) == root
    except ValueError:
        return False  # different drives on Windows


def nearest_root(
    start: str,
    markers: Iterable[str],
    *,
    excludes: Optional[Iterable[str]] = None,
    ceiling: Optional[str] = None,
) -> Optional[str]:
    """Walk up from ``start`` for the directory containing the first matched marker.

    Returns ``None`` past ``ceiling`` (or the filesystem root), or when an exclude marker is found
    first — the server is gated off for that file (e.g. typescript skips deno projects when
    ``deno.json`` precedes ``package.json``).  Marker names are exact filenames — no globs.
    """
    start_path = _start_dir(start)
    if start_path is None:
        return None
    ceiling_path = Path(normalize_path(ceiling)) if ceiling else None
    markers_list = list(markers)
    excludes_list = list(excludes) if excludes else []

    def present(cur: Path, names: list) -> bool:
        for name in names:
            try:
                if (cur / name).exists():
                    return True
            except OSError:
                continue
        return False

    for cur in _walk_up(start_path):
        # Excludes are checked before markers at each level.
        if present(cur, excludes_list):
            return None
        # A directory holding __init__.py is a Python package, never a project root (hermes_cli/setup.py
        # matched the python marker list and gave every package dir its own pyright).
        if not present(cur, ["__init__.py"]) and present(cur, markers_list):
            return str(cur)
        if ceiling_path is not None and cur == ceiling_path:
            return None
    return None


def resolve_workspace_for_file(file_path: str, *, cwd: Optional[str] = None) -> Tuple[Optional[str], bool]:
    """Return ``(workspace_root, gated_in)`` for a file.  The cwd's worktree wins when the file is
    inside it; otherwise the file's own worktree is the fallback anchor (monorepos / unrelated
    checkouts).  ``(None, False)`` when neither is in a git worktree."""
    cwd_root = find_git_worktree(cwd or os.getcwd())
    if cwd_root is not None and is_inside_workspace(file_path, cwd_root):
        return cwd_root, True
    file_root = find_git_worktree(file_path)
    if file_root is not None:
        return file_root, True
    return None, False


def clear_cache() -> None:
    """Clear the workspace-resolution cache (on service shutdown, so re-init doesn't see stale results)."""
    _workspace_cache.clear()


__all__ = [
    "find_git_worktree", "is_inside_workspace", "nearest_root", "normalize_path", "resolve_workspace_for_file",
    "clear_cache",
]
