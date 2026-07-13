"""Workspace and project-root resolution for LSP.

Two concerns live here:

1. **Workspace gate** — the upper-level "is this directory a project?"
   check.  Hermes only runs LSP when the cwd (or the file being edited)
   sits inside a git worktree.  Files outside any git root never
   trigger LSP, even if a server is configured.  This keeps Telegram
   gateway users on user-home cwd's from spawning daemons.

2. **NearestRoot** — the per-server project-root walk.  Each language
   server cares about a different marker (``pyproject.toml`` for
   Python, ``Cargo.toml`` for Rust, ``go.mod`` for Go, etc.) and
   wants the directory containing that marker.  ``nearest_root()``
   walks up from a starting path looking for any of a list of marker
   files, optionally bailing if an exclude marker shows up first.
"""
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Iterable, Optional, Tuple

logger = logging.getLogger("agent.lsp.workspace")

# Cache: cwd → (worktree_root, is_git) so repeated calls don't re-stat.
# Cleared on shutdown.  Keyed by absolute resolved path so symlink
# folds collapse to one entry.
_workspace_cache: dict = {}


def normalize_path(path: str) -> str:
    """Normalize a path for use as a stable map key.

    Resolves ``~``, makes absolute, and collapses ``.``/``..``.  We do
    NOT resolve symlinks here — symlink stability matters for some
    LSP servers (rust-analyzer cares about Cargo workspace identity)
    and we want the canonical path the user typed when possible.
    """
    return os.path.abspath(os.path.expanduser(path))


def find_git_worktree(start: str) -> Optional[str]:
    """Walk up from ``start`` looking for a ``.git`` entry (file or dir).

    Returns the directory containing ``.git``, or ``None`` if no git
    root is found before hitting the filesystem root.

    A ``.git`` *file* (not directory) means we're inside a git
    worktree set up via ``git worktree add`` — both forms count.
    """
    try:
        start_path = Path(normalize_path(start))
        if start_path.is_file():
            start_path = start_path.parent
    except (OSError, RuntimeError, ValueError):
        # Pathological input (loop in symlinks, encoding error, etc.) —
        # bail out rather than crash the lint hook.
        return None

    # Cache check
    cached = _workspace_cache.get(str(start_path))
    if cached is not None:
        root, _is_git = cached
        return root

    cur = start_path
    # Defensive cap: the deepest reasonable monorepo is well under 64
    # levels.  Caps the walk so a pathological cwd or a symlink cycle
    # we somehow traverse can't keep us looping.
    for _ in range(64):
        git_marker = cur / ".git"
        try:
            if git_marker.exists():
                resolved = str(cur)
                _workspace_cache[str(start_path)] = (resolved, True)
                return resolved
        except OSError:
            # Permission error on a parent dir — bail out cleanly.
            break
        parent = cur.parent
        if parent == cur:
            break
        cur = parent

    _workspace_cache[str(start_path)] = (None, False)
    return None


def is_inside_workspace(path: str, workspace_root: str) -> bool:
    """Return True iff ``path`` is inside (or equal to) ``workspace_root``.

    Uses absolute paths but does not resolve symlinks — a file accessed
    via a symlink that points outside the workspace still counts as
    outside.  This is the conservative interpretation; matches LSP
    behaviour where servers reject didOpen for unrelated files.
    """
    p = normalize_path(path)
    root = normalize_path(workspace_root)
    if p == root:
        return True
    # Use os.path.commonpath to handle case-insensitive filesystems
    # correctly on macOS/Windows.
    try:
        common = os.path.commonpath([p, root])
    except ValueError:
        # Different drives on Windows.
        return False
    return common == root


def nearest_root(
    start: str,
    markers: Iterable[str],
    *,
    excludes: Optional[Iterable[str]] = None,
    ceiling: Optional[str] = None,
) -> Optional[str]:
    """Walk up from ``start`` looking for any of the given marker files.

    Returns the **directory containing** the first matched marker, or
    ``None`` if no marker is found before hitting ``ceiling`` (or the
    filesystem root if no ceiling).

    If ``excludes`` is provided and an exclude marker matches *first*
    in the upward walk, returns ``None`` — the server is gated off
    for that file.  Mirrors OpenCode's NearestRoot exclude semantics
    (e.g. typescript skips deno projects when ``deno.json`` is found
    before ``package.json``).
    """
    start_path = Path(normalize_path(start))
    try:
        if start_path.is_file():
            start_path = start_path.parent
    except (OSError, RuntimeError, ValueError):
        return None
    ceiling_path = Path(normalize_path(ceiling)) if ceiling else None

    markers_list = list(markers)
    excludes_list = list(excludes) if excludes else []

    cur = start_path
    # Defensive cap matching ``find_git_worktree``.  Bounded walk
    # protects against pathological inputs even though the
    # parent-equality stop normally terminates within ~10 steps.
    for _ in range(64):
        # Check excludes first — if an exclude is found at this level,
        # the server is gated off for this file.
        for exc in excludes_list:
            try:
                if (cur / exc).exists():
                    return None
            except OSError:
                continue
        # Then check markers.
        for marker in markers_list:
            try:
                if (cur / marker).exists():
                    return str(cur)
            except OSError:
                continue
        # Stop conditions.
        if ceiling_path is not None and cur == ceiling_path:
            return None
        parent = cur.parent
        if parent == cur:
            return None
        cur = parent
    return None


def resolve_workspace_for_file(
    file_path: str,
    *,
    cwd: Optional[str] = None,
) -> Tuple[Optional[str], bool]:
    """Resolve the workspace root for a file.

    Returns ``(workspace_root, gated_in)`` where ``gated_in`` is True
    iff LSP should run for this file at all.  Currently the gate is
    "file is inside a git worktree found by walking up from cwd OR
    from the file itself".

    The cwd path takes precedence — if the agent was launched in a
    git project, that worktree is the workspace, and any edit inside
    it (regardless of where the file lives) is in-scope.  If the cwd
    isn't in a git worktree, we try the file's own location as a
    fallback.

    Returns ``(None, False)`` when neither path is in a git worktree.
    """
    cwd = cwd or os.getcwd()
    cwd_root = find_git_worktree(cwd)
    if cwd_root is not None:
        if is_inside_workspace(file_path, cwd_root):
            return cwd_root, True
        # File is outside the cwd's worktree — try the file's own
        # location as a secondary anchor.  Useful for monorepos where
        # the user opens an unrelated checkout.
    file_root = find_git_worktree(file_path)
    if file_root is not None:
        return file_root, True
    return None, False


def clear_cache() -> None:
    """Clear the workspace-resolution cache.

    Called on service shutdown so a subsequent re-init doesn't pick
    up stale results from a previous session.
    """
    _workspace_cache.clear()


__all__ = [
    "find_git_worktree",
    "is_inside_workspace",
    "nearest_root",
    "normalize_path",
    "resolve_workspace_for_file",
    "clear_cache",
]
