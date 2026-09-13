"""Current git branch for the CLI status bar.

Reads ``.git/HEAD`` directly (no subprocess) so status-bar repaints stay cheap, with a
short per-directory TTL cache. Worktrees and submodules (``.git`` as a ``gitdir:``
pointer file) resolve to their private git dir, whose HEAD is per-worktree.
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Optional

_TTL_SECONDS = 5.0
_cache: dict = {}


def _resolve_git_dir(start: Path) -> Optional[Path]:
    """Nearest enclosing git dir for ``start``, following worktree pointer files."""
    for parent in (start, *start.parents):
        dotgit = parent / ".git"
        if dotgit.is_dir():
            return dotgit
        if dotgit.is_file():
            try:
                line = dotgit.read_text(encoding="utf-8", errors="replace").strip()
            except OSError:
                return None
            if line.startswith("gitdir:"):
                target = (parent / line.split(":", 1)[1].strip()).resolve()
                return target if target.is_dir() else None
            return None
    return None


def current_git_branch(cwd: Optional[str] = None) -> str:
    """Branch name for ``cwd`` (defaults to the process cwd); ``""`` outside a repo.

    A detached HEAD renders as the abbreviated commit (``a1b2c3d…``). Results are
    cached ~5s per directory so per-repaint calls never re-walk the tree.
    """
    try:
        base = Path(cwd or os.getcwd()).resolve()
    except OSError:
        return ""
    key = str(base)
    now = time.monotonic()
    hit = _cache.get(key)
    if hit and now - hit[0] < _TTL_SECONDS:
        return hit[1]
    label = ""
    git_dir = _resolve_git_dir(base)
    if git_dir is not None:
        try:
            head = (git_dir / "HEAD").read_text(encoding="utf-8", errors="replace").strip()
        except OSError:
            head = ""
        if head.startswith("ref:"):
            ref = head.split(":", 1)[1].strip()
            label = ref[len("refs/heads/"):] if ref.startswith("refs/heads/") else ref.rsplit("/", 1)[-1]
        elif head:
            label = f"{head[:8]}…"
    _cache[key] = (now, label)
    return label
