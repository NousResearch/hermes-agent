"""Helpers for locating the mini-swe-agent source tree.

Hermes often runs from git worktrees. In that layout the worktree root may have
an empty ``mini-swe-agent/`` placeholder while the real populated submodule
lives under the main checkout that owns the shared ``.git`` directory.

These helpers locate a usable ``mini-swe-agent/src`` directory and optionally
prepend it to ``sys.path`` so imports like ``import minisweagent`` work from
both normal checkouts and worktrees.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Optional


def _read_gitdir(repo_root: Path) -> Optional[Path]:
    """Resolve the gitdir referenced by ``repo_root/.git`` when it is a file."""
    git_marker = repo_root / ".git"
    if not git_marker.is_file():
        return None

    try:
        raw = git_marker.read_text(encoding="utf-8").strip()
    except OSError:
        return None

    prefix = "gitdir:"
    if not raw.lower().startswith(prefix):
        return None

    target = raw[len(prefix):].strip()
    gitdir = Path(target)
    if not gitdir.is_absolute():
        gitdir = (repo_root / gitdir).resolve()
    else:
        gitdir = gitdir.resolve()
    return gitdir


def discover_minisweagent_src(repo_root: Optional[Path] = None) -> Optional[Path]:
    """Return the best available ``mini-swe-agent/src`` path, if any.

    Search order:
    1. Current checkout/worktree root
    2. Main checkout that owns the shared ``.git`` directory (for worktrees)
    """
    repo_root = (repo_root or Path(__file__).resolve().parent).resolve()

    candidates: list[Path] = [repo_root / "mini-swe-agent" / "src"]

    gitdir = _read_gitdir(repo_root)
    if gitdir is not None:
        # Worktree layout: <main>/.git/worktrees/<name>
        if len(gitdir.parents) >= 3 and gitdir.parent.name == "worktrees":
            candidates.append(gitdir.parents[2] / "mini-swe-agent" / "src")
        # Direct checkout with .git file pointing elsewhere
        elif gitdir.name == ".git":
            candidates.append(gitdir.parent / "mini-swe-agent" / "src")

    seen = set()
    for candidate in candidates:
        candidate = candidate.resolve()
        if candidate in seen:
            continue
        seen.add(candidate)
        if candidate.exists() and candidate.is_dir():
            return candidate

    return None


def ensure_minisweagent_on_path(repo_root: Optional[Path] = None) -> Optional[Path]:
    """Ensure ``minisweagent`` is importable by prepending its src dir to sys.path.

    Returns the inserted/discovered path, or ``None`` if the package is already
    importable or no local source tree could be found.
    """
    if importlib.util.find_spec("minisweagent") is not None:
        return None

    src = discover_minisweagent_src(repo_root)
    if src is None:
        return None

    src_str = str(src)
    if src_str not in sys.path:
        sys.path.insert(0, src_str)
    return src
