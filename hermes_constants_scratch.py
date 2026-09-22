"""Scratch-dir retention: idle detection plus the reaping an idle tree needs before it goes.

Deleting an idle ``cache/scratch/<entry>`` is not enough on its own: a lane's e2e run leaves
headless browsers whose cwd was inside the tree (they survived for days with a ``(deleted)``
cwd), and a repo whose linked worktree lived in the tree keeps a dangling registration until
someone runs ``git worktree prune``. Both are reaped here, right before ``rmtree``.
"""
from __future__ import annotations

import logging
import os
import shutil
import subprocess
import time
from pathlib import Path

logger = logging.getLogger(__name__)

# How long a TERMed process gets before KILL; browsers exit well within this.
_REAP_GRACE_SECONDS = 3.0
# ``.git`` files (linked worktrees) are looked for this deep; lanes nest repo/tree/subtree.
_GIT_FILE_MAX_DEPTH = 4


def subtree_touched_since(path: Path, cutoff: float) -> bool:
    """True when *path* or anything beneath it has an mtime at or after *cutoff*.

    Stops at the first recent entry, so a live tree costs one hit and only a truly idle
    tree pays for the full walk (once, right before it is deleted). Symlinks are never
    followed: a link into the repo would make the target's activity keep the entry alive.
    """
    try:
        if os.lstat(path).st_mtime >= cutoff:
            return True
    except OSError:
        return False
    if not path.is_dir() or path.is_symlink():
        return False
    stack = [str(path)]
    while stack:
        try:
            with os.scandir(stack.pop()) as it:
                for child in it:
                    try:
                        if child.stat(follow_symlinks=False).st_mtime >= cutoff:
                            return True
                    except OSError:
                        continue
                    if child.is_dir(follow_symlinks=False):
                        stack.append(child.path)
        except OSError:
            continue
    return False


def _under(path: str, root: str) -> bool:
    return path == root or path.startswith(root.rstrip(os.sep) + os.sep)


def _own_lineage() -> set[int]:
    """This process and its ancestors: never reap the shell that is running the prune."""
    import psutil

    pids: set[int] = set()
    try:
        proc = psutil.Process()
        while proc is not None and proc.pid not in pids:
            pids.add(proc.pid)
            proc = proc.parent()
    except (psutil.Error, OSError):
        pass
    return pids


def reap_processes_rooted_in(scratch_root: Path, doomed: list[Path]) -> int:
    """TERM (then KILL) same-user processes whose cwd is inside an entry about to be
    pruned, or is a path under *scratch_root* that no longer exists. Returns the count.

    Only the cwd is consulted: a process merely holding a file open inside scratch (an
    editor, a log tail) is not ours to kill, but one *living* in a directory we are
    about to delete, or in one already gone, has nothing left to run for.
    """
    import psutil

    root = os.path.realpath(str(scratch_root))
    targets = [os.path.realpath(str(p)) for p in doomed]
    skip = _own_lineage()
    uid = os.getuid() if hasattr(os, "getuid") else None
    victims: list[psutil.Process] = []
    for proc in psutil.process_iter(["pid"]):
        if proc.pid in skip:
            continue
        try:
            if uid is not None and proc.uids().real != uid:
                continue
            cwd = proc.cwd()
        except (psutil.Error, OSError):
            continue
        if not cwd:
            continue
        deleted = cwd.endswith(" (deleted)")
        cwd_path = cwd[: -len(" (deleted)")] if deleted else cwd
        if not _under(cwd_path, root):
            continue
        if deleted or not os.path.exists(cwd_path) or any(_under(cwd_path, t) for t in targets):
            victims.append(proc)
    if not victims:
        return 0
    for proc in victims:
        try:
            proc.terminate()
        except (psutil.Error, OSError):
            continue
    _, alive = psutil.wait_procs(victims, timeout=_REAP_GRACE_SECONDS)
    for proc in alive:
        try:
            proc.kill()
        except (psutil.Error, OSError):
            continue
    logger.info("scratch prune: reaped %d process(es) rooted in pruned entries", len(victims))
    return len(victims)


def _linked_worktree_repos(entry: Path) -> set[str]:
    """Common Git directories whose linked worktrees live inside *entry*."""
    repos: set[str] = set()
    stack = [(str(entry), 0)]
    while stack:
        current, depth = stack.pop()
        try:
            with os.scandir(current) as it:
                for child in it:
                    if child.name == ".git" and child.is_file(follow_symlinks=False):
                        try:
                            line = Path(child.path).read_text(encoding="utf-8", errors="replace").strip()
                        except OSError:
                            continue
                        if line.startswith("gitdir:"):
                            gitdir = Path(line[len("gitdir:"):].strip())
                            if not gitdir.is_absolute():
                                gitdir = Path(child.path).parent / gitdir
                            # Prune from the common Git directory: it need not be
                            # named .git (bare repos and separate git dirs).
                            if gitdir.parent.name == "worktrees":
                                repos.add(os.path.abspath(gitdir.parent.parent))
                    elif child.is_dir(follow_symlinks=False) and depth < _GIT_FILE_MAX_DEPTH \
                            and child.name not in ("node_modules", ".venv", "venv"):
                        stack.append((child.path, depth + 1))
        except OSError:
            continue
    return repos


def release_git_worktrees(repos: set[str]) -> None:
    """``git worktree prune`` in each repo: drops registrations whose tree we just deleted."""
    for repo in sorted(repos):
        if not os.path.isdir(repo):
            continue
        try:
            subprocess.run(
                ["git", "-C", repo, "worktree", "prune"],
                stdin=subprocess.DEVNULL, capture_output=True, text=True, encoding="utf-8", errors="replace",
                timeout=15, check=False,
            )
        except (OSError, subprocess.SubprocessError) as exc:
            logger.debug("git worktree prune in %s failed: %s", repo, exc)


def prune_idle_entries(root: Path, max_idle_hours: float, skip_names: frozenset[str]) -> int:
    """Delete top-level entries of *root* with no write anywhere in their subtree for
    *max_idle_hours*, reaping processes and worktree registrations rooted in them first.
    Returns the count removed."""
    cutoff = time.time() - max_idle_hours * 3600
    try:
        entries = [e for e in root.iterdir() if e.name not in skip_names]
    except OSError:
        return 0
    doomed = [e for e in entries if not subtree_touched_since(e, cutoff)]
    # Runs even with nothing to delete: orphans whose cwd was removed by an earlier pass
    # (or by hand) are found by the deleted-cwd rule, not by membership in ``doomed``.
    try:
        reap_processes_rooted_in(root, doomed)
    except Exception as exc:  # psutil missing or restricted host: the deletion still proceeds
        logger.debug("scratch prune: process reap skipped: %s", exc)
    if not doomed:
        return 0
    repos: set[str] = set()
    removed = 0
    for entry in doomed:
        try:
            if entry.is_dir() and not entry.is_symlink():
                repos |= _linked_worktree_repos(entry)
                shutil.rmtree(entry, ignore_errors=True)
            else:
                entry.unlink()
            removed += 1
        except OSError:
            continue
    release_git_worktrees(repos)
    return removed
