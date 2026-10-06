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
from logging.handlers import RotatingFileHandler
from pathlib import Path

logger = logging.getLogger(__name__)

# How long a TERMed process gets before KILL; browsers exit well within this.
_REAP_GRACE_SECONDS = 3.0
# ``.git`` files (linked worktrees) are looked for this deep; lanes nest repo/tree/subtree.
_GIT_FILE_MAX_DEPTH = 4
# logs/scratch-prune.log: one line per entry deleted and per process signalled (#132401). Same
# fixed rotation as tool_calls.log.
_PRUNE_LOG_MAX_BYTES = 5 * 1024 * 1024
_PRUNE_LOG_BACKUPS = 3


def _open_prune_log(log_file: Path | None) -> logging.Logger:
    """Logger for one prune's records, with its own RotatingFileHandler on *log_file*, so a
    prune that runs at boot (before ``setup_logging()``) still leaves them on disk. Records also
    propagate through this module's logger to whatever logging is set up.

    One unregistered logger per prune, closed when it ends: no process holds the file open
    between prunes (an open handle blocks rollover on Windows), and prunes of two homes never
    share a handler. Without a *log_file* (test-runner temp roots) this module's logger is used.
    """
    if log_file is None:
        return logger
    try:
        log_file.parent.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        logger.warning("scratch prune: cannot create %s: %s", log_file.parent, exc)
        return logger
    audit = logging.Logger(logger.name, logging.INFO)
    audit.parent = logger
    handler = RotatingFileHandler(
        log_file, maxBytes=_PRUNE_LOG_MAX_BYTES, backupCount=_PRUNE_LOG_BACKUPS, encoding="utf-8", delay=True,
    )
    handler.setFormatter(logging.Formatter("%(asctime)s %(message)s"))
    audit.addHandler(handler)
    return audit


def _tree_bytes(path: Path) -> int:
    """Apparent size of *path* and everything under it, symlinks not followed."""
    try:
        total = os.lstat(path).st_size
    except OSError:
        return 0
    stack = [str(path)] if path.is_dir() and not path.is_symlink() else []
    while stack:
        try:
            with os.scandir(stack.pop()) as it:
                for child in it:
                    try:
                        total += child.stat(follow_symlinks=False).st_size
                    except OSError:
                        continue
                    if child.is_dir(follow_symlinks=False):
                        stack.append(child.path)
        except OSError:
            continue
    return total


def subtree_touched_since(path: Path, cutoff: float) -> bool:
    """True when *path* or anything beneath it has an mtime at or after *cutoff*.

    Stops at the first recent entry, so a live tree costs one hit and only a truly idle
    tree pays for the full walk (once, right before it is deleted). Symlinks are never
    followed: a link into the repo would make the target's activity keep the entry alive.
    An unreadable entry is kept: an incomplete scan cannot establish that it is idle.
    """
    try:
        if os.lstat(path).st_mtime >= cutoff:
            return True
        if not path.is_dir() or path.is_symlink():
            return False
    except OSError:
        return True
    stack = [str(path)]
    while stack:
        try:
            with os.scandir(stack.pop()) as it:
                for child in it:
                    try:
                        if child.stat(follow_symlinks=False).st_mtime >= cutoff:
                            return True
                    except OSError:
                        return True
                    if child.is_dir(follow_symlinks=False):
                        stack.append(child.path)
        except OSError:
            return True
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


def reap_processes_rooted_in(scratch_root: Path, doomed: list[Path], audit: logging.Logger = logger) -> int:
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
    victims: list[tuple[psutil.Process, str]] = []
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
            victims.append((proc, cwd_path))
    if not victims:
        return 0
    for proc, cwd_path in victims:
        try:
            name = proc.name()
            proc.terminate()
        except (psutil.Error, OSError):
            continue
        audit.info("scratch prune: sent TERM pid=%d name=%s cwd=%s", proc.pid, name, cwd_path)
    _, alive = psutil.wait_procs([proc for proc, _ in victims], timeout=_REAP_GRACE_SECONDS)
    for proc in alive:
        try:
            # A zombie already exited; only its parent has not collected it yet.
            if proc.status() == psutil.STATUS_ZOMBIE:
                continue
            proc.kill()
        except (psutil.Error, OSError):
            continue
        audit.info("scratch prune: sent KILL pid=%d (still running after TERM)", proc.pid)
    logger.info("scratch prune: reaped %d process(es) rooted in pruned entries", len(victims))
    return len(victims)


def _linked_worktree_repos(entry: Path) -> set[str]:
    """Repos whose linked worktrees live inside *entry* (``.git`` FILES, ``gitdir: <repo>/.git/worktrees/<n>``)."""
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
                            # <repo>/.git/worktrees/<name> -> <repo>
                            if gitdir.parent.name == "worktrees" and gitdir.parent.parent.name == ".git":
                                repos.add(str(gitdir.parent.parent.parent))
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


def prune_idle_entries(
    root: Path, max_idle_hours: float, skip_names: frozenset[str], log_file: Path | None = None,
) -> int:
    """Delete top-level entries of *root* with no write anywhere in their subtree for
    *max_idle_hours*, reaping processes and worktree registrations rooted in them first.
    Each removal and each signalled process is recorded in *log_file*. Returns the count
    actually removed."""
    audit = _open_prune_log(log_file)
    try:
        return _prune_idle_entries(root, max_idle_hours, skip_names, audit)
    finally:
        if audit is not logger:
            for handler in audit.handlers:
                handler.close()


def _prune_idle_entries(
    root: Path, max_idle_hours: float, skip_names: frozenset[str], audit: logging.Logger,
) -> int:
    cutoff = time.time() - max_idle_hours * 3600
    try:
        entries = [e for e in root.iterdir() if e.name not in skip_names]
    except OSError:
        return 0
    doomed = [e for e in entries if not subtree_touched_since(e, cutoff)]
    # Runs even with nothing to delete: orphans whose cwd was removed by an earlier pass
    # (or by hand) are found by the deleted-cwd rule, not by membership in ``doomed``.
    try:
        reap_processes_rooted_in(root, doomed, audit)
    except Exception as exc:  # psutil missing or restricted host: the deletion still proceeds
        logger.debug("scratch prune: process reap skipped: %s", exc, exc_info=True)
    if not doomed:
        return 0
    repos: set[str] = set()
    removed = 0
    for entry in doomed:
        size = _tree_bytes(entry)
        try:
            if entry.is_dir() and not entry.is_symlink():
                repos |= _linked_worktree_repos(entry)
                shutil.rmtree(entry, ignore_errors=True)
            else:
                entry.unlink()
        except OSError as exc:
            audit.info("scratch prune: could not remove %s: %s", entry, exc)
            continue
        # rmtree ignores errors, so only a path that is really gone counts as removed.
        if os.path.lexists(entry):
            audit.info("scratch prune: could not fully remove %s", entry)
            continue
        audit.info("scratch prune: removed %s (%d bytes)", entry, size)
        removed += 1
    release_git_worktrees(repos)
    return removed
