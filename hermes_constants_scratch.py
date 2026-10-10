"""Scratch-dir retention: idle detection plus the reaping an idle tree needs before it goes.

Deleting an idle ``cache/scratch/<entry>`` is not enough on its own: a lane's e2e run leaves
headless browsers whose cwd was inside the tree (they survived for days with a ``(deleted)``
cwd), and a repo whose linked worktree lived in the tree keeps a dangling registration until
someone runs ``git worktree prune``. Both are reaped here, right before ``rmtree``.
"""
from __future__ import annotations

import contextlib
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
# logs/scratch-prune.log: one line per entry deleted, per process signalled, and per
# entry rescued or vanished mid-prune (#132401). Same fixed rotation as tool_calls.log.
_PRUNE_LOG_MAX_BYTES = 5 * 1024 * 1024
_PRUNE_LOG_BACKUPS = 3
# Paths and process names go in with %r: a control character (a newline in a legal POSIX name)
# is written escaped, so it cannot split a record or forge one.


class _PruneLogHandler(RotatingFileHandler):
    """Rollover that tolerates a concurrent prune of the same home rotating first: its rename
    of ``scratch-prune.log`` can land between this handler's existence check and its own
    rename. The rotation is then already done, so the record goes to the new file instead of
    being dropped with a FileNotFoundError."""

    def doRollover(self) -> None:
        with contextlib.suppress(FileNotFoundError):
            super().doRollover()


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
    # backslashreplace: a legal POSIX name that is not UTF-8 is written escaped instead of
    # dropping the whole record.
    handler = _PruneLogHandler(
        log_file, maxBytes=_PRUNE_LOG_MAX_BYTES, backupCount=_PRUNE_LOG_BACKUPS, encoding="utf-8",
        errors="backslashreplace", delay=True,
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


_TOUCHED = "touched"
_SCAN_FAILED = "scan-failed"
_IDLE = "idle"


def subtree_touch_state(path: Path, cutoff: float) -> str:
    """Tri-state form of :func:`subtree_touched_since`: ``touched`` (recent write seen),
    ``scan-failed`` (unreadable — idle cannot be established, so the entry is kept, but
    a failed scan is not activity and is never reported as such), or ``idle``.
    Same walk, same stop-at-first-recent shortcut, same symlink rules."""
    try:
        if os.lstat(path).st_mtime >= cutoff:
            return _TOUCHED
        if not path.is_dir() or path.is_symlink():
            return _IDLE
    except OSError:
        return _SCAN_FAILED
    stack = [str(path)]
    while stack:
        try:
            with os.scandir(stack.pop()) as it:
                for child in it:
                    try:
                        if child.stat(follow_symlinks=False).st_mtime >= cutoff:
                            return _TOUCHED
                    except OSError:
                        return _SCAN_FAILED
                    if child.is_dir(follow_symlinks=False):
                        stack.append(child.path)
        except OSError:
            return _SCAN_FAILED
    return _IDLE


def subtree_touched_since(path: Path, cutoff: float) -> bool:
    """True when *path* or anything beneath it has an mtime at or after *cutoff*.

    Stops at the first recent entry, so a live tree costs one hit and only a truly idle
    tree pays for the full walk (once, right before it is deleted). Symlinks are never
    followed: a link into the repo would make the target's activity keep the entry alive.
    An unreadable entry is kept: an incomplete scan cannot establish that it is idle —
    so True here means "touched **or** uninspectable", and callers that need to tell the
    two apart (the mid-prune audit) use :func:`subtree_touch_state` instead.
    """
    return subtree_touch_state(path, cutoff) != _IDLE


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
        audit.info("scratch prune: sent TERM pid=%d name=%r cwd=%r", proc.pid, name, cwd_path)
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
    Each removal, each signalled process and each entry rescued mid-prune (touched
    again after selection) is recorded in *log_file*. Returns the count actually
    removed."""
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
        # Worktree scan first: a registration whose tree an earlier pass (or hand)
        # removed is stale even in a rescued entry, and the re-check below sits as
        # close to the delete as the loop allows.
        if entry.is_dir() and not entry.is_symlink():
            repos |= _linked_worktree_repos(entry)
        # Last-moment re-validation (#132401 C1/F1): ``doomed`` is a snapshot from
        # before the reap ran, and a writer whose cwd is outside scratch — invisible
        # to the reap — can land fresh work in that window. The snapshot is a
        # candidate list, never a verdict: anything touched since selection is
        # rescued here, not deleted. A re-scan that cannot read the entry is also
        # fail-closed (the entry is kept), but a failed scan is not activity — the
        # audit names the failed re-scan instead of inventing a touch that did not
        # happen, so operators can tell an uninspectable candidate from verified
        # recent activity (P3 on #134173).
        state = subtree_touch_state(entry, cutoff)
        if state != _IDLE:
            # Only a missing path is a vanish: a stat that raises PermissionError means
            # the entry is there but unreadable, and reporting it as vanished would be
            # the same invented reason the audit exists to prevent.
            try:
                os.lstat(entry)
                entry_gone = False
            except FileNotFoundError:
                entry_gone = True
            except OSError:
                entry_gone = False
            if entry_gone:
                audit.info("scratch prune: entry %r vanished since selection", os.fspath(entry))
            elif state == _TOUCHED:
                audit.info(
                    "scratch prune: rescued %r — touched since selection, kept",
                    os.fspath(entry),
                )
            else:
                audit.info(
                    "scratch prune: kept %r — activity re-scan failed, left untouched",
                    os.fspath(entry),
                )
            continue
        size = _tree_bytes(entry)
        try:
            if entry.is_dir() and not entry.is_symlink():
                shutil.rmtree(entry, ignore_errors=True)
            else:
                entry.unlink()
        except OSError as exc:
            audit.info("scratch prune: could not remove %r: %s", os.fspath(entry), exc)
            continue
        # rmtree ignores errors, so only a path that is really gone counts as removed.
        # Only an explicit lstat proves absence: lexists() swallows OSError from lstat,
        # so an entry that stayed (rmtree left residue) but cannot be inspected would be
        # counted and recorded as removed — the same invented reason the vanish rule
        # above refuses to invent (P2 on #134173).
        try:
            os.lstat(entry)
            audit.info("scratch prune: could not fully remove %r", os.fspath(entry))
            continue
        except FileNotFoundError:
            pass
        except OSError:
            audit.info("scratch prune: could not confirm removal of %r", os.fspath(entry))
            continue
        audit.info("scratch prune: removed %r (%d bytes)", os.fspath(entry), size)
        removed += 1
    release_git_worktrees(repos)
    return removed
