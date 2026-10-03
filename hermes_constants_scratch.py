"""Scratch-dir retention: idle detection plus the reaping an idle tree needs before it goes.

Deleting an idle ``cache/scratch/<entry>`` is not enough on its own: a lane's e2e run leaves
headless browsers whose cwd was inside the tree (they survived for days with a ``(deleted)``
cwd), and a repo whose linked worktree lived in the tree keeps a dangling registration until
someone runs ``git worktree prune``. Both are reaped here, right before ``rmtree``.

Every departure leaves a record: each pruned entry logs its name, its newest mtime, and its
size where cheap, each reaped process its pid and cmdline, and the boot caller logs the
summary count instead of dropping it. A delete with no trace was the #132401 lesson: the
prune itself is policy, the silence was the defect.
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

    Thin alias for the first element of :func:`_scan_entry_idleness` — one walk is the
    primitive, the audit fields ride the same stat calls. See there for the exact
    semantics (early stop, no symlink following, unreadable = kept).
    """
    return _scan_entry_idleness(path, cutoff)[0]


def _scan_entry_idleness(path: Path, cutoff: float) -> tuple[bool, float | None, int | None]:
    """One walk, three answers: (touched_since_cutoff, newest_mtime, bytes).

    Early-stops at the first recent mtime — a live tree costs one hit (bytes stay
    ``None``: untouched entries are not audited). A walk that completes returns the
    newest mtime seen and the byte total collected from the *same* stat calls — the
    audit record never costs a second walk of a doomed tree, the #132401 review's
    boot-cost objection answered by construction. Symlinks are never followed: a link
    into the repo would make the target's activity keep the entry alive. An unreadable
    entry reports ``(True, None, None)``: an incomplete scan cannot establish that it
    is idle, so it is kept (the pinned scan-failure semantics). The bytes figure is
    approximate by design: file bytes plus the root entry's own inode size.
    """
    try:
        st = os.lstat(path)
    except OSError:
        return True, None, None
    if st.st_mtime >= cutoff:
        return True, None, None
    if not path.is_dir() or path.is_symlink():
        return False, st.st_mtime, st.st_size
    newest = st.st_mtime
    total = st.st_size
    stack = [str(path)]
    while stack:
        try:
            with os.scandir(stack.pop()) as it:
                for child in it:
                    try:
                        cst = child.stat(follow_symlinks=False)
                    except OSError:
                        return True, None, None
                    if cst.st_mtime >= cutoff:
                        return True, None, None
                    if cst.st_mtime > newest:
                        newest = cst.st_mtime
                    if child.is_dir(follow_symlinks=False):
                        stack.append(child.path)
                    else:
                        total += cst.st_size
        except OSError:
            return True, None, None
    return False, newest, total


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
    # Evidence is captured BEFORE the kill: a terminated process's cmdline is gone,
    # so the audit record reads from this snapshot, not from post-mortem queries.
    evidence: list[tuple[int, str]] = []
    for proc in victims:
        try:
            evidence.append((proc.pid, " ".join(proc.cmdline() or [])[:200]))
        except (psutil.Error, OSError):
            evidence.append((proc.pid, "<unavailable>"))
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
    for pid, cmdline in evidence:
        logger.info("scratch prune: reaped pid=%d cmdline=%s", pid, cmdline)
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


def prune_idle_entries(root: Path, max_idle_hours: float, skip_names: frozenset[str]) -> int:
    """Delete top-level entries of *root* with no write anywhere in their subtree for
    *max_idle_hours*, reaping processes and worktree registrations rooted in them first.
    Returns the count removed.

    Every departure leaves a per-entry record at INFO — name, newest mtime, and size —
    so a delete is never silent (#132401): the count alone hid five multi-day work
    products being destroyed with no log, no quarantine, no trace. The reap detail
    (pid + cmdline) and the boot caller's summary log alongside.
    """
    cutoff = time.time() - max_idle_hours * 3600
    try:
        entries = [e for e in root.iterdir() if e.name not in skip_names]
    except OSError:
        return 0
    doomed: list[tuple[Path, float | None, int | None]] = []
    for e in entries:
        touched, newest, bytes_ = _scan_entry_idleness(e, cutoff)
        if not touched:
            doomed.append((e, newest, bytes_))
    # Runs even with nothing to delete: orphans whose cwd was removed by an earlier pass
    # (or by hand) are found by the deleted-cwd rule, detail via the reap's own record.
    try:
        reap_processes_rooted_in(root, [e for e, _, _ in doomed])
    except Exception as exc:  # psutil missing or restricted host: the deletion still proceeds
        logger.debug("scratch prune: process reap skipped: %s", exc)
    if not doomed:
        return 0
    repos: set[str] = set()
    removed = 0
    for entry, newest, bytes_ in doomed:
        kind = "dir" if (entry.is_dir() and not entry.is_symlink()) else "file"
        try:
            if kind == "dir":
                repos |= _linked_worktree_repos(entry)
                shutil.rmtree(entry, ignore_errors=True)
            else:
                entry.unlink()
        except OSError as exc:
            # Attempted, failed outright (permissions, vanished mid-run): the record
            # is the audit; the count only ever claims confirmed removals.
            logger.info(
                "scratch prune: removal failed entry=%r kind=%s error=%s",
                entry.name, kind, exc,
            )
            continue
        if entry.exists():
            # ``rmtree(ignore_errors=True)`` can leave residue — a partial removal is
            # not a removal. Recorded, never counted (#132401 review round 2).
            logger.info(
                "scratch prune: removal left residue entry=%r kind=%s bytes=%d newest_mtime=%.0f",
                entry.name, kind, bytes_ or 0, newest or 0.0,
            )
            continue
        logger.info(
            "scratch prune: removed entry=%r kind=%s bytes=%d newest_mtime=%.0f",
            entry.name, kind, bytes_ or 0, newest or 0.0,
        )
        removed += 1
    release_git_worktrees(repos)
    return removed
