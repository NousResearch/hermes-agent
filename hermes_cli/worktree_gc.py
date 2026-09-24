"""On-demand worktree + branch reclaim (``hermes worktree`` / ``/worktree prune``).

The startup pruner (``cli._prune_stale_worktrees``) is conservative and silent — clean, fully
merged scratch past an age tier only. This module also reclaims trees whose only "dirt" is
untracked scratch (archived first) and branches whose content is on upstream.
"""

from __future__ import annotations

import contextlib
import logging
import os
import re
import shutil
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

logger = logging.getLogger(__name__)

# Branches never considered for deletion, in any mode.
_PROTECTED_BRANCHES = {"main", "master", "develop", "dev", "trunk"}

# Trees owned by another lifecycle (kanban dispatcher gc) — never touched.
_KANBAN_RE = re.compile(r"^t_[0-9a-f]+$")

# Bounded cherry probe: a branch this far ahead of upstream is a stale-base
# lane, not merged scratch; checking it is expensive and it stays preserved.
_MAX_CHERRY_AHEAD = 50


@dataclass
class TreeRecord:
    name: str
    path: str
    branch: str
    age_days: float
    size_mb: Optional[int]
    verdict: str          # reap | reap-archive | keep
    reason: str
    untracked: List[str] = field(default_factory=list)


@dataclass
class BranchRecord:
    name: str
    verdict: str          # delete | keep
    reason: str


def _run(cmd: list, timeout: int, cwd: Optional[str] = None) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace",
                          timeout=timeout, cwd=cwd)


def _git(args: list, cwd: str, timeout: int = 15) -> subprocess.CompletedProcess:
    """Run git, translating timeouts into returncode 124. Every verdict fails safe toward "keep"
    on nonzero, so a slow ``git cherry`` on a huge repo degrades to keep instead of aborting the
    audit mid-list."""
    try:
        return _run(["git", *args], timeout, cwd)
    except subprocess.TimeoutExpired:
        return subprocess.CompletedProcess(args=["git", *args], returncode=124, stdout="",
                                           stderr=f"timeout after {timeout}s")


def _tree_size_mb(path: Path, timeout: int = 30) -> Optional[int]:
    """Cheap directory size via ``du -sm`` — best-effort, None on failure."""
    try:
        result = _run(["du", "-sm", str(path)], timeout)
        return int(result.stdout.split()[0]) if result.returncode == 0 and result.stdout.strip() else None
    except Exception:
        return None


def _dirty_split(path: str) -> tuple[bool, List[str]]:
    """(has_tracked_modifications, untracked_paths) — tracked = real work, untracked = archivable."""
    try:
        result = _git(["status", "--porcelain"], cwd=path, timeout=10)
        if result.returncode != 0:
            return True, []  # fail safe: treat as real work
        lines = [line for line in result.stdout.splitlines() if line.strip()]
        untracked = [line[3:].strip() for line in lines if line.startswith("??")]
        return len(untracked) != len(lines), untracked
    except Exception:
        return True, []


def _archive_untracked(tree: Path, untracked: List[str]) -> Optional[Path]:
    """Copy untracked files out of a doomed tree; None on any failure (caller must then keep)."""
    stamp = time.strftime("%Y%m%d-%H%M%S")
    dest = Path.home() / ".hermes" / "archive" / "worktree-prune" / f"{tree.name}-{stamp}"
    try:
        for rel in untracked:
            src = tree / rel
            if not src.exists() or src.is_symlink():
                continue
            (dest / rel).parent.mkdir(parents=True, exist_ok=True)
            if src.is_dir():
                shutil.copytree(src, dest / rel, dirs_exist_ok=True)
            else:
                shutil.copy2(src, dest / rel)
        return dest if dest.exists() else None
    except Exception as exc:
        logger.warning("Could not archive untracked files from %s: %s", tree, exc)
        return None


def _classify_tree(_ops, repo_root: str, entry: Path, merge_cache, remote_heads) -> tuple[str, str, List[str]]:
    """Return (verdict, reason, untracked) for one tree under ``.worktrees/``."""
    path = str(entry)
    if _KANBAN_RE.match(entry.name):
        return "keep", "kanban task tree (owned by kanban gc)", []
    if _ops._worktree_lock_is_live(repo_root, path, timeout=5) == "live":
        return "keep", "in use by a running hermes session", []
    tracked_dirty, untracked = _dirty_split(path)
    if tracked_dirty:
        return "keep", "uncommitted tracked changes (real work)", []
    archive_note = f"{len(untracked)} untracked file(s) will be archived"
    if _ops._worktree_has_unpushed_commits(path, timeout=5) and not _ops._worktree_commits_all_merged_upstream(
        path, timeout=30, cache=merge_cache, max_ahead=_MAX_CHERRY_AHEAD):
        # Pushed-branch tier: single-branch fetch refspecs (managed-install default) leave pushed
        # PR branches with no refs/remotes/* entry, so `git log HEAD --not --remotes` reads them
        # as unpushed forever. A head EXACTLY matching the remote branch has nothing origin lacks.
        if not _ops._worktree_branch_pushed_exact(path, remote_heads, timeout=10):
            return "keep", "unpushed commits not found upstream", []
        if untracked:
            return "reap-keep-branch", f"pushed to origin (open-PR lane); branch kept; {archive_note}", untracked
        return "reap-keep-branch", "pushed to origin (open-PR lane); branch kept", []
    if untracked:
        return "reap-archive", f"merged/pushed; {archive_note}", untracked
    return "reap", "clean and fully merged/pushed", []


def audit_worktrees(repo_root: str, *, with_sizes: bool = True) -> List[TreeRecord]:
    """Classify every tree under ``.worktrees/`` without mutating anything."""
    from hermes_cli import worktree_ops as _ops
    worktrees_dir = Path(repo_root) / ".worktrees"
    if not worktrees_dir.exists():
        return []

    if _ops._repo_is_shallow(repo_root):
        _ops._deepen_shallow_repo(repo_root)

    merge_cache = _ops._load_worktree_merge_cache()
    cache_size_before = len(merge_cache)
    # One ls-remote for the whole sweep; None (offline) degrades pushed-tier verdicts to keep.
    remote_heads = _ops._fetch_remote_branch_heads(repo_root)

    now = time.time()
    records: List[TreeRecord] = []
    for entry in sorted(worktrees_dir.iterdir()):
        if not entry.is_dir():
            continue
        try:
            age_days = (now - entry.stat().st_mtime) / 86400.0
        except Exception:
            continue
        try:
            branch = _git(["branch", "--show-current"], cwd=str(entry), timeout=5).stdout.strip()
        except Exception:
            branch = ""
        verdict, reason, untracked = _classify_tree(_ops, repo_root, entry, merge_cache, remote_heads)
        records.append(TreeRecord(
            name=entry.name, path=str(entry), branch=branch,
            age_days=age_days, size_mb=_tree_size_mb(entry) if with_sizes else None,
            verdict=verdict, reason=reason, untracked=untracked))

    if len(merge_cache) != cache_size_before:
        _ops._save_worktree_merge_cache(merge_cache)
    return records


_REAP_VERDICTS = {"reap", "reap-archive", "reap-keep-branch"}


def reclaim_worktrees(
    repo_root: str, *, dry_run: bool = False, records: Optional[List[TreeRecord]] = None
) -> List[str]:
    """Remove every reap-verdict tree from a frozen audit list — never re-globs inside the
    destructive loop, so trees created by concurrent sessions after the audit are out of scope."""
    if records is None:
        records = audit_worktrees(repo_root, with_sizes=False)
    actions: List[str] = []
    for record in records:
        if record.verdict not in _REAP_VERDICTS:
            continue
        if dry_run:
            actions.append(f"would remove {record.name} ({record.reason})")
            continue

        entry = Path(record.path)
        if record.untracked:
            archive = _archive_untracked(entry, record.untracked)
            if archive is None:
                actions.append(f"kept {record.name} (archive of untracked files failed)")
                continue
            actions.append(f"archived {len(record.untracked)} untracked file(s) → {archive}")

        # Dead-pid locks must be unlocked or `remove --force` refuses.
        with contextlib.suppress(Exception):
            _git(["worktree", "unlock", record.path], cwd=repo_root, timeout=10)
        try:
            remove_result = _git(["worktree", "remove", record.path, "--force"], cwd=repo_root, timeout=30)
            if remove_result.returncode != 0:
                actions.append(f"failed to remove {record.name}: {remove_result.stderr.strip()}")
                continue
            if record.verdict == "reap-keep-branch":
                actions.append(f"removed {record.name} (branch {record.branch} kept — pushed open-PR lane)")
                continue
            if record.branch and record.branch not in _PROTECTED_BRANCHES:
                _git(["branch", "-D", record.branch], cwd=repo_root, timeout=10)
            actions.append(f"removed {record.name}")
        except Exception as exc:
            actions.append(f"failed to remove {record.name}: {exc}")

    if not dry_run:
        with contextlib.suppress(Exception):
            _git(["worktree", "prune"], cwd=repo_root, timeout=15)
    return actions


def audit_branches(repo_root: str) -> List[BranchRecord]:
    """Classify EVERY local branch: deletable when fully merged OR every commit is patch-equivalent
    upstream (``git cherry``) and not checked out. The gate is content reachability, not name."""
    from hermes_cli import worktree_ops as _ops
    if _ops._repo_is_shallow(repo_root):
        _ops._deepen_shallow_repo(repo_root)

    def _lines(result) -> List[str]:
        return [b.strip() for b in result.stdout.splitlines() if b.strip()]

    upstream = next(
        (c for c in ("origin/HEAD", "origin/main", "origin/master")
         if _git(["rev-parse", "--verify", "--quiet", c], cwd=repo_root, timeout=5).returncode == 0),
        None)
    if upstream is None:
        return []

    result = _git(["branch", "--format=%(refname:short)"], cwd=repo_root, timeout=10)
    if result.returncode != 0:
        return []
    branches = _lines(result)

    wt = _git(["worktree", "list", "--porcelain"], cwd=repo_root, timeout=10)
    active = {
        line.removeprefix("branch refs/heads/").strip()
        for line in wt.stdout.splitlines() if line.startswith("branch refs/heads/")}
    merged = set(_lines(_git(["branch", "--merged", upstream, "--format=%(refname:short)"], cwd=repo_root, timeout=15)))

    def _classify_branch(branch: str) -> BranchRecord:
        if branch in _PROTECTED_BRANCHES or branch in active:
            return BranchRecord(branch, "keep", "protected or checked out")
        if branch in merged:
            return BranchRecord(branch, "delete", "fully merged into " + upstream)
        # Rebase merges rewrite SHAs, so --merged misses them; cherry patch-equivalence catches
        # the dominant leak. Bounded: a branch far ahead is a stale-base lane, keep it.
        ahead = _git(["rev-list", "--count", f"{upstream}..{branch}"], cwd=repo_root, timeout=10)
        try:
            ahead_count = int(ahead.stdout.strip() or "0")
        except ValueError:
            ahead_count = _MAX_CHERRY_AHEAD + 1
        if ahead_count == 0:
            return BranchRecord(branch, "delete", "no commits beyond " + upstream)
        if ahead_count > _MAX_CHERRY_AHEAD:
            return BranchRecord(branch, "keep", f"{ahead_count} commits ahead (stale-base lane)")
        cherry = _git(["cherry", upstream, branch], cwd=repo_root, timeout=30)
        if cherry.returncode != 0:
            return BranchRecord(branch, "keep", "could not verify (git cherry failed)")
        lines = _lines(cherry)
        if lines and all(ln.startswith("-") for ln in lines):
            return BranchRecord(branch, "delete", "all commits patch-equivalent upstream")
        unique = sum(1 for ln in lines if ln.startswith("+"))
        return BranchRecord(branch, "keep", f"{unique} unique commit(s) not upstream")

    # Read-only, so parallel: hundreds of local branches × ~0.2-1s cherry probes would be minutes.
    import concurrent.futures
    workers = max(1, min(8, (os.cpu_count() or 4), len(branches)))
    if workers > 1:
        try:
            with concurrent.futures.ThreadPoolExecutor(max_workers=workers, thread_name_prefix="hermes-branch-gc") as pool:
                return list(pool.map(_classify_branch, branches))
        except Exception:
            pass
    return [_classify_branch(b) for b in branches]


def reclaim_branches(
    repo_root: str, *, dry_run: bool = False, records: Optional[List[BranchRecord]] = None
) -> List[str]:
    """Delete every delete-verdict branch from a frozen audit list."""
    if records is None:
        records = audit_branches(repo_root)
    actions: List[str] = []
    for record in records:
        if record.verdict != "delete":
            continue
        if dry_run:
            actions.append(f"would delete branch {record.name} ({record.reason})")
            continue
        result = _git(["branch", "-D", record.name], cwd=repo_root, timeout=10)
        actions.append(
            f"deleted branch {record.name}" if result.returncode == 0
            else f"failed to delete {record.name}: {result.stderr.strip()}")
    return actions


def worktrees_summary(repo_root: str) -> tuple[int, Optional[int]]:
    """(tree_count, total_size_mb) for the escalation notice; size is best-effort with a timeout."""
    worktrees_dir = Path(repo_root) / ".worktrees"
    if not worktrees_dir.exists():
        return 0, None
    try:
        count = sum(1 for e in worktrees_dir.iterdir() if e.is_dir())
    except Exception:
        return 0, None
    return count, _tree_size_mb(worktrees_dir, timeout=20)
