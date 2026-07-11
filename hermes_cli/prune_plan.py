"""Ancestry-gated prune planning for task worktrees.

REQ-028 (saga decision 0004): a task worktree may be removed only when its
branch is fully merged into the repo default branch (merge-base ancestry,
not name matching), its tree is clean, AND it is at least one day old.
Unmerged or dirty worktrees are NEVER pruned — they are reported instead.
Merged+pruned branches are deleted with ``git branch -d`` (safe delete)
only; ``git worktree remove`` is never forced.

:func:`plan_worktree_prune` is the pure decision function (no git, no I/O)
so the gate is unit-testable in isolation; :func:`prune_task_worktrees`
does the scanning/removal around it.
"""

from __future__ import annotations

import subprocess
import time
from pathlib import Path

from hermes_cli.worktree_safety import (
    detect_default_branch,
    is_branch_unmerged,
    is_worktree_dirty,
)

__all__ = ["plan_worktree_prune", "prune_task_worktrees"]

_REQUIRED_KEYS = ("id", "merged", "dirty", "age_days")

_GIT_TIMEOUT = 60


def plan_worktree_prune(worktrees: list[dict]) -> list[str]:
    """Return the sorted ids of worktree entries that are safe to prune.

    Each entry MUST have keys ``id`` (str), ``merged`` (bool), ``dirty``
    (bool), ``age_days`` (int|float); a missing key raises
    :class:`ValueError` naming the entry's ``id`` when present, else its
    index. An entry is pruned iff ``merged`` is True AND ``dirty`` is
    False AND ``age_days >= 1`` (exactly 1 day IS pruned). Extra keys are
    ignored; the input is not mutated and its order does not matter.
    """
    planned: list[str] = []
    for index, entry in enumerate(worktrees):
        missing = [k for k in _REQUIRED_KEYS if k not in entry]
        if missing:
            label = entry["id"] if "id" in entry else str(index)
            raise ValueError(
                f"worktree entry {label} is missing required key(s): "
                f"{', '.join(missing)}"
            )
        if entry["merged"] is True and entry["dirty"] is False and entry["age_days"] >= 1:
            planned.append(entry["id"])
    return sorted(planned)


def _run_git(repo: Path, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["git", "-C", str(repo), *args],
        capture_output=True,
        text=True,
        timeout=_GIT_TIMEOUT,
        check=False,
    )


def _worktree_branch(wt: Path) -> str | None:
    """Current branch of ``wt``, or ``None`` when detached / not a checkout."""
    try:
        result = _run_git(wt, "branch", "--show-current")
    except Exception:
        return None
    if result.returncode != 0:
        return None
    branch = (result.stdout or "").strip()
    return branch or None


def prune_task_worktrees(
    repo_root: Path, *, dry_run: bool = True, now: float | None = None
) -> dict:
    """Scan ``<repo_root>/.worktrees/*`` and prune the ancestry-safe entries.

    Per entry: ``merged`` = branch fully merged into the repo default branch
    (detached HEAD, missing branch, or no default branch → treated as
    unmerged, i.e. NEVER prunable), ``dirty`` = ``git status --porcelain``
    non-empty, ``age_days`` = directory mtime age. The prune decision is
    delegated to :func:`plan_worktree_prune`.

    With ``dry_run=True`` (the default) nothing is removed. With
    ``dry_run=False`` planned worktrees are removed via ``git worktree
    remove`` (never ``--force``) and their branches deleted with ``git
    branch -d`` (safe delete only). A git failure on one entry is recorded
    under ``errors`` and never aborts the others.

    Returns ``{"planned", "removed", "kept_unmerged", "kept_dirty",
    "kept_young", "errors"}``.
    """
    if now is None:
        now = time.time()
    report: dict = {
        "planned": [],
        "removed": [],
        "kept_unmerged": [],
        "kept_dirty": [],
        "kept_young": [],
        "errors": {},
    }
    wt_root = Path(repo_root) / ".worktrees"
    if not wt_root.is_dir():
        return report

    base = detect_default_branch(repo_root)
    entries: list[dict] = []
    for wt_dir in sorted(p for p in wt_root.iterdir() if p.is_dir()):
        wt_id = wt_dir.name
        try:
            branch = _worktree_branch(wt_dir)
            if branch is None or base is None:
                # Detached HEAD / missing branch / no default branch: we
                # cannot prove ancestry, so the entry is NEVER prunable.
                merged = False
            else:
                merged = not is_branch_unmerged(repo_root, branch, base)
            entries.append(
                {
                    "id": wt_id,
                    "merged": merged,
                    "dirty": is_worktree_dirty(wt_dir),
                    "age_days": (now - wt_dir.stat().st_mtime) / 86400.0,
                    "branch": branch,
                    "path": wt_dir,
                }
            )
        except Exception as exc:  # one bad entry must not abort the scan
            report["errors"][wt_id] = str(exc)

    planned = plan_worktree_prune(entries)
    report["planned"] = planned
    planned_set = set(planned)
    for entry in entries:
        if entry["id"] in planned_set:
            continue
        if not entry["merged"]:
            report["kept_unmerged"].append(entry["id"])
        elif entry["dirty"]:
            report["kept_dirty"].append(entry["id"])
        else:
            report["kept_young"].append(entry["id"])

    if dry_run:
        return report

    by_id = {e["id"]: e for e in entries}
    for wt_id in planned:
        entry = by_id[wt_id]
        try:
            result = _run_git(
                Path(repo_root), "worktree", "remove", str(entry["path"])
            )
            if result.returncode != 0:
                stderr = (result.stderr or result.stdout or "").strip()
                report["errors"][wt_id] = f"git worktree remove failed: {stderr}"
                continue
            branch = entry["branch"]
            if branch and branch != base:
                result = _run_git(Path(repo_root), "branch", "-d", branch)
                if result.returncode != 0:
                    stderr = (result.stderr or result.stdout or "").strip()
                    report["errors"][wt_id] = f"git branch -d {branch} failed: {stderr}"
                    # The worktree itself was removed; still count it.
            report["removed"].append(wt_id)
        except Exception as exc:
            report["errors"][wt_id] = str(exc)
    return report
