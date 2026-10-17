"""Snapshot-then-delete sweep for stale worktrees (``worktrees.archive_dirty``, default on).

The classic pruner (``worktree_ops._prune_stale_worktree_trees``) never removes a dirty or
unpushed tree at any age, so leftovers of finished lanes pile up forever. This sweep keeps its
tiers and classifier, then removes an idle dirty/unpushed tree only after archiving its full
working state to ``refs/hermes/archive/<tree>`` (``worktree_archive``); a failed archive keeps
the tree. Never touched, at any age or budget: Kanban ``t_<hex>`` trees, trees whose ``hermes``
lock owner still runs, trees any live process sits in (cwd or open file), and trees with file or
ref activity younger than their tier. With ``archive_dirty: false`` and no size budget the
classic pruner runs unchanged.

``plan_sweep`` is read-only (``hermes worktree prune --dry-run`` prints it); ``execute_plan``
applies a frozen plan.
"""

from __future__ import annotations

import logging
import os
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

logger = logging.getLogger("cli")

_DEFAULTS = {"archive_dirty": True, "archive_max_file_mb": 50, "archive_retention_days": 30, "max_total_gb": 0}
_PROTECTED_BRANCHES = {"main", "master", "develop", "dev", "trunk"}
_ARCHIVE_REASONS = {"dirty": "uncommitted changes", "unpushed": "unpushed commits"}


@dataclass
class Decision:
    name: str
    path: str
    action: str   # remove | remove-keep-branch | archive-remove | skip-* | keep-*
    reason: str
    age_hours: float
    branch: Optional[str] = None
    mtime: float = 0.0
    force: bool = False
    verdict: str = ""
    lock_state: Optional[str] = None
    size_mb: Optional[int] = None

    def as_dict(self) -> dict:
        return asdict(self)


def sweep_settings() -> dict:
    """``worktrees.*`` from the active profile's config (defaults on any read failure)."""
    settings = dict(_DEFAULTS)
    try:
        from hermes_cli.config import load_config_readonly
        section = load_config_readonly().get("worktrees") or {}
        if isinstance(section, dict):
            settings.update({k: section[k] for k in _DEFAULTS if section.get(k) is not None})
    except Exception:
        logger.debug("worktrees config unreadable; using defaults", exc_info=True)
    settings["archive_dirty"] = bool(settings["archive_dirty"])
    for key in ("archive_max_file_mb", "archive_retention_days", "max_total_gb"):
        try:
            settings[key] = max(0.0, float(settings[key]))
        except (TypeError, ValueError):
            settings[key] = float(_DEFAULTS[key])
    return settings


def classic_mode(settings: dict) -> bool:
    """True when the sweep must behave exactly like the pre-archive pruner."""
    return not settings["archive_dirty"] and not settings["max_total_gb"]


# ---------------------------------------------------------------- live-work detection


def _inside(target: str, roots: list[str]) -> Optional[str]:
    for root in roots:
        if target == root or target.startswith(root + os.sep):
            return root
    return None


def _proc_links(pid_dir: Path):
    yield pid_dir / "cwd"
    try:
        yield from (pid_dir / "fd").iterdir()
    except OSError:
        return


def _linux_live_pids(roots: list[str], exclude: frozenset) -> dict[str, list[int]]:
    hits: dict[str, list[int]] = {}
    for pid_dir in Path("/proc").iterdir():
        if not pid_dir.name.isdigit() or int(pid_dir.name) in exclude:
            continue
        for link in _proc_links(pid_dir):
            try:
                target = os.readlink(link)
            except OSError:
                continue
            root = _inside(target.removesuffix(" (deleted)"), roots)
            if root is not None:
                hits.setdefault(root, []).append(int(pid_dir.name))
                break
    return hits


def _psutil_live_pids(roots: list[str], exclude: frozenset) -> dict[str, list[int]]:
    hits: dict[str, list[int]] = {}
    try:
        import psutil
    except ImportError:
        return hits
    for proc in psutil.process_iter(["pid"]):
        try:
            if proc.pid in exclude:
                continue
            root = _inside(os.path.realpath(proc.cwd()), roots)
        except (psutil.Error, OSError):
            continue
        if root is not None:
            hits.setdefault(root, []).append(proc.pid)
    return hits


def live_process_pids(paths, exclude=()) -> dict[str, list[int]]:
    """``{path: [pid, ...]}`` for every *path* some live process works in: cwd or an open fd on
    Linux (``/proc``), cwd elsewhere (psutil, best-effort). Keys are the paths as given;
    *exclude* pids are ignored (an exiting session's own process)."""
    real = {os.path.realpath(str(p)): str(p) for p in paths}
    if not real:
        return {}
    roots = sorted(real, key=len, reverse=True)
    try:
        skip = frozenset(int(p) for p in exclude)
        hits = (_linux_live_pids(roots, skip) if sys.platform.startswith("linux")
                else _psutil_live_pids(roots, skip))
    except OSError:
        hits = {}
    return {real[root]: pids for root, pids in hits.items()}


def _last_activity(path: str) -> float:
    """Newest mtime among the tree's ref/reflog files and every changed or untracked file.

    The tree dir's own mtime only moves when a top-level entry is created or removed, so a lane
    editing files deep inside for days still looks idle by it. ``GIT_OPTIONAL_LOCKS=0`` keeps
    this probe from refreshing (and re-stamping) the index.
    """
    from hermes_cli._subprocess_compat import noninteractive_repo_git_env
    newest = 0.0
    wt = Path(path)
    env = noninteractive_repo_git_env(path)
    if env is None:
        return time.time()  # unverifiable: treat as active
    env["GIT_OPTIONAL_LOCKS"] = "0"
    try:
        git_dir = subprocess.run(["git", "rev-parse", "--absolute-git-dir"], cwd=path, env=env, capture_output=True,
                                 text=True, timeout=10, stdin=subprocess.DEVNULL, check=False).stdout.strip()
        status = subprocess.run(["git", "status", "--porcelain", "-z", "-uall", "--no-renames"], cwd=path, env=env,
                                capture_output=True, text=True, encoding="utf-8", errors="surrogateescape",
                                timeout=60, stdin=subprocess.DEVNULL, check=False)
    except (OSError, subprocess.SubprocessError):
        return time.time()
    if status.returncode != 0:
        return time.time()
    candidates = [Path(git_dir) / name for name in ("HEAD", "logs/HEAD")] if git_dir else []
    candidates += [wt / entry[3:] for entry in status.stdout.split("\0") if len(entry) > 3]
    for candidate in candidates:
        try:
            newest = max(newest, candidate.lstat().st_mtime)
        except OSError:
            continue
    return newest


# ---------------------------------------------------------------- planning


def _tier_hours(name: str, max_age_hours: float) -> float:
    return max_age_hours if name.startswith("hermes-") else max_age_hours * 3


def _plan_survivor(repo_root: str, entry: Path, verdict: str, lock_state, settings: dict,
                   tier_hours: float, now: float) -> tuple[str, str, Optional[str]]:
    """(action, reason, lock_state) for a dirty/unpushed tree past its tier."""
    from hermes_cli import worktree_ops as ops
    from hermes_cli.worktree_archive import archive_preflight
    what = _ARCHIVE_REASONS[verdict]
    lock_state = ops._worktree_lock_is_live(repo_root, str(entry), timeout=5)
    if lock_state == "live":
        return "skip-live-lock", "lock owner still running", lock_state
    active = _last_activity(str(entry))
    if active > now - tier_hours * 3600:
        return "skip-young", f"{what}; last activity {(now - active) / 3600:.1f}h ago", lock_state
    if not settings["archive_dirty"]:
        return f"keep-{verdict}", what, lock_state  # idle: only the size budget may evict it
    unsafe = archive_preflight(str(entry))
    if unsafe:
        return "keep-unarchivable", unsafe, lock_state
    return "archive-remove", what, lock_state


def _apply_budget(decisions: list[Decision], max_total_gb: float) -> None:
    """Evict the oldest idle kept trees (archive first) until they fit *max_total_gb*.

    Only idle past-tier trees count or move; live, young and kanban trees are invisible here.
    """
    from hermes_cli.worktree_archive import archive_preflight
    from hermes_cli.worktree_gc import _tree_size_mb
    eligible = [d for d in decisions if d.action in ("keep-dirty", "keep-unpushed")]
    for d in eligible:
        d.size_mb = _tree_size_mb(Path(d.path), timeout=120) or 0
    total = sum(d.size_mb or 0 for d in eligible)
    budget_mb = max_total_gb * 1024
    for d in sorted(eligible, key=lambda x: x.mtime):
        if total <= budget_mb:
            break
        unsafe = archive_preflight(d.path)
        if unsafe:
            d.action, d.reason = "keep-unarchivable", unsafe
            continue
        d.action, d.reason = "archive-remove", f"{d.reason}; over worktrees.max_total_gb={max_total_gb:g}"
        total -= d.size_mb or 0


def plan_sweep(repo_root: str, max_age_hours: int = 24, settings: Optional[dict] = None,
               now: Optional[float] = None) -> list[Decision]:
    """Read-only decision for every tree under ``<repo>/.worktrees`` (kanban/young included)."""
    import re

    from hermes_cli import worktree_ops as ops
    settings = settings or sweep_settings()
    now = now or time.time()
    worktrees_dir = Path(repo_root) / ".worktrees"
    if not worktrees_dir.is_dir():
        return []
    candidates = ops._prune_candidates(worktrees_dir, max_age_hours, now)
    doomed_names = {entry.name for entry, _m, _f in candidates}
    decisions: list[Decision] = []
    for entry in sorted(worktrees_dir.iterdir()):
        if not entry.is_dir() or entry.name in doomed_names:
            continue
        try:
            age = (now - entry.stat().st_mtime) / 3600
        except OSError:
            continue
        kanban = re.match(r"^t_[0-9a-f]+$", entry.name)
        decisions.append(Decision(entry.name, str(entry), "skip-kanban" if kanban else "skip-young",
                                  "owned by kanban gc" if kanban else "younger than its age tier", round(age, 1)))
    verdicts = ops._classify_prune_candidates(repo_root, candidates) if candidates else []
    # archive_dirty: false is the classic pruner verbatim (I6), which has no process check.
    live = {} if classic_mode(settings) else live_process_pids([str(v[0]) for v in verdicts])
    for entry, mtime, force, verdict, lock_state in verdicts:
        d = Decision(entry.name, str(entry), "", "", round((now - mtime) / 3600, 1), mtime=mtime, force=force,
                     verdict=verdict, lock_state=lock_state)
        if live.get(str(entry)):
            d.action, d.reason = "skip-live-process", f"in use by pid {', '.join(map(str, live[str(entry)][:5]))}"
        elif verdict == "locked-live":
            d.action, d.reason = "skip-live-lock", "lock owner still running"
        elif verdict in _ARCHIVE_REASONS and classic_mode(settings):
            d.action, d.reason = f"keep-{verdict}", _ARCHIVE_REASONS[verdict]
        elif verdict in _ARCHIVE_REASONS:
            d.action, d.reason, d.lock_state = _plan_survivor(
                repo_root, entry, verdict, lock_state, settings, _tier_hours(entry.name, max_age_hours), now)
        else:
            d.action = "remove-keep-branch" if verdict == "reap-keep-branch" else "remove"
            d.reason = "pushed to origin; branch kept" if verdict == "reap-keep-branch" else "clean and merged/pushed"
        decisions.append(d)
    if settings["max_total_gb"]:
        _apply_budget(decisions, settings["max_total_gb"])
    return decisions


# ---------------------------------------------------------------- execution


def _branch_published(repo_root: str, branch: str, remote_heads: Optional[dict]) -> bool:
    """Whether *branch* exists on a remote. Unknown (``ls-remote`` failed) counts as published:
    deleting a branch is never needed for safety, so doubt keeps it."""
    from hermes_cli.worktree_ops import _git_out
    try:
        if _git_out(["remote"], repo_root) == "":
            return False  # no remote at all: nothing could have been published
        if remote_heads is None or branch in remote_heads:
            return True
        refs = _git_out(["for-each-ref", "--format=%(refname)", f"refs/remotes/*/{branch}"], repo_root)
    except (OSError, subprocess.SubprocessError):
        return True  # unknown: keep the branch
    return refs is None or bool(refs.strip())


def archive_then_remove(repo_root: str, path: str, *, reason: str, max_untracked_mb: float,
                        unlock: bool, remote_heads_fn=None, keep_branch: bool = False,
                        ) -> tuple[str, Optional[str], bool]:
    """Snapshot *path*, then remove it -> ``(archive_ref, kept_branch, removed)``.

    Raises ``ArchiveError`` (tree untouched) when the snapshot fails. The branch survives when it
    is protected or published on a remote (an open PR may still need it); otherwise the archive
    ref anchors its commits and the local branch is deleted.
    """
    from hermes_cli.worktree_archive import snapshot_worktree
    from hermes_cli.worktree_ops import _git, _git_quiet, release_lsp_clients
    from hermes_cli._subprocess_compat import noninteractive_git_env
    arc = snapshot_worktree(repo_root, path, reason=reason, max_untracked_mb=max_untracked_mb)
    release_lsp_clients(path)
    if unlock:
        _git_quiet(["worktree", "unlock", path], repo_root, log=f"unlock {path}")
    removed = _git(["worktree", "remove", "--force", path], repo_root, timeout=60)
    if removed.returncode != 0:
        logger.warning("Archived %s to %s but could not remove it: %s", path, arc.ref, removed.stderr.strip())
        return arc.ref, None, False
    branch = arc.meta.get("branch")
    if not branch or keep_branch or branch in _PROTECTED_BRANCHES:
        return arc.ref, branch, True
    if remote_heads_fn is None:
        from hermes_cli.worktree_ops import _fetch_remote_branch_heads
        heads = _fetch_remote_branch_heads(repo_root, timeout=10)
    else:
        heads = remote_heads_fn()
    if _branch_published(repo_root, branch, heads):
        return arc.ref, branch, True
    _git(["branch", "-D", branch], repo_root, stdin=subprocess.DEVNULL, env=noninteractive_git_env())
    return arc.ref, None, True


def execute_plan(repo_root: str, plan: list[Decision], settings: dict) -> tuple[list[str], set, list[str]]:
    """Apply a frozen plan -> ``(preserved_stale, kept_branches, archived_refs)``."""
    from hermes_cli import worktree_ops as ops
    from hermes_cli.worktree_archive import ArchiveError
    doomed = [d for d in plan if d.action in ("remove", "remove-keep-branch", "archive-remove")]
    # Classification can take minutes on a big board: re-check liveness right before acting.
    live = live_process_pids([d.path for d in doomed])
    memo: dict = {}

    def remote_heads():
        if "heads" not in memo:
            memo["heads"] = ops._fetch_remote_branch_heads(repo_root, timeout=10)
        return memo["heads"]

    reap = [(Path(d.path), d.mtime, d.force, d.verdict, d.lock_state) for d in doomed
            if d.action != "archive-remove" and not live.get(d.path)]
    _, kept_branches = ops._reap_prune_verdicts(repo_root, reap, 0.0)
    archived: list[str] = []
    preserved: list[str] = [f"{d.name} ({d.reason})" for d in plan if d.action.startswith("keep-")]
    for d in doomed:
        if d.action != "archive-remove" or live.get(d.path):
            continue
        try:
            ref, kept, _removed = archive_then_remove(
                repo_root, d.path, reason=f"sweep: {d.reason}", max_untracked_mb=settings["archive_max_file_mb"],
                unlock=d.lock_state == "dead", remote_heads_fn=remote_heads)
        except (ArchiveError, OSError, subprocess.SubprocessError) as exc:
            logger.warning("Keeping worktree %s: archive failed (%s)", d.name, exc)
            preserved.append(f"{d.name} (archive failed)")
            continue
        archived.append(ref)
        if kept:
            kept_branches.add(kept)
    return preserved, kept_branches, archived


def run_sweep(repo_root: str, max_age_hours: int, settings: dict) -> None:
    """Plan + execute + archive retention for one repo (the archive-mode pruner)."""
    from hermes_cli import worktree_ops as ops
    from hermes_cli.worktree_archive import prune_expired_archives
    plan = plan_sweep(repo_root, max_age_hours, settings)
    preserved, kept_branches, archived = execute_plan(repo_root, plan, settings)
    if archived:
        logger.warning("Archived and removed %d idle worktree(s) with unmerged work; restore any with "
                       "`hermes worktree restore <name>` (`hermes worktree list` shows archives).", len(archived))
    if preserved:
        logger.warning("Preserving %d stale worktree(s) (run `hermes worktree prune --dry-run` for reasons): %s",
                       len(preserved), ", ".join(sorted(preserved)[:20]))
    expired = prune_expired_archives(repo_root, settings["archive_retention_days"])
    if expired:
        logger.info("Deleted %d worktree archive(s) older than %g days", len(expired), settings["archive_retention_days"])
    ops._prune_orphaned_branches(repo_root, protect=kept_branches)
