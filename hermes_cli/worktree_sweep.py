"""Snapshot-then-delete sweep for stale worktrees (``worktrees.archive_dirty``, default on).

The classic pruner (``worktree_ops._prune_stale_worktree_trees``) never removes a dirty or
unpushed tree at any age, so leftovers of finished lanes pile up forever. This sweep keeps its
tiers and classifier, then removes an idle dirty/unpushed tree only after archiving its full
working state to ``refs/hermes/archive/<tree>`` (``worktree_archive``); a failed archive keeps
the tree. Never touched, at any age or budget: Kanban ``t_<hex>`` trees, trees whose ``hermes``
lock owner still runs, trees any live process sits in (cwd or open file), and trees with file or
ref activity younger than their tier, and trees holding anything an archive cannot carry
(``worktree_retention.removal_blocker``: a submodule, an embedded repository, a per-worktree ref,
an untracked file over the size cap). With
``archive_dirty: false`` and no size budget the classic pruner runs unchanged.

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
    tier_hours: float = 0.0
    planned_at: float = 0.0

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
    """Newest mtime among the tree's HEAD/reflog, every changed or untracked file, and the
    directory holding each changed path.

    The tree dir's own mtime only moves when a top-level entry is created or removed, so a lane
    editing files deep inside for days still looks idle by it. A deleted file has no mtime; its
    nearest surviving parent directory was stamped by the unlink. The index is not a signal:
    the classifier's own ``git status`` may rewrite it. ``GIT_OPTIONAL_LOCKS=0`` keeps this probe from refreshing (and re-stamping) the index.
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
                                 text=True, encoding="utf-8", errors="surrogateescape", timeout=10, stdin=subprocess.DEVNULL, check=False).stdout.strip()
        status = subprocess.run(["git", "status", "--porcelain", "-z", "-uall", "--no-renames"], cwd=path, env=env,
                                capture_output=True, text=True, encoding="utf-8", errors="surrogateescape",
                                timeout=60, stdin=subprocess.DEVNULL, check=False)
    except (OSError, subprocess.SubprocessError):
        return time.time()
    if status.returncode != 0:
        return time.time()
    candidates = [Path(git_dir) / name for name in ("HEAD", "logs/HEAD")] if git_dir else []
    changed = [wt / entry[3:] for entry in status.stdout.split("\0") if len(entry) > 3]
    candidates += changed
    for item in changed:
        parent = item.parent
        while parent != wt and not parent.exists() and wt in parent.parents:
            parent = parent.parent  # a whole deleted subtree: its surviving ancestor moved
        candidates.append(parent)
    for candidate in dict.fromkeys(candidates):
        try:
            newest = max(newest, candidate.lstat().st_mtime)
        except OSError:
            continue
    return newest


# ---------------------------------------------------------------- planning


def _tier_hours(name: str, max_age_hours: float) -> float:
    from hermes_cli.worktree_ops import prune_tier_hours
    return prune_tier_hours(name, max_age_hours)


def _plan_survivor(repo_root: str, entry: Path, verdict: str, lock_state, settings: dict,
                   tier_hours: float, now: float) -> tuple[str, str, Optional[str], Optional[float]]:
    """(action, reason, lock_state, last_activity) for a dirty/unpushed tree past its tier."""
    from hermes_cli import worktree_ops as ops
    from hermes_cli.worktree_archive import archive_preflight
    what = _ARCHIVE_REASONS[verdict]
    lock_state = ops._worktree_lock_is_live(repo_root, str(entry), timeout=5)
    if lock_state == "live":
        return "skip-live-lock", "lock owner still running", lock_state, None
    active = _last_activity(str(entry))
    if active > now - tier_hours * 3600:
        return "skip-young", f"{what}; last activity {(now - active) / 3600:.1f}h ago", lock_state, active
    if not settings["archive_dirty"]:
        return f"keep-{verdict}", what, lock_state, active  # idle: only the size budget may evict it
    unsafe = archive_preflight(str(entry), settings["archive_max_file_mb"])
    if unsafe:
        return unsafe[0], unsafe[1], lock_state, active
    return "archive-remove", what, lock_state, active


def _apply_budget(decisions: list[Decision], max_total_gb: float, max_file_mb: float = 50) -> None:
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
        unsafe = archive_preflight(d.path, max_file_mb)
        if unsafe:
            d.action, d.reason = unsafe
            continue
        d.action, d.reason = "archive-remove", f"{d.reason}; over worktrees.max_total_gb={max_total_gb:g}"
        total -= d.size_mb or 0


_DESTRUCTIVE = ("remove", "remove-keep-branch", "archive-remove")


def plan_sweep(repo_root: str, max_age_hours: int = 24, settings: Optional[dict] = None,
               now: Optional[float] = None, older_than_days: Optional[float] = None) -> list[Decision]:
    """Read-only decision for every tree under ``<repo>/.worktrees`` (kanban/young included).

    *older_than_days* (``hermes worktree prune --older-than``) only restricts: a tree idle for
    less than that is never archived or removed, whatever its tier.
    """
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
        tier = _tier_hours(entry.name, max_age_hours)
        d = Decision(entry.name, str(entry), "", "", round((now - mtime) / 3600, 1), mtime=mtime, force=force,
                     verdict=verdict, lock_state=lock_state, tier_hours=tier, planned_at=now)
        if live.get(str(entry)):
            d.action, d.reason = "skip-live-process", f"in use by pid {', '.join(map(str, live[str(entry)][:5]))}"
        elif verdict == "locked-live":
            d.action, d.reason = "skip-live-lock", "lock owner still running"
        elif verdict in _ARCHIVE_REASONS and classic_mode(settings):
            d.action, d.reason = f"keep-{verdict}", _ARCHIVE_REASONS[verdict]
        elif verdict in _ARCHIVE_REASONS:
            d.action, d.reason, d.lock_state, active = _plan_survivor(
                repo_root, entry, verdict, lock_state, settings, tier, now)
            if active:
                d.age_hours = min(d.age_hours, round((now - active) / 3600, 1))
        elif blocked := ops.removal_blocked(str(entry)):
            d.action, d.reason = blocked
        else:
            d.action = "remove-keep-branch" if verdict == "reap-keep-branch" else "remove"
            d.reason = "pushed to origin; branch kept" if verdict == "reap-keep-branch" else "clean and merged/pushed"
        decisions.append(d)
    if settings["max_total_gb"]:
        _apply_budget(decisions, settings["max_total_gb"], settings["archive_max_file_mb"])
    if older_than_days is not None:
        for d in decisions:
            if d.action in _DESTRUCTIVE and d.age_hours < older_than_days * 24:
                d.action, d.reason = "skip-young", f"idle {d.age_hours:.1f}h (--older-than {older_than_days:g})"
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


def _in_use_since(path: str, since: float, exclude=()) -> Optional[str]:
    """Why *path* must not be removed now: a live process works in it, or a file/ref in it
    changed at or after *since*. None = still idle."""
    pids = live_process_pids([path], exclude=exclude).get(path)
    if pids:
        return f"in use by pid {', '.join(map(str, pids[:5]))}"
    active = _last_activity(path)
    if active >= since:
        return f"changed {max(0.0, time.time() - active):.0f}s ago"
    return None


def _index_moved(path: str, captured: str) -> Optional[str]:
    """Why the staged state of *path* no longer matches the archive (``git add``/``apply
    --cached`` after the capture: the index is not a file-mtime signal), or None. A stat-only
    refresh by ``git status`` does not count. Unreadable index = moved (keep the tree)."""
    from hermes_cli.worktree_archive import ArchiveError, staged_state
    try:
        if staged_state(path) == captured:
            return None
    except (ArchiveError, OSError, subprocess.SubprocessError):
        pass
    return "staged changes since the snapshot"


def archive_then_remove(repo_root: str, path: str, *, reason: str, max_untracked_mb: float,
                        unlock: bool, remote_heads_fn=None, keep_branch: bool = False,
                        exclude_pids=(),
                        ) -> tuple[str, Optional[str], bool, Optional[str]]:
    """Snapshot *path*, then remove it -> ``(archive_ref, kept_branch, removed, why_kept)``.

    Raises ``ArchiveError`` (tree untouched) when the snapshot fails. Right after the snapshot the
    tree is checked again: a process (other than *exclude_pids*) now working in it, any edit
    made while the snapshot ran, staged entries that differ from the captured index, or anything
    the archive does not hold (``worktree_retention.removal_blocker``: a file the snapshot skipped
    for size, an embedded repository, a per-worktree ref, a submodule) keep the tree (``removed`` False, *why_kept* says why; the archive stays). The branch
    survives when it is protected or published on a remote (an open PR may still need it);
    otherwise the archive ref anchors its commits and the local branch is deleted.
    """
    from hermes_cli.worktree_archive import snapshot_worktree
    from hermes_cli.worktree_ops import _git, _git_quiet, release_lsp_clients
    from hermes_cli.worktree_retention import removal_blocker
    from hermes_cli._subprocess_compat import noninteractive_git_env
    started = time.time()
    arc = snapshot_worktree(repo_root, path, reason=reason, max_untracked_mb=max_untracked_mb)
    blocked = removal_blocker(path, skipped=arc.meta.get("skipped_untracked"), gitlinks=True)
    busy = (_in_use_since(path, started, exclude=exclude_pids) or _index_moved(path, arc.staged)
            or (blocked and blocked[1]))
    if busy:
        logger.warning("Archived %s to %s but keeping it: %s", path, arc.ref, busy)
        return arc.ref, None, False, busy
    release_lsp_clients(path)
    if unlock:
        _git_quiet(["worktree", "unlock", path], repo_root, log=f"unlock {path}")
    removed = _git(["worktree", "remove", "--force", path], repo_root, timeout=60)
    if removed.returncode != 0:
        logger.warning("Archived %s to %s but could not remove it: %s", path, arc.ref, removed.stderr.strip())
        return arc.ref, None, False, removed.stderr.strip() or "git worktree remove failed"
    branch = arc.meta.get("branch")
    if not branch or keep_branch or branch in _PROTECTED_BRANCHES:
        return arc.ref, branch, True, None
    if remote_heads_fn is None:
        from hermes_cli.worktree_ops import _fetch_remote_branch_heads
        heads = _fetch_remote_branch_heads(repo_root, timeout=10)
    else:
        heads = remote_heads_fn()
    if _branch_published(repo_root, branch, heads):
        return arc.ref, branch, True, None
    _git(["branch", "-D", branch], repo_root, stdin=subprocess.DEVNULL, env=noninteractive_git_env())
    return arc.ref, None, True, None


def _recheck(repo_root: str, d: Decision) -> tuple[Optional[str], Optional[str]]:
    """Fresh gate for ONE planned tree, run immediately before it is touched -> (why_keep,
    lock_state). Earlier trees in the batch can take minutes to archive; a user may have entered
    this one meanwhile."""
    from hermes_cli import worktree_ops as ops
    if not os.path.isdir(d.path):
        return "gone", None
    lock_state = ops._worktree_lock_is_live(repo_root, d.path, timeout=5)
    if lock_state == "live":
        return "lock owner still running", lock_state
    since = d.planned_at or time.time()
    if d.action == "archive-remove" and d.tier_hours:
        since = min(since, time.time() - d.tier_hours * 3600)  # back inside its idle tier
    busy = _in_use_since(d.path, since)
    if busy:
        return busy, lock_state
    if d.action != "archive-remove" and ops._worktree_is_dirty(d.path, repo_root, timeout=10):
        return "new uncommitted changes", lock_state
    return None, lock_state


def execute_plan(repo_root: str, plan: list[Decision], settings: dict
                 ) -> tuple[list[str], set, list[tuple[str, bool, Optional[str]]]]:
    """Apply a frozen plan -> ``(preserved_stale, kept_branches, [(archive_ref, removed, why_kept)])``.
    Each tree is re-gated (lock, live process, fresh activity) right before it is archived or
    removed; ``removed`` is False when the archive was saved but the tree was kept (*why_kept*)."""
    from hermes_cli import worktree_ops as ops
    from hermes_cli.worktree_archive import ArchiveError
    memo: dict = {}

    def remote_heads():
        if "heads" not in memo:
            memo["heads"] = ops._fetch_remote_branch_heads(repo_root, timeout=10)
        return memo["heads"]

    kept_branches: set = set()
    archived: list[tuple[str, bool, Optional[str]]] = []
    preserved: list[str] = [f"{d.name} ({d.reason})" for d in plan if d.action.startswith("keep-")]
    for d in (d for d in plan if d.action in _DESTRUCTIVE):
        why_keep, lock_state = _recheck(repo_root, d)
        if why_keep:
            logger.info("Sweep: keeping %s (%s)", d.name, why_keep)
            continue
        if d.action != "archive-remove":
            _, kept = ops._reap_prune_verdicts(
                repo_root, [(Path(d.path), d.mtime, d.force, d.verdict, lock_state)], 0.0)
            kept_branches |= kept
            continue
        try:
            ref, kept, removed, why = archive_then_remove(
                repo_root, d.path, reason=f"sweep: {d.reason}", max_untracked_mb=settings["archive_max_file_mb"],
                unlock=lock_state == "dead", remote_heads_fn=remote_heads)
        except (ArchiveError, OSError, subprocess.SubprocessError) as exc:
            logger.warning("Keeping worktree %s: archive failed (%s)", d.name, exc)
            preserved.append(f"{d.name} (archive failed)")
            continue
        archived.append((ref, removed, why))
        if not removed:
            preserved.append(f"{d.name} (archived to {ref}, kept: {why})")
        if kept:
            kept_branches.add(kept)
    return preserved, kept_branches, archived


def run_sweep(repo_root: str, max_age_hours: int, settings: dict) -> None:
    """Plan + execute + archive retention for one repo (the archive-mode pruner)."""
    from hermes_cli import worktree_ops as ops
    from hermes_cli.worktree_archive import prune_expired_archives
    plan = plan_sweep(repo_root, max_age_hours, settings)
    preserved, kept_branches, archived = execute_plan(repo_root, plan, settings)
    removed = [ref for ref, gone, _why in archived if gone]
    if removed:
        logger.warning("Archived and removed %d idle worktree(s) with unmerged work; restore any with "
                       "`hermes worktree restore <name>` (`hermes worktree list` shows archives).", len(removed))
    if preserved:
        logger.warning("Preserving %d stale worktree(s) (run `hermes worktree prune --dry-run` for reasons): %s",
                       len(preserved), ", ".join(sorted(preserved)[:20]))
    expired = prune_expired_archives(repo_root, settings["archive_retention_days"])
    if expired:
        logger.info("Deleted %d worktree archive(s) older than %g days", len(expired), settings["archive_retention_days"])
    ops._prune_orphaned_branches(repo_root, protect=kept_branches)
