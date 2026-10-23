"""``hermes worktree`` — audit (``list [--json] [--older-than DAYS]``), reclaim
(``prune [--dry-run] [--json] [--older-than DAYS] [--trees-only | --branches-only]``) and
``restore <name> [--path DIR]`` accumulated git worktrees/branches. ``--json`` output is the only
thing written to stdout in that mode so scripts can consume it."""

from __future__ import annotations

import json
from dataclasses import asdict
from typing import Optional


def _fmt_size(size_mb: Optional[int]) -> str:
    if size_mb is None:
        return "?"
    return f"{size_mb / 1024:.1f}G" if size_mb >= 1024 else f"{size_mb}M"


def _list(worktree_gc, repo_root: str, args) -> int:
    older_than = getattr(args, "older_than", None)
    records = worktree_gc.audit_worktrees(repo_root, older_than_days=older_than)
    external = worktree_gc.audit_external_trees(repo_root)
    branch_records = worktree_gc.audit_branches(repo_root)
    from hermes_cli.worktree_archive import list_archives
    archives = list_archives(repo_root)
    if getattr(args, "json", False):
        print(json.dumps({
            "repo": repo_root,
            "trees": [asdict(r) for r in records],
            "external_trees": [asdict(r) for r in external],
            "branches": [asdict(b) for b in branch_records],
            "archives": [asdict(a) for a in archives],
        }, indent=2))
        return 0
    _print_archives(archives)
    if not records and not external:
        print("No worktrees under .worktrees/ — nothing to reclaim.")
        return 0
    if records:
        total_mb = sum(r.size_mb or 0 for r in records)
        reapable_mb = sum(r.size_mb or 0 for r in records if r.verdict.startswith("reap"))
        print(f"{'TREE':32} {'AGE':>6} {'SIZE':>6} {'VERDICT':13} REASON")
        for r in sorted(records, key=lambda x: -(x.size_mb or 0)):
            print(f"{r.name[:32]:32} {r.age_days:>5.1f}d {_fmt_size(r.size_mb):>6} {r.verdict:13} {r.reason}")
        print(
            f"\n{len(records)} tree(s), {_fmt_size(total_mb)} total — "
            f"{_fmt_size(reapable_mb)} reclaimable now via `hermes worktree prune`.")
    if external:
        print(f"\n{len(external)} externally-registered worktree(s) (never touched by prune):")
        for e in external:
            state = "MISSING" if e.missing else ("locked" if e.locked else "ok")
            print(f"  {e.path}  [{e.branch or '?'}]  {state}")
        if any(e.missing and not e.locked for e in external):
            print("  Stale registrations (MISSING) are cleaned by `hermes worktree prune` (metadata only).")
    deletable = [b for b in branch_records if b.verdict == "delete"]
    if deletable:
        print(f"{len(deletable)} local branch(es) fully merged/patch-equivalent upstream would also be deleted.")
    return 0


def _print_archives(archives) -> None:
    if not archives:
        return
    import time
    print(f"{len(archives)} archived worktree(s) (restore with `hermes worktree restore <name>`):")
    for a in archives:
        age = (time.time() - a.created) / 86400 if a.created else 0
        skipped = len(a.meta.get("skipped_untracked") or [])
        extra = f", {skipped} large untracked file(s) not captured" if skipped else ""
        print(f"  {a.name:40} {age:5.1f}d  {a.meta.get('branch') or 'detached'}  "
              f"({a.meta.get('reason') or '?'}{extra})")
    print()


def _sweep_lines(plan) -> list[str]:
    lines = [f"{'TREE':40} {'IDLE':>7}  {'DECISION':20} REASON"]
    for d in sorted(plan, key=lambda x: (x.action, x.name)):
        lines.append(f"{d.name[:40]:40} {d.age_hours:>6.1f}h  {d.action:20} {d.reason}")
    counts: dict = {}
    for d in plan:
        counts[d.action] = counts.get(d.action, 0) + 1
    lines.append("Sweep totals: " + (", ".join(f"{k}={v}" for k, v in sorted(counts.items())) or "nothing to do"))
    return lines


def _sweep(repo_root: str, dry_run: bool, older_than: Optional[float] = None) -> tuple[list, list[str]]:
    """The automatic sweep's plan for this repo (applied unless *dry_run*) -> (plan, actions).
    *older_than* (days) bounds what the plan may archive or remove before anything runs."""
    from hermes_cli import worktree_sweep
    settings = worktree_sweep.sweep_settings()
    plan = worktree_sweep.plan_sweep(repo_root, settings=settings, older_than_days=older_than)
    if dry_run or worktree_sweep.classic_mode(settings):
        return plan, []
    _, _, archived = worktree_sweep.execute_plan(repo_root, plan, settings)
    from hermes_cli.worktree_archive import prune_expired_archives
    expired = prune_expired_archives(repo_root, settings["archive_retention_days"])
    from agent.i18n import t
    saved = [t("cli.worktree.sweep_archived_removed", ref=ref) if removed
             else t("cli.worktree.sweep_archived_kept", ref=ref, reason=why or "?")
             for ref, removed, why in archived]
    return plan, saved + [f"deleted expired archive {r}" for r in expired]


def _attended_records(records, plan) -> list:
    """The attended reclaim's records minus trees the sweep already decided (archived, removed,
    kanban) and minus any tree a live process works in: those are never touched here either."""
    from hermes_cli.worktree_sweep import classic_mode, live_process_pids, sweep_settings
    if classic_mode(sweep_settings()):
        return records
    owned = {d.path for d in plan if d.action != "skip-young"}
    records = [r for r in records if r.path not in owned]
    live = live_process_pids([r.path for r in records])
    return [r for r in records if not live.get(r.path)]


def _prune(worktree_gc, repo_root: str, args) -> int:
    dry_run = bool(getattr(args, "dry_run", False))
    as_json = bool(getattr(args, "json", False))
    older_than = getattr(args, "older_than", None)
    actions: list = []
    kept: list = []
    plan: list = []
    if not getattr(args, "branches_only", False):
        plan, actions = _sweep(repo_root, dry_run, older_than)
        if not as_json:
            print("\n".join(_sweep_lines(plan)) + "\n")
        actions += worktree_gc.prune_missing_registrations(repo_root, dry_run=dry_run)
        tree_records = _attended_records(
            worktree_gc.audit_worktrees(repo_root, with_sizes=False, older_than_days=older_than), plan)
        actions += worktree_gc.reclaim_worktrees(repo_root, dry_run=dry_run, records=tree_records)
        kept = [r for r in tree_records if r.verdict == "keep"
                and "kanban" not in r.reason and "in use" not in r.reason]
        if kept and not as_json:
            print(f"Preserved {len(kept)} tree(s) with real work:")
            for r in kept:
                print(f"  {r.name}: {r.reason}")
    if not getattr(args, "trees_only", False):
        actions += worktree_gc.reclaim_branches(repo_root, dry_run=dry_run)

    if as_json:
        print(json.dumps({
            "repo": repo_root,
            "dry_run": dry_run,
            "actions": actions,
            "preserved": [asdict(r) for r in kept],
            "sweep": [d.as_dict() for d in plan],
        }, indent=2))
        return 0
    if actions:
        print("Attended reclaim (`hermes worktree prune` only; the automatic sweep is the table above):")
        for line in actions:
            print(f"  {line}")
        print(f"{len(actions)} action(s) {'planned' if dry_run else 'done'}.")
    else:
        print("Nothing to reclaim — all trees/branches carry real work or are in use.")
    return 0


def _restore(_worktree_gc, repo_root: str, args) -> int:
    from hermes_cli.worktree_archive import ArchiveError, restore_archive
    name = getattr(args, "name", None)
    if not name:
        print("Usage: hermes worktree restore <name> [--path DIR]")
        return 1
    try:
        result = restore_archive(repo_root, name, getattr(args, "path", None))
    except ArchiveError as exc:
        print(f"✗ {exc}")
        return 1
    print(f"✓ Restored {result['ref']} → {result['path']}")
    if result["detached"]:
        print(f"  Detached at {result['head'][:12]} (branch {result['branch'] or '-'} moved or is checked out "
              "elsewhere); create a branch with `git switch -c <name>`.")
    else:
        print(f"  Branch: {result['branch']}")
    for item in result["skipped_untracked"]:
        print(f"  Not captured (over the size cap): {item['path']} ({item['bytes'] // (1024 * 1024)} MB)")
    return 0


_ACTIONS = {"list": _list, "prune": _prune, "restore": _restore}


def cmd_worktree(args) -> int:
    from hermes_cli import worktree_gc
    repo_root = getattr(args, "repo", None)
    if not repo_root:
        import cli as _cli
        repo_root = _cli._git_repo_root()
    if not repo_root:
        print("Not inside a git repository (or pass --repo <path>).")
        return 1
    older_than = getattr(args, "older_than", None)
    if older_than is not None and older_than < 0:
        print("--older-than must be a non-negative number of days.")
        return 1
    action = getattr(args, "worktree_action", None) or "list"
    handler = _ACTIONS.get(action)
    if handler is None:
        print(f"Unknown worktree action: {action}")
        return 1
    return handler(worktree_gc, repo_root, args)
