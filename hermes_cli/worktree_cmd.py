"""``hermes worktree`` — audit (``list``) and reclaim (``prune [--dry-run] [--trees-only |
--branches-only]``) accumulated git worktrees/branches."""

from __future__ import annotations

from typing import Optional


def _fmt_size(size_mb: Optional[int]) -> str:
    if size_mb is None:
        return "?"
    return f"{size_mb / 1024:.1f}G" if size_mb >= 1024 else f"{size_mb}M"


def _list(worktree_gc, repo_root: str, args) -> int:
    records = worktree_gc.audit_worktrees(repo_root)
    if not records:
        print("No worktrees under .worktrees/ — nothing to reclaim.")
        return 0
    total_mb = sum(r.size_mb or 0 for r in records)
    reapable_mb = sum(r.size_mb or 0 for r in records if r.verdict.startswith("reap"))
    print(f"{'TREE':32} {'AGE':>6} {'SIZE':>6} {'VERDICT':13} REASON")
    for r in sorted(records, key=lambda x: -(x.size_mb or 0)):
        print(f"{r.name[:32]:32} {r.age_days:>5.1f}d {_fmt_size(r.size_mb):>6} {r.verdict:13} {r.reason}")
    print(
        f"\n{len(records)} tree(s), {_fmt_size(total_mb)} total — "
        f"{_fmt_size(reapable_mb)} reclaimable now via `hermes worktree prune`.")
    deletable = [b for b in worktree_gc.audit_branches(repo_root) if b.verdict == "delete"]
    if deletable:
        print(f"{len(deletable)} local branch(es) fully merged/patch-equivalent upstream would also be deleted.")
    return 0


def _prune(worktree_gc, repo_root: str, args) -> int:
    dry_run = bool(getattr(args, "dry_run", False))
    actions: list = []
    if not getattr(args, "branches_only", False):
        tree_records = worktree_gc.audit_worktrees(repo_root, with_sizes=False)
        actions += worktree_gc.reclaim_worktrees(repo_root, dry_run=dry_run, records=tree_records)
        kept = [r for r in tree_records if r.verdict == "keep"
                and "kanban" not in r.reason and "in use" not in r.reason]
        if kept:
            print(f"Preserved {len(kept)} tree(s) with real work:")
            for r in kept:
                print(f"  {r.name}: {r.reason}")
    if not getattr(args, "trees_only", False):
        actions += worktree_gc.reclaim_branches(repo_root, dry_run=dry_run)

    if actions:
        for line in actions:
            print(f"  {line}")
        print(f"{len(actions)} action(s) {'planned' if dry_run else 'done'}.")
    else:
        print("Nothing to reclaim — all trees/branches carry real work or are in use.")
    return 0


_ACTIONS = {"list": _list, "prune": _prune}


def cmd_worktree(args) -> int:
    from hermes_cli import worktree_gc
    repo_root = getattr(args, "repo", None)
    if not repo_root:
        import cli as _cli
        repo_root = _cli._git_repo_root()
    if not repo_root:
        print("Not inside a git repository (or pass --repo <path>).")
        return 1
    action = getattr(args, "worktree_action", None) or "list"
    handler = _ACTIONS.get(action)
    if handler is None:
        print(f"Unknown worktree action: {action}")
        return 1
    return handler(worktree_gc, repo_root, args)
