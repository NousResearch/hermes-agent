"""``hermes worktree`` — audit and reclaim accumulated git worktrees/branches.

Attended counterpart of the silent startup pruner (see
``hermes_cli/worktree_gc.py`` for the policy and shared invariants). Usage:

    hermes worktree list             # audit: verdict + reason per tree
    hermes worktree prune            # reap safe trees + merged branches
    hermes worktree prune --dry-run  # show the plan, change nothing
    hermes worktree prune --trees-only / --branches-only
"""

from __future__ import annotations

from typing import Optional


def conversation_cleanup_status(
    verdict,
    record,
    *,
    removed: bool = False,
    failure_phase: str | None = None,
    failure_message: str | None = None,
) -> dict:
    """Return the complete UI-safe cleanup status without dropping blockers."""
    status = {
        "allowed": bool(verdict.allowed),
        "reasons": [str(reason) for reason in verdict.reasons],
        "removed": bool(removed),
        "root_session_id": str(record.root_session_id),
        "path": str(record.worktree_path),
        "branch": str(record.branch),
        "base_commit": str(record.base_commit),
        "state": str(record.state),
    }
    if failure_phase is not None:
        status["failure_phase"] = str(failure_phase)
    if failure_message is not None:
        status["failure_message"] = str(failure_message)
    return status


def _repo_root() -> Optional[str]:
    import cli as _cli

    return _cli._git_repo_root()


def _fmt_size(size_mb: Optional[int]) -> str:
    if size_mb is None:
        return "?"
    if size_mb >= 1024:
        return f"{size_mb / 1024:.1f}G"
    return f"{size_mb}M"


def cmd_worktree(args) -> int:
    from hermes_cli import worktree_gc

    repo_root = getattr(args, "repo", None) or _repo_root()
    if not repo_root:
        print("Not inside a git repository (or pass --repo <path>).")
        return 1

    action = getattr(args, "worktree_action", None) or "list"

    if action == "list":
        records = worktree_gc.audit_worktrees(repo_root)
        if not records:
            print("No worktrees under .worktrees/ — nothing to reclaim.")
            return 0
        total_mb = sum(r.size_mb or 0 for r in records)
        reapable_mb = sum(
            r.size_mb or 0 for r in records if r.verdict.startswith("reap")
        )
        print(f"{'TREE':32} {'AGE':>6} {'SIZE':>6} {'VERDICT':13} REASON")
        for r in sorted(records, key=lambda x: -(x.size_mb or 0)):
            print(
                f"{r.name[:32]:32} {r.age_days:>5.1f}d {_fmt_size(r.size_mb):>6} "
                f"{r.verdict:13} {r.reason}"
            )
        print(
            f"\n{len(records)} tree(s), {_fmt_size(total_mb)} total — "
            f"{_fmt_size(reapable_mb)} reclaimable now via `hermes worktree prune`."
        )
        branch_records = worktree_gc.audit_branches(repo_root)
        deletable = [b for b in branch_records if b.verdict == "delete"]
        if deletable:
            print(
                f"{len(deletable)} local branch(es) fully merged/patch-equivalent "
                f"upstream would also be deleted."
            )
        return 0

    if action == "prune":
        dry_run = bool(getattr(args, "dry_run", False))
        trees_only = bool(getattr(args, "trees_only", False))
        branches_only = bool(getattr(args, "branches_only", False))

        actions: list = []
        if not branches_only:
            tree_records = worktree_gc.audit_worktrees(repo_root, with_sizes=False)
            actions += worktree_gc.reclaim_worktrees(
                repo_root, dry_run=dry_run, records=tree_records
            )
            kept = [r for r in tree_records if r.verdict == "keep"
                    and "kanban" not in r.reason and "in use" not in r.reason]
            if kept:
                print(f"Preserved {len(kept)} tree(s) with real work:")
                for r in kept:
                    print(f"  {r.name}: {r.reason}")
        if not trees_only:
            actions += worktree_gc.reclaim_branches(repo_root, dry_run=dry_run)

        if actions:
            for line in actions:
                print(f"  {line}")
            verb = "planned" if dry_run else "done"
            print(f"{len(actions)} action(s) {verb}.")
        else:
            print("Nothing to reclaim — all trees/branches carry real work or are in use.")
        return 0

    print(f"Unknown worktree action: {action}")
    return 1
