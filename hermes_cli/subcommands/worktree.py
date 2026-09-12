"""``hermes worktree`` subcommand parser."""

from __future__ import annotations


def build_worktree_parser(subparsers) -> None:
    """Attach the ``worktree`` subcommand to ``subparsers``."""
    worktree_parser = subparsers.add_parser(
        "worktree", help="Audit and reclaim accumulated git worktrees and merged branches",
        description="Attended reclaim for the .worktrees/ directory hermes -w sessions "
            "accumulate. Never deletes uncommitted tracked changes, unique "
            "unpushed commits, or in-use trees; untracked-only scratch is "
            "archived to ~/.hermes/archive/worktree-prune/ before removal. See: "
            "https://hermes-agent.nousresearch.com/docs/user-guide/cli#worktree-cleanup")
    worktree_subparsers = worktree_parser.add_subparsers(dest="worktree_action")
    worktree_list = worktree_subparsers.add_parser(
        "list", aliases=["ls", "audit"],
        help="Classify every tree: age, size, verdict, reason (default action)")
    worktree_list.add_argument("--repo", help="Repo root (default: current repo)")
    worktree_prune = worktree_subparsers.add_parser(
        "prune", help="Remove safe trees and delete fully-merged local branches")
    worktree_prune.add_argument("--repo", help="Repo root (default: current repo)")
    worktree_prune.add_argument(
        "--dry-run", action="store_true", help="Show the plan without changing anything")
    worktree_prune.add_argument(
        "--trees-only", action="store_true",
        help="Only remove worktrees; leave local branches alone")
    worktree_prune.add_argument(
        "--branches-only", action="store_true",
        help="Only delete merged local branches; leave worktrees alone")

    def _dispatch_worktree(_args):
        from hermes_cli.worktree_cmd import cmd_worktree

        # argparse aliases set dest to the literal typed string ("ls"/"audit").
        action = getattr(_args, "worktree_action", None)
        if action in ("ls", "audit"):
            _args.worktree_action = "list"
        return cmd_worktree(_args)

    worktree_parser.set_defaults(func=_dispatch_worktree)
