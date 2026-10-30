"""``hermes worktree`` subcommand parser."""

from __future__ import annotations


def build_worktree_parser(subparsers) -> None:
    """Attach the ``worktree`` subcommand to ``subparsers``."""
    worktree_parser = subparsers.add_parser(
        "worktree", help="Audit and reclaim accumulated git worktrees and merged branches",
        description="Audit, reclaim and restore the .worktrees/ trees Hermes sessions "
            "accumulate. Idle trees with uncommitted changes or unpushed commits are "
            "archived to refs/hermes/archive/<tree> before removal "
            "(worktrees.archive_dirty); `restore` brings one back. In-use trees are "
            "never touched. See: "
            "https://hermes-agent.nousresearch.com/docs/user-guide/git-worktrees")
    worktree_subparsers = worktree_parser.add_subparsers(dest="worktree_action")
    worktree_list = worktree_subparsers.add_parser(
        "list", aliases=["ls", "audit"],
        help="Classify every tree: age, size, verdict, reason (default action)")
    worktree_list.add_argument("--repo", help="Repo root (default: current repo)")
    worktree_list.add_argument(
        "--json", action="store_true",
        help="Machine-readable audit output (trees, external trees, branches)")
    worktree_list.add_argument(
        "--older-than", type=float, metavar="DAYS", dest="older_than",
        help="Treat reapable trees younger than DAYS as keep")
    worktree_prune = worktree_subparsers.add_parser(
        "prune", help="Remove safe trees and delete fully-merged local branches")
    worktree_prune.add_argument("--repo", help="Repo root (default: current repo)")
    worktree_prune.add_argument(
        "--json", action="store_true",
        help="Machine-readable result (actions taken/planned, preserved trees)")
    worktree_prune.add_argument(
        "--older-than", type=float, metavar="DAYS", dest="older_than",
        help="Only reap trees idle for at least DAYS days (safety gates still apply)")
    worktree_prune.add_argument(
        "--dry-run", action="store_true", help="Show the plan without changing anything")
    worktree_prune.add_argument(
        "--trees-only", action="store_true",
        help="Only remove worktrees; leave local branches alone")
    worktree_prune.add_argument(
        "--branches-only", action="store_true",
        help="Only delete merged local branches; leave worktrees alone")

    worktree_restore = worktree_subparsers.add_parser(
        "restore", help="Recreate an archived worktree (tracked, staged and untracked files)")
    worktree_restore.add_argument("name", help="Archive name or tree name (see `hermes worktree list`)")
    worktree_restore.add_argument("--path", help="Restore here instead of the original path")
    worktree_restore.add_argument("--repo", help="Repo root (default: current repo)")

    worktree_merge = worktree_subparsers.add_parser(
        "merge", help="Merge a branch (e.g. a subagent's) into this worktree only if it passes checks",
        description="Merge <branch> into the current worktree's HEAD, all-or-nothing. Refuses when "
            "tracked files are dirty, a changed file is outside --allow, or commits landed after "
            "--tested; a conflict aborts; a failing or timed-out --gate resets to the pre-merge commit.")
    worktree_merge.add_argument("branch", help="Branch or commit to merge (e.g. hermes-subagent/subagent-<id>)")
    worktree_merge.add_argument(
        "--allow", action="append", metavar="PATH",
        help="Repo-relative file or directory the branch may change (repeatable)")
    worktree_merge.add_argument(
        "--tested", metavar="SHA", help="Commit the branch was verified at; it must be the branch tip")
    worktree_merge.add_argument("--gate", metavar="CMD", help="Shell command that must pass after the merge")
    worktree_merge.add_argument(
        "--gate-timeout", type=float, default=1800, metavar="SECONDS", dest="gate_timeout",
        help="Kill the gate and reset after SECONDS (default 1800)")
    worktree_merge.add_argument("--json", action="store_true", help="Machine-readable result")
    worktree_merge.add_argument("--repo", help="Worktree to merge into (default: current worktree)")

    def _dispatch_worktree(_args):
        from hermes_cli.worktree_cmd import cmd_worktree

        # argparse aliases set dest to the literal typed string ("ls"/"audit").
        action = getattr(_args, "worktree_action", None)
        if action in ("ls", "audit"):
            _args.worktree_action = "list"
        return cmd_worktree(_args)

    worktree_parser.set_defaults(func=_dispatch_worktree)
