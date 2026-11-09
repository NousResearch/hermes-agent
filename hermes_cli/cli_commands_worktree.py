"""``/worktree`` subcommand bodies split out of ``cli_commands_mixin``: ``new``'s step into the tree it
just created, ``prune`` and ``remove``."""

from __future__ import annotations

import os

from agent.i18n import t
from hermes_cli.cli_commands_session_tools import _pr, _t


def enter_new_worktree(cli, wt_info) -> bool:
    """Move *cli*'s session into the fresh tree -> True when the session's tools now run there.

    A gateway refusal (e.g. ``session_busy``) moved nothing: the fresh tree is removed now instead of
    sitting unlisted until quit, and an earlier active tree stays active. A chdir failure after the
    gateway accepted the move leaves only this process behind, so it is reported as that."""
    import cli as _cli
    from hermes_cli.cli_chat_error_copy import refusal_text
    from hermes_cli.gateway_automation import gateway_connection
    from hermes_cli.gateway_client import GatewayClientError
    from hermes_cli.gateway_session_store import move_cli_workspace
    try:
        move_cli_workspace(cli, wt_info)  # the gateway session's tools, then this process
    except GatewayClientError as e:
        print(f"  {_t('worktree.enter_failed', error=refusal_text(e))}")
        active = _cli._active_worktree
        _cli._cleanup_worktree(wt_info)
        _cli._active_worktree = active
        return False
    except OSError as e:
        if gateway_connection(cli) is None:  # a local session: nothing moved
            print(f"  {_t('worktree.enter_failed', error=e)}")
            return False
        print(f"  {t('cli.session.cwd_enter_failed', path=wt_info['path'], error=e)}")
    return True


def worktree_prune(repo_root: str, rest: str) -> None:
    """``/worktree prune [--dry-run]``: reap every reapable tree and merged branch, never this session's tree."""
    import cli as _cli
    from hermes_cli import worktree_gc
    rest = rest.lower()
    dry_run = "--dry-run" in rest or "-n" in rest.split()
    active = _cli._active_worktree
    tree_records = worktree_gc.audit_worktrees(repo_root, with_sizes=False)
    if active:
        # Never reap the tree this session is sitting in, even if judged clean+merged.
        active_path = str(active.get("path") or "")
        tree_records = [record for record in tree_records if record.path != active_path]
    actions = worktree_gc.reclaim_worktrees(repo_root, dry_run=dry_run, records=tree_records)
    actions += worktree_gc.reclaim_branches(repo_root, dry_run=dry_run)
    if actions:
        _pr(*(f"  {line}" for line in actions),
            "  " + _t("worktree.actions_planned" if dry_run else "worktree.actions_done", count=len(actions)))
    else:
        print(f"  {_t('worktree.nothing_to_reclaim')}")
    kept = [r for r in tree_records
            if r.verdict == "keep" and "kanban" not in r.reason and "in use" not in r.reason]
    if kept:
        _pr(f"  {_t('worktree.preserved', count=len(kept))}",
            *(f"    {record.name}: {record.reason}" for record in kept))


def worktree_remove(repo_root: str, rest: str) -> None:
    """``/worktree remove <name|path>``: one tree under the prune policy (never this session's own)."""
    import cli as _cli
    from hermes_cli import worktree_gc
    name = rest.strip()
    if not name:
        return print(f"  {_t('worktree.remove_usage')}")
    repo_root, path = worktree_gc.resolve_tree_target(repo_root, name)
    target = os.path.realpath(path)
    active = _cli._active_worktree
    own = [os.path.realpath(str(active.get("path") or ""))] if active else []
    own.append(os.path.realpath(os.getcwd()))  # also this process's cwd when no tree is "active"
    if any(p == target or p.startswith(target + os.sep) for p in own):
        return print(f"  {_t('worktree.remove_own_tree')}")
    result = worktree_gc.reclaim_worktree(repo_root, path)
    _pr(*(f"  {line}" for line in result["actions"]))
    if not result["removed"]:
        print(f"  {_t('worktree.kept', path=path, reason=result['reason'])}")
