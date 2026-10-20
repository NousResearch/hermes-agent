"""``/worktree new``'s step into the tree it just created, split out of ``cli_commands_mixin``."""

from __future__ import annotations

from agent.i18n import t
from hermes_cli.cli_commands_session_tools import _t


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
