"""Classic CLI commands whose state lives in the gateway session, answered there.

Rewired, the session's tools, checkpoints, environment and handoff watcher run in the gateway, so
main's handlers (which read this process's own agent and environment) answer nothing or the wrong
thing:

- ``/diff session``: the checkpoint baseline is the gateway tools'. ``rollback.session_diff`` reads it
  under the session's frozen checkpoint policy (``--checkpoints`` included) and returns it as data
  for main's terminal renderer. Working-tree modes stay main's local git read of the same checkout.
- ``/reload``: ``reload.env`` re-reads ``.env`` where turns and tools run (this process re-reads too).
- ``/handoff``: the pending row is written in the owner's store (``handoff.request``), where the
  gateway's handoff watcher claims it; this process's own store write would never be seen.
"""
from __future__ import annotations

from hermes_agent.agent.i18n import t
from hermes_cli.gateway_client import GatewayClientError


def _diff(cli, connection, cmd):
    args = cmd.partition(" ")[2].strip().lower().split()
    if "session" not in args:
        return False
    from hermes_cli.cli_commands_session_tools import _lines, _pr, _t, print_session_diff
    result = connection.call("rollback.session_diff", session_id=cli.session_id) or {}
    if not result.get("enabled"):
        _pr(*_lines(_t("diff.not_enabled")))
    else:
        print_session_diff(cli, result, stat_only=bool({"--stat", "stat"} & set(args)))
    return True


def _reload(cli, connection, cmd):
    from hermes_cli.config import reload_env
    reload_env()
    updated = (connection.call("reload.env", session_id=cli.session_id) or {}).get("updated", 0)
    print(f"  {t('cli.reload.env_reloaded', count=updated)}")
    return True


# A route's verdict for a command that ended the CLI session (a completed /handoff): handled, and
# ``process_command`` answers False as it does for /quit, so the running app exits.
EXIT_SESSION = object()


def _handoff(cli, connection, cmd):
    """``/handoff <platform>`` on the owner: ``handoff.request`` writes the pending row in the owner's
    store, where its watcher claims it (this process's own store write would never be seen), and
    main's two-phase wait (``HermesCLI._handoff_wait``) polls it through the owner."""
    from hermes_cli.cli_commands_session_tools import _command_arg, _cp, _t
    platform_name = _command_arg(cmd, lower=True)
    if not platform_name:
        _cp(f"  {_t('handoff.usage')}", f"  {_t('handoff.usage_detail_1')}", f"  {_t('handoff.usage_detail_2')}")
        return True
    if getattr(cli, "_agent_running", False):
        _cp(f"  {_t('shared.agent_busy', command='/handoff')}")
        return True
    queued = connection.call("handoff.request", session_id=cli.session_id, platform=platform_name) or {}
    _cp(f"  {_t('handoff.queued', title=cli.session_id[:8], platform=platform_name, home=queued.get('home_name', ''))}",
        f"  {_t('handoff.waiting')}")
    kept = cli._handoff_wait(
        platform_name, cli.session_id,
        read_state=lambda: connection.call("handoff.state", session_id=cli.session_id),
        fail_pending=lambda reason: connection.call("handoff.fail", session_id=cli.session_id, error=reason))
    return True if kept else EXIT_SESSION


_ROUTES = {"diff": _diff, "reload": _reload, "handoff": _handoff}
# Named refusals: nothing runs, so the gateway is not told (no ``pre_command``).
REFUSED = frozenset()


def run(cli, canonical, cmd):
    """Truthy when the gateway answered the command (the local handler must not run); ``EXIT_SESSION``
    when that answer ended the CLI session."""
    route = _ROUTES.get(canonical)
    if route is None:
        return False
    from hermes_cli.gateway_side import connection_for
    connection = connection_for(cli)
    if connection is None:
        return False
    try:
        return route(cli, connection, (cmd or "").strip())
    except GatewayClientError as exc:
        from cli import _cprint
        # A refusal's code alone (``invalid_params``) does not say what to fix; its detail does.
        _cprint(f"  /{canonical}: {(getattr(exc, 'data', None) or {}).get('detail') or exc}")
        return True
