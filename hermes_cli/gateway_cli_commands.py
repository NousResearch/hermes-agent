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

from agent.i18n import t
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


def _handoff(cli, connection, cmd):
    """``/handoff <platform>`` on the owner: ``handoff.request`` writes the pending row in the owner's
    store and its watcher performs the transfer, polled here with main's two-phase wait
    (``HermesCLI._handoff_wait``: 60 s to be claimed, then 15 min while the transfer runs)."""
    import time
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
    pending_deadline, running_deadline, next_heartbeat = time.time() + cli._HANDOFF_PENDING_TIMEOUT, None, None
    while True:
        record = connection.call("handoff.state", session_id=cli.session_id) or {}
        current = record.get("state") or "pending"
        if current == "completed":
            _cp("", f"  {_t('handoff.complete', platform=platform_name)}",
                f"  {_t('handoff.resume_hint', title=cli.session_id)}", "")
            from cli import _handed_off_session_ids
            _handed_off_session_ids.add(cli.session_id)  # the gateway owns the row now (#88234)
            cli._should_exit = True  # same exit semantics as /quit
            return True
        if current == "failed":
            _cp(f"  {_t('handoff.failed', error=record.get('error') or _t('handoff.unknown_error'))}",
                f"  {_t('handoff.session_intact_retry')}")
            return True
        now = time.time()
        if current == "running":
            if running_deadline is None:
                _cp(f"  {_t('handoff.picked_up')}")
                running_deadline = now + cli._HANDOFF_RUNNING_TIMEOUT
                next_heartbeat = now + cli._HANDOFF_HEARTBEAT_EVERY
            elif now >= running_deadline:
                # The watcher owns a claimed row: failing it here would split the session.
                _cp(f"  {_t('handoff.taking_long')}", f"  {_t('handoff.check_platform', platform=platform_name)}",
                    f"  {_t('handoff.no_longer_waiting')}", f"  {_t('handoff.retry_when_settled')}")
                return True
            elif now >= next_heartbeat:
                _cp(f"  {_t('handoff.still_transferring')}")
                next_heartbeat = now + cli._HANDOFF_HEARTBEAT_EVERY
        elif now >= pending_deadline:
            # Unclaimed: CAS-fail it so the user can retry (a claim racing this instant wins).
            if (connection.call("handoff.fail", session_id=cli.session_id,
                                error="timed out waiting for gateway") or {}).get("state") == "completed":
                continue
            _cp(f"  {_t('handoff.timed_out')}", f"  {_t('handoff.session_intact')}")
            return True
        time.sleep(0.5)


_ROUTES = {"diff": _diff, "reload": _reload, "handoff": _handoff}
# Named refusals: nothing runs, so the gateway is not told (no ``pre_command``).
REFUSED = frozenset()


def run(cli, canonical, cmd):
    """True when the gateway answered the command (the local handler must not run)."""
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
        _cprint(f"  /{canonical}: {exc}")
        return True
