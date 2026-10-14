"""Classic CLI commands whose state lives in the gateway session, answered there.

Rewired, the session's tools, checkpoints, environment and handoff watcher run in the gateway, so
main's handlers (which read this process's own agent and environment) answer nothing or the wrong
thing:

- ``/diff session``: the checkpoint baseline is the gateway tools'; its ``/diff`` handler (a reviewed
  ``slash.exec`` read, run in this session's cwd) renders it. Working-tree modes stay main's local
  git read of the same checkout.
- ``/reload``: ``reload.env`` re-reads ``.env`` where turns and tools run (this process re-reads too).
- ``/handoff``: no gateway verb carries a handoff yet; refused as Ink and Desktop refuse it, instead
  of the store's ``request_handoff`` refusal dying silently in the process loop.
"""
from __future__ import annotations

from agent.i18n import t
from hermes_cli.gateway_client import GatewayClientError


def _diff(cli, connection, cmd):
    args = cmd.partition(" ")[2].strip()
    if "session" not in args.lower().split():
        return False
    from cli import _cprint
    reply = connection.call("slash.exec", session_id=cli.session_id, command=f"diff {args}")
    # The messaging handler fences its blocks for chat; a terminal shows the text itself.
    for line in str((reply or {}).get("output") or "").splitlines():
        if not line.startswith("```"):
            _cprint(f"  {line}" if line else "")
    return True


def _reload(cli, connection, cmd):
    from hermes_cli.config import reload_env
    reload_env()
    updated = (connection.call("reload.env", session_id=cli.session_id) or {}).get("updated", 0)
    print(f"  {t('cli.reload.env_reloaded', count=updated)}")
    return True


def _handoff(cli, connection, cmd):
    from cli import _cprint
    from hermes_cli.gateway_chat_commands import refusal
    _cprint(f"  {refusal('handoff')}")
    return True


_ROUTES = {"diff": _diff, "reload": _reload, "handoff": _handoff}


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
