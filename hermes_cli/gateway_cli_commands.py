"""Classic CLI commands whose state lives in the gateway session, answered there.

Rewired, the session's tools, checkpoints, environment and handoff watcher run in the gateway, so
main's handlers (which read this process's own agent and environment) answer nothing or the wrong
thing:

- ``/diff session``: the checkpoint baseline is the gateway tools'. ``rollback.session_diff`` reads it
  under the session's frozen checkpoint policy (``--checkpoints`` included) and returns it as data
  for main's terminal renderer. Working-tree modes stay main's local git read of the same checkout.
- ``/reload``: ``reload.env`` re-reads ``.env`` where turns and tools run (this process re-reads too).
- ``/handoff``: no gateway verb carries a handoff yet; refused as Ink and Desktop refuse it, instead
  of the store's ``request_handoff`` refusal dying silently in the process loop.
- ``/discard [admission_id ...]``: acknowledges the turns a gateway restart left ``unknown``
  (``prompt.resolve_unknown``), the in-chat form of ``hermes sessions discard``.
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
    from cli import _cprint
    from hermes_cli.gateway_chat_commands import refusal
    _cprint(f"  {refusal('handoff')}")
    return True


def _discard(cli, connection, cmd):
    """Acknowledge the turns a gateway restart lost (``prompt.resolve_unknown``, fenced by the
    generation the owner stamped): they are not replayed, and the session FIFO runs again.

    A bare ``/discard`` acknowledges EVERY lost turn in the session, so it asks first, as
    ``hermes sessions discard`` does; naming admission ids is the explicit form that does not ask.
    """
    import sys

    from cli import _cprint
    wanted = set(cmd.partition(" ")[2].split())
    snapshot = connection.call("session.resume", session_id=cli.session_id) or {}
    lost = [row for row in snapshot.get("pending", []) if row.get("status") == "unknown"
            and (not wanted or row["admission_id"] in wanted)]
    if not lost:
        _cprint("  /discard: no lost turn in this session; nothing to discard.")
        return True
    if not wanted:
        ids = ", ".join(row["admission_id"] for row in lost)
        if not sys.stdin.isatty():
            _cprint(f"  /discard: {len(lost)} lost turn(s) here ({ids}); this session is not interactive,"
                    " so name the admission id(s) to discard: /discard <id>")
            return True
        _cprint(f"  This discards {len(lost)} lost turn(s) ({ids}) without replaying them.")
        if input("  Proceed? [y/N] ").strip().lower() not in {"y", "yes"}:
            _cprint("  Nothing discarded.")
            return True
    for row in lost:
        connection.call("prompt.resolve_unknown", session_id=cli.session_id, admission_id=row["admission_id"],
                        execution_generation=row["execution_generation"])
        _cprint(f"  Discarded lost turn {row['admission_id']} (not replayed).")
    # The next prompt reads this state: refresh it now rather than wait for the session.info frame.
    after = connection.call("session.resume", session_id=cli.session_id) or {}
    connection.state(cli.session_id).pending = list(after.get("pending") or [])
    _cprint("  Queued messages in this session can run again.")
    return True


_ROUTES = {"diff": _diff, "reload": _reload, "handoff": _handoff, "discard": _discard}
# Named refusals: nothing runs, so the gateway is not told (no ``pre_command``).
REFUSED = frozenset({"handoff"})


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
