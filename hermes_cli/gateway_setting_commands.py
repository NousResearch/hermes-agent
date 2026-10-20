"""Per-session setting commands of the classic CLI on the shared gateway (``slash.exec``).

On main ``/personality`` ``/busy`` ``/approvals`` ``/footer`` ``/fast`` changed this process's own
agent or config. Rewired, the session's agent lives in the gateway, so the gateway's handler
decides and applies the change (gateway/session_setting_commands.py; RULING §1: one behaviour per
command, every surface) and the CLI prints its reply. Client-side state the CLI itself enforces
(``/busy``: what Enter does while a turn runs) is mirrored locally after the owner accepts it.
``/refine`` and ``/review`` take the same route: their background reviewer must run beside the
session's agent in the gateway (as on messaging), where its result is delivered back to the session.
``/cron`` too: the profile's cron store belongs to the gateway that fires its jobs, so the gateway runs
main's handler (one writer); ``/cron run`` lasts as long as the job, so a write waits with no deadline.
``/toolsets`` too: the gateway builds the report from its own registry (MCP and plugin toolsets
included) with this session's toolsets marked (gateway/session_info_reads.py).
"""
from __future__ import annotations

import shlex

# Canonical command -> mirror(cli, arg) run after the owner applied it (None: nothing local).
_BUSY_MODES = frozenset({"queue", "steer", "interrupt"})


def _mirror_busy(cli, arg):
    if arg.lower() in _BUSY_MODES:
        cli.busy_input_mode = arg.lower()


def _watch_reviewer(cli, _arg):
    """/review spawns its reviewer off-turn: have the live-work dock poll for it right away."""
    from hermes_cli.gateway_tools import process_view
    view = process_view()
    if view is not None:
        view.watch_subagents()


# Reports the owner already lays out (main's own line builders): printed as is.
_VERBATIM = frozenset({"toolsets"})
GATEWAY_SETTINGS = {"personality": None, "busy": _mirror_busy, "approvals": None, "footer": None, "fast": None,
                    "refine": None, "review": _watch_reviewer, "cron": None,
                    "toolsets": None}


def _unbounded(canonical, arg):
    """True for a form answered only once its work is done, however long: a /cron write (``run`` runs the
    job). Its reads keep the deadline, and so does an unclosed quote, which the owner refuses at once."""
    if canonical != "cron":
        return False
    from gateway.session_cron_commands import cron_read
    try:
        return cron_read(shlex.split(arg)) is None
    except ValueError:
        return False


def _call(cli, connection, canonical, arg):
    command = canonical + (" " + arg if arg else "")
    if not _unbounded(canonical, arg):
        return connection.call("slash.exec", session_id=cli.session_id, command=command)
    # No deadline, on a socket of its own so the session's prompts never queue behind it
    # (GatewayConnection.call_until); Ctrl+C stops waiting.
    from hermes_cli.gateway_plugin_commands import call_unbounded
    return call_unbounded(cli, connection, canonical, command)


def run(cli, canonical, cmd):
    """True when the gateway answered the command (the local handler must not run)."""
    from hermes_cli.gateway_side import connection_for
    if canonical not in GATEWAY_SETTINGS:
        return False
    connection = connection_for(cli)
    if connection is None:
        return False
    from cli import _cprint
    from hermes_cli.gateway_client import GatewayClientError
    from hermes_cli.gateway_plugin_commands import STOPPED
    parts = (cmd or "").strip().split(None, 1)
    arg = parts[1].strip() if len(parts) > 1 else ""
    try:
        reply = _call(cli, connection, canonical, arg)
    except GatewayClientError as exc:
        _cprint(f"  /{canonical}: {exc}")
        return True
    if reply is STOPPED:
        return True
    for line in str((reply or {}).get("output") or "").splitlines():
        _cprint(f"  {line}" if line and canonical not in _VERBATIM else line)
    mirror = GATEWAY_SETTINGS[canonical]
    if mirror is not None:
        mirror(cli, arg)
    return True
