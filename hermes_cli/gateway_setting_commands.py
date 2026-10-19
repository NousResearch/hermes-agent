"""Per-session setting commands of the classic CLI on the shared gateway (``slash.exec``).

On main ``/personality`` ``/busy`` ``/approvals`` ``/footer`` ``/fast`` changed this process's own
agent or config. Rewired, the session's agent lives in the gateway, so the gateway's handler
decides and applies the change (gateway/session_setting_commands.py; RULING §1: one behaviour per
command, every surface) and the CLI prints its reply. Client-side state the CLI itself enforces
(``/busy``: what Enter does while a turn runs) is mirrored locally after the owner accepts it.
``/refine`` and ``/review`` take the same route: their background reviewer must run beside the
session's agent in the gateway (as on messaging), where its result is delivered back to the session.
"""
from __future__ import annotations

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


GATEWAY_SETTINGS = {"personality": None, "busy": _mirror_busy, "approvals": None, "footer": None, "fast": None,
                    "refine": None, "review": _watch_reviewer}


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
    parts = (cmd or "").strip().split(None, 1)
    arg = parts[1].strip() if len(parts) > 1 else ""
    try:
        reply = connection.call("slash.exec", session_id=cli.session_id,
                                command=canonical + (" " + arg if arg else ""))
    except GatewayClientError as exc:
        _cprint(f"  /{canonical}: {exc}")
        return True
    for line in str((reply or {}).get("output") or "").splitlines():
        _cprint(f"  {line}" if line else "")
    mirror = GATEWAY_SETTINGS[canonical]
    if mirror is not None:
        mirror(cli, arg)
    return True

