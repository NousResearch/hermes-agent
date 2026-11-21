"""``hermes sessions interrupt``: stop a gateway-hosted session's running turn by id, without a REPL.

An external supervisor (cron watchdog, kanban dispatcher, CI wrapper) that launched
``hermes chat -q`` knows the session id the client printed but has no way to stop a hung
turn short of killing the client, which only *detaches*: accepted work keeps running at the
gateway. This is the same fenced control as interactive ``/stop`` (``session.interrupt`` with
the execution generation the owner reports) for scripts.

Exit codes: 0 interrupt delivered, 3 the session has no running turn, 1 error.
"""
from __future__ import annotations

import json

from hermes_cli.sessions.target import attach, paused, run_gateway_verb, session_argument


def _execution_state(snapshot) -> str:
    """The owner's resume reply says ``running`` (a claimed turn) and lists ``pending`` rows; an
    unknown row (a turn lost in a crash) pauses the session instead."""
    if snapshot.get("running") or any(row.get("status") == "started" for row in snapshot.get("pending") or ()):
        return "running"
    return "unknown" if paused(snapshot.get("pending")) else "idle"


def _report(args, payload: dict) -> None:
    if getattr(args, "json", False):
        print(json.dumps(payload, sort_keys=True))
    elif payload["interrupted"]:
        print(f"Interrupted session {payload['session_id']} (generation {payload['execution_generation']})")
    else:
        print(f"Nothing to interrupt: session {payload['session_id']} is {payload['execution_state']}.")


async def _interrupt(client, args):
    snapshot = await attach(client, args.session, omit_messages=True)
    session_id = snapshot.get("session_id") or snapshot["stored_session_id"]
    state = _execution_state(snapshot)
    generation = snapshot.get("execution_generation", 0)
    payload = {"session_id": session_id, "execution_state": state,
               "execution_generation": generation, "interrupted": False}
    if state != "running":
        _report(args, payload)
        return 3
    await client.rpc("session.interrupt", session_id=session_id, execution_generation=generation)
    payload["interrupted"] = True
    _report(args, payload)
    return 0


def add_interrupt_parser(sessions_subparsers) -> None:
    parser = sessions_subparsers.add_parser(
        "interrupt", aliases=["stop"], help="Stop a gateway-hosted session's running turn (non-interactive /stop)",
        description="Deliver /stop to the running turn of a gateway-hosted session by id, id prefix, title or '-', so a "
            "supervisor can end a hung `hermes chat -q` run. Killing the client only detaches; the turn "
            "keeps running at the gateway. Exit 0 = interrupted, 3 = no running turn, 1 = error.")
    session_argument(parser)
    parser.add_argument("--json", action="store_true", help="Print a single JSON result line")
    parser.set_defaults(sessions_action="interrupt")


def cmd_interrupt(args) -> int:
    return run_gateway_verb(_interrupt, args)
