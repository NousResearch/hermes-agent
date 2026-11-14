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
import sys

_RUNNING_STATES = frozenset({"running", "waiting"})


async def _resume(client, name):
    """Exact session id first, then title, as ``hermes chat --resume`` / ``-c`` resolve it."""
    from hermes_cli.gateway_client import GatewayClientError
    for key in ("session_id", "title"):
        try:
            return await client.rpc("session.resume", **{key: name})
        except GatewayClientError as exc:
            if str(exc) != "not_found":
                raise
    raise GatewayClientError(f"No session found matching '{name}'. Use 'hermes sessions list' to see available sessions.")


def _report(args, payload: dict) -> None:
    if getattr(args, "json", False):
        print(json.dumps(payload, sort_keys=True))
    elif payload["interrupted"]:
        print(f"Interrupted session {payload['session_id']} (generation {payload['execution_generation']})")
    else:
        print(f"Nothing to interrupt: session {payload['session_id']} is {payload['execution_state']}.")


async def _interrupt(args):
    from hermes_cli.gateway_client import connect_gateway
    async with connect_gateway() as client:
        snapshot = await _resume(client, args.session)
        session_id = snapshot.get("session_id") or snapshot["stored_session_id"]
        state = snapshot.get("execution_state", "unknown")
        generation = snapshot.get("execution_generation", 0)
        payload = {"session_id": session_id, "execution_state": state,
                   "execution_generation": generation, "interrupted": False}
        if state not in _RUNNING_STATES:
            _report(args, payload)
            return 3
        await client.rpc("session.interrupt", session_id=session_id, execution_generation=generation)
        payload["interrupted"] = True
        _report(args, payload)
        return 0


def add_interrupt_parser(sessions_subparsers) -> None:
    parser = sessions_subparsers.add_parser(
        "interrupt", aliases=["stop"], help="Stop a gateway-hosted session's running turn (non-interactive /stop)",
        description="Deliver /stop to the running turn of a gateway-hosted session by id or title, so a "
            "supervisor can end a hung `hermes chat -q` run. Killing the client only detaches; the turn "
            "keeps running at the gateway. Exit 0 = interrupted, 3 = no running turn, 1 = error.")
    parser.add_argument("session", help="Session ID or title")
    parser.add_argument("--json", action="store_true", help="Print a single JSON result line")
    parser.set_defaults(sessions_action="interrupt")


def cmd_interrupt(args) -> int:
    import asyncio
    from websockets.exceptions import WebSocketException
    from hermes_cli.gateway_client import GatewayClientError
    try:
        return asyncio.run(_interrupt(args))
    except (GatewayClientError, OSError, TimeoutError, WebSocketException) as exc:
        message = str(exc) if isinstance(exc, GatewayClientError) else "Gateway connection/read failed"
        print("Error: " + message, file=sys.stderr)
        return 1
