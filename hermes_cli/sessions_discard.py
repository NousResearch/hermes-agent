"""``hermes sessions discard``: acknowledge a turn lost across a gateway crash, without a REPL.

After an owner restart mid-turn, the session's ``started`` admission is fenced ``unknown`` and
everything queued behind it waits. A finite ``hermes chat --resume <id> -q`` refuses (exit 3)
rather than queue behind it. Interactively, ``/discard <admission>`` clears it. This is the same
fenced control (``prompt.resolve_unknown`` with the generation the owner stamped) for scripts.
The lost input is not replayed; it stays in the transcript to resend.
"""
from __future__ import annotations

import sys


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


def _confirm(session_id, rows):
    print(f"Discard {len(rows)} lost turn(s) on session {session_id}? They are acknowledged, not replayed.")
    for row in rows:
        print(f"  {row['admission_id']}")
    try:
        return input("Proceed? [y/N] ").strip().lower() in {"y", "yes"}
    except EOFError:
        return False


async def _discard(args):
    from hermes_cli.gateway_client import GatewayClientError, connect_gateway
    async with connect_gateway() as client:
        snapshot = await _resume(client, args.session)
        session_id = snapshot.get("session_id") or snapshot["stored_session_id"]
        unknown = [row for row in snapshot.get("pending", []) if row.get("status") == "unknown"]
        if args.admission:
            chosen = [row for row in unknown if row["admission_id"] in set(args.admission)]
            missing = sorted(set(args.admission) - {row["admission_id"] for row in chosen})
            if missing:
                raise GatewayClientError("Not an unknown (lost) admission of this session: " + ", ".join(missing))
        else:
            chosen = unknown
        if not chosen:
            print(f"Nothing to discard: session {session_id} has no unknown admission.")
            return 0
        if not args.yes:
            if not sys.stdin.isatty():
                print("Refusing to discard without confirmation on a non-interactive stdin; pass --yes.", file=sys.stderr)
                return 2
            if not _confirm(session_id, chosen):
                print("Cancelled; nothing was discarded.")
                return 1
        for row in chosen:
            await client.rpc("prompt.resolve_unknown", session_id=session_id, admission_id=row["admission_id"],
                             execution_generation=row["execution_generation"])
            print(f"Discarded {row['admission_id']}")
        return 0


def add_discard_parser(sessions_subparsers) -> None:
    from hermes_cli.subcommands._shared import add_yes_flag
    parser = sessions_subparsers.add_parser(
        "discard", help="Acknowledge turns lost in a gateway crash (unknown admissions) so the session runs again",
        description="After a gateway crash mid-turn, the lost turn is fenced 'unknown' and blocks the "
            "session: `hermes chat --resume <id> -q` exits 3 until it is acknowledged. This is the "
            "non-interactive form of `/discard <admission>`. The lost input is not replayed.")
    parser.add_argument("session", help="Session ID or title")
    parser.add_argument("--admission", action="append", metavar="ID",
        help="Discard only this unknown admission (repeatable; default: every unknown admission)")
    add_yes_flag(parser, "Do not ask for confirmation (required when stdin is not a TTY)")


def cmd_discard(args) -> int:
    import asyncio
    from websockets.exceptions import WebSocketException
    from hermes_cli.gateway_client import GatewayClientError
    try:
        return asyncio.run(_discard(args))
    except (GatewayClientError, OSError, TimeoutError, WebSocketException) as exc:
        message = str(exc) if isinstance(exc, GatewayClientError) else "Gateway connection/read failed"
        print("Error: " + message, file=sys.stderr)
        return 1
