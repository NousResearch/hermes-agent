"""Owner-mediated compression-route repair for an existing local conversation."""

import asyncio
import sys
import uuid


def add_reset_compression_parser(subparsers):
    parser = subparsers.add_parser(
        "reset-compression",
        help="Use this profile's current compression model in an existing chat",
        description="Replace the saved auxiliary.compression settings of an idle local chat with "
        "this profile's current settings. Keeps history and the main model. "
        "An absent compression override restores automatic selection.",
    )
    parser.add_argument("session", help="Session ID (from hermes sessions list)")


async def _reset(args):
    from hermes_cli.gateway_client import connect_gateway

    async with connect_gateway() as client:
        snapshot = await client.rpc(
            "session.resume", session_id=args.session, omit_messages=True
        )
        sid = snapshot["session_id"]
        try:
            await client.rpc(
                "session.mutate",
                session_id=sid,
                request_id=uuid.uuid4().hex,
                operation="compression_model",
                payload={},
                expected_revision=snapshot["revision"],
                expected_generation=snapshot["execution_generation"],
            )
        finally:
            await client.rpc(
                "session.detach",
                session_id=sid,
                subscription_id=snapshot["subscription_id"],
            )
    print(
        f"Session {sid} now uses this profile's current compression settings. History and main model kept."
    )
    return 0


def cmd_reset_compression(args):
    from websockets.exceptions import WebSocketException
    from hermes_cli.gateway_client import GatewayClientError

    try:
        return asyncio.run(_reset(args))
    except (GatewayClientError, OSError, TimeoutError, WebSocketException) as exc:
        message = (
            str(exc)
            if isinstance(exc, GatewayClientError)
            else "Gateway connection/read failed"
        )
        print("Error: " + message, file=sys.stderr)
        return 1
