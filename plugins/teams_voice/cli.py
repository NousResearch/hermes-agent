"""``hermes teams-voice`` CLI subcommands."""

from __future__ import annotations

import argparse
import asyncio
import json

from .config import resolve_config


def register_cli(subparser: argparse.ArgumentParser) -> None:
    """Build the ``hermes teams-voice`` argparse tree."""
    subs = subparser.add_subparsers(dest="teams_voice_command")
    subs.add_parser("status", help="Print bridge configuration and readiness")
    serve_p = subs.add_parser("serve", help="Run the bridge WebSocket server (foreground)")
    serve_p.add_argument("--host", default=None, help="Override bind host")
    serve_p.add_argument("--port", type=int, default=None, help="Override bind port")
    serve_p.add_argument(
        "--handler",
        choices=("logging", "echo", "realtime"),
        default="logging",
        help=(
            "Call brain: 'logging' (no audio back), 'echo' (smile + echo caller "
            "audio so the avatar lip-syncs — smoke test), 'realtime' (OpenAI/Azure "
            "speech-to-speech; needs OPENAI_API_KEY)."
        ),
    )


def teams_voice_command(args) -> int:
    """Dispatch ``hermes teams-voice`` subcommands. Returns an exit code."""
    command = getattr(args, "teams_voice_command", None)

    if command == "status":
        cfg = resolve_config()
        print(
            json.dumps(
                {
                    "configured": cfg.configured,
                    "host": cfg.host,
                    "port": cfg.port,
                    "path": cfg.path,
                },
                indent=2,
            )
        )
        return 0

    if command == "serve":
        from .bridge_server import BridgeServer, CallSessionHandler

        cfg = resolve_config()
        if not cfg.configured:
            print("error: no shared secret (set TEAMS_VOICE_SHARED_SECRET)")
            return 1
        if args.host or args.port:
            from dataclasses import replace

            cfg = replace(cfg, host=args.host or cfg.host, port=args.port or cfg.port)

        handler_kind = getattr(args, "handler", "logging")
        factory = CallSessionHandler  # default: log only, no audio back
        if handler_kind == "echo":
            from .handlers import EchoCallSessionHandler

            factory = EchoCallSessionHandler
        elif handler_kind == "realtime":
            from .handlers import RealtimeCallSessionHandler
            from .realtime.openai_client import realtime_config_from_env

            rt_cfg = realtime_config_from_env()
            if not rt_cfg.configured:
                print(
                    "error: realtime handler needs an API key "
                    "(OPENAI_API_KEY, or AZURE_FOUNDRY_API_KEY / TEAMS_VOICE_REALTIME_API_KEY for Azure)"
                )
                return 1
            factory = lambda: RealtimeCallSessionHandler(rt_cfg, bridge_config=cfg)  # noqa: E731

        async def _run() -> None:
            server = BridgeServer(config=cfg, handler_factory=factory)
            await server.start()
            try:
                await asyncio.Future()
            finally:
                await server.stop()

        try:
            asyncio.run(_run())
        except KeyboardInterrupt:
            pass
        return 0

    print("usage: hermes teams-voice {status|serve}")
    return 2
