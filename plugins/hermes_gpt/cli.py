from __future__ import annotations

import argparse
import json
from typing import Any

from . import server


def register_cli(subparser: argparse.ArgumentParser) -> None:
    subs = subparser.add_subparsers(dest="hermes_gpt_command")

    serve = subs.add_parser("serve", help="Run the MCP sidecar")
    serve.add_argument("--http", action="store_true", help="Run streamable HTTP transport")
    serve.add_argument("--sse", action="store_true", help="Run legacy SSE transport")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=7677)
    serve.add_argument("--cert", default="", help="TLS certificate path for HTTP/SSE")
    serve.add_argument("--key", default="", help="TLS key path for HTTP/SSE")
    serve.add_argument(
        "--profile",
        choices=[server.LOCAL_DEV_PROFILE, server.REMOTE_PROFILE],
        default=server.LOCAL_DEV_PROFILE,
    )
    serve.add_argument(
        server.UNSAFE_REMOTE_ACK,
        action="store_true",
        dest="unsafe_remote_ack",
        help="Allow remote no-auth mode for temporary experiments only",
    )

    subs.add_parser("status", help="Show import status and visible MCP tools")

    subparser.set_defaults(func=hermes_gpt_command)


def hermes_gpt_command(args: Any) -> int:
    command = getattr(args, "hermes_gpt_command", None)
    if command == "serve":
        argv = _server_argv(args)
        return server.main(argv)
    if command == "status":
        print(json.dumps(server.status(), ensure_ascii=False, indent=2))
        return 0 if server.IMPORT_ERROR is None else 1
    print("usage: hermes hermes-gpt {status,serve}")
    return 2


def _server_argv(args: Any) -> list[str]:
    argv: list[str] = []
    if getattr(args, "http", False):
        argv.append("--http")
    if getattr(args, "sse", False):
        argv.append("--sse")
    argv.extend(["--host", str(getattr(args, "host", "127.0.0.1"))])
    argv.extend(["--port", str(getattr(args, "port", 7677))])
    cert = getattr(args, "cert", "") or ""
    key = getattr(args, "key", "") or ""
    if cert:
        argv.extend(["--cert", cert])
    if key:
        argv.extend(["--key", key])
    argv.extend(["--profile", getattr(args, "profile", server.LOCAL_DEV_PROFILE)])
    if getattr(args, "unsafe_remote_ack", False):
        argv.append(server.UNSAFE_REMOTE_ACK)
    return argv
