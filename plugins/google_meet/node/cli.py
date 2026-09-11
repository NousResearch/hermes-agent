"""`hermes meet node ...` subcommand tree, wired under the ``hermes meet`` parser."""

from __future__ import annotations

import argparse
import asyncio
import json
import sys

from plugins.google_meet.node.client import NodeClient
from plugins.google_meet.node.registry import NodeRegistry
from plugins.google_meet.node.server import NodeServer


def register_cli(subparser: argparse.ArgumentParser) -> None:
    """Add ``run / list / approve / remove / status / ping`` subparsers to the ``node`` parser."""
    sp = subparser.add_subparsers(dest="node_cmd", required=True)
    run = sp.add_parser("run", help="Start a node server on this machine.")
    run.add_argument("--host", default="0.0.0.0")
    run.add_argument("--port", type=int, default=18789)
    run.add_argument("--display-name", default="hermes-meet-node")
    sp.add_parser("list", help="List approved remote nodes.")
    app = sp.add_parser("approve", help="Register a remote node on the gateway.")
    for arg in ("name", "url", "token"):
        app.add_argument(arg)
    for name, help_ in (("remove", "Forget a registered node."), ("status", "Ping a registered node."),
                        ("ping", "Alias for status.")):
        sp.add_parser(name, help=help_).add_argument("name")
    for p in sp.choices.values():
        p.set_defaults(func=node_command)


def _cmd_run(args: argparse.Namespace, reg: NodeRegistry) -> int:
    server = NodeServer(host=args.host, port=args.port, display_name=args.display_name)
    token = server.ensure_token()
    print(f"[meet-node] display_name={server.display_name}\n"
          f"[meet-node] listening on ws://{args.host}:{args.port}\n"
          f"[meet-node] token (copy to gateway): {token}\n[meet-node] approve with:\n"
          f"             hermes meet node approve <name> ws://<host>:{args.port} {token}")
    try:
        asyncio.run(server.serve())
    except KeyboardInterrupt:
        pass
    except RuntimeError as exc:
        print(f"[meet-node] error: {exc}", file=sys.stderr)
        return 2
    return 0


def _cmd_list(args: argparse.Namespace, reg: NodeRegistry) -> int:
    nodes = reg.list_all()
    if not nodes:
        print("no nodes registered")
    for n in nodes:
        print(f"{n['name']}\t{n['url']}\ttoken={n['token'][:6]}…")
    return 0


def _cmd_approve(args: argparse.Namespace, reg: NodeRegistry) -> int:
    reg.add(args.name, args.url, args.token)
    print(f"approved node {args.name!r} at {args.url}")
    return 0


def _cmd_remove(args: argparse.Namespace, reg: NodeRegistry) -> int:
    ok = reg.remove(args.name)
    print(f"removed {args.name!r}" if ok else f"no such node: {args.name!r}")
    return 0 if ok else 1


def _cmd_ping(args: argparse.Namespace, reg: NodeRegistry) -> int:
    entry = reg.get(args.name)
    if entry is None:
        print(f"no such node: {args.name!r}", file=sys.stderr)
        return 1
    try:
        result = NodeClient(entry["url"], entry["token"]).ping()
    except Exception as exc:  # noqa: BLE001 — surface any connection error
        print(json.dumps({"ok": False, "error": str(exc)}))
        return 1
    print(json.dumps({"ok": True, "node": args.name, **result}))
    return 0


_COMMANDS = {
    "run": _cmd_run,
    "list": _cmd_list,
    "approve": _cmd_approve,
    "remove": _cmd_remove,
    "status": _cmd_ping,
    "ping": _cmd_ping}


def node_command(args: argparse.Namespace) -> int:
    """Dispatch for ``hermes meet node ...``; returns a process exit code."""
    cmd = getattr(args, "node_cmd", None)
    handler = _COMMANDS.get(cmd or "")
    if handler is None:
        print(f"unknown node command: {cmd!r}", file=sys.stderr)
        return 2
    # ``run`` never touches the registry; constructing it is side-effect free.
    return handler(args, NodeRegistry())


# ---- BEGIN PLUGIN-COMPAT (revert-scheduled; see COMPAT_MANIFEST.md) ----
# Names external plugins imported from this module before the Sep 2026 decomposition.
# Internal code MUST NOT use these (scripts/check_compat_pointers.py fails CI if it does).
# The whole block is removed by reverting the commit that added it.
from typing import Any  # noqa: F401,E402
# ---- END PLUGIN-COMPAT ----
