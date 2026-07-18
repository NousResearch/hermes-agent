"""Plugin-local ``hermes a2a`` configuration and named-peer CLI."""

from __future__ import annotations

import argparse
import asyncio
import getpass
import json
import sys
from typing import Any

from . import auth
from . import config
from . import setup

_SUCCESS = 0
_FAILURE = 1
_USAGE = 2


def register_cli(parser: argparse.ArgumentParser) -> None:
    subs = parser.add_subparsers(dest="a2a_command", required=False)
    p_setup = subs.add_parser("setup", help="Enable the A2A platform safely")
    p_setup.add_argument("--public-url")
    subs.add_parser("status", help="Show non-secret A2A configuration status")

    p_peer = subs.add_parser("peer", help="Manage named outbound A2A peers")
    peer_subs = p_peer.add_subparsers(dest="peer_command", required=True)
    peer_subs.add_parser("list")
    p_peer_add = peer_subs.add_parser("add")
    p_peer_add.add_argument("name")
    p_peer_add.add_argument("url")
    p_peer_add.add_argument(
        "--token-stdin",
        action="store_true",
        help="Read the bearer from stdin instead of a hidden prompt",
    )
    p_peer_remove = peer_subs.add_parser("remove")
    p_peer_remove.add_argument("name")

    p_principal = subs.add_parser("principal", help="Manage inbound peer identities")
    principal_subs = p_principal.add_subparsers(dest="principal_command", required=True)
    principal_subs.add_parser("list")
    p_principal_add = principal_subs.add_parser("add")
    p_principal_add.add_argument("name")
    p_principal_add.add_argument("--profile", required=True)
    p_principal_remove = principal_subs.add_parser("remove")
    p_principal_remove.add_argument("name")

    p_credential = subs.add_parser("credential", help="Rotate inbound credentials")
    credential_subs = p_credential.add_subparsers(dest="credential_command", required=True)
    p_rotate = credential_subs.add_parser("rotate")
    p_rotate.add_argument("principal")

    p_card = subs.add_parser("card", help="Fetch a configured peer's Agent Card")
    p_card.add_argument("peer")
    p_card.add_argument("--json", action="store_true")

    p_ask = subs.add_parser("ask", help="Send text to a configured peer")
    p_ask.add_argument("peer")
    p_ask.add_argument("message", nargs="?")
    p_ask.add_argument("--stdin", action="store_true", help="Read the full message from stdin")
    context = p_ask.add_mutually_exclusive_group()
    context.add_argument("--new-context", action="store_true")
    context.add_argument("--context-id")
    p_ask.add_argument("--json", action="store_true")

    for name, help_text in (
        ("get", "Get a task from a configured peer"),
        ("cancel", "Cancel a task on a configured peer"),
    ):
        task_parser = subs.add_parser(name, help=help_text)
        task_parser.add_argument("peer")
        task_parser.add_argument("task_id")
        task_parser.add_argument("--json", action="store_true")

    p_list = subs.add_parser("list", help="List tasks from a configured peer")
    p_list.add_argument("peer")
    p_list.add_argument("--json", action="store_true")
    parser.set_defaults(func=dispatch)


def _read_peer_token(*, from_stdin: bool) -> str:
    if from_stdin:
        value = sys.stdin.readline()
    else:
        value = getpass.getpass("Peer bearer token: ")
    return value.strip()


def _show_status() -> int:
    settings = config.load_a2a_settings()
    summary = auth.credential_summary()
    print(f"enabled: {'yes' if settings.enabled else 'no'}")
    print(f"principals: {len(settings.principals)}")
    print(f"peers: {len(settings.peers)}")
    print(f"inbound credentials: {len(summary['inbound'])}")
    print(f"outbound credentials: {len(summary['outbound'])}")
    return _SUCCESS


def _message_to_dict(message: Any) -> dict[str, Any]:
    # Lazy so configuration-only commands do not require the optional SDK.
    from google.protobuf.json_format import MessageToDict

    return MessageToDict(message)


def _load_client_class():
    # Importing client imports the optional A2A SDK, so keep it off P1 commands.
    from .client import NamedPeerClient

    return NamedPeerClient


def _state_name(task: Any) -> str:
    field = task.status.DESCRIPTOR.fields_by_name["state"]
    value = field.enum_type.values_by_number.get(task.status.state)
    return value.name if value else str(task.status.state)


def _artifact_text(task: Any) -> list[str]:
    return [
        part.text
        for artifact in getattr(task, "artifacts", ())
        for part in artifact.parts
        if part.WhichOneof("content") == "text"
    ]


def _print_task(task: Any, *, texts: list[str] | None = None) -> None:
    print(f"task id: {task.id}")
    print(f"context id: {task.context_id}")
    print(f"state: {_state_name(task)}")
    for text in texts if texts is not None else _artifact_text(task):
        print(text)


async def _close_even_if_cancelled(api: Any) -> None:
    cleanup = asyncio.create_task(api.aclose(), name="a2a-cli-close")
    try:
        await asyncio.shield(cleanup)
    except asyncio.CancelledError:
        # The cleanup task is independently owned and must be observed before
        # propagating cancellation; this also prevents dangling HTTP clients.
        await cleanup
        raise


async def _run_outbound(args: argparse.Namespace) -> int:
    peer = config.validate_name(args.peer, label="peer")
    message = None
    if args.a2a_command == "ask":
        if args.message is not None and args.stdin:
            raise ValueError("provide either MESSAGE or --stdin, not both")
        message = sys.stdin.read() if args.stdin else args.message
        if message is None or not message.strip():
            raise ValueError("MESSAGE is required (or pass --stdin)")
    api = _load_client_class()()
    try:
        command = args.a2a_command
        if command == "card":
            result = await api.fetch_card(peer)
        elif command == "ask":
            result, texts = await api.ask(
                peer,
                message,
                new_context=bool(args.new_context),
                context_id=args.context_id,
            )
            if args.json:
                print(json.dumps(_message_to_dict(result), sort_keys=True))
            else:
                _print_task(result, texts=texts)
            return _SUCCESS
        elif command == "get":
            result = await api.get_task(peer, args.task_id)
        elif command == "list":
            result = await api.list_tasks(peer)
        elif command == "cancel":
            result = await api.cancel(peer, args.task_id)
        else:  # pragma: no cover - parser owns this invariant
            raise ValueError("unsupported outbound command")

        if args.json:
            print(json.dumps(_message_to_dict(result), sort_keys=True))
        elif command == "card":
            print(f"name: {result.name}")
            print(f"version: {result.version}")
            for interface in result.supported_interfaces:
                print(f"interface: {interface.protocol_binding} {interface.protocol_version}")
        elif command == "list":
            for task in result.tasks:
                _print_task(task)
        else:
            _print_task(result)
        return _SUCCESS
    finally:
        await _close_even_if_cancelled(api)


def _dispatch_outbound(args: argparse.Namespace) -> int:
    try:
        return asyncio.run(_run_outbound(args))
    except (ValueError, TypeError) as exc:
        print(f"hermes a2a: {exc}", file=sys.stderr)
        return _USAGE
    except KeyboardInterrupt:
        print("hermes a2a: interrupted", file=sys.stderr)
        return _FAILURE
    except asyncio.CancelledError:
        print("hermes a2a: interrupted", file=sys.stderr)
        return _FAILURE
    except Exception:
        # Client exceptions are deliberately sanitized at their source. Do not
        # leak URLs, credentials, response bodies, or nested SDK exceptions.
        print("hermes a2a: peer request failed", file=sys.stderr)
        return _FAILURE


def dispatch(args: argparse.Namespace) -> int:
    command = getattr(args, "a2a_command", None) or "status"
    if command in {"card", "ask", "get", "list", "cancel"}:
        return _dispatch_outbound(args)
    try:
        if command == "setup":
            setup.ensure_a2a_platform_config(public_url=getattr(args, "public_url", None))
            print("A2A enabled with zero default tools.")
            return _SUCCESS
        if command == "status":
            return _show_status()
        if command == "peer":
            action = args.peer_command
            if action == "list":
                for name, entry in sorted(config.load_a2a_settings().peers.items()):
                    print(f"{name}\t{entry.get('url', '')}")
                return _SUCCESS
            if action == "add":
                token = _read_peer_token(from_stdin=bool(args.token_stdin))
                setup.add_peer(args.name, url=args.url, token=token)
                print(f"Added peer {args.name}.")
                return _SUCCESS
            if action == "remove":
                return _SUCCESS if setup.remove_peer(args.name) else _FAILURE
        if command == "principal":
            action = args.principal_command
            if action == "list":
                for name, entry in sorted(config.load_a2a_settings().principals.items()):
                    print(f"{name}\tprofile={entry.get('profile', '')}")
                return _SUCCESS
            if action == "add":
                token = setup.add_principal(args.name, profile=args.profile)
                print("Inbound bearer (shown once; store it securely):")
                print(token)
                return _SUCCESS
            if action == "remove":
                return _SUCCESS if setup.remove_principal(args.name) else _FAILURE
        if command == "credential" and args.credential_command == "rotate":
            token = setup.rotate_principal_credential(args.principal)
            print("Replacement inbound bearer (shown once; store it securely):")
            print(token)
            return _SUCCESS
    except (KeyError, RuntimeError, ValueError) as exc:
        print(f"hermes a2a: {exc}", file=sys.stderr)
        return _FAILURE
    print(f"hermes a2a: unsupported command {command}", file=sys.stderr)
    return _USAGE
