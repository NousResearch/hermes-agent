from __future__ import annotations

import argparse
import json

from . import core


def register_cli(subparser: argparse.ArgumentParser) -> None:
    subs = subparser.add_subparsers(dest="sillytavern_command")
    subs.add_parser("status", help="Show submodule, process, and endpoint readiness")
    subs.add_parser("capabilities", help="Show supported bridge features and safety gates")

    start = subs.add_parser("start", help="Start the pinned local SillyTavern server")
    start.add_argument("--acknowledge-side-effects", action="store_true")

    stop = subs.add_parser("stop", help="Stop the managed local SillyTavern server")
    stop.add_argument("--acknowledge-side-effects", action="store_true")

    generate = subs.add_parser("generate", help="Send one chat-completions request")
    generate.add_argument("--prompt", required=True)
    generate.add_argument("--model", default="")
    generate.add_argument("--source", default="")
    generate.add_argument("--max-tokens", type=int, default=None)
    generate.add_argument("--temperature", type=float, default=None)
    generate.add_argument("--acknowledge-side-effects", action="store_true")


def _print(payload: dict[str, object]) -> int:
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0 if payload.get("success", True) else 1


def sillytavern_command(args: argparse.Namespace) -> int:
    command = getattr(args, "sillytavern_command", None)
    if command == "status":
        return _print(core.status_payload({}))
    if command == "capabilities":
        return _print(core.capabilities())
    if command == "start":
        return _print(
            core.start_server(
                {"acknowledge_side_effects": args.acknowledge_side_effects}
            )
        )
    if command == "stop":
        return _print(
            core.stop_server(
                {"acknowledge_side_effects": args.acknowledge_side_effects}
            )
        )
    if command == "generate":
        values = {
            "prompt": args.prompt,
            "model": args.model,
            "chat_completion_source": args.source,
            "max_tokens": args.max_tokens,
            "temperature": args.temperature,
            "acknowledge_side_effects": args.acknowledge_side_effects,
        }
        return _print(core.generate(values))
    print("usage: hermes sillytavern {status,capabilities,start,stop,generate}")
    return 2
