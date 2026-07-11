from __future__ import annotations

import argparse
import json

from . import core


def register_cli(subparser: argparse.ArgumentParser) -> None:
    subs = subparser.add_subparsers(dest="openmanus_command")
    subs.add_parser("status", help="Show the pinned submodule and runtime readiness")
    subs.add_parser("capabilities", help="Show supported agent modes and safety policy")

    run = subs.add_parser("run", help="Plan or run one OpenManus task")
    run.add_argument("--prompt", required=True)
    run.add_argument("--workspace", default="")
    run.add_argument("--execute", action="store_true", help="Run instead of planning")
    run.add_argument("--acknowledge-side-effects", action="store_true")
    run.add_argument("--allow-network", action="store_true")
    run.add_argument("--agent-mode", choices=["manus", "data_analysis"], default="manus")
    run.add_argument("--max-steps", type=int, default=20)
    run.add_argument("--timeout-seconds", type=int, default=600)

    wide = subs.add_parser("wide-research", help="Run bounded independent OpenManus workers")
    wide.add_argument("--item", action="append", required=True, dest="items")
    wide.add_argument("--workspace", default="")
    wide.add_argument("--execute", action="store_true")
    wide.add_argument("--acknowledge-side-effects", action="store_true")
    wide.add_argument("--allow-network", action="store_true")
    wide.add_argument("--max-parallel", type=int, default=4)
    wide.add_argument("--synthesize", action="store_true")


def _print(payload: dict) -> int:
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0 if payload.get("ok", True) else 1


def openmanus_command(args: argparse.Namespace) -> int:
    command = getattr(args, "openmanus_command", None)
    if command == "status" or command == "capabilities":
        return _print(core.capabilities())
    if command == "run":
        payload = core.run_task(
            {
                "task": args.prompt,
                "workspace": args.workspace,
                "dry_run": not args.execute,
                "allow_side_effects": args.execute,
                "acknowledge_side_effects": args.acknowledge_side_effects,
                "allow_network": args.allow_network,
                "agent_mode": args.agent_mode,
                "max_steps": args.max_steps,
                "timeout_seconds": args.timeout_seconds,
            }
        )
        return _print(payload)
    if command == "wide-research":
        return _print(
            core.wide_research(
                {
                    "items": args.items,
                    "workspace": args.workspace,
                    "dry_run": not args.execute,
                    "allow_side_effects": args.execute,
                    "acknowledge_side_effects": args.acknowledge_side_effects,
                    "allow_network": args.allow_network,
                    "max_parallel": args.max_parallel,
                    "synthesize": args.synthesize,
                }
            )
        )
    print("usage: hermes openmanus {status,capabilities,run,wide-research}")
    return 2
