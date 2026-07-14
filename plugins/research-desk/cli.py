from __future__ import annotations

import argparse
import json


def register_cli(subparser: argparse.ArgumentParser) -> None:
    subs = subparser.add_subparsers(dest="research_desk_command")
    subs.add_parser("status", help="Show profile, workspace, and readiness")

    plan = subs.add_parser("plan", help="Validate a report plan without network access")
    plan.add_argument("--topic", required=True)
    plan.add_argument("--target", action="append", default=[])
    plan.add_argument("--source-domain", action="append", default=[])
    plan.add_argument("--frequency", choices=["ad_hoc", "weekly", "monthly"], default="weekly")
    plan.add_argument("--workers", type=int, default=2)
    plan.add_argument("--format", choices=["markdown", "json", "csv"], default="markdown")

    run = subs.add_parser("run", help="Run an approved public-research plan")
    run.add_argument("--plan-id", required=True)
    run.add_argument("--approved", action="store_true")
    run.add_argument("--acknowledge-side-effects", action="store_true")

    export = subs.add_parser("export", help="Export an explicitly approved report")
    export.add_argument("--run-id", required=True)
    export.add_argument("--format", choices=["markdown", "json", "csv"], default="markdown")
    export.add_argument("--approved", action="store_true")


def _print(payload) -> int:
    if isinstance(payload, str):
        try:
            payload = json.loads(payload)
        except json.JSONDecodeError:
            print(payload)
            return 0
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0 if payload.get("ok", True) else 1


def research_desk_command(args: argparse.Namespace, *, ctx) -> int:
    command = getattr(args, "research_desk_command", None)
    if command == "status":
        return _print(ctx.dispatch_tool("research_desk_status", {}))
    if command == "plan":
        return _print(
            ctx.dispatch_tool(
                "research_desk_plan",
                {
                    "topic": args.topic,
                    "targets": args.target,
                    "source_domains": args.source_domain,
                    "frequency": args.frequency,
                    "worker_count": args.workers,
                    "output_format": args.format,
                },
            )
        )
    if command == "run":
        return _print(
            ctx.dispatch_tool(
                "research_desk_run",
                {
                    "plan_id": args.plan_id,
                    "approved": args.approved,
                    "acknowledge_side_effects": args.acknowledge_side_effects,
                },
            )
        )
    if command == "export":
        return _print(
            ctx.dispatch_tool(
                "research_desk_export",
                {
                    "run_id": args.run_id,
                    "format": args.format,
                    "approved": args.approved,
                },
            )
        )
    print("usage: hermes research-desk {status,plan,run,export}")
    return 2
