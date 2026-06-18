"""CLI for osint-agent unified OSINT plugin."""

from __future__ import annotations

import argparse
import json

from . import core
from . import cron_setup
from . import stack


def register_cli(subparser: argparse.ArgumentParser) -> None:
    subs = subparser.add_subparsers(dest="osint_agent_command")

    subs.add_parser("status", help="Stack readiness probe")

    brief = subs.add_parser("brief", help="Run integrated OSINT brief (markdown)")
    brief.add_argument("--slot", choices=("morning", "evening"), default="morning")
    brief.add_argument("--topic", default="日本の安全保障と世界情勢")
    brief.add_argument("--wm-tier", choices=("auto", "free", "pro"), default="free")
    brief.add_argument("--source-mode", choices=("mock", "real"), default="real")
    brief.add_argument("--llm-summary", action="store_true")
    brief.add_argument("--max-scenarios", type=int, default=4)
    brief.add_argument("--no-sitdeck", action="store_true")
    brief.add_argument("--no-mhlw", action="store_true")
    brief.add_argument("--no-save", action="store_true")
    brief.add_argument("--cron-stdout", action="store_true", help="Cron: print markdown only")

    setup = subs.add_parser("setup", help="Enable OSINT stack + plugins")
    setup.add_argument("--dry-run", action="store_true")

    cron = subs.add_parser("cron", help="Scheduled integrated briefs")
    cron_sub = cron.add_subparsers(dest="osint_agent_cron_command")
    inst = cron_sub.add_parser("install", help="08:00 + 18:00 integrated OSINT cron")
    inst.add_argument("--morning-schedule", default=cron_setup.DEFAULT_MORNING_SCHEDULE)
    inst.add_argument("--evening-schedule", default=cron_setup.DEFAULT_EVENING_SCHEDULE)
    inst.add_argument("--deliver", default="telegram,discord")
    inst.add_argument("--wm-tier", default="free")
    inst.add_argument("--source-mode", choices=("mock", "real"), default="real")
    inst.add_argument("--llm-summary", action="store_true")
    inst.add_argument("--dry-run", action="store_true")
    inst.add_argument("--paused", action="store_true")
    inst.add_argument("--force", action="store_true")


def osint_agent_command(args: argparse.Namespace) -> int:
    cmd = getattr(args, "osint_agent_command", None) or "status"
    if cmd == "status":
        print(core.handle_status({}))
        return 0
    if cmd == "brief":
        return core.run_brief_cli(
            slot=args.slot,
            topic=args.topic,
            source_mode=args.source_mode,
            wm_tier=args.wm_tier,
            llm_summary=args.llm_summary,
            max_scenarios=args.max_scenarios,
            include_sitdeck=not args.no_sitdeck,
            include_mhlw=not args.no_mhlw,
            save=not args.no_save,
            cron_stdout=args.cron_stdout,
        )
    if cmd == "setup":
        result = stack.enable_osint_agent_stack(dry_run=args.dry_run)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0 if result.get("success") else 1
    if cmd == "cron":
        sub = getattr(args, "osint_agent_cron_command", None)
        if sub == "install":
            result = cron_setup.install_integrated_cron(
                morning_schedule=args.morning_schedule,
                evening_schedule=args.evening_schedule,
                deliver=args.deliver,
                wm_tier=args.wm_tier,
                source_mode=args.source_mode,
                llm_summary=args.llm_summary,
                dry_run=args.dry_run,
                paused=args.paused,
                force=args.force,
            )
            print(json.dumps(result, ensure_ascii=False, indent=2))
            return 0 if result.get("success") else 1
        print("Usage: hermes osint-agent cron install")
        return 2
    print("Unknown subcommand")
    return 2
