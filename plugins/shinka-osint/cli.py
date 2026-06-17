"""CLI for the ShinkaEvolve-OSINT Hermes plugin."""

from __future__ import annotations

import argparse
import json

from . import bridge, core


def register_cli(subparser: argparse.ArgumentParser) -> None:
    subs = subparser.add_subparsers(dest="shinka_osint_command")

    subs.add_parser("status", help="Show ShinkaEvolve-OSINT readiness")

    setup = subs.add_parser("setup", help="Save checkout path and default example")
    setup.add_argument(
        "--root",
        required=True,
        help="Absolute path to ShinkaEvolve-OSINT checkout",
    )
    setup.add_argument(
        "--default-example",
        default="milspec_security_jp",
        help="Default example directory for briefings",
    )

    scenarios = subs.add_parser("scenarios", help="List MILSPEC scenarios")
    scenarios.add_argument("--example", default=None)
    scenarios.add_argument("--domain", default="")

    analyze = subs.add_parser("analyze", help="Run one OSINT scenario")
    analyze.add_argument("scenario_id")
    analyze.add_argument("--example", default=None)
    analyze.add_argument("--source-mode", choices=("mock", "real"), default="mock")

    briefing = subs.add_parser("briefing", help="Run a multi-scenario daily briefing")
    briefing.add_argument("topic", nargs="*", default=[])
    briefing.add_argument("--domain", default="")
    briefing.add_argument("--scenario-id", action="append", default=[])
    briefing.add_argument("--max-scenarios", type=int, default=3)
    briefing.add_argument("--example", default=None)
    briefing.add_argument("--source-mode", choices=("mock", "real"), default="mock")
    briefing.add_argument("--save", action="store_true")

    verify = subs.add_parser("verify", help="Verify corpus and audit-chain integrity")
    verify.add_argument("--example", default=None)

    audit = subs.add_parser("audit", help="Show recent MILSPEC audit log entries")
    audit.add_argument("--example", default=None)
    audit.add_argument("--last-n", type=int, default=20)

    stack = subs.add_parser(
        "setup-stack",
        help="Enable OSINT stack (shinka + worldmonitor plugins, toolsets, egov-law MCP)",
    )
    stack.add_argument("--dry-run", action="store_true")
    stack.add_argument("--skip-egov", action="store_true")

    cron = subs.add_parser("cron", help="Install a recurring Hermes cron briefing job")
    cron_subs = cron.add_subparsers(dest="cron_command")
    install_cron = cron_subs.add_parser("install", help="Create a daily briefing cron job")
    install_cron.add_argument("--schedule", default="every 9am")
    install_cron.add_argument("--topic", default="世界情勢 安全保障")
    install_cron.add_argument("--domain", default="")
    install_cron.add_argument("--max-scenarios", type=int, default=3)
    install_cron.add_argument("--source-mode", choices=("mock", "real"), default="mock")
    install_cron.add_argument("--deliver", default="local")
    install_cron.add_argument("--profile", default=None)
    install_cron.add_argument("--dry-run", action="store_true")
    install_cron.add_argument("--paused", action="store_true")

    subparser.set_defaults(func=shinka_osint_command)


def _print(payload: dict) -> int:
    print(json.dumps(payload, ensure_ascii=False, indent=2, default=str))
    return 0 if payload.get("success", True) else 1


def _example_name(value: str | None) -> str:
    return (value or bridge.resolve_default_example()).strip()


def shinka_osint_command(args: argparse.Namespace) -> int:
    command = getattr(args, "shinka_osint_command", None)
    if not command:
        print(
            "usage: hermes shinka-osint "
            "{status,setup,setup-stack,scenarios,analyze,briefing,verify,audit,cron}"
        )
        return 2

    if command == "status":
        return _print(core.status())
    if command == "setup":
        try:
            root = bridge.save_root(args.root)
            example = bridge.save_default_example(args.default_example)
            return _print(
                {
                    "success": True,
                    "root": str(root),
                    "default_example": example,
                }
            )
        except (FileNotFoundError, OSError) as exc:
            return _print({"success": False, "error": str(exc)})
    if command == "scenarios":
        return _print(
            core.list_scenarios(_example_name(args.example), domain=args.domain or "")
        )
    if command == "analyze":
        return _print(
            core.analyze(
                args.scenario_id,
                example=_example_name(args.example),
                source_mode=args.source_mode,
            )
        )
    if command == "briefing":
        topic = " ".join(args.topic or []).strip()
        return _print(
            core.briefing(
                topic=topic,
                domain=args.domain or "",
                scenario_ids=args.scenario_id or None,
                max_scenarios=args.max_scenarios,
                example=_example_name(args.example),
                source_mode=args.source_mode,
                save_report=bool(args.save),
            )
        )
    if command == "setup-stack":
        try:
            from worldmonitor_osint_stack import enable_osint_stack  # type: ignore
        except ImportError:
            import importlib.util
            from pathlib import Path

            stack_path = (
                Path(__file__).resolve().parent.parent
                / "worldmonitor-osint"
                / "stack.py"
            )
            spec = importlib.util.spec_from_file_location("wm_osint_stack", stack_path)
            assert spec and spec.loader
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            enable_osint_stack = mod.enable_osint_stack  # type: ignore
        return _print(
            enable_osint_stack(
                install_egov=not args.skip_egov,
                dry_run=bool(args.dry_run),
            )
        )
    if command == "verify":
        return _print(core.verify(_example_name(args.example)))
    if command == "audit":
        return _print(core.audit(_example_name(args.example), last_n=args.last_n))
    if command == "cron":
        return _cron_command(args)
    return 2


def _cron_command(args: argparse.Namespace) -> int:
    cron_command = getattr(args, "cron_command", None)
    if cron_command != "install":
        print("usage: hermes shinka-osint cron install")
        return 2

    topic = (args.topic or "").strip() or "世界情勢 安全保障"
    domain_flag = f' --domain="{args.domain}"' if args.domain else ""
    save_flag = " --save" if not args.dry_run else ""
    prompt = (
        "Run a ShinkaEvolve-OSINT daily security briefing.\n"
        f"1. Call tool `shinka_osint_briefing` with topic={topic!r}, "
        f"max_scenarios={args.max_scenarios}, source_mode={args.source_mode!r}"
        f"{domain_flag}{save_flag}.\n"
        "2. Summarize each scenario result in Japanese: domain, score, key judgments, "
        "evidence count, allowlist violations (if any).\n"
        "3. End with a short executive summary and recommended follow-up OSINT tasks."
    )

    if args.dry_run:
        return _print(
            {
                "success": True,
                "dry_run": True,
                "schedule": args.schedule,
                "prompt": prompt,
            }
        )

    try:
        from cron.jobs import create_job, pause_job
    except ImportError as exc:
        return _print(
            {
                "success": False,
                "error": f"cron.jobs unavailable: {exc}",
            }
        )

    job = create_job(
        prompt=prompt,
        schedule=args.schedule,
        name="shinka-osint-daily-briefing",
        deliver=args.deliver,
        skills=["research/osint-investigation"],
        enabled_toolsets=["shinka_osint", "worldmonitor_osint", "web", "search"],
    )
    if args.paused and job.get("id"):
        pause_job(job["id"], reason="installed paused by shinka-osint cron install")
    return _print({"success": True, "job": job})
