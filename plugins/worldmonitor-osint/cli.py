"""CLI for the World Monitor OSINT Hermes plugin."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from . import core
from . import auth_setup
from .stack import enable_osint_stack
from . import cron_setup
from . import situation_report
from . import dev_server


def register_cli(subparser: argparse.ArgumentParser) -> None:
    subs = subparser.add_subparsers(dest="worldmonitor_osint_command")

    subs.add_parser("status", help="Show World Monitor + OSINT stack status")

    snap = subs.add_parser("snapshot", help="Fetch Japan-security snapshot")
    snap.add_argument("--country", default="JP")
    snap.add_argument("--region", default="east-asia")
    snap.add_argument("--news-lang", default="en")
    snap.add_argument("--news-limit", type=int, default=12)
    snap.add_argument(
        "--tier",
        choices=("auto", "free", "pro"),
        default="auto",
        help="auto: paid when configured else Free web crawl",
    )

    free = subs.add_parser("free-crawl", help="Free-tier web JSON crawl (no Pro key)")
    free.add_argument("--focus", default="japan_security")
    free.add_argument("--news-lang", default="en")
    free.add_argument("--news-limit", type=int, default=20)
    free.add_argument("--no-shell", action="store_true", help="Skip HTML metadata crawl")

    brief = subs.add_parser("country-brief", help="Country intel brief")
    brief.add_argument("country_code")
    brief.add_argument("--framework", default="")

    fusion = subs.add_parser("fusion", help="Fusion report (WM + Shinka MILSPEC)")
    fusion.add_argument("topic", nargs="*", default=[])
    fusion.add_argument("--domain", default="")
    fusion.add_argument("--country", default="JP")
    fusion.add_argument("--max-scenarios", type=int, default=3)
    fusion.add_argument("--source-mode", choices=("mock", "real"), default="real")
    fusion.add_argument("--save", action="store_true")
    fusion.add_argument(
        "--wm-tier",
        choices=("auto", "free", "pro"),
        default="auto",
        help="WM data: auto=sidecar/key else Free web (default)",
    )
    fusion.add_argument(
        "--llm-summary",
        action="store_true",
        help="Japanese executive summary via Hermes LLM (no google-generativeai)",
    )

    sitrep = subs.add_parser(
        "situation-report",
        help="PDB-style 24h national-security situation report (WM + Shinka)",
    )
    sitrep.add_argument(
        "--slot",
        choices=("morning", "evening"),
        default="morning",
        help="Briefing slot label (morning=08:00, evening=18:00)",
    )
    sitrep.add_argument("--topic", default="日本の安全保障と世界情勢")
    sitrep.add_argument("--country", default="JP")
    sitrep.add_argument("--max-scenarios", type=int, default=4)
    sitrep.add_argument("--source-mode", choices=("mock", "real"), default="mock")
    sitrep.add_argument("--wm-tier", choices=("auto", "free", "pro"), default="auto")
    sitrep.add_argument("--llm-summary", action="store_true")
    sitrep.add_argument("--no-primary-backfill", action="store_true", help="Skip e-Gov + site: backfill")
    sitrep.add_argument("--skip-egov", action="store_true", help="Skip e-Gov Law API citation fetch")
    sitrep.add_argument("--skip-github", action="store_true", help="Skip GitHub toolchain provenance")
    sitrep.add_argument("--skip-gov-feeds", action="store_true", help="Skip government RSS direct-read (scrapling-feeds)")
    sitrep.add_argument("--max-headline-backfill", type=int, default=5)
    sitrep.add_argument("--save", action="store_true", default=True)
    sitrep.add_argument("--no-save", action="store_true")
    sitrep.add_argument(
        "--cron-stdout",
        action="store_true",
        help="Print markdown to stdout only (for no-agent cron delivery)",
    )

    cron = subs.add_parser("cron", help="Install PDB situation-report cron jobs")
    cron_subs = cron.add_subparsers(dest="wm_cron_command")
    cron_install = cron_subs.add_parser(
        "install",
        help="Register 08:00 and 18:00 PDB-style situation reports",
    )
    cron_install.add_argument(
        "--morning-schedule",
        default=cron_setup.DEFAULT_MORNING_SCHEDULE,
        help="Cron expr for morning brief (default: 0 8 * * *)",
    )
    cron_install.add_argument(
        "--evening-schedule",
        default=cron_setup.DEFAULT_EVENING_SCHEDULE,
        help="Cron expr for evening brief (default: 0 18 * * *)",
    )
    cron_install.add_argument(
        "--deliver",
        default="local",
        help='Delivery target: local, telegram, discord, or comma-separated (e.g. "telegram,discord")',
    )
    cron_install.add_argument("--source-mode", choices=("mock", "real"), default="mock")
    cron_install.add_argument("--wm-tier", choices=("auto", "free", "pro"), default="auto")
    cron_install.add_argument("--max-scenarios", type=int, default=4)
    cron_install.add_argument("--llm-summary", action="store_true")
    cron_install.add_argument("--dry-run", action="store_true")
    cron_install.add_argument("--paused", action="store_true")
    cron_install.add_argument("--force", action="store_true")

    setup = subs.add_parser(
        "setup-stack",
        help="Enable shinka-osint + worldmonitor-osint, toolsets, and egov-law MCP",
    )
    setup.add_argument("--dry-run", action="store_true")
    setup.add_argument("--skip-egov", action="store_true")
    setup.add_argument("--skip-worldmonitor-mcp", action="store_true")

    auth = subs.add_parser("setup-auth", help="Configure WM API key, OAuth MCP, or local sidecar")
    auth.add_argument(
        "--mode",
        choices=("auto", "oauth", "key", "sidecar", "dev"),
        default="auto",
        help="auto: dev → sidecar → key → oauth MCP",
    )
    auth.add_argument("--api-key", default="", help="wm_… World Monitor API key (mode=key)")
    auth.add_argument("--port", type=int, default=46123, help="Local sidecar port (mode=sidecar)")
    auth.add_argument("--dry-run", action="store_true")
    auth.add_argument("--skip-mcp", action="store_true", help="Do not register worldmonitor OAuth MCP")

    dev = subs.add_parser("dev", help="Clone, install, and run local World Monitor (npm run dev)")
    dev_subs = dev.add_subparsers(dest="wm_dev_command")
    dev_subs.add_parser("status", help="Show repo + Vite dev server status")
    dev_setup = dev_subs.add_parser("setup", help="Clone → npm install → npm run dev → configure API")
    dev_setup.add_argument("--repo-url", default=dev_server.DEFAULT_REPO_URL)
    dev_setup.add_argument("--repo", default="", help="Checkout path (default: sibling or ~/.hermes/worldmonitor)")
    dev_setup.add_argument("--port", type=int, default=dev_server.DEFAULT_DEV_PORT)
    dev_setup.add_argument(
        "--variant",
        choices=("", "tech", "finance", "happy", "commodity", "energy"),
        default="",
        help="Vite variant (default full dev server)",
    )
    dev_setup.add_argument("--skip-clone", action="store_true")
    dev_setup.add_argument("--skip-install", action="store_true")
    dev_setup.add_argument("--skip-start", action="store_true")
    dev_setup.add_argument("--dry-run", action="store_true")
    dev_setup.add_argument(
        "--bind",
        default="",
        help="Vite listen address (default localhost; 0.0.0.0 for LAN/Tailscale)",
    )
    dev_setup.add_argument(
        "--host",
        default="",
        help="Client-facing URL host (Tailscale IP or MagicDNS name)",
    )
    dev_setup.add_argument(
        "--tailscale",
        action="store_true",
        help="Bind 0.0.0.0 and expose via this machine's Tailscale IPv4",
    )
    dev_setup.add_argument(
        "--tailscale-serve",
        action="store_true",
        help="Register tailscale serve /worldmonitor → localhost (HTTPS tailnet)",
    )
    dev_clone = dev_subs.add_parser("clone", help="git clone koala73/worldmonitor")
    dev_clone.add_argument("--repo-url", default=dev_server.DEFAULT_REPO_URL)
    dev_clone.add_argument("--repo", default="")
    dev_clone.add_argument("--dry-run", action="store_true")
    dev_install = dev_subs.add_parser("install", help="npm install in checkout")
    dev_install.add_argument("--repo", default="")
    dev_install.add_argument("--dry-run", action="store_true")
    dev_start = dev_subs.add_parser("start", help="Start npm run dev in background")
    dev_start.add_argument("--repo", default="")
    dev_start.add_argument("--port", type=int, default=dev_server.DEFAULT_DEV_PORT)
    dev_start.add_argument(
        "--variant",
        choices=("", "tech", "finance", "happy", "commodity", "energy"),
        default="",
    )
    dev_start.add_argument("--wait-seconds", type=float, default=45.0)
    dev_start.add_argument("--no-configure-api", action="store_true")
    dev_start.add_argument("--dry-run", action="store_true")
    dev_start.add_argument("--bind", default="", help="Vite listen address (0.0.0.0 for LAN/Tailscale)")
    dev_start.add_argument("--host", default="", help="Client-facing URL host override")
    dev_start.add_argument(
        "--tailscale",
        action="store_true",
        help="Bind 0.0.0.0 and use Tailscale IPv4 in saved API base URL",
    )
    dev_start.add_argument(
        "--tailscale-serve",
        action="store_true",
        help="Also run tailscale serve /worldmonitor → localhost",
    )
    dev_stop = dev_subs.add_parser("stop", help="Stop recorded dev server process")
    dev_stop.add_argument("--pid", type=int, default=0)

    subparser.set_defaults(func=worldmonitor_osint_command)


def _print(payload: dict) -> int:
    print(json.dumps(payload, ensure_ascii=False, indent=2, default=str))
    return 0 if payload.get("success", True) else 1


def worldmonitor_osint_command(args: argparse.Namespace) -> int:
    command = getattr(args, "worldmonitor_osint_command", None)
    if not command:
        print(
            "usage: hermes worldmonitor-osint "
            "{status,snapshot,free-crawl,country-brief,fusion,situation-report,"
            "setup-stack,setup-auth,dev,cron}"
        )
        return 2

    if command == "status":
        return _print(core.status())
    if command == "snapshot":
        return _print(
            core.snapshot(
                country_code=args.country,
                region_id=args.region,
                news_lang=args.news_lang,
                news_limit=args.news_limit,
                tier_mode=args.tier,
            )
        )
    if command == "free-crawl":
        return _print(
            core.free_crawl(
                focus=args.focus,
                news_lang=args.news_lang,
                news_limit=args.news_limit,
                include_shell=not args.no_shell,
            )
        )
    if command == "country-brief":
        return _print(core.country_brief(args.country_code, framework=args.framework))
    if command == "fusion":
        topic = " ".join(args.topic or []).strip() or "日本の安全保障と世界情勢"
        return _print(
            core.fusion_report(
                topic=topic,
                domain=args.domain,
                country_code=args.country,
                max_scenarios=args.max_scenarios,
                source_mode=args.source_mode,
                save_report=bool(args.save),
                wm_tier=args.wm_tier,
                llm_summary=bool(getattr(args, "llm_summary", False)),
            )
        )
    if command == "situation-report":
        if getattr(args, "cron_stdout", False):
            code = situation_report.run_for_cron_stdout(
                slot=args.slot,
                topic=args.topic,
                country_code=args.country,
                max_scenarios=args.max_scenarios,
                source_mode=args.source_mode,
                wm_tier=args.wm_tier,
                llm_summary=bool(getattr(args, "llm_summary", False)),
                save=not bool(getattr(args, "no_save", False)),
                primary_backfill=not bool(getattr(args, "no_primary_backfill", False)),
                fetch_egov=not bool(getattr(args, "skip_egov", False)),
                fetch_github=not bool(getattr(args, "skip_github", False)),
                fetch_gov_feeds=not bool(getattr(args, "skip_gov_feeds", False)),
                max_headline_backfill=int(getattr(args, "max_headline_backfill", 5) or 5),
            )
            return code
        return _print(
            situation_report.generate_situation_report(
                slot=args.slot,
                topic=args.topic,
                country_code=args.country,
                max_scenarios=args.max_scenarios,
                source_mode=args.source_mode,
                wm_tier=args.wm_tier,
                llm_summary=bool(getattr(args, "llm_summary", False)),
                save=not bool(getattr(args, "no_save", False)),
                primary_backfill=not bool(getattr(args, "no_primary_backfill", False)),
                fetch_egov=not bool(getattr(args, "skip_egov", False)),
                fetch_github=not bool(getattr(args, "skip_github", False)),
                fetch_gov_feeds=not bool(getattr(args, "skip_gov_feeds", False)),
                max_headline_backfill=int(getattr(args, "max_headline_backfill", 5) or 5),
            )
        )
    if command == "cron":
        sub = getattr(args, "wm_cron_command", None)
        if sub != "install":
            print("usage: hermes worldmonitor-osint cron install")
            return 2
        return _print(
            cron_setup.install_pdb_cron(
                morning_schedule=args.morning_schedule,
                evening_schedule=args.evening_schedule,
                deliver=args.deliver,
                source_mode=args.source_mode,
                wm_tier=args.wm_tier,
                llm_summary=bool(getattr(args, "llm_summary", False)),
                max_scenarios=args.max_scenarios,
                dry_run=bool(args.dry_run),
                paused=bool(args.paused),
                force=bool(args.force),
            )
        )
    if command == "setup-stack":
        return _print(
            enable_osint_stack(
                install_egov=not args.skip_egov,
                install_worldmonitor_mcp=not getattr(args, "skip_worldmonitor_mcp", False),
                dry_run=bool(args.dry_run),
            )
        )
    if command == "setup-auth":
        return _print(
            auth_setup.setup_auth(
                mode=args.mode,
                api_key=args.api_key or "",
                port=args.port,
                install_mcp=not args.skip_mcp,
                dry_run=bool(args.dry_run),
            )
        )
    if command == "dev":
        sub = getattr(args, "wm_dev_command", None)
        repo = Path(args.repo).expanduser() if getattr(args, "repo", "") else None
        if sub == "status":
            return _print(dev_server.dev_status())
        if sub == "clone":
            return _print(
                dev_server.clone_repo(
                    repo_url=args.repo_url,
                    target=repo,
                    dry_run=bool(args.dry_run),
                )
            )
        if sub == "install":
            return _print(dev_server.install_deps(repo=repo, dry_run=bool(args.dry_run)))
        if sub == "start":
            return _print(
                dev_server.start_dev(
                    repo=repo,
                    port=args.port,
                    variant=args.variant,
                    wait_seconds=args.wait_seconds,
                    configure_api=not args.no_configure_api,
                    dry_run=bool(args.dry_run),
                    bind=args.bind,
                    host=args.host,
                    tailscale=bool(args.tailscale),
                    tailscale_serve=bool(args.tailscale_serve),
                )
            )
        if sub == "stop":
            return _print(dev_server.stop_dev(pid=args.pid or None))
        if sub == "setup":
            return _print(
                dev_server.setup_dev_stack(
                    repo_url=args.repo_url,
                    repo=repo,
                    clone=not args.skip_clone,
                    install=not args.skip_install,
                    start=not args.skip_start,
                    port=args.port,
                    variant=args.variant,
                    dry_run=bool(args.dry_run),
                    bind=args.bind,
                    host=args.host,
                    tailscale=bool(args.tailscale),
                    tailscale_serve=bool(args.tailscale_serve),
                )
            )
        print("usage: hermes worldmonitor-osint dev {status|setup|clone|install|start|stop}")
        return 2
    return 2
