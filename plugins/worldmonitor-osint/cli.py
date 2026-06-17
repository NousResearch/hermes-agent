"""CLI for the World Monitor OSINT Hermes plugin."""

from __future__ import annotations

import argparse
import json

from . import core
from . import auth_setup
from .stack import enable_osint_stack


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
        choices=("auto", "oauth", "key", "sidecar"),
        default="auto",
        help="auto: sidecar → key → oauth MCP",
    )
    auth.add_argument("--api-key", default="", help="wm_… World Monitor API key (mode=key)")
    auth.add_argument("--port", type=int, default=46123, help="Local sidecar port (mode=sidecar)")
    auth.add_argument("--dry-run", action="store_true")
    auth.add_argument("--skip-mcp", action="store_true", help="Do not register worldmonitor OAuth MCP")

    subparser.set_defaults(func=worldmonitor_osint_command)


def _print(payload: dict) -> int:
    print(json.dumps(payload, ensure_ascii=False, indent=2, default=str))
    return 0 if payload.get("success", True) else 1


def worldmonitor_osint_command(args: argparse.Namespace) -> int:
    command = getattr(args, "worldmonitor_osint_command", None)
    if not command:
        print(
            "usage: hermes worldmonitor-osint "
            "{status,snapshot,free-crawl,country-brief,fusion,setup-stack,setup-auth}"
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
    return 2
