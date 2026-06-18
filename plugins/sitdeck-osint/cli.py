"""CLI for sitdeck-osint Hermes plugin."""

from __future__ import annotations

import argparse
import json

from . import core
from . import stack


def register_cli(subparser: argparse.ArgumentParser) -> None:
    subs = subparser.add_subparsers(dest="sitdeck_osint_command")

    subs.add_parser("status", help="Credentials, Playwright, last crawl")

    crawl = subs.add_parser("crawl", help="Browser login + dashboard crawl")
    crawl.add_argument("--no-headless", action="store_true")
    crawl.add_argument("--no-session", action="store_true")
    crawl.add_argument("--json", action="store_true")

    digest = subs.add_parser("digest", help="Crawl and print markdown digest")

    setup = subs.add_parser(
        "setup",
        help="Enable plugin, disable World Monitor MCP, optional .env email",
    )
    setup.add_argument("--email", default="", help="Gmail local-part or full address")
    setup.add_argument("--skip-env", action="store_true")
    setup.add_argument("--dry-run", action="store_true")

    disable_wm = subs.add_parser(
        "disable-wm-mcp",
        help="Disable worldmonitor OAuth MCP in config (keep free-crawl tools)",
    )
    disable_wm.add_argument("--dry-run", action="store_true")


def sitdeck_osint_command(args: argparse.Namespace) -> int:
    cmd = getattr(args, "sitdeck_osint_command", None) or "status"
    if cmd == "status":
        print(core.handle_status({}))
        return 0
    if cmd == "crawl":
        payload = core.handle_crawl(
            {
                "headless": not args.no_headless,
                "reuse_session": not args.no_session,
            }
        )
        if args.json:
            print(payload)
        else:
            data = json.loads(payload)
            if data.get("success"):
                print(data.get("body_text") or "(no body text)")
            else:
                print(payload)
        return 0 if json.loads(payload).get("success") else 1
    if cmd == "digest":
        print(core.handle_digest({"headless": True}))
        return 0
    if cmd == "setup":
        result = stack.setup_sitdeck_stack(
            email=args.email or None,
            write_env=not args.skip_env,
            dry_run=args.dry_run,
        )
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0 if result.get("success") else 1
    if cmd == "disable-wm-mcp":
        result = stack.disable_worldmonitor_mcp(dry_run=args.dry_run)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0 if result.get("success") else 1
    print("Unknown subcommand. Try: status, crawl, digest, setup, disable-wm-mcp")
    return 2
