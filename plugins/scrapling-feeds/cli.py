"""CLI for scrapling-feeds Hermes plugin."""

from __future__ import annotations

import argparse
import json

from . import core
from . import cron_setup
from . import feeds_catalog
from . import gov_digest
from . import mhlw_designated


def register_cli(subparser: argparse.ArgumentParser) -> None:
    subs = subparser.add_subparsers(dest="scrapling_feeds_command")

    subs.add_parser("status", help="Catalog + Scrapling availability")

    fetch = subs.add_parser("fetch", help="Fetch one government feed")
    fetch.add_argument("feed_id", choices=sorted(feeds_catalog.GOV_FEEDS.keys()))
    fetch.add_argument("--hours", type=int, default=24)
    fetch.add_argument("--max-items", type=int, default=15)
    fetch.add_argument("--include-disabled", action="store_true")

    digest = subs.add_parser("digest", help="Digest all enabled government feeds")
    digest.add_argument("--hours", type=int, default=24)
    digest.add_argument("--max-per-feed", type=int, default=10)
    digest.add_argument(
        "--feed",
        action="append",
        dest="feed_ids",
        metavar="FEED_ID",
        help="Repeat to limit to specific feed ids",
    )
    digest.add_argument("--include-disabled", action="store_true")
    digest.add_argument("--json", action="store_true", help="Raw JSON output")

    mhlw = subs.add_parser(
        "mhlw-check",
        help="Check 厚労省 指定薬物部会 notices and 指定/施行 announcements",
    )
    mhlw.add_argument(
        "--record-baseline",
        action="store_true",
        help="Register current items as seen (first-time setup)",
    )
    mhlw.add_argument("--no-enforcement-scan", action="store_true")
    mhlw.add_argument("--json", action="store_true")
    mhlw.add_argument(
        "--cron-stdout",
        action="store_true",
        help="Cron mode: short message if no new items",
    )
    mhlw.add_argument("--show-known", action="store_true")

    cron = subs.add_parser("cron", help="Install scheduled checks")
    cron_subs = cron.add_subparsers(dest="scrapling_cron_command")
    cron_mhlw = cron_subs.add_parser(
        "install-mhlw",
        help="Weekly 指定薬物部会 / 施行チェック (default: Mon 09:00, cron 0 9 * * 1)",
    )
    cron_mhlw.add_argument("--schedule", default=cron_setup.DEFAULT_SCHEDULE)
    cron_mhlw.add_argument("--deliver", default="local")
    cron_mhlw.add_argument("--dry-run", action="store_true")
    cron_mhlw.add_argument("--paused", action="store_true")
    cron_mhlw.add_argument("--force", action="store_true")
    cron_mhlw.add_argument(
        "--record-baseline",
        action="store_true",
        help="Baseline state on first install (avoid flooding alerts)",
    )


def scrapling_feeds_command(args: argparse.Namespace) -> int:
    cmd = getattr(args, "scrapling_feeds_command", None) or "status"
    if cmd == "status":
        print(core.handle_status({}))
        return 0
    if cmd == "fetch":
        result = gov_digest.fetch_single_feed(
            args.feed_id,
            hours=args.hours,
            max_items=args.max_items,
            include_disabled=args.include_disabled,
        )
        print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
        return 0 if result.get("success") else 1
    if cmd == "digest":
        result = gov_digest.digest_feeds(
            hours=args.hours,
            max_per_feed=args.max_per_feed,
            feed_ids=args.feed_ids,
            include_disabled=args.include_disabled,
        )
        if args.json:
            print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
            return 0 if result.get("success") else 1
        _print_digest_markdown(result)
        return 0 if result.get("success") else 1
    if cmd == "mhlw-check":
        if getattr(args, "cron_stdout", False):
            return mhlw_designated.run_for_cron_stdout(
                record_baseline=bool(args.record_baseline),
                scan_enforcement=not bool(args.no_enforcement_scan),
            )
        result = mhlw_designated.check_mhlw_designated(
            record_baseline=bool(args.record_baseline),
            scan_enforcement=not bool(args.no_enforcement_scan),
        )
        if args.json:
            print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
        else:
            print(mhlw_designated.build_report_markdown(result, include_known=args.show_known))
        return 0 if result.get("success") else 1
    if cmd == "cron":
        sub = getattr(args, "scrapling_cron_command", None)
        if sub == "install-mhlw":
            out = cron_setup.install_mhlw_cron(
                schedule=args.schedule,
                deliver=args.deliver,
                dry_run=bool(args.dry_run),
                paused=bool(args.paused),
                force=bool(args.force),
                record_baseline=bool(args.record_baseline),
            )
            print(json.dumps(out, ensure_ascii=False, indent=2, default=str))
            return 0 if out.get("success") else 1
        print("usage: hermes scrapling-feeds cron install-mhlw")
        return 2
    return 1


def _print_digest_markdown(result: dict) -> None:
    print("# Government Feed Digest (PRIMARY)")
    print()
    print(f"- Window: {result.get('window_hours')}h")
    print(f"- Feeds OK: {result.get('feeds_ok')}/{result.get('feed_count')}")
    print(f"- Entries: {result.get('total_entries')}")
    print(f"- Backend: Scrapling={result.get('scrapling_available')}")
    print()
    for feed in result.get("feeds") or []:
        print(f"## {feed.get('name')} ({feed.get('feed_id')})")
        if not feed.get("success"):
            print(f"_Error: {feed.get('error')}_")
            print()
            continue
        for entry in feed.get("entries") or []:
            print(f"- {entry.get('title')}")
            if entry.get("published_at"):
                print(f"  _{entry.get('published_at')}_")
            print(f"  {entry.get('citation') or entry.get('url')}")
        print()
