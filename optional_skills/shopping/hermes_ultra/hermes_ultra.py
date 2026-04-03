#!/usr/bin/env python3
"""Hermes Ultra: Deal Scout & Anti-Scalper — Standalone Runner.

A standalone entry point that can be used independently from the
hermes-agent CLI.  Run with::

    python hermes_ultra.py --help
    python hermes_ultra.py track "https://amazon.com/dp/XXXX" --target 500
    python hermes_ultra.py search "RTX 4090" --target 1200
    python hermes_ultra.py check --id 1
    python hermes_ultra.py history --id 1
    python hermes_ultra.py list
    python hermes_ultra.py watch --interval 30

This file is a thin entry point — all business logic lives under
``optional-skills/shopping/hermes-ultra/scripts/``.
"""

import argparse
import json
import os
import sys
import time

# ---------------------------------------------------------------------------
# Path setup — ensure both project root and skill scripts are importable
# ---------------------------------------------------------------------------

_skill_root = os.path.dirname(os.path.abspath(__file__))
_project_root = os.path.abspath(os.path.join(_skill_root, os.pardir, os.pardir, os.pardir))

for p in (_project_root, _skill_root):
    if p not in sys.path:
        sys.path.insert(0, p)

# Force UTF-8 encoding on Windows to support emojis
if hasattr(sys.stdout, "encoding") and sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")


def _print_banner() -> None:
    """Print the Hermes Ultra banner."""
    try:
        from rich.console import Console
        from rich.panel import Panel

        console = Console()
        banner = """
[bold gold1]   ██╗  ██╗███████╗██████╗ ███╗   ███╗███████╗███████╗
   ██║  ██║██╔════╝██╔══██╗████╗ ████║██╔════╝██╔════╝
   ███████║█████╗  ██████╔╝██╔████╔██║█████╗  ███████╗
   ██╔══██║██╔══╝  ██╔══██╗██║╚██╔╝██║██╔══╝  ╚════██║
   ██║  ██║███████╗██║  ██║██║ ╚═╝ ██║███████╗███████║
   ╚═╝  ╚═╝╚══════╝╚═╝  ╚═╝╚═╝     ╚═╝╚══════╝╚══════╝[/bold gold1]
[bold cyan]    ⚡ U L T R A — Deal Scout & Anti-Scalper ⚡[/bold cyan]
"""
        console.print(Panel(banner, border_style="gold1", width=65))
    except ImportError:
        print("=" * 60)
        print("  HERMES ULTRA — Deal Scout & Anti-Scalper")
        print("=" * 60)


# ---------------------------------------------------------------------------
# Commands — each imports from the skill's scripts package
# ---------------------------------------------------------------------------

def cmd_track(args: argparse.Namespace) -> None:
    """Add a product to tracking."""
    from optional_skills.shopping.hermes_ultra.scripts.scraper import StealthScraper
    from optional_skills.shopping.hermes_ultra.scripts.parsers import get_parser
    from optional_skills.shopping.hermes_ultra.scripts.database import PriceTrackerDB, Product

    url = args.url
    if not url:
        print("❌ Error: URL is required.")
        return

    db = PriceTrackerDB()
    existing = db.get_product_by_url(url)
    if existing:
        print(f"ℹ️ Product already tracked (ID: {existing.id})")
        return

    # Scrape and parse
    scraper = StealthScraper()
    result = scraper.scrape(url)

    product = Product(url=url, name=args.name or "")
    if result.success:
        parser = get_parser(url)
        if parser:
            data = parser.parse(result.html, url)
            product.name = product.name or data.name
            product.current_price = data.price
            product.original_price = data.original_price
            product.stock_status = data.stock_status
            product.seller = data.seller
            product.site = parser.get_site_name()

    product.target_price = args.target
    product = db.add_product(product)

    if product.current_price:
        db.update_product_price(
            product.id, product.current_price,
            product.original_price, product.stock_status, product.seller,
        )

    print(f"\n✅ Product tracked (ID: {product.id})")
    if product.current_price:
        print(f"💰 Current price: ${product.current_price:,.2f}")
    else:
        print("⚠️ Price info could not be retrieved.")


def cmd_search(args: argparse.Namespace) -> None:
    """Search for a product across stores and display intelligence."""
    from optional_skills.shopping.hermes_ultra.scripts.searcher import ProductSearcher
    from optional_skills.shopping.hermes_ultra.scripts.database import PriceTrackerDB, Product
    from optional_skills.shopping.hermes_ultra.scripts.scoring import DealScorer
    from optional_skills.shopping.hermes_ultra.scripts.scalper_detector import ScalperDetector
    from optional_skills.shopping.hermes_ultra.scripts.trend_predictor import TrendPredictor
    from optional_skills.shopping.hermes_ultra.scripts.reasoning import analyze_price_reasoning
    from optional_skills.shopping.hermes_ultra.scripts.alerts import format_market_overview, format_full_report

    print("⚡ INITIALIZING ULTRA PROTOCOL")
    print(f"📡 Scanning global grids for '{args.query}'...")
    print("⏳ Synchronizing multi-source intelligence (10-20s)...\n")

    searcher = ProductSearcher()
    all_results = searcher.search_all_sources(args.query)

    if not all_results:
        print("❌ Product not found across any store, or blocked by anti-bot systems.")
        return

    priced = [r for r in all_results if r.get("price") and r["price"] > 0]

    if priced:
        overview = format_market_overview(all_results, args.query)
        print(overview)

        # Cross-site scalper check
        detector = ScalperDetector()
        if len(priced) >= 2:
            scalper_report = detector.check_cross_site(priced)
            if scalper_report and scalper_report.is_suspicious:
                min_r = min(priced, key=lambda r: r["price"])
                max_r = max(priced, key=lambda r: r["price"])
                from optional_skills.shopping.hermes_ultra.scripts.alerts import format_cross_site_scalper_alert
                print(format_cross_site_scalper_alert(
                    args.query, min_r["price"], min_r["site"],
                    max_r["price"], max_r["site"],
                    scalper_report.deviation_pct,
                ))

        # Auto-track best price
        best = priced[0]
        db = PriceTrackerDB()
        existing = db.get_product_by_url(best["url"])
        if not existing:
            product = Product(
                url=best["url"], name=best.get("name", args.query),
                site=best.get("site", ""), current_price=best["price"],
                original_price=best.get("original_price"),
                stock_status=best.get("stock_status", "unknown"),
                seller=best.get("seller", ""), target_price=args.target,
            )
            product = db.add_product(product)
            db.update_product_price(
                product.id, product.current_price,
                product.original_price, product.stock_status, product.seller,
            )
            print(f"\n🛡️ Best deal auto-tracked (ID: {product.id})")
        else:
            product = existing

        # Deal score + analysis
        scorer = DealScorer()
        deal = scorer.calculate(
            current_price=best["price"],
            target_price=args.target,
            original_price=best.get("original_price"),
            stock_status=best.get("stock_status", "unknown"),
        )

        scalper = detector.check(
            current_price=best["price"],
            original_price=best.get("original_price"),
        )

        predictor = TrendPredictor()
        trend = predictor.predict([best["price"]])

        reasoning = analyze_price_reasoning(
            product_name=best.get("name", args.query),
            current_price=best["price"],
            original_price=best.get("original_price"),
            market_prices=priced,
        )

        report = format_full_report(
            product_name=best.get("name", args.query),
            current_price=best["price"],
            target_price=args.target,
            deal_score=deal.total_score,
            deal_label=deal.label,
            scalper_risk=scalper.risk_level,
            scalper_text=scalper.analysis_text,
            trend_direction=trend.direction,
            trend_text=trend.analysis_text,
            url=best["url"],
            market_prices=priced,
            volatility_warning=trend.volatility_warning,
            reasoning_text=reasoning.reasoning,
        )
        print(report)
    else:
        print("🔍 All sources currently shielded. Try again later.")


def cmd_check(args: argparse.Namespace) -> None:
    """Check a product's current status."""
    from optional_skills.shopping.hermes_ultra.scripts.scraper import StealthScraper
    from optional_skills.shopping.hermes_ultra.scripts.parsers import get_parser
    from optional_skills.shopping.hermes_ultra.scripts.database import PriceTrackerDB
    from optional_skills.shopping.hermes_ultra.scripts.scoring import DealScorer
    from optional_skills.shopping.hermes_ultra.scripts.scalper_detector import ScalperDetector
    from optional_skills.shopping.hermes_ultra.scripts.trend_predictor import TrendPredictor
    from optional_skills.shopping.hermes_ultra.scripts.reasoning import analyze_price_reasoning
    from optional_skills.shopping.hermes_ultra.scripts.alerts import format_full_report

    db = PriceTrackerDB()
    product = None
    if args.id:
        product = db.get_product(args.id)
    elif args.url:
        product = db.get_product_by_url(args.url)

    if not product:
        print("❌ Product not found.")
        return

    # Re-scrape
    scraper = StealthScraper()
    result = scraper.scrape(product.url)
    if result.success:
        parser = get_parser(product.url)
        if parser:
            data = parser.parse(result.html, product.url)
            if data.price:
                db.update_product_price(
                    product.id, data.price,
                    data.original_price, data.stock_status, data.seller, data.name,
                )
                product.current_price = data.price
                product.original_price = data.original_price

    if not product.current_price:
        print("⚠️ Could not retrieve current price.")
        return

    history = db.get_price_history(product.id)
    price_list = [h.price for h in history]

    scorer = DealScorer()
    deal = scorer.calculate(
        current_price=product.current_price,
        target_price=product.target_price,
        original_price=product.original_price,
        stock_status=product.stock_status,
        price_history=price_list,
    )

    detector = ScalperDetector()
    scalper = detector.check(
        current_price=product.current_price,
        original_price=product.original_price,
        price_history=price_list,
    )

    predictor = TrendPredictor()
    trend = predictor.predict(price_list, product.current_price)

    reasoning = analyze_price_reasoning(
        product_name=product.name,
        current_price=product.current_price,
        original_price=product.original_price,
        price_history=price_list,
    )

    report = format_full_report(
        product_name=product.name,
        current_price=product.current_price,
        target_price=product.target_price,
        deal_score=deal.total_score,
        deal_label=deal.label,
        scalper_risk=scalper.risk_level,
        scalper_text=scalper.analysis_text,
        trend_direction=trend.direction,
        trend_text=trend.analysis_text,
        url=product.url,
        volatility_warning=trend.volatility_warning,
        reasoning_text=reasoning.reasoning,
    )
    print(report)


def cmd_history(args: argparse.Namespace) -> None:
    """Show price history."""
    from optional_skills.shopping.hermes_ultra.scripts.database import PriceTrackerDB

    db = PriceTrackerDB()
    product = db.get_product(args.id)
    if not product:
        print("❌ Product not found.")
        return

    history = db.get_price_history(product.id, limit=args.limit)
    print(f"\n📋 {product.name} — Price History")
    print(f"🔗 {product.url}")
    print("-" * 50)

    for entry in history:
        import datetime
        ts = datetime.datetime.fromtimestamp(entry.timestamp).strftime("%Y-%m-%d %H:%M") if entry.timestamp else "?"
        stock_icon = {"in_stock": "✅", "out_of_stock": "❌", "limited": "⚠️"}.get(entry.stock_status, "❓")
        print(f"  {ts}  |  ${entry.price:>10,.2f}  | {stock_icon}")


def cmd_list(args: argparse.Namespace) -> None:
    """List all tracked products."""
    from optional_skills.shopping.hermes_ultra.scripts.database import PriceTrackerDB
    from optional_skills.shopping.hermes_ultra.scripts.alerts import format_price_table

    db = PriceTrackerDB()
    products = db.list_products()

    if not products:
        print("\n📭 No tracked products yet. Use the 'track' command to add one.")
        return

    table_text = format_price_table([p.to_dict() for p in products])
    print(table_text)


def cmd_watch(args: argparse.Namespace) -> None:
    """Start a simple polling watcher."""
    from optional_skills.shopping.hermes_ultra.scripts.database import PriceTrackerDB

    db = PriceTrackerDB()
    products = db.list_products()
    if not products:
        print("📭 No products to watch. Track something first.")
        return

    interval = args.interval * 60
    print(f"👀 Watching {len(products)} product(s) every {args.interval} minute(s).")
    print("Press Ctrl+C to stop.\n")

    try:
        while True:
            for p in products:
                sys.argv_backup = sys.argv
                # Simulate a check
                ns = argparse.Namespace(id=p.id, url=None)
                cmd_check(ns)
            time.sleep(interval)
            # Refresh product list
            products = db.list_products()
    except KeyboardInterrupt:
        print("\n\n✅ Price tracker stopped.")


def main() -> None:
    _print_banner()

    parser = argparse.ArgumentParser(
        description="Hermes Ultra — Deal Scout & Anti-Scalper",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command", help="Commands")

    # track
    p_track = subparsers.add_parser("track", help="Track a product")
    p_track.add_argument("url", help="Product URL")
    p_track.add_argument("--target", "-t", type=float, help="Target price")
    p_track.add_argument("--name", "-n", help="Product name (optional)")

    # search
    p_search = subparsers.add_parser("search", help="Search by name and track")
    p_search.add_argument("query", help="Product name to search")
    p_search.add_argument("--target", "-t", type=float, help="Target price")

    # check
    p_check = subparsers.add_parser("check", help="Check price and analyze")
    p_check.add_argument("--id", "-i", type=int, help="Product ID")
    p_check.add_argument("--url", "-u", help="Product URL")

    # history
    p_hist = subparsers.add_parser("history", help="Price history")
    p_hist.add_argument("--id", "-i", type=int, required=True, help="Product ID")
    p_hist.add_argument("--limit", "-l", type=int, default=20, help="Record limit")

    # list
    subparsers.add_parser("list", help="List all tracked products")

    # watch
    p_watch = subparsers.add_parser("watch", help="Start background watcher")
    p_watch.add_argument("--interval", "-i", type=int, default=30, help="Scan interval (minutes)")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return

    commands = {
        "track": cmd_track,
        "search": cmd_search,
        "check": cmd_check,
        "history": cmd_history,
        "list": cmd_list,
        "watch": cmd_watch,
    }

    try:
        commands[args.command](args)
    except KeyboardInterrupt:
        print("\n\n✅ Operation cancelled.")
    except Exception as e:
        print(f"\n❌ An unexpected error occurred: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
