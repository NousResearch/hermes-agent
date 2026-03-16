#!/usr/bin/env python3
"""Hermes Ultra: Deal Scout & Anti-Scalper — Standalone Runner.

A standalone entry point that can be used independently from the
hermes-agent CLI.  Run with::

    python hermes_ultra.py --help
    python hermes_ultra.py track "https://amazon.com/dp/XXXX" --target 500
    python hermes_ultra.py check --id 1
    python hermes_ultra.py history --id 1
    python hermes_ultra.py list
    python hermes_ultra.py watch --interval 30
"""

import argparse
import json
import os
import sys
import time

# Ensure project root is on the path
_project_root = os.path.dirname(os.path.abspath(__file__))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

# Force UTF-8 encoding on Windows to support emojis
if sys.stdout.encoding.lower() != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')


def _print_banner():
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


def cmd_track(args):
    """Add a product to tracking."""
    from tools.price_tracker_tool import _handle_price_track
    result = json.loads(_handle_price_track({
        "url": args.url,
        "target_price": args.target,
        "name": args.name or "",
    }))

    if "error" in result:
        print(f"\n❌ Error: {result['error']}")
        return

    print(f"\n✅ {result.get('message', 'Product added.')}")
    if result.get("price_found"):
        print(f"💰 Current price: ${result['current_price']:,.2f}")
    else:
        print(f"⚠️ {result.get('note', 'Price info could not be retrieved.')}")


def cmd_search(args):
    """Search for a product across global grids and display intelligence."""
    from tools.price_tracker_tool import _handle_price_search_and_track, _handle_price_check

    print("⚡ INITIALIZING ULTRA PROTOCOL")
    print(f"📡 Scanning global grids for '{args.query}'...")
    print("⏳ Synchronizing multi-source intelligence (10-20s)...\n")

    result_str = _handle_price_search_and_track({
        "query": args.query,
        "target_price": args.target,
    })
    result = json.loads(result_str)

    if "error" in result:
        print(f"\n❌ Error: {result['error']}")
        return

    # --- MARKET OVERVIEW TABLE ---
    market_prices = result.get("market_prices", [])
    product_name = result.get("product_name", args.query)

    # Filter: only stores with valid prices
    priced_results = [r for r in market_prices if r.get("price") and r["price"] > 0]

    if priced_results:
        from tools.price_tracker.alerts import format_market_overview, format_cross_site_scalper_alert
        from tools.price_tracker.scalper_detector import ScalperDetector

        # Render the market comparison table (only successful stores)
        overview = format_market_overview(market_prices, product_name)
        print(overview)

        # Cross-site scalper detection (only on priced results)
        detector = ScalperDetector()
        scalper_report = detector.check_cross_site(priced_results)
        if scalper_report and scalper_report.is_suspicious:
            min_r = min(priced_results, key=lambda r: r["price"])
            max_r = max(priced_results, key=lambda r: r["price"])
            alert = format_cross_site_scalper_alert(
                product_name=product_name,
                min_price=min_r["price"],
                min_site=min_r["site"],
                max_price=max_r["price"],
                max_site=max_r["site"],
                spread_pct=scalper_report.deviation_pct,
            )
            print(alert)
    elif market_prices:
        # All stores returned results but none had valid prices (captcha/blocked)
        print("🔍 Scanning global markets... All sources currently shielded. Retrying in background.")
    else:
        print(f"✅ Target acquired: {result.get('found_site')}")

    # Show tracking status
    print(f"🛡️ {result.get('message', 'Target active for tracking.')}")
    if result.get("price_found"):
        print(f"📌 Locked price: ${result['current_price']:,.2f}")

    # --- ONE-SHOT DEAL ANALYSIS on the Best Price ---
    product_id = result.get("product_id")
    if product_id:
        print("\n🔍 Running deep intelligence analysis on best price...")
        check_args = {"product_id": product_id}
        if market_prices:
            check_args["market_prices"] = market_prices
        check_result = json.loads(_handle_price_check(check_args))
        if "error" in check_result:
            print(f"\n⚠️ {check_result['error']}")

def cmd_check(args):
    """Check a product's current status."""
    from tools.price_tracker_tool import _handle_price_check
    params = {}
    if args.id:
        params["product_id"] = args.id
    elif args.url:
        params["url"] = args.url
    else:
        print("❌ --id or --url parameter is required.")
        return

    result = json.loads(_handle_price_check(params))
    if "error" in result:
        print(f"\n❌ Error: {result['error']}")
    # Alert was already rendered via hooks


def cmd_history(args):
    """Show price history."""
    from tools.price_tracker_tool import _handle_price_history
    result = json.loads(_handle_price_history({
        "product_id": args.id,
        "limit": args.limit,
    }))

    if "error" in result:
        print(f"\n❌ Error: {result['error']}")
        return

    print(f"\n📋 {result['product_name']} — Price History")
    print(f"🔗 {result['product_url']}")
    print("-" * 50)

    for entry in result.get("history", []):
        stock_icon = {"in_stock": "✅", "out_of_stock": "❌", "limited": "⚠️"}.get(
            entry.get("stock_status", ""), "❓"
        )
        print(f"  {entry['timestamp']}  |  ${entry['price']:>10,.2f}  | {stock_icon}")


def cmd_portfolio(args):
    """List tracked assets with P/L and Deal Scores."""
    from tools.price_tracker_tool import _handle_portfolio
    result = json.loads(_handle_portfolio({}))
    if "table" in result:
        print(result["table"])
    else:
        print(f"\n📊 Potential Savings Tracking: ${result.get('lifetime_savings', 0):,.2f}")


def cmd_buy(args):
    """Mark a product as purchased and realize savings."""
    from tools.price_tracker_tool import _handle_price_buy
    result = json.loads(_handle_price_buy({"product_id": args.id}))
    
    if "error" in result:
        print(f"\n❌ Error: {result['error']}")
        return

    print(f"\n✅ Target Acquired: {result['product_name']}")
    print(f"💰 Purchase Price: ${result['buy_price']:,.2f}")
    if result['savings'] > 0:
        print(f"💎 Intelligence Savings: ${result['savings']:,.2f}")
    print(f"🏆 Lifetime Asset Savings: ${result['lifetime_savings']:,.2f}")


def cmd_list(args):
    """List all tracked products."""
    from tools.price_tracker.database import PriceTrackerDB
    from tools.price_tracker.alerts import format_price_table
    db = PriceTrackerDB()
    products = db.list_products()

    if not products:
        print("\n📭 No tracked products yet. Use the 'track' command to add one.")
        return

    table_text = format_price_table([p.to_dict() for p in products])
    print(table_text)


def cmd_watch(args):
    """Start the background watcher."""
    from tools.price_tracker_tool import _handle_watcher_start
    result = json.loads(_handle_watcher_start({
        "interval_minutes": args.interval,
    }))

    print(f"\n{result.get('message', '')}")

    if result.get("status") == "started":
        print(f"⏱️  Scan interval: {args.interval} minutes")
        print("Press Ctrl+C to stop.")
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            from tools.price_tracker_tool import _handle_watcher_stop
            _handle_watcher_stop({})
            print("\n\n✅ Price tracker stopped.")


def main():
    _print_banner()

    parser = argparse.ArgumentParser(
        description="Hermes Ultra — Deal Scout & Anti-Scalper",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command", help="Commands")

    # track
    p_track = subparsers.add_parser("track", help="Track a product")
    p_track.add_argument("url", help="Product URL")
    p_track.add_argument("--target", "-t", type=float, help="Target price (USD/EUR)")
    p_track.add_argument("--name", "-n", help="Product name (optional)")

    # search
    p_search = subparsers.add_parser("search", help="Search by name and track")
    p_search.add_argument("query", help="Product name to search (e.g. 'iPhone 15 128GB')")
    p_search.add_argument("--target", "-t", type=float, help="Target price (USD/EUR)")

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

    # portfolio / dashboard
    p_port = subparsers.add_parser("portfolio", help="Show investment portfolio and savings")
    subparsers.add_parser("dashboard", help="Alias for portfolio")

    # buy
    p_buy = subparsers.add_parser("buy", help="Mark product as bought and record savings")
    p_buy.add_argument("id", type=int, help="Product ID")

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
        "portfolio": cmd_portfolio,
        "dashboard": cmd_portfolio,
        "buy": cmd_buy,
        "watch": cmd_watch,
    }

    try:
        commands[args.command](args)
    except KeyboardInterrupt:
        print("\n\n✅ Operation cancelled.")
    except Exception as e:
        print(f"\n❌ An unexpected error occurred: {e}")
        print("   Please report this issue at https://github.com/NousResearch/hermes-agent/issues")


if __name__ == "__main__":
    main()
