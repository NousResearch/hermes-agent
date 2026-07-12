"""Warashibe Reselling — CLI
hermes warashibe <subcommand>
"""
from __future__ import annotations
import argparse, json, sys
import pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import core


def register_cli(parser) -> None:
    """Add warashibe subcommands to the given warashibe command parser."""
    sub = parser.add_subparsers(dest="cmd")

    # license
    sp = sub.add_parser("license", help="古物商許可申請パッケージ生成")
    sp.add_argument("--output", "-o", default=None)

    # reroll
    sp = sub.add_parser("reroll", help="AI社員をsedori- prefixでリロール")
    sp.add_argument("--dry-run", action="store_true", default=True)
    sp.add_argument("--execute", action="store_true")

    # research
    sp = sub.add_parser("research", help="リサーチ試算シート生成")
    sp.add_argument("--keyword", "-k", required=True)
    sp.add_argument("--budget", "-b", type=int, default=10000)
    sp.add_argument("--output", "-o", default=None)

    # public market price research via CloakBrowser / official APIs
    sp = sub.add_parser("price", help="公開価格調査 (メルカリ/ヤフオク/eBay/Amazon公式)")
    sp.add_argument("--keyword", "-k", required=True)
    sp.add_argument(
        "--platforms",
        nargs="+",
        default=["mercari", "yahoo_auction", "ebay", "amazon_jp"],
    )
    sp.add_argument("--limit", type=int, default=10)
    sp.add_argument("--dry-run", action="store_true")
    sp.add_argument("--arbitrage", action="store_true", help="黒字売買ルートを同時評価")

    # cross-market arbitrage scan
    sp = sub.add_parser("arb", help="横断裁定: 黒字になり得る売買組み合わせ")
    sp.add_argument("--keyword", "-k", required=True)
    sp.add_argument(
        "--platforms",
        nargs="+",
        default=["mercari", "yahoo_auction", "ebay", "amazon_jp"],
    )
    sp.add_argument("--limit", type=int, default=8)
    sp.add_argument("--budget", "-b", type=int, default=None)
    sp.add_argument("--min-profit", type=int, default=None)
    sp.add_argument("--min-rate", type=float, default=None)
    sp.add_argument("--dry-run", action="store_true")

    # shipping
    sub.add_parser("shipping", help="発送API検証")

    # platforms
    sub.add_parser("platforms", help="プラットフォーム比較表")

    # sop
    sp = sub.add_parser("sop", help="SOPテンプレート生成")
    sp.add_argument("--output", "-o", default=None)

    # ledger
    sp = sub.add_parser("ledger", help="古物台帳初期化")
    sp.add_argument("--path", default=None)

    # profit
    sp = sub.add_parser("profit", help="利益計算")
    sp.add_argument("--buy", type=int, required=True)
    sp.add_argument("--sell", type=int, required=True)
    sp.add_argument("--platform", default="mercari")
    sp.add_argument("--shipping", type=int, default=0)

    # status
    sub.add_parser("status", help="全体ステータス")


def main(argv=None):
    if argv is None or isinstance(argv, (list, tuple)):
        p = argparse.ArgumentParser(prog="hermes warashibe", description="わらしべ長者式せどり")
        register_cli(p)
        args = p.parse_args(argv)
    else:
        args = argv
    if args.cmd == "license":
        out = args.output or str(core.pathlib.Path.home() / "Documents" / "ops" / "sedori" / "license_pkg")
        r = core.generate_license_package(out)
        print(json.dumps(r, ensure_ascii=False, indent=2))
    elif args.cmd == "reroll":
        dry = not args.execute
        r = core.reroll_ai_employees(dry_run=dry)
        print(json.dumps(r, ensure_ascii=False, indent=2))
    elif args.cmd == "research":
        r = core.generate_research_sheet(args.keyword, args.budget, args.output)
        print(json.dumps(r, ensure_ascii=False, indent=2))
    elif args.cmd == "price":
        from .price_research import search_markets, find_arbitrage
        if getattr(args, "arbitrage", False):
            r = find_arbitrage(
                args.keyword,
                args.platforms,
                args.limit,
                dry_run=args.dry_run,
            )
        else:
            r = search_markets(args.keyword, args.platforms, args.limit, dry_run=args.dry_run)
        print(json.dumps(r, ensure_ascii=False, indent=2))
    elif args.cmd == "arb":
        from .price_research import find_arbitrage
        r = find_arbitrage(
            args.keyword,
            args.platforms,
            args.limit,
            dry_run=args.dry_run,
            min_profit_yen=args.min_profit,
            min_profit_rate=args.min_rate,
            budget_yen=args.budget,
        )
        print(json.dumps(r, ensure_ascii=False, indent=2))
    elif args.cmd == "shipping":
        r = core.verify_shipping_apis()
        print(json.dumps(r, ensure_ascii=False, indent=2))
    elif args.cmd == "platforms":
        r = core.platform_comparison()
        print(json.dumps(r, ensure_ascii=False, indent=2))
    elif args.cmd == "sop":
        out = args.output or str(core.pathlib.Path.home() / "Documents" / "ops" / "sedori" / "sop")
        r = core.generate_sop_templates(out)
        print(json.dumps(r, ensure_ascii=False, indent=2))
    elif args.cmd == "ledger":
        r = core.init_ledger(args.path)
        print(json.dumps(r, ensure_ascii=False, indent=2))
    elif args.cmd == "profit":
        r = core.calc_profit(args.buy, args.sell, args.platform, args.shipping)
        print(json.dumps(r, ensure_ascii=False, indent=2))
    elif args.cmd == "status":
        r = {
            "platforms": len(core.PLATFORMS),
            "sedori_roles": list(core.SEDORI_ROLES.keys()),
            "shipping_apis": list(core.SHIPPING_APIS.keys()),
            "defaults": core.DEFAULTS,
        }
        print(json.dumps(r, ensure_ascii=False, indent=2))
    else:
        p.print_help()


if __name__ == "__main__":
    main()