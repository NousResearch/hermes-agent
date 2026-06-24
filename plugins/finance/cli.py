"""``hermes finance`` CLI: connect, sync, status, accounts, transactions, disconnect.

The CLI is the trusted local owner of the data, so it prints exact figures
(privacy buckets apply to the *model-facing* tools, not the owner's terminal).
Onboarding opens a Plaid Hosted Link URL and polls until the user finishes in
their browser — no local web server required.
"""

from __future__ import annotations

import argparse
import json
import webbrowser
from typing import Any, List, Optional

from hermes_constants import display_hermes_home

from plugins.finance import privacy, settings
from plugins.finance.errors import FinanceConfigError, FinanceError
from plugins.finance.provider import available_providers, get_provider
from plugins.finance.store import FinanceStore


def register_cli(subparser: argparse.ArgumentParser) -> None:
    subs = subparser.add_subparsers(dest="finance_action")

    connect_p = subs.add_parser("connect", help="Link a financial institution")
    connect_p.add_argument("provider", nargs="?", default="plaid", help="Provider (default: plaid)")
    connect_p.add_argument("--sandbox", action="store_true", help="Use the provider sandbox (test data)")
    connect_p.add_argument("--institution-id", default="", help="Institution id (sandbox/advanced)")
    connect_p.add_argument("--products", default="", help="Comma-separated products (default: transactions)")
    connect_p.add_argument("--timeout", type=int, default=300, help="Seconds to wait for Link completion")
    connect_p.add_argument("--no-sync", action="store_true", help="Skip the initial sync after linking")
    connect_p.add_argument("--no-browser", action="store_true", help="Print the Link URL but don't open a browser")

    sync_p = subs.add_parser("sync", help="Refresh data from the provider")
    sync_p.add_argument("provider", nargs="?", default="plaid")
    sync_p.add_argument("--item-id", default="", help="Sync only this linked item")

    status_p = subs.add_parser("status", help="Show linked items and last sync")
    status_p.add_argument("provider", nargs="?", default="plaid")

    accounts_p = subs.add_parser("accounts", help="List linked accounts and balances")
    accounts_p.add_argument("provider", nargs="?", default="plaid")

    txns_p = subs.add_parser("transactions", aliases=["txns"], help="List recent transactions")
    txns_p.add_argument("provider", nargs="?", default="plaid")
    txns_p.add_argument("--account-id", default="")
    txns_p.add_argument("--days", type=int, default=30)
    txns_p.add_argument("--limit", type=int, default=25)
    txns_p.add_argument("--search", default="")

    disconnect_p = subs.add_parser("disconnect", aliases=["unlink"], help="Unlink an item")
    disconnect_p.add_argument("provider", nargs="?", default="plaid")
    disconnect_p.add_argument("item_id", help="Item id to disconnect (see `hermes finance status`)")
    disconnect_p.add_argument("--purge", action="store_true", help="Also delete the item's cached data")

    subparser.set_defaults(func=finance_command)


def finance_command(args: argparse.Namespace) -> int:
    action = getattr(args, "finance_action", None)
    if not action:
        print("Usage: hermes finance {connect|sync|status|accounts|transactions|disconnect}")
        print(f"Providers: {', '.join(available_providers())}")
        return 2
    try:
        if action == "connect":
            return _cmd_connect(args)
        if action == "sync":
            return _cmd_sync(args)
        if action == "status":
            return _cmd_status(args)
        if action == "accounts":
            return _cmd_accounts(args)
        if action in {"transactions", "txns"}:
            return _cmd_transactions(args)
        if action in {"disconnect", "unlink"}:
            return _cmd_disconnect(args)
        print(f"Unknown finance action: {action}")
        return 2
    except FinanceConfigError as exc:
        print(str(exc))
        _print_setup_hint(getattr(args, "provider", "plaid"))
        return 1
    except FinanceError as exc:
        print(f"Finance error: {exc}")
        return 1


def _print_setup_hint(provider: str) -> None:
    if provider == "plaid":
        print()
        print(f"  Configure Plaid by adding to {display_hermes_home()}/.env:")
        print("    PLAID_CLIENT_ID=...")
        print("    PLAID_SECRET=...")
        print("  Keys: https://dashboard.plaid.com/developers/keys")
        print("  Set the environment in config.yaml under finance.plaid.environment (sandbox|production).")


def _parse_products(raw: str) -> Optional[List[str]]:
    items = [p.strip() for p in (raw or "").split(",") if p.strip()]
    return items or None


def _money(value: Any, currency: Optional[str]) -> str:
    return privacy.format_money(value, currency, mode=privacy.FULL)


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------

def _cmd_connect(args: argparse.Namespace) -> int:
    provider = get_provider(args.provider)
    products = _parse_products(args.products)
    with FinanceStore() as store:
        if args.sandbox:
            print(f"Linking a {args.provider} sandbox item…")
            result = provider.connect(
                store=store, mode="sandbox",
                institution_id=(args.institution_id or None),
                products=products, do_sync=not args.no_sync,
            )
        else:
            def _on_url(url: str) -> None:
                print("\nOpen this URL to connect your institution:\n")
                print(f"  {url}\n")
                if not args.no_browser:
                    try:
                        webbrowser.open(url)
                    except Exception:
                        pass
                print(f"Waiting up to {args.timeout}s for you to finish in the browser…")

            result = provider.connect(
                store=store, mode="hosted", products=products,
                timeout=args.timeout, on_link_url=_on_url, do_sync=not args.no_sync,
            )
        print()
        print(json.dumps(result, indent=2, sort_keys=True))
    if result.get("status") == "pending":
        return 1
    return 0


def _cmd_sync(args: argparse.Namespace) -> int:
    provider = get_provider(args.provider)
    with FinanceStore() as store:
        summary = provider.sync(store, item_id=(args.item_id or None))
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


def _cmd_status(args: argparse.Namespace) -> int:
    provider = get_provider(args.provider)
    with FinanceStore() as store:
        status = provider.status(store)
    items = status.get("items") or []
    if not items:
        print(f"No {args.provider} items linked. Run `hermes finance connect {args.provider}`.")
        return 0
    print(f"\n{len(items)} linked item(s) via {args.provider}:\n")
    for item in items:
        print(f"  ◆ {item.get('institution_name') or item.get('item_id')}")
        print(f"    item_id: {item.get('item_id')}")
        print(f"    environment: {item.get('environment')}")
        print(f"    status: {item.get('status')}  accounts: {item.get('accounts')}")
        if item.get("last_sync_at"):
            print(f"    last sync: {item.get('last_sync_at')}")
        if item.get("last_error"):
            print(f"    last error: {item.get('last_error')}")
        print()
    return 0


def _cmd_accounts(args: argparse.Namespace) -> int:
    with FinanceStore() as store:
        accounts = store.get_accounts()
        net_worth = store.net_worth()
    if not accounts:
        print("No accounts cached yet. Run `hermes finance sync`.")
        return 0
    print(f"\n{len(accounts)} account(s):\n")
    for acct in accounts:
        currency = acct.get("currency")
        print(f"  ◆ {acct.get('name')} (…{acct.get('mask') or '----'}) [{acct.get('type')}/{acct.get('subtype')}]")
        print(f"    current: {_money(acct.get('current_balance'), currency)}", end="")
        if acct.get("available_balance") is not None:
            print(f"   available: {_money(acct.get('available_balance'), currency)}", end="")
        print()
    for currency, vals in net_worth.items():
        print(
            f"\n  Net worth ({currency}): {_money(vals['net_worth'], currency)} "
            f"(assets {_money(vals['assets'], currency)}, liabilities {_money(vals['liabilities'], currency)})"
        )
    return 0


def _cmd_transactions(args: argparse.Namespace) -> int:
    from datetime import date, timedelta

    start = (date.today() - timedelta(days=max(1, args.days))).isoformat()
    from plugins.finance import categorize

    with FinanceStore() as store:
        txns = store.iter_transactions(
            account_id=(args.account_id or None),
            start_date=start,
            search=(args.search or None),
            limit=max(1, args.limit),
        )
        categorizer = categorize.Categorizer(store)
    if not txns:
        print("No transactions cached for that window. Run `hermes finance sync`.")
        return 0
    print(f"\n{len(txns)} transaction(s) since {start}:\n")
    for txn in txns:
        currency = txn.get("currency")
        pending = " (pending)" if txn.get("pending") else ""
        merchant = categorizer.normalize_merchant(txn)
        category = categorizer.resolve_category(txn)
        print(f"  {txn.get('date')}  {_money(txn.get('amount'), currency):>14}  {merchant} — {category}{pending}")
    return 0


def _cmd_disconnect(args: argparse.Namespace) -> int:
    provider = get_provider(args.provider)
    with FinanceStore() as store:
        result = provider.disconnect(store, args.item_id, purge=args.purge)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0
