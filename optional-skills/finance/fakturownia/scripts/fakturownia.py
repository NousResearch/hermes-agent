"""Read-only Fakturownia client. Stdlib only; secrets come from the environment and are never printed."""

from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.error
import urllib.parse
import urllib.request

INVOICE_FIELDS = ("id", "number", "kind", "buyer_name", "issue_date", "payment_to", "price_gross",
                  "currency", "status", "paid", "paid_date", "oid")
CLIENT_FIELDS = ("id", "name", "tax_no", "email", "city", "country")


def base_url(subdomain: str) -> str:
    if not subdomain or not all(c.isalnum() or c == "-" for c in subdomain):
        raise ValueError("FAKTUROWNIA_SUBDOMAIN must be the part before .fakturownia.pl")
    return f"https://{subdomain}.fakturownia.pl"


def build_url(subdomain: str, path: str, token: str, params: dict[str, str]) -> str:
    query = {k: v for k, v in params.items() if v}
    query["api_token"] = token
    return f"{base_url(subdomain)}{path}?{urllib.parse.urlencode(query)}"


def pick(item: dict, fields: tuple[str, ...]) -> dict:
    return {f: item[f] for f in fields if f in item}


def fetch(url: str) -> object:
    request = urllib.request.Request(url, headers={"Accept": "application/json"})
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            return json.load(response)
    except urllib.error.HTTPError as exc:
        raise SystemExit(f"Fakturownia answered HTTP {exc.code}") from None
    except urllib.error.URLError as exc:
        raise SystemExit(f"Could not reach Fakturownia: {exc.reason}") from None


def _credentials() -> tuple[str, str]:
    subdomain = os.environ.get("FAKTUROWNIA_SUBDOMAIN", "")
    token = os.environ.get("FAKTUROWNIA_API_TOKEN", "")
    if not subdomain or not token:
        raise SystemExit("Set FAKTUROWNIA_SUBDOMAIN and FAKTUROWNIA_API_TOKEN first.")
    return subdomain, token


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)
    inv = sub.add_parser("invoices")
    inv.add_argument("--period", default="this_month", choices=["this_month", "last_month", "this_year", "all"])
    inv.add_argument("--client", default="")
    inv.add_argument("--page", default="1")
    one = sub.add_parser("invoice")
    one.add_argument("id")
    cli = sub.add_parser("clients")
    cli.add_argument("--name", default="")
    args = parser.parse_args(argv)
    subdomain, token = _credentials()

    if args.cmd == "invoices":
        params = {"period": args.period, "page": args.page, "client_name": args.client}
        data = fetch(build_url(subdomain, "/invoices.json", token, params))
        result = [pick(i, INVOICE_FIELDS) for i in data] if isinstance(data, list) else data
    elif args.cmd == "invoice":
        if not args.id.isdigit():
            raise SystemExit("invoice id must be a number")
        result = fetch(build_url(subdomain, f"/invoices/{args.id}.json", token, {}))
    else:
        data = fetch(build_url(subdomain, "/clients.json", token, {"name": args.name}))
        result = [pick(c, CLIENT_FIELDS) for c in data] if isinstance(data, list) else data
    json.dump(result, sys.stdout, ensure_ascii=False, indent=2)
    sys.stdout.write("\n")


if __name__ == "__main__":
    main()
