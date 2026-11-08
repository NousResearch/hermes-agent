#!/usr/bin/env python3
"""
ECcompass E-Commerce Intelligence CLI

Usage:
  python3 query.py search "pet food"
  python3 query.py search "coffee" --country CN --platform shopify
  python3 query.py search --country US --platform shopify --sort gmvLast12month
  python3 query.py domain ooni.com
  python3 query.py historical ooni.com
  python3 query.py apps ooni.com
  python3 query.py contacts ooni.com
"""

__version__ = "1.2.18"

import os
import sys
import json
import argparse
import urllib.request
import urllib.error
import urllib.parse

TOKEN = os.environ.get("APEX_TOKEN", "")
API_BASE = "https://api.eccompass.ai"


def api_request(method, path, body=None):
    url = f"{API_BASE}{path}"
    data = json.dumps(body).encode("utf-8") if body else None
    req = urllib.request.Request(url, data=data, method=method, headers={
        "APEX_TOKEN": TOKEN,
        "Content-Type": "application/json",
    })
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            return json.loads(resp.read().decode())
    except urllib.error.HTTPError as e:
        raw = e.read().decode() if e.fp else ""
        try:
            msg = json.loads(raw).get("message", raw)
        except Exception:
            msg = raw or str(e)
        print(f"Error {e.code}: {msg}", file=sys.stderr)
        sys.exit(1)
    except urllib.error.URLError as e:
        print(f"Connection failed: {e}", file=sys.stderr)
        sys.exit(1)


def fmt_money(n):
    if n is None:
        return "N/A"
    try:
        n = float(n)
        if n >= 1e9:
            return f"${n / 1e9:.1f}B"
        if n >= 1e6:
            return f"${n / 1e6:.1f}M"
        if n >= 1e3:
            return f"${n / 1e3:.0f}K"
        return f"${n:.0f}"
    except (ValueError, TypeError):
        return str(n) if n else "N/A"


def fmt_num(n):
    if n is None:
        return "N/A"
    try:
        n = float(n)
        if n >= 1e6:
            return f"{n / 1e6:.1f}M"
        if n >= 1e3:
            return f"{n / 1e3:.0f}K"
        return str(int(n))
    except (ValueError, TypeError):
        return "0"


def fmt_growth(g):
    if g is None:
        return "N/A"
    try:
        return f"{float(g):+.1f}%"
    except (ValueError, TypeError):
        return "N/A"


def build_search_body(args):
    body = {"page": args.page, "size": args.size}

    if args.keyword:
        body["keyword"] = args.keyword

    filters = {}
    if args.country:
        vals = [c.strip().upper() for c in args.country.split(",")]
        filters["countryCode4"] = vals if len(vals) > 1 else vals[0]
    if args.platform:
        vals = [p.strip().lower() for p in args.platform.split(",")]
        filters["platform"] = vals if len(vals) > 1 else vals[0]
    if args.region:
        vals = [r.strip() for r in args.region.split(",")]
        filters["region"] = vals if len(vals) > 1 else vals[0]
    if args.status:
        filters["status"] = args.status
    if args.language:
        filters["languageCode"] = args.language
    if filters:
        body["filters"] = filters

    ranges = {}
    if args.min_gmv is not None:
        ranges["gmvLast12month"] = {"min": args.min_gmv}
    if args.min_growth is not None:
        ranges["growth"] = {"min": args.min_growth}
    if args.min_employees is not None:
        ranges["employeeCount"] = {"min": args.min_employees}
    if args.min_instagram is not None:
        ranges["instagramFollowers"] = {"min": args.min_instagram}
    if args.min_tiktok is not None:
        ranges["tiktokFollowers"] = {"min": args.min_tiktok}
    if args.min_visits is not None:
        ranges["estimatedVisits"] = {"min": args.min_visits}
    if ranges:
        body["ranges"] = ranges

    if args.exists:
        body["exists"] = [f.strip() for f in args.exists.split(",")]

    if args.sort:
        body["sortField"] = args.sort
        body["sortOrder"] = args.order or "desc"

    return body


def cmd_search(args):
    body = build_search_body(args)
    resp = api_request("POST", "/public/api/v1/search", body)

    if not resp.get("success"):
        print(f"API error: {resp.get('message', 'Unknown error')}", file=sys.stderr)
        sys.exit(1)

    data = resp["data"]
    records = data.get("records", [])
    total = data.get("total", 0)
    page = data.get("page", 1)
    total_pages = data.get("totalPages", 1)

    if args.json:
        print(json.dumps(resp, indent=2, ensure_ascii=False, default=str))
        return

    label_parts = []
    if args.keyword:
        label_parts.append(f"keyword='{args.keyword}'")
    if args.country:
        label_parts.append(f"country={args.country.upper()}")
    if args.region:
        label_parts.append(f"region={args.region}")
    if args.platform:
        label_parts.append(f"platform={args.platform}")
    label = " | ".join(label_parts) if label_parts else "all"

    print(f"\nSearch: {label} | {total} results | page {page}/{total_pages}\n")
    print(f"{'#':<4} {'Domain':<30} {'Platform':<12} {'Country':<8} {'GMV(12m)':<12} {'Growth':<8}")
    print("-" * 78)

    for i, r in enumerate(records, (page - 1) * args.size + 1):
        domain = r.get("domain", "N/A")
        platform = r.get("platform", "")
        country = r.get("countryCode4", "")
        gmv = fmt_money(r.get("gmvLast12month"))
        growth = fmt_growth(r.get("growth"))
        print(f"{i:<4} {domain:<30} {platform:<12} {country:<8} {gmv:<12} {growth:<8}")
        title = r.get("title") or r.get("merchantName") or ""
        if title:
            print(f"     {title[:70]}")
        desc = (r.get("description") or "")[:90]
        if desc:
            print(f"     {desc}")
        print()

    if not records:
        print("  No results found.")
    print()


def cmd_historical(args):
    resp = api_request("GET", f"/public/api/v1/historical/{urllib.parse.quote(args.domain, safe='.')}")

    if not resp.get("success"):
        print(f"API error: {resp.get('message', 'Unknown error')}", file=sys.stderr)
        sys.exit(1)

    data = resp["data"]

    if args.json:
        print(json.dumps(resp, indent=2, ensure_ascii=False, default=str))
        return

    records = data.get("records", [])
    months = data.get("months", [])
    total = data.get("total", 0)

    print(f"\nHistorical Data: {args.domain} | {total} months\n")
    print(f"{'Month':<12} {'Monthly GMV':<14} {'Yearly GMV':<14} {'UV':<12} {'PV':<12} {'Avg Price':<10}")
    print("-" * 76)

    for r in records:
        month = r.get("dataMonth", "")
        mgmv = fmt_money(r.get("monthlyGmv"))
        ygmv = fmt_money(r.get("yearlyGmv"))
        uv = fmt_num(r.get("uv"))
        pv = fmt_num(r.get("pv"))
        avg = f"${r.get('avgPriceUsd', 0)}" if r.get("avgPriceUsd") else "N/A"
        print(f"{month:<12} {mgmv:<14} {ygmv:<14} {uv:<12} {pv:<12} {avg:<10}")

    if not records:
        print("  No historical data found.")
    print()


def cmd_installed_apps(args):
    resp = api_request("GET", f"/public/api/v1/installed-apps/{urllib.parse.quote(args.domain, safe='.')}")

    if not resp.get("success"):
        print(f"API error: {resp.get('message', 'Unknown error')}", file=sys.stderr)
        sys.exit(1)

    if args.json:
        print(json.dumps(resp, indent=2, ensure_ascii=False, default=str))
        return

    apps = resp.get("data", [])
    total = resp.get("total", len(apps))

    print(f"\nInstalled Apps: {args.domain} | {total} apps\n")
    print(f"{'#':<4} {'App Name':<35} {'Rating':<8} {'Installs':<10} {'State':<8}")
    print("-" * 70)

    for i, app in enumerate(apps, 1):
        name = (app.get("name") or "")[:34]
        rating = app.get("averageRating") or "N/A"
        installs = fmt_num(app.get("installs"))
        state = app.get("state") or ""
        print(f"{i:<4} {name:<35} {rating:<8} {installs:<10} {state:<8}")
        vendor = app.get("vendorName") or ""
        if vendor:
            print(f"     Vendor: {vendor}")

    if not apps:
        print("  No installed apps found.")
    print()


def cmd_contacts(args):
    resp = api_request("GET", f"/public/api/v1/contacts/{urllib.parse.quote(args.domain, safe='.')}")

    if not resp.get("success"):
        print(f"API error: {resp.get('message', 'Unknown error')}", file=sys.stderr)
        sys.exit(1)

    if args.json:
        print(json.dumps(resp, indent=2, ensure_ascii=False, default=str))
        return

    data = resp.get("data", {})
    contacts = data.get("contacts", [])
    total = data.get("total", 0)

    print(f"\nLinkedIn Contacts: {args.domain} | {total} contacts\n")
    print(f"{'#':<4} {'Name':<25} {'Position':<40} {'Email':<30}")
    print("-" * 100)

    for i, c in enumerate(contacts, 1):
        name = f"{c.get('firstName', '')} {c.get('lastName', '')}".strip()
        position = (c.get("position") or "")[:39]
        email = c.get("email") or ""
        print(f"{i:<4} {name:<25} {position:<40} {email:<30}")

    if not contacts:
        print("  No LinkedIn contacts found.")
    print()


def cmd_domain(args):
    resp = api_request("GET", f"/public/api/v1/domain/{urllib.parse.quote(args.domain, safe='.')}")

    if not resp.get("success"):
        print(f"API error: {resp.get('message', 'Unknown error')}", file=sys.stderr)
        sys.exit(1)

    s = resp["data"]

    if args.json:
        print(json.dumps(resp, indent=2, ensure_ascii=False, default=str))
        return

    print(f"\n{'=' * 64}")
    print(f"  Domain:    {s.get('domain', 'N/A')}")
    print(f"  Brand:     {s.get('merchantName') or s.get('title') or 'N/A'}")
    print(f"  Platform:  {s.get('platform', 'N/A')}  |  Plan: {s.get('plan', 'N/A')}")
    print(f"  Country:   {s.get('countryCode4', 'N/A')}  |  City: {s.get('city', 'N/A')}  |  State: {s.get('state', 'N/A')}")
    print(f"  Status:    {s.get('status', 'N/A')}  |  Created: {str(s.get('created', ''))[:10]}")
    print(f"  Employees: {s.get('employeeCount', 'N/A')}")
    print(f"{'=' * 64}")

    print(f"\n  Revenue (GMV)")
    print(f"  {'2023:':<10} {fmt_money(s.get('gmv2023'))}")
    print(f"  {'2024:':<10} {fmt_money(s.get('gmv2024'))}")
    print(f"  {'2025:':<10} {fmt_money(s.get('gmv2025'))}")
    print(f"  {'2026:':<10} {fmt_money(s.get('gmv2026'))}")
    print(f"  {'Last 12m:':<10} {fmt_money(s.get('gmvLast12month'))}")
    print(f"  {'Growth:':<10} {fmt_growth(s.get('growth'))}")
    print(f"  {'Monthly:':<10} {fmt_money(s.get('estimatedMonthlySales'))}")

    print(f"\n  Products")
    print(f"  Count: {s.get('productCount', 'N/A')}  |  Avg Price: {s.get('avgPriceFormatted') or s.get('avgPriceUsd') or 'N/A'}")

    print(f"\n  Traffic")
    print(f"  Monthly Visits: {fmt_num(s.get('estimatedVisits'))}  |  Page Views: {fmt_num(s.get('estimatedPageViews'))}")
    print(f"  Alexa Rank: #{s.get('alexaRank', 'N/A')}  |  Platform Rank: #{s.get('platformRank', 'N/A')}")

    print(f"\n  Social Media")
    socials = [
        ("Instagram", "instagramFollowers", "instagramFollowers30d"),
        ("TikTok", "tiktokFollowers", "tiktokFollowers30d"),
        ("Twitter/X", "twitterFollowers", "twitterFollowers30d"),
        ("YouTube", "youtubeFollowers", "youtubeFollowers30d"),
        ("Facebook", "facebookFollowers", "facebookFollowers30d"),
        ("Pinterest", "pinterestFollowers", "pinterestFollowers30d"),
    ]
    for name, fld, fld_30d in socials:
        val = s.get(fld)
        d30 = s.get(fld_30d)
        if val is not None:
            d30_str = f"  (30d: {d30:+,})" if d30 is not None else ""
            print(f"  {name:<12} {fmt_num(val)}{d30_str}")

    desc = s.get("description", "")
    if desc:
        print(f"\n  Description")
        print(f"  {desc[:200]}")

    tags = s.get("tagsV5") or s.get("tagsFirst") or ""
    if tags:
        print(f"\n  Tags: {tags[:120]}")

    cats = s.get("categoriesV1") or s.get("categories") or ""
    if cats:
        print(f"  Categories: {cats[:120]}")

    emails = s.get("emails", "")
    phones = s.get("phones", "")
    if emails or phones:
        print(f"\n  Contact")
        if emails:
            print(f"  Email: {emails[:100]}")
        if phones:
            print(f"  Phone: {phones[:60]}")

    print()


def main():
    parser = argparse.ArgumentParser(
        description="ECcompass E-Commerce Intelligence CLI"
    )
    parser.add_argument(
        "--version", action="version",
        version=f"%(prog)s {__version__}"
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_search = sub.add_parser("search", help="Search e-commerce domains")
    p_search.add_argument("keyword", nargs="?", default=None, help="Search keyword (optional if filters provided)")
    p_search.add_argument("--country", help="Country code(s), comma-separated for OR, e.g. US,GB,DE")
    p_search.add_argument("--platform", help="Platform(s), comma-separated for OR, e.g. shopify,woocommerce")
    p_search.add_argument("--region", help="Region(s), comma-separated for OR, e.g. Europe,Africa")
    p_search.add_argument("--status", help="Site status filter")
    p_search.add_argument("--language", help="Language code filter")
    p_search.add_argument("--min-gmv", type=float, help="Minimum GMV last 12 months (USD)")
    p_search.add_argument("--min-growth", type=float, help="Minimum YoY growth rate (percentage, e.g. 20 = 20%)")
    p_search.add_argument("--min-employees", type=float, help="Minimum employee count")
    p_search.add_argument("--min-instagram", type=float, help="Minimum Instagram followers")
    p_search.add_argument("--min-tiktok", type=float, help="Minimum TikTok followers")
    p_search.add_argument("--min-visits", type=float, help="Minimum monthly visits")
    p_search.add_argument("--exists", help="Fields that must exist, comma-separated, e.g. tiktokUrl,emails")
    p_search.add_argument("--sort", help="Sort field (camelCase), e.g. gmvLast12month, growth")
    p_search.add_argument("--order", choices=["asc", "desc"], default="desc", help="Sort order (default: desc)")
    p_search.add_argument("--page", type=int, default=1, help="Page number (default: 1)")
    p_search.add_argument("--size", type=int, default=20, help="Results per page (max 100, default: 20)")
    p_search.add_argument("--json", action="store_true", help="Output raw JSON response")

    p_domain = sub.add_parser("domain", help="Get full analytics for a domain")
    p_domain.add_argument("domain", help="Domain name, e.g. ooni.com")
    p_domain.add_argument("--json", action="store_true", help="Output raw JSON response")

    p_hist = sub.add_parser("historical", help="Get historical GMV and traffic data")
    p_hist.add_argument("domain", help="Domain name, e.g. ooni.com")
    p_hist.add_argument("--json", action="store_true", help="Output raw JSON response")

    p_apps = sub.add_parser("apps", help="Get installed apps/plugins for a domain")
    p_apps.add_argument("domain", help="Domain name, e.g. ooni.com")
    p_apps.add_argument("--json", action="store_true", help="Output raw JSON response")

    p_contacts = sub.add_parser("contacts", help="Get LinkedIn contacts for a domain")
    p_contacts.add_argument("domain", help="Domain name, e.g. ooni.com")
    p_contacts.add_argument("--json", action="store_true", help="Output raw JSON response")

    args = parser.parse_args()

    if not TOKEN:
        print("Error: APEX_TOKEN environment variable is required.", file=sys.stderr)
        print("", file=sys.stderr)
        print("  1. Sign up at https://eccompass.ai", file=sys.stderr)
        print("  2. Go to Dashboard → API Access → Create Token", file=sys.stderr)
        print("  3. Run: export APEX_TOKEN='your_token_here'", file=sys.stderr)
        sys.exit(1)

    if args.cmd == "search":
        cmd_search(args)
    elif args.cmd == "domain":
        cmd_domain(args)
    elif args.cmd == "historical":
        cmd_historical(args)
    elif args.cmd == "apps":
        cmd_installed_apps(args)
    elif args.cmd == "contacts":
        cmd_contacts(args)


if __name__ == "__main__":
    main()
