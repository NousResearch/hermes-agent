#!/usr/bin/env python3
"""Search OpenCorporates company registry data.

OpenCorporates aggregates ~200M companies from 130+ jurisdictions. The
public API requires an API token (free tier: 500 calls/month). Set
OPENCORPORATES_API_TOKEN in env or pass --token.

Without a token, this script falls back to scraping the public HTML
search page (limited fields, more brittle, no jurisdiction filter).
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
import urllib.parse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from _http import get, get_json  # noqa: E402

API_URL = "https://api.opencorporates.com/v0.4/companies/search"
HTML_URL = "https://opencorporates.com/companies"

COLUMNS = [
    "name",
    "company_number",
    "jurisdiction_code",
    "jurisdiction_name",
    "incorporation_date",
    "dissolution_date",
    "company_type",
    "status",
    "registered_address",
    "opencorporates_url",
    "officers_count",
    "source",
]


def _via_api(query: str, jurisdiction: str | None, token: str, limit: int) -> list[dict]:
    params = {
        "q": query,
        "api_token": token,
        "per_page": str(min(limit, 100)),
    }
    if jurisdiction:
        params["jurisdiction_code"] = jurisdiction
    url = f"{API_URL}?{urllib.parse.urlencode(params)}"
    payload = get_json(url)
    if not isinstance(payload, dict):
        return []
    results = payload.get("results", {}).get("companies", []) or []
    return [r.get("company", {}) for r in results if isinstance(r, dict)]


def _via_html(query: str, limit: int) -> list[dict]:
    """Best-effort HTML fallback when no API token is available."""
    params = {"q": query, "utf8": "✓"}
    url = f"{HTML_URL}?{urllib.parse.urlencode(params)}"
    body = get(url, user_agent="Mozilla/5.0 hermes-osint").decode("utf-8", errors="replace")
    # Each result is in <li class="company"> ... </li> with name, url, status
    pattern = re.compile(
        r'<li[^>]*class="[^"]*company[^"]*"[^>]*>.*?'
        r'<a[^>]+href="(?P<url>/companies/[^"]+)"[^>]*>(?P<name>[^<]+)</a>'
        r'(?:.*?<span[^>]*class="[^"]*jurisdiction[^"]*"[^>]*>(?P<jur>[^<]+)</span>)?'
        r"(?:.*?<dt[^>]*>(?:Company\s+Number|Number)</dt>\s*<dd[^>]*>(?P<num>[^<]+)</dd>)?",
        re.DOTALL | re.IGNORECASE,
    )
    out = []
    for m in pattern.finditer(body):
        if len(out) >= limit:
            break
        url_path = m.group("url").strip()
        out.append(
            {
                "name": (m.group("name") or "").strip(),
                "opencorporates_url": f"https://opencorporates.com{url_path}",
                "jurisdiction_code": (m.group("jur") or "").strip(),
                "company_number": (m.group("num") or "").strip(),
                "_via": "html",
            }
        )
    return out


def fetch(
    query: str,
    jurisdiction: str | None,
    token: str | None,
    limit: int,
    out_path: str,
) -> int:
    if token:
        try:
            companies = _via_api(query, jurisdiction, token, limit)
            source_tag = "api"
        except Exception as e:  # noqa: BLE001
            print(
                f"OpenCorporates API call failed ({e}); falling back to HTML.",
                file=sys.stderr,
            )
            companies = _via_html(query, limit)
            source_tag = "html-fallback"
    else:
        print(
            "OPENCORPORATES_API_TOKEN not set — using HTML fallback (limited fields). "
            "Get a free token at https://opencorporates.com/api_accounts/new",
            file=sys.stderr,
        )
        companies = _via_html(query, limit)
        source_tag = "html"

    rows: list[dict[str, str]] = []
    for c in companies[:limit]:
        if c.get("_via") == "html":
            rows.append(
                {
                    "name": c.get("name", ""),
                    "company_number": c.get("company_number", ""),
                    "jurisdiction_code": c.get("jurisdiction_code", ""),
                    "jurisdiction_name": "",
                    "incorporation_date": "",
                    "dissolution_date": "",
                    "company_type": "",
                    "status": "",
                    "registered_address": "",
                    "opencorporates_url": c.get("opencorporates_url", ""),
                    "officers_count": "",
                    "source": source_tag,
                }
            )
            continue
        addr = c.get("registered_address_in_full") or ""
        rows.append(
            {
                "name": c.get("name", "") or "",
                "company_number": c.get("company_number", "") or "",
                "jurisdiction_code": c.get("jurisdiction_code", "") or "",
                "jurisdiction_name": "",
                "incorporation_date": c.get("incorporation_date", "") or "",
                "dissolution_date": c.get("dissolution_date", "") or "",
                "company_type": c.get("company_type", "") or "",
                "status": c.get("current_status", "") or c.get("inactive", "") or "",
                "registered_address": addr,
                "opencorporates_url": c.get("opencorporates_url", "") or "",
                "officers_count": str(c.get("officers", {}).get("total_count", "") if c.get("officers") else ""),
                "source": source_tag,
            }
        )

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=COLUMNS)
        w.writeheader()
        w.writerows(rows)
    if not rows:
        print(
            f"OpenCorporates: 0 matches for query={query!r}"
            f"{f' jurisdiction={jurisdiction!r}' if jurisdiction else ''}.",
            file=sys.stderr,
        )
    return len(rows)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--query", required=True, help="Company name search")
    p.add_argument(
        "--jurisdiction",
        help="Jurisdiction code, e.g. 'us_ny', 'us_de', 'gb', 'sg' (lowercased OpenCorporates style)",
    )
    p.add_argument("--limit", type=int, default=50)
    p.add_argument("--token", default=os.environ.get("OPENCORPORATES_API_TOKEN"))
    p.add_argument("--out", required=True)
    a = p.parse_args()
    n = fetch(
        query=a.query,
        jurisdiction=a.jurisdiction,
        token=a.token,
        limit=a.limit,
        out_path=a.out,
    )
    print(f"Wrote {n} OpenCorporates rows to {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
