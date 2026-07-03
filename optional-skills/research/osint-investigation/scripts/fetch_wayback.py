#!/usr/bin/env python3
"""Search the Internet Archive Wayback Machine via the CDX server.

The CDX API indexes ~900B+ archived web pages. Anonymous read access,
no auth required. Useful for finding deleted / changed pages by URL,
domain, or substring match.
"""
from __future__ import annotations

import argparse
import csv
import sys
import urllib.parse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from _http import get_json  # noqa: E402

BASE = "https://web.archive.org/cdx/search/cdx"

COLUMNS = [
    "url",
    "timestamp",
    "wayback_url",
    "mimetype",
    "status",
    "digest",
    "length",
]


def fetch(
    url_or_host: str,
    match_type: str,
    from_date: str | None,
    to_date: str | None,
    status: str | None,
    mime: str | None,
    collapse: str | None,
    limit: int,
    out_path: str,
) -> int:
    params: dict[str, str] = {
        "url": url_or_host,
        "matchType": match_type,
        "output": "json",
        "limit": str(limit),
    }
    if from_date:
        params["from"] = from_date.replace("-", "")
    if to_date:
        params["to"] = to_date.replace("-", "")
    if status:
        params["filter"] = f"statuscode:{status}"
    if mime:
        params.setdefault("filter", "")
        # Multiple filters: CDX accepts repeated filter params via urlencode list
        params["filter"] = f"mimetype:{mime}"
    if collapse:
        params["collapse"] = collapse

    url = f"{BASE}?{urllib.parse.urlencode(params)}"
    try:
        payload = get_json(url)
    except Exception as e:  # noqa: BLE001
        print(f"Wayback CDX error: {e}", file=sys.stderr)
        payload = []

    rows: list[dict[str, str]] = []
    if isinstance(payload, list) and len(payload) > 1:
        header = payload[0]
        idx = {h: i for i, h in enumerate(header)}
        for entry in payload[1:]:
            ts = entry[idx["timestamp"]] if "timestamp" in idx else ""
            orig = entry[idx["original"]] if "original" in idx else ""
            rows.append(
                {
                    "url": orig,
                    "timestamp": ts,
                    "wayback_url": f"https://web.archive.org/web/{ts}/{orig}" if ts and orig else "",
                    "mimetype": entry[idx["mimetype"]] if "mimetype" in idx else "",
                    "status": entry[idx["statuscode"]] if "statuscode" in idx else "",
                    "digest": entry[idx["digest"]] if "digest" in idx else "",
                    "length": entry[idx["length"]] if "length" in idx else "",
                }
            )

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=COLUMNS)
        w.writeheader()
        w.writerows(rows)
    if not rows:
        print(
            f"Wayback Machine: 0 captures for {url_or_host!r} matchType={match_type}.",
            file=sys.stderr,
        )
    return len(rows)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--url", required=True, help="URL or host to look up in the archive")
    p.add_argument(
        "--match",
        default="exact",
        choices=["exact", "prefix", "host", "domain"],
        help=(
            "exact: this URL only. "
            "prefix: this URL's path-prefix. "
            "host: any URL on this host. "
            "domain: any URL on this domain or subdomains."
        ),
    )
    p.add_argument("--from-date", help="Earliest capture YYYY-MM-DD")
    p.add_argument("--to-date", help="Latest capture YYYY-MM-DD")
    p.add_argument("--status", help="HTTP status filter (e.g. 200)")
    p.add_argument("--mime", help="MIME type filter (e.g. text/html)")
    p.add_argument(
        "--collapse",
        help="Collapse adjacent identical entries (e.g. 'digest' for unique-content captures)",
    )
    p.add_argument("--limit", type=int, default=200)
    p.add_argument("--out", required=True)
    a = p.parse_args()
    n = fetch(
        url_or_host=a.url,
        match_type=a.match,
        from_date=a.from_date,
        to_date=a.to_date,
        status=a.status,
        mime=a.mime,
        collapse=a.collapse,
        limit=a.limit,
        out_path=a.out,
    )
    print(f"Wrote {n} Wayback capture rows to {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
