#!/usr/bin/env python3
"""中国农历黄历吉凶 · Zhongguo Nongli Huangli Jixiong · China Lunar Almanac API query helper for Hermes skill."""

import argparse
import json
import os
import sys
import urllib.error
import urllib.request
from datetime import date, datetime, timedelta

BASE = os.environ.get("HUANGLI_BASE", "https://api.nongli.skill.4glz.com").rstrip("/")
TOKEN = os.environ.get("HUANGLI_TOKEN", "").strip()


def auth_headers() -> dict:
    if not TOKEN:
        print("Error: HUANGLI_TOKEN is required", file=sys.stderr)
        print("Hint: export HUANGLI_TOKEN or run huangli_auth.py first", file=sys.stderr)
        sys.exit(1)
    return {"Authorization": f"Bearer {TOKEN}", "Content-Type": "application/json"}


def get_json(url: str, headers: dict):
    req = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(req) as resp:
        return json.loads(resp.read().decode("utf-8"))


def post_json(url: str, payload: dict, headers: dict):
    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=body, headers=headers, method="POST")
    with urllib.request.urlopen(req) as resp:
        return json.loads(resp.read().decode("utf-8"))


def fmt_day(rec: dict) -> str:
    return json.dumps(
        {
            "solar_date": rec.get("solar_date"),
            "lunar_info": rec.get("lunar_info"),
            "ganzhi_day": rec.get("ganzhi_day"),
            "suitable_activities": rec.get("suitable_activities", []),
            "unsuitable_activities": rec.get("unsuitable_activities", []),
            "solar_term": rec.get("solar_term"),
        },
        ensure_ascii=False,
        indent=2,
    )


def parse_date(s: str) -> date:
    return datetime.strptime(s, "%Y-%m-%d").date()


def by_date(target: str):
    headers = auth_headers()
    out = get_json(f"{BASE}/api/lunar/date/{target}", headers)
    print(fmt_day(out))


def daterange(start: date, end: date):
    cur = start
    while cur <= end:
        yield cur.isoformat()
        cur += timedelta(days=1)


def batch(start: str, end: str, activity_filter: str | None):
    s = parse_date(start)
    e = parse_date(end)
    if e < s:
        raise ValueError("end date must be >= start date")

    dates = list(daterange(s, e))
    headers = auth_headers()
    out = post_json(f"{BASE}/api/lunar/batch", {"dates": dates}, headers)
    results = out.get("results", [])

    if activity_filter:
        results = [
            r
            for r in results
            if activity_filter in (r.get("suitable_activities") or [])
            or activity_filter in (r.get("unsuitable_activities") or [])
            or activity_filter in (r.get("ganzhi_day") or "")
            or activity_filter in (r.get("lunar_info") or "")
        ]

    print(json.dumps({"count": len(results), "results": results}, ensure_ascii=False, indent=2))


def search(keyword: str, year: int | None, start: str | None, end: str | None):
    if year is not None:
        start = f"{year}-01-01"
        end = f"{year}-12-31"
    if not start or not end:
        raise ValueError("provide --year or both --start and --end")
    batch(start, end, keyword)


def main():
    p = argparse.ArgumentParser(description="Huangli API toolkit")
    sub = p.add_subparsers(dest="cmd", required=True)

    p_date = sub.add_parser("by-date")
    p_date.add_argument("date", help="YYYY-MM-DD")

    p_batch = sub.add_parser("batch")
    p_batch.add_argument("start", help="YYYY-MM-DD")
    p_batch.add_argument("end", help="YYYY-MM-DD")
    p_batch.add_argument("--filter", default=None)

    p_search = sub.add_parser("search")
    p_search.add_argument("keyword")
    p_search.add_argument("--year", type=int, default=None)
    p_search.add_argument("--start", default=None)
    p_search.add_argument("--end", default=None)

    args = p.parse_args()

    try:
        if args.cmd == "by-date":
            by_date(args.date)
        elif args.cmd == "batch":
            batch(args.start, args.end, args.filter)
        elif args.cmd == "search":
            search(args.keyword, args.year, args.start, args.end)
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="replace")
        print(f"HTTP {e.code}: {body}", file=sys.stderr)
        sys.exit(1)
    except urllib.error.URLError as e:
        print(f"Network error: {e.reason}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
