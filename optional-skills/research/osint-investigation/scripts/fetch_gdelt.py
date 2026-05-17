#!/usr/bin/env python3
"""Search the GDELT 2.0 DOC API for news mentions.

GDELT monitors world news in 100+ languages and indexes the full text.
Free, anonymous, ~15-minute update frequency. Covers ~2015→present.

Useful for surfacing news mentions of a person, company, or topic across
international media — much wider net than Google News.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
import urllib.parse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from _http import get_json  # noqa: E402

BASE = "https://api.gdeltproject.org/api/v2/doc/doc"

COLUMNS = [
    "title",
    "url",
    "seen_date",
    "domain",
    "language",
    "source_country",
    "tone",
    "social_image",
]


def fetch(
    query: str,
    mode: str,
    timespan: str | None,
    start_datetime: str | None,
    end_datetime: str | None,
    source_country: str | None,
    source_lang: str | None,
    limit: int,
    out_path: str,
) -> int:
    params: dict[str, str] = {
        "query": query,
        "mode": mode,
        "format": "json",
        "maxrecords": str(min(limit, 250)),
        "sort": "datedesc",
    }
    if timespan:
        params["timespan"] = timespan
    if start_datetime:
        params["startdatetime"] = start_datetime.replace("-", "").replace(":", "").replace(" ", "")
    if end_datetime:
        params["enddatetime"] = end_datetime.replace("-", "").replace(":", "").replace(" ", "")
    if source_country:
        params["sourcecountry"] = source_country
    if source_lang:
        params["sourcelang"] = source_lang

    url = f"{BASE}?{urllib.parse.urlencode(params)}"
    payload: dict | list = {}
    for attempt in range(3):
        try:
            payload = get_json(url)
            break
        except RuntimeError as e:
            # GDELT requires 1 request per 5 seconds; back off and retry.
            if "429" in str(e) and attempt < 2:
                print(
                    f"GDELT throttle hit; sleeping 6s before retry "
                    f"(attempt {attempt + 1}/3)",
                    file=sys.stderr,
                )
                time.sleep(6)
                continue
            print(f"GDELT error: {e}", file=sys.stderr)
            payload = {}
            break
        except Exception as e:  # noqa: BLE001
            print(f"GDELT error: {e}", file=sys.stderr)
            payload = {}
            break

    rows: list[dict[str, str]] = []
    if isinstance(payload, dict):
        articles = payload.get("articles", []) or []
        for a in articles[:limit]:
            seen = (a.get("seendate") or "")
            # GDELT format: 20260319T083000Z → 2026-03-19 08:30:00Z
            if len(seen) == 16 and "T" in seen:
                seen = f"{seen[0:4]}-{seen[4:6]}-{seen[6:8]} {seen[9:11]}:{seen[11:13]}:{seen[13:15]}Z"
            rows.append(
                {
                    "title": (a.get("title") or "").replace("\n", " ").strip(),
                    "url": a.get("url") or "",
                    "seen_date": seen,
                    "domain": a.get("domain") or "",
                    "language": a.get("language") or "",
                    "source_country": a.get("sourcecountry") or "",
                    "tone": str(a.get("tone") or ""),
                    "social_image": a.get("socialimage") or "",
                }
            )

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=COLUMNS)
        w.writeheader()
        w.writerows(rows)
    if not rows:
        print(
            f"GDELT: 0 articles for query={query!r}. "
            "GDELT indexes ~2015→present. Try widening the timespan or "
            "checking the query syntax (https://blog.gdeltproject.org/gdelt-doc-2-0-api-debuts/).",
            file=sys.stderr,
        )
    return len(rows)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--query", required=True, help='Search query (supports GDELT operators: quoted phrases, AND/OR/NOT, sourcecountry:, theme:)')
    p.add_argument(
        "--mode",
        default="ArtList",
        choices=["ArtList", "ImageCollage", "TimelineVol", "TimelineTone", "ToneChart"],
        help="GDELT mode (default ArtList for article list)",
    )
    p.add_argument(
        "--timespan",
        help="Relative window: e.g. '1d', '1w', '1m', '3m', '1y' (overrides start/end)",
    )
    p.add_argument("--start", help="Absolute start YYYY-MM-DD or YYYY-MM-DDTHH:MM:SS")
    p.add_argument("--end", help="Absolute end YYYY-MM-DD or YYYY-MM-DDTHH:MM:SS")
    p.add_argument("--source-country", help="2-letter source country (e.g. US, UK)")
    p.add_argument("--source-lang", help="Source language (e.g. English, Spanish)")
    p.add_argument("--limit", type=int, default=100)
    p.add_argument("--out", required=True)
    a = p.parse_args()
    n = fetch(
        query=a.query,
        mode=a.mode,
        timespan=a.timespan,
        start_datetime=a.start,
        end_datetime=a.end,
        source_country=a.source_country,
        source_lang=a.source_lang,
        limit=a.limit,
        out_path=a.out,
    )
    print(f"Wrote {n} GDELT article rows to {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
