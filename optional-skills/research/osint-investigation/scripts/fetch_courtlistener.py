#!/usr/bin/env python3
"""Search court records via CourtListener (Free Law Project).

Covers ~10M federal and state court opinions, plus PACER docket data
where available. Public REST API v4 supports anonymous read access for
search; some endpoints require a token (free at courtlistener.com).

Set COURTLISTENER_TOKEN to authenticate (raises rate limits).
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
import urllib.parse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from _http import get_json  # noqa: E402

BASE = "https://www.courtlistener.com/api/rest/v4/search/"

COLUMNS = [
    "case_name",
    "court",
    "court_id",
    "date_filed",
    "docket_number",
    "judge",
    "citation",
    "result_type",
    "snippet",
    "absolute_url",
]

SEARCH_TYPES = {
    "opinions": "o",       # Court opinions
    "dockets": "r",        # PACER dockets (may require auth depending on coverage)
    "oral": "oa",          # Oral arguments
    "people": "p",         # Judges / people
    "recap": "r",          # Same as dockets in v4
}


def fetch(
    query: str,
    search_type: str,
    court: str | None,
    date_from: str | None,
    date_to: str | None,
    token: str | None,
    limit: int,
    out_path: str,
) -> int:
    type_code = SEARCH_TYPES.get(search_type, search_type)
    params = {
        "q": query,
        "type": type_code,
    }
    if court:
        params["court"] = court
    if date_from:
        params["filed_after"] = date_from
    if date_to:
        params["filed_before"] = date_to
    headers = {"Authorization": f"Token {token}"} if token else None

    rows: list[dict[str, str]] = []
    next_url: str | None = f"{BASE}?{urllib.parse.urlencode(params)}"
    while next_url and len(rows) < limit:
        try:
            payload = get_json(next_url, headers=headers)
        except Exception as e:  # noqa: BLE001
            print(f"CourtListener error: {e}", file=sys.stderr)
            break
        if not isinstance(payload, dict):
            break
        results = payload.get("results", [])
        for r in results:
            if len(rows) >= limit:
                break
            rows.append(
                {
                    "case_name": r.get("caseName", "") or r.get("case_name", "") or "",
                    "court": r.get("court", "") or "",
                    "court_id": r.get("court_id", "") or "",
                    "date_filed": (r.get("dateFiled", "") or r.get("date_filed", "") or "")[:10],
                    "docket_number": r.get("docketNumber", "") or r.get("docket_number", "") or "",
                    "judge": r.get("judge", "") or "",
                    "citation": "; ".join(r.get("citation", []) or []) if isinstance(r.get("citation"), list) else (r.get("citation") or ""),
                    "result_type": search_type,
                    "snippet": (r.get("snippet", "") or "").replace("\n", " ")[:500],
                    "absolute_url": (
                        f"https://www.courtlistener.com{r.get('absolute_url', '')}"
                        if r.get("absolute_url", "").startswith("/")
                        else r.get("absolute_url", "")
                    ),
                }
            )
        next_url = payload.get("next")

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=COLUMNS)
        w.writeheader()
        w.writerows(rows)
    if not rows:
        print(
            f"CourtListener: 0 results for type={search_type!r} q={query!r}. "
            "Most private individuals don't appear in published court records "
            "unless they were party to a federal or state appellate case.",
            file=sys.stderr,
        )
    return len(rows)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--query", required=True, help="Search query (party name, case name, keyword)")
    p.add_argument(
        "--type",
        default="opinions",
        choices=list(SEARCH_TYPES.keys()),
        help="Search type (default: opinions)",
    )
    p.add_argument("--court", help="Court ID filter (e.g. 'nysd' = SDNY, 'scotus' = Supreme Court)")
    p.add_argument("--date-from", help="Filed-after date YYYY-MM-DD")
    p.add_argument("--date-to", help="Filed-before date YYYY-MM-DD")
    p.add_argument("--token", default=os.environ.get("COURTLISTENER_TOKEN"))
    p.add_argument("--limit", type=int, default=100)
    p.add_argument("--out", required=True)
    a = p.parse_args()
    n = fetch(
        query=a.query,
        search_type=a.type,
        court=a.court,
        date_from=a.date_from,
        date_to=a.date_to,
        token=a.token,
        limit=a.limit,
        out_path=a.out,
    )
    print(f"Wrote {n} CourtListener rows to {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
