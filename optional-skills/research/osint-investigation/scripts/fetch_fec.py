#!/usr/bin/env python3
"""Fetch FEC individual contributions via the OpenFEC API.

Defaults to DEMO_KEY (30 req/hour). Set FEC_API_KEY for 1000 req/hour.

The OpenFEC `schedule_a/` endpoint indexes contributions by date range.
Filtering by `two_year_transaction_period` AND `sort` AND `contributor_name`
puts the query plan on a slow path that the upstream gateway times out (25s).
This script uses `min_date`/`max_date` instead, drops the explicit sort, and
prints a clear warning when no rows return.
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from _http import get_json  # noqa: E402

BASE = "https://api.open.fec.gov/v1/schedules/schedule_a/"
COLUMNS = [
    "contributor_name",
    "contributor_employer",
    "contributor_occupation",
    "contributor_city",
    "contributor_state",
    "contributor_zip",
    "recipient_name",
    "recipient_committee_id",
    "amount",
    "date",
    "cycle",
    "transaction_id",
]


def _cycle_dates(cycle: int) -> tuple[str, str]:
    """Election-cycle window: Jan 1 of (cycle-1) through Dec 31 of cycle."""
    return (f"{cycle - 1}-01-01", f"{cycle}-12-31")


def fetch(
    api_key: str,
    contributor: str | None,
    committee: str | None,
    employer: str | None,
    state: str | None,
    cycle: int,
    out_path: str,
    page_size: int = 100,
    max_pages: int = 50,
) -> int:
    min_date, max_date = _cycle_dates(cycle)
    params: dict[str, str | int] = {
        "api_key": api_key,
        "per_page": page_size,
        "min_date": min_date,
        "max_date": max_date,
    }
    if contributor:
        params["contributor_name"] = contributor
    if employer:
        params["contributor_employer"] = employer
    if committee:
        params["committee_id"] = committee
    if state:
        params["contributor_state"] = state

    rows: list[dict[str, str]] = []
    last_index = None
    last_contribution_receipt_date = None
    pages = 0
    while pages < max_pages:
        if last_index is not None:
            params["last_index"] = last_index
        if last_contribution_receipt_date is not None:
            params["last_contribution_receipt_date"] = last_contribution_receipt_date
        try:
            payload = get_json(BASE, params=params)
        except Exception as e:  # noqa: BLE001
            print(f"FEC error on page {pages + 1}: {e}", file=sys.stderr)
            break
        if not isinstance(payload, dict):
            break
        results = payload.get("results", [])
        if not results:
            break
        for r in results:
            rows.append(
                {
                    "contributor_name": r.get("contributor_name", "") or "",
                    "contributor_employer": r.get("contributor_employer", "") or "",
                    "contributor_occupation": r.get("contributor_occupation", "") or "",
                    "contributor_city": r.get("contributor_city", "") or "",
                    "contributor_state": r.get("contributor_state", "") or "",
                    "contributor_zip": r.get("contributor_zip", "") or "",
                    "recipient_name": r.get("committee", {}).get("name", "") if r.get("committee") else "",
                    "recipient_committee_id": r.get("committee_id", "") or "",
                    "amount": str(r.get("contribution_receipt_amount", "") or ""),
                    "date": (r.get("contribution_receipt_date") or "")[:10],
                    "cycle": str(r.get("two_year_transaction_period", cycle)),
                    "transaction_id": r.get("transaction_id", "") or "",
                }
            )
        pagination = payload.get("pagination", {}) or {}
        last_indexes = pagination.get("last_indexes") or {}
        last_index = last_indexes.get("last_index")
        last_contribution_receipt_date = last_indexes.get("last_contribution_receipt_date")
        if not last_index:
            break
        pages += 1
        time.sleep(1.0)

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=COLUMNS)
        w.writeheader()
        w.writerows(rows)

    if not rows:
        filters = []
        if contributor:
            filters.append(f"contributor={contributor!r}")
        if employer:
            filters.append(f"employer={employer!r}")
        if committee:
            filters.append(f"committee={committee!r}")
        if state:
            filters.append(f"state={state!r}")
        print(
            f"Warning: FEC returned 0 rows for cycle={cycle} ({', '.join(filters)}). "
            f"Date window was {min_date}..{max_date}. "
            f"Names use 'LAST, FIRST' format (uppercase). "
            f"Note: contributions <$200 are NOT itemized — only itemized contributions appear here.",
            file=sys.stderr,
        )
    return len(rows)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument(
        "--contributor",
        help="Contributor (donor) name filter, e.g. 'SMITH, JOHN'. Use uppercase 'LAST, FIRST'.",
    )
    p.add_argument(
        "--candidate",
        dest="contributor",
        help="Deprecated alias for --contributor (FEC searches by contributor name, not candidate name).",
    )
    p.add_argument("--committee", help="FEC committee ID (e.g. C00580100)")
    p.add_argument("--employer", help="Filter by contributor employer (substring OK)")
    p.add_argument("--state", help="Two-letter contributor state, e.g. NY")
    p.add_argument("--cycle", type=int, default=2024)
    p.add_argument("--out", required=True)
    p.add_argument("--api-key", default=os.environ.get("FEC_API_KEY", "DEMO_KEY"))
    p.add_argument("--max-pages", type=int, default=50)
    a = p.parse_args()
    if not (a.contributor or a.committee or a.employer):
        p.error("must supply at least one of --contributor / --committee / --employer")
    n = fetch(
        api_key=a.api_key,
        contributor=a.contributor,
        committee=a.committee,
        employer=a.employer,
        state=a.state,
        cycle=a.cycle,
        out_path=a.out,
        max_pages=a.max_pages,
    )
    print(f"Wrote {n} FEC contribution rows to {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
