#!/usr/bin/env python3
"""Fetch federal contracts/awards from USAspending.gov API v2.

No auth required. POST to /api/v2/search/spending_by_award/ with filters.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
import urllib.request
from pathlib import Path

ENDPOINT = "https://api.usaspending.gov/api/v2/search/spending_by_award/"
COLUMNS = [
    "award_id",
    "recipient_name",
    "recipient_uei",
    "recipient_duns",
    "recipient_parent_name",
    "recipient_state",
    "awarding_agency",
    "awarding_sub_agency",
    "award_type",
    "award_amount",
    "award_date",
    "period_of_performance_start",
    "period_of_performance_end",
    "naics_code",
    "psc_code",
    "competition_extent",
    "description",
]

# USAspending result column "code" → human label mapping for output.
_FIELDS = [
    "Award ID",
    "Recipient Name",
    "Recipient UEI",
    "Recipient DUNS Number",
    "Recipient Parent Name",
    "Recipient State Code",
    "Awarding Agency",
    "Awarding Sub Agency",
    "Award Type",
    "Award Amount",
    "Start Date",
    "End Date",
    "NAICS Code",
    "PSC Code",
    "Type of Set Aside",
    "Description",
]


def _post(body: dict) -> dict:
    req = urllib.request.Request(
        ENDPOINT,
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json", "User-Agent": "hermes-agent osint-investigation"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=60) as resp:
        return json.loads(resp.read().decode("utf-8"))


def fetch(
    recipient: str | None,
    agency: str | None,
    fy: int,
    sole_source_only: bool,
    out_path: str,
    page_size: int = 100,
    max_pages: int = 20,
) -> int:
    filters: dict = {
        "time_period": [{"start_date": f"{fy - 1}-10-01", "end_date": f"{fy}-09-30"}],
        # Contracts only by default; adjust award_type_codes for grants/loans.
        "award_type_codes": ["A", "B", "C", "D"],
    }
    if recipient:
        filters["recipient_search_text"] = [recipient]
    if agency:
        filters["agencies"] = [{"type": "awarding", "tier": "toptier", "name": agency}]

    rows: list[dict[str, str]] = []
    page = 1
    while page <= max_pages:
        body = {
            "filters": filters,
            "fields": _FIELDS,
            "page": page,
            "limit": page_size,
            "sort": "Award Amount",
            "order": "desc",
        }
        try:
            payload = _post(body)
        except Exception as e:  # noqa: BLE001
            print(f"USAspending error on page {page}: {e}", file=sys.stderr)
            break
        results = payload.get("results", [])
        if not results:
            break
        for r in results:
            set_aside = r.get("Type of Set Aside", "") or ""
            if sole_source_only and "sole" not in set_aside.lower():
                continue
            rows.append(
                {
                    "award_id": r.get("Award ID", "") or "",
                    "recipient_name": r.get("Recipient Name", "") or "",
                    "recipient_uei": r.get("Recipient UEI", "") or "",
                    "recipient_duns": r.get("Recipient DUNS Number", "") or "",
                    "recipient_parent_name": r.get("Recipient Parent Name", "") or "",
                    "recipient_state": r.get("Recipient State Code", "") or "",
                    "awarding_agency": r.get("Awarding Agency", "") or "",
                    "awarding_sub_agency": r.get("Awarding Sub Agency", "") or "",
                    "award_type": r.get("Award Type", "") or "",
                    "award_amount": str(r.get("Award Amount", "") or ""),
                    "award_date": r.get("Start Date", "") or "",
                    "period_of_performance_start": r.get("Start Date", "") or "",
                    "period_of_performance_end": r.get("End Date", "") or "",
                    "naics_code": str(r.get("NAICS Code", "") or ""),
                    "psc_code": str(r.get("PSC Code", "") or ""),
                    "competition_extent": set_aside,
                    "description": r.get("Description", "") or "",
                }
            )
        meta = payload.get("page_metadata", {})
        if not meta.get("hasNext"):
            break
        page += 1
        time.sleep(0.5)

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=COLUMNS)
        w.writeheader()
        w.writerows(rows)
    return len(rows)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--recipient", help="Recipient name search")
    p.add_argument("--agency", help="Awarding agency (top-tier)")
    p.add_argument("--fy", type=int, default=2024, help="Federal fiscal year")
    p.add_argument("--sole-source-only", action="store_true")
    p.add_argument("--max-pages", type=int, default=20)
    p.add_argument("--out", required=True)
    a = p.parse_args()
    if not (a.recipient or a.agency):
        p.error("must supply at least one of --recipient / --agency")
    n = fetch(
        recipient=a.recipient,
        agency=a.agency,
        fy=a.fy,
        sole_source_only=a.sole_source_only,
        out_path=a.out,
        max_pages=a.max_pages,
    )
    print(f"Wrote {n} USAspending rows to {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
