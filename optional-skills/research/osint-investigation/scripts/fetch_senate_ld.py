#!/usr/bin/env python3
"""Fetch Senate Lobbying Disclosure (LD-1 / LD-2) filings.

Anonymous: 120 req/hour. Token (SENATE_LDA_TOKEN): 1200 req/hour.
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

ENDPOINT = "https://lda.senate.gov/api/v1/filings/"
COLUMNS = [
    "filing_uuid",
    "filing_type",
    "filing_year",
    "filing_period",
    "registrant_name",
    "registrant_id",
    "client_name",
    "client_id",
    "client_general_description",
    "income",
    "expenses",
    "lobbyists",
    "issues",
    "government_entities",
    "filing_date",
]


def fetch(
    client: str | None,
    registrant: str | None,
    year: int,
    token: str | None,
    out_path: str,
    page_size: int = 100,
    max_pages: int = 25,
) -> int:
    params: dict = {"filing_year": year, "page_size": page_size}
    if client:
        params["client_name"] = client
    if registrant:
        params["registrant_name"] = registrant

    headers = {"Authorization": f"Token {token}"} if token else None
    rows: list[dict[str, str]] = []
    url = ENDPOINT
    page = 0
    while page < max_pages:
        try:
            payload = get_json(url, params=params if page == 0 else None, headers=headers)
        except Exception as e:  # noqa: BLE001
            print(f"Senate LDA error on page {page + 1}: {e}", file=sys.stderr)
            break
        if not isinstance(payload, dict):
            break
        results = payload.get("results", [])
        for r in results:
            client_obj = r.get("client") or {}
            registrant_obj = r.get("registrant") or {}
            lobbying_activities = r.get("lobbying_activities") or []
            lobbyists = []
            issues = []
            entities = []
            for la in lobbying_activities:
                for lob in la.get("lobbyists") or []:
                    lob_obj = lob.get("lobbyist") or {}
                    name = " ".join(
                        x for x in (lob_obj.get("first_name", ""), lob_obj.get("last_name", "")) if x
                    )
                    if name:
                        lobbyists.append(name)
                desc = la.get("description") or ""
                if desc:
                    issues.append(desc)
                for ge in la.get("government_entities") or []:
                    nm = ge.get("name") or ""
                    if nm:
                        entities.append(nm)
            rows.append(
                {
                    "filing_uuid": r.get("filing_uuid", "") or "",
                    "filing_type": r.get("filing_type", "") or "",
                    "filing_year": str(r.get("filing_year", "") or year),
                    "filing_period": r.get("filing_period", "") or "",
                    "registrant_name": registrant_obj.get("name", "") or "",
                    "registrant_id": str(registrant_obj.get("id", "") or ""),
                    "client_name": client_obj.get("name", "") or "",
                    "client_id": str(client_obj.get("id", "") or ""),
                    "client_general_description": client_obj.get("general_description", "") or "",
                    "income": str(r.get("income", "") or ""),
                    "expenses": str(r.get("expenses", "") or ""),
                    "lobbyists": "; ".join(sorted(set(lobbyists))),
                    "issues": "; ".join(issues),
                    "government_entities": "; ".join(sorted(set(entities))),
                    "filing_date": (r.get("dt_posted") or "")[:10],
                }
            )
        next_url = payload.get("next")
        if not next_url:
            break
        url = next_url
        page += 1
        time.sleep(1.0 if not token else 0.3)

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=COLUMNS)
        w.writeheader()
        w.writerows(rows)
    return len(rows)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--client", help="Client name filter")
    p.add_argument("--registrant", help="Registrant (lobbying firm) name filter")
    p.add_argument("--year", type=int, default=2024)
    p.add_argument("--token", default=os.environ.get("SENATE_LDA_TOKEN"))
    p.add_argument("--max-pages", type=int, default=25)
    p.add_argument("--out", required=True)
    a = p.parse_args()
    if not (a.client or a.registrant):
        p.error("must supply at least one of --client / --registrant")
    n = fetch(
        client=a.client,
        registrant=a.registrant,
        year=a.year,
        token=a.token,
        out_path=a.out,
        max_pages=a.max_pages,
    )
    print(f"Wrote {n} Senate LDA rows to {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
