#!/usr/bin/env python3
"""Search NYC property records via ACRIS (Automated City Register Information System).

Uses the city's Socrata-backed open data API. No auth required for read access.

Datasets:
  bnx9-e6tj — Real Property Master (one row per recorded document)
  636b-3b5g — Real Property Parties (names — grantor, grantee, etc.)
  8h5j-fqxa — Real Property Legal (lot / property identifiers)
  uqqa-hym2 — Real Property References

The Parties dataset has the names. We search by name and optionally join to
Master to get the doc type and date.
"""
from __future__ import annotations

import argparse
import csv
import sys
import urllib.parse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from _http import get_json  # noqa: E402

PARTIES_URL = "https://data.cityofnewyork.us/resource/636b-3b5g.json"
MASTER_URL = "https://data.cityofnewyork.us/resource/bnx9-e6tj.json"

PARTY_TYPE = {
    "1": "grantor (seller / mortgagor / debtor)",
    "2": "grantee (buyer / mortgagee / creditor)",
    "3": "other party",
}

BOROUGH = {
    "1": "Manhattan",
    "2": "Bronx",
    "3": "Brooklyn",
    "4": "Queens",
    "5": "Staten Island",
}

COLUMNS = [
    "document_id",
    "name",
    "party_type",
    "party_role",
    "address_1",
    "address_2",
    "city",
    "state",
    "zip",
    "country",
    "doc_type",
    "doc_date",
    "recorded_date",
    "borough",
    "amount",
    "filing_url",
]


def _filing_url(document_id: str) -> str:
    if not document_id:
        return ""
    return (
        f"https://a836-acris.nyc.gov/DS/DocumentSearch/DocumentImageView?doc_id={document_id}"
    )


def fetch(
    name: str | None,
    address: str | None,
    party_type: str | None,
    limit: int,
    out_path: str,
    enrich: bool = True,
) -> int:
    if not (name or address):
        raise SystemExit("must supply --name or --address")

    where_clauses: list[str] = []
    if name:
        safe = name.upper().replace("'", "''")
        where_clauses.append(f"upper(name) like '%{safe}%'")
    if address:
        safe_addr = address.upper().replace("'", "''")
        where_clauses.append(f"upper(address_1) like '%{safe_addr}%'")
    if party_type and party_type in {"1", "2", "3"}:
        where_clauses.append(f"party_type='{party_type}'")

    params = {
        "$where": " AND ".join(where_clauses),
        "$limit": str(limit),
    }
    url = f"{PARTIES_URL}?{urllib.parse.urlencode(params)}"
    parties = get_json(url)
    if not isinstance(parties, list):
        raise SystemExit(f"Unexpected ACRIS response: {parties!r}")

    # Enrich with master record (doc_type, dates, borough, amount).
    doc_ids: list[str] = sorted({
        d for d in (p.get("document_id") for p in parties) if d
    })
    masters: dict[str, dict] = {}
    if enrich and doc_ids:
        # Batch up to 100 doc_ids per request (Socrata IN-list is fine for this).
        for i in range(0, len(doc_ids), 100):
            chunk = doc_ids[i : i + 100]
            id_list = ",".join(f"'{d}'" for d in chunk)
            master_params = {
                "$where": f"document_id in ({id_list})",
                "$limit": "100",
            }
            url = f"{MASTER_URL}?{urllib.parse.urlencode(master_params)}"
            try:
                rows = get_json(url)
            except Exception as e:  # noqa: BLE001
                print(f"ACRIS master lookup failed for chunk: {e}", file=sys.stderr)
                continue
            if isinstance(rows, list):
                for r in rows:
                    did = r.get("document_id", "")
                    if did:
                        masters[did] = r

    out_rows: list[dict[str, str]] = []
    for p in parties:
        did = p.get("document_id", "") or ""
        m = masters.get(did, {})
        out_rows.append(
            {
                "document_id": did,
                "name": p.get("name", "") or "",
                "party_type": p.get("party_type", "") or "",
                "party_role": PARTY_TYPE.get(p.get("party_type", ""), ""),
                "address_1": p.get("address_1", "") or "",
                "address_2": p.get("address_2", "") or "",
                "city": p.get("city", "") or "",
                "state": p.get("state", "") or "",
                "zip": p.get("zip", "") or "",
                "country": p.get("country", "") or "",
                "doc_type": m.get("doc_type", "") or "",
                "doc_date": (m.get("document_date", "") or "")[:10],
                "recorded_date": (m.get("recorded_datetime", "") or "")[:10],
                "borough": BOROUGH.get(m.get("recorded_borough", ""), m.get("recorded_borough", "")),
                "amount": m.get("document_amt", "") or "",
                "filing_url": _filing_url(did),
            }
        )

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=COLUMNS)
        w.writeheader()
        w.writerows(out_rows)

    if not out_rows:
        filters = []
        if name:
            filters.append(f"name={name!r}")
        if address:
            filters.append(f"address={address!r}")
        print(
            f"NYC ACRIS: 0 records for {', '.join(filters)}. "
            "ACRIS covers ONLY NYC (5 boroughs). For property records elsewhere, "
            "search the relevant county recorder directly.",
            file=sys.stderr,
        )
    return len(out_rows)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--name", help="Party name substring (case-insensitive)")
    p.add_argument("--address", help="Address line 1 substring")
    p.add_argument(
        "--party-type",
        choices=["1", "2", "3"],
        help="Filter party type: 1=grantor (seller/mortgagor), 2=grantee (buyer/mortgagee), 3=other",
    )
    p.add_argument("--limit", type=int, default=200)
    p.add_argument(
        "--no-enrich",
        action="store_true",
        help="Skip the master-document lookup that adds doc_type/date/amount",
    )
    p.add_argument("--out", required=True)
    a = p.parse_args()
    n = fetch(
        name=a.name,
        address=a.address,
        party_type=a.party_type,
        limit=a.limit,
        out_path=a.out,
        enrich=not a.no_enrich,
    )
    print(f"Wrote {n} NYC ACRIS rows to {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
