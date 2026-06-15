#!/usr/bin/env python3
"""Fetch OFAC SDN list (CSV format) and normalize.

Public endpoint: https://www.treasury.gov/ofac/downloads/sdn.csv
Format reference: https://ofac.treasury.gov/specially-designated-nationals-and-blocked-persons-list-sdn-human-readable-lists

The SDN CSV uses a specific 12-column format with no header row:
    ent_num, sdn_name, sdn_type, program, title, call_sign, vess_type,
    tonnage, grt, vess_flag, vess_owner, remarks
Address and AKA records live in separate files. We fetch all three and join.
"""
from __future__ import annotations

import argparse
import csv
import io
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from _http import get  # noqa: E402

SDN_URL = "https://www.treasury.gov/ofac/downloads/sdn.csv"
ADD_URL = "https://www.treasury.gov/ofac/downloads/add.csv"
ALT_URL = "https://www.treasury.gov/ofac/downloads/alt.csv"

SDN_COLS = [
    "ent_num", "sdn_name", "sdn_type", "program", "title",
    "call_sign", "vess_type", "tonnage", "grt", "vess_flag",
    "vess_owner", "remarks",
]
ADD_COLS = [
    "ent_num", "add_num", "address", "city_state_zip", "country", "add_remarks",
]
ALT_COLS = [
    "ent_num", "alt_num", "alt_type", "alt_name", "alt_remarks",
]

COLUMNS = [
    "entity_id",
    "name",
    "entity_type",
    "program_list",
    "title",
    "nationalities",
    "aka_list",
    "addresses",
    "dob",
    "pob",
    "remarks",
    "last_updated",
]

_TYPE_MAP = {
    "individual": "individual",
    "entity": "entity",
    "vessel": "vessel",
    "aircraft": "aircraft",
}


def _read_csv(url: str, columns: list[str]) -> list[dict[str, str]]:
    body = get(url, timeout=60).decode("latin-1", errors="replace")
    reader = csv.reader(io.StringIO(body))
    out = []
    for row in reader:
        if not row:
            continue
        # Pad/truncate to expected width.
        row = row[: len(columns)] + [""] * (len(columns) - len(row))
        out.append(dict(zip(columns, row)))
    return out


def _strip_quotes(s: str) -> str:
    s = s.strip()
    if s.startswith('"') and s.endswith('"'):
        s = s[1:-1]
    if s == "-0-":
        return ""
    return s


def fetch(
    program: str | None,
    entity_type: str | None,
    out_path: str,
) -> int:
    sdn = _read_csv(SDN_URL, SDN_COLS)
    addresses = _read_csv(ADD_URL, ADD_COLS)
    akas = _read_csv(ALT_URL, ALT_COLS)

    addr_by_ent: dict[str, list[str]] = defaultdict(list)
    for a in addresses:
        ent = _strip_quotes(a["ent_num"])
        parts = [
            _strip_quotes(a[c])
            for c in ("address", "city_state_zip", "country")
            if _strip_quotes(a[c])
        ]
        if parts:
            addr_by_ent[ent].append(", ".join(parts))

    aka_by_ent: dict[str, list[str]] = defaultdict(list)
    for k in akas:
        ent = _strip_quotes(k["ent_num"])
        name = _strip_quotes(k["alt_name"])
        if name:
            aka_by_ent[ent].append(name)

    rows: list[dict[str, str]] = []
    for r in sdn:
        ent_num = _strip_quotes(r["ent_num"])
        if not ent_num:
            continue
        sdn_type = _TYPE_MAP.get(_strip_quotes(r["sdn_type"]).lower(), _strip_quotes(r["sdn_type"]))
        if entity_type and sdn_type != entity_type:
            continue
        progs = _strip_quotes(r["program"])
        if program and program.upper() not in progs.upper().split(";"):
            continue
        remarks = _strip_quotes(r["remarks"])
        # DOB / POB are commonly embedded in remarks for individuals.
        dob = ""
        pob = ""
        if sdn_type == "individual" and remarks:
            for chunk in remarks.split(";"):
                ch = chunk.strip()
                if ch.upper().startswith("DOB"):
                    dob = ch.split(maxsplit=1)[1] if " " in ch else ""
                elif ch.upper().startswith("POB"):
                    pob = ch.split(maxsplit=1)[1] if " " in ch else ""
        rows.append(
            {
                "entity_id": ent_num,
                "name": _strip_quotes(r["sdn_name"]),
                "entity_type": sdn_type,
                "program_list": "; ".join(p.strip() for p in progs.split(";") if p.strip()),
                "title": _strip_quotes(r["title"]),
                "nationalities": "",  # not in this CSV; available in XML format
                "aka_list": "; ".join(aka_by_ent.get(ent_num, [])),
                "addresses": "; ".join(addr_by_ent.get(ent_num, [])),
                "dob": dob,
                "pob": pob,
                "remarks": remarks,
                "last_updated": "",
            }
        )

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=COLUMNS)
        w.writeheader()
        w.writerows(rows)
    return len(rows)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--program", help="Filter to specific sanctions program (e.g. SDGT, IRAN)")
    p.add_argument(
        "--entity-type",
        choices=["individual", "entity", "vessel", "aircraft"],
        help="Filter to a specific entity type",
    )
    p.add_argument("--out", required=True)
    a = p.parse_args()
    n = fetch(program=a.program, entity_type=a.entity_type, out_path=a.out)
    print(f"Wrote {n} OFAC SDN rows to {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
