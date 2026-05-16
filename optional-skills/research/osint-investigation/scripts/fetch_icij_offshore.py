#!/usr/bin/env python3
"""Search ICIJ Offshore Leaks via the bulk CSV database.

The old reconcile endpoint (https://offshoreleaks.icij.org/reconcile) returns
404 — ICIJ has removed it. The remaining stable access path is the public
bulk download:

    https://offshoreleaks-data.icij.org/offshoreleaks/csv/full-oldb.LATEST.zip

~70 MB, ~6 CSVs inside (nodes-entities, nodes-officers, nodes-intermediaries,
nodes-addresses, relationships, ...). We cache it under
$HERMES_OSINT_CACHE/icij/ (default: ~/.cache/hermes-osint/icij/) and search
locally so the agent doesn't re-download for every query.

Output CSV columns match the original `fetch_icij_offshore.py` contract.
"""
from __future__ import annotations

import argparse
import csv
import io
import os
import re
import sys
import time
import urllib.request
import zipfile
from pathlib import Path

BULK_URL = "https://offshoreleaks-data.icij.org/offshoreleaks/csv/full-oldb.LATEST.zip"

COLUMNS = [
    "node_id",
    "name",
    "node_type",
    "country_codes",
    "countries",
    "jurisdiction",
    "incorporation_date",
    "inactivation_date",
    "source",
    "entity_url",
    "connections",
]


def _cache_dir() -> Path:
    base = os.environ.get("HERMES_OSINT_CACHE")
    if base:
        return Path(base) / "icij"
    return Path.home() / ".cache" / "hermes-osint" / "icij"


def _download(dest: Path, force: bool = False) -> Path:
    """Download (or reuse cached) ICIJ bulk ZIP."""
    dest.mkdir(parents=True, exist_ok=True)
    zip_path = dest / "full-oldb.zip"
    if zip_path.exists() and not force:
        # Re-check age: refetch if older than 30 days.
        age_days = (time.time() - zip_path.stat().st_mtime) / 86400
        if age_days < 30:
            return zip_path
    print(f"Downloading ICIJ bulk database (~70 MB) to {zip_path}", file=sys.stderr)
    req = urllib.request.Request(
        BULK_URL,
        headers={"User-Agent": "hermes-agent osint-investigation skill"},
    )
    with urllib.request.urlopen(req, timeout=120) as resp:  # noqa: S310
        tmp = zip_path.with_suffix(".zip.tmp")
        with open(tmp, "wb") as fh:
            while True:
                chunk = resp.read(1 << 16)
                if not chunk:
                    break
                fh.write(chunk)
    tmp.replace(zip_path)
    return zip_path


def _open_csv(zf: zipfile.ZipFile, name_pattern: str):
    """Open the first CSV matching name_pattern (case-insensitive substring)."""
    for info in zf.infolist():
        if name_pattern.lower() in info.filename.lower() and info.filename.lower().endswith(".csv"):
            return zf.open(info), info.filename
    return None, None


def _match(needle_norm: str, hay: str) -> bool:
    return needle_norm in (hay or "").upper()


def _normalize_query(s: str) -> str:
    s = s.upper()
    s = re.sub(r"[^\w\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def fetch(
    entity: str | None,
    officer: str | None,
    jurisdiction: str | None,
    out_path: str,
    cache_dir: Path,
    force_refresh: bool = False,
    limit: int = 500,
) -> int:
    zip_path = _download(cache_dir, force=force_refresh)
    rows: list[dict[str, str]] = []
    needles: list[tuple[str, str]] = []  # (kind, normalized needle)
    if entity:
        needles.append(("Entity", _normalize_query(entity)))
    if officer:
        needles.append(("Officer", _normalize_query(officer)))
    jur_norm = _normalize_query(jurisdiction) if jurisdiction else None

    targets = [
        ("Entity", "nodes-entities"),
        ("Officer", "nodes-officers"),
        ("Intermediary", "nodes-intermediaries"),
    ]

    with zipfile.ZipFile(zip_path) as zf:
        for node_type, csv_substring in targets:
            relevant_needles = [n for (k, n) in needles if k in (node_type, "Entity", "Officer")] or []
            # Only scan a CSV if we have a needle that could plausibly match it,
            # or if we have ONLY a jurisdiction filter.
            applicable_needles = [n for (k, n) in needles if k == node_type]
            if needles and not applicable_needles and not jur_norm:
                continue
            stream, fname = _open_csv(zf, csv_substring)
            if not stream:
                continue
            with stream:
                text = io.TextIOWrapper(stream, encoding="utf-8", errors="replace")
                reader = csv.DictReader(text)
                for row in reader:
                    name = (row.get("name") or "").strip()
                    if not name:
                        continue
                    name_u = name.upper()
                    matched = False
                    for n in applicable_needles or relevant_needles:
                        if _match(n, name_u):
                            matched = True
                            break
                    if not needles:
                        matched = True  # jurisdiction-only sweep
                    if not matched:
                        continue
                    jur = (row.get("jurisdiction_description") or row.get("country_codes") or "").strip()
                    if jur_norm and jur_norm not in jur.upper() and jur_norm not in (row.get("countries") or "").upper():
                        continue
                    node_id = (row.get("node_id") or "").strip()
                    rows.append(
                        {
                            "node_id": node_id,
                            "name": name,
                            "node_type": node_type,
                            "country_codes": row.get("country_codes", "") or "",
                            "countries": row.get("countries", "") or "",
                            "jurisdiction": jur,
                            "incorporation_date": row.get("incorporation_date", "") or "",
                            "inactivation_date": row.get("inactivation_date", "") or "",
                            "source": row.get("sourceID", "") or row.get("source", "") or "",
                            "entity_url": (
                                f"https://offshoreleaks.icij.org/nodes/{node_id}" if node_id else ""
                            ),
                            "connections": "",
                        }
                    )
                    if len(rows) >= limit:
                        break
            if len(rows) >= limit:
                break

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=COLUMNS)
        w.writeheader()
        w.writerows(rows)
    if not rows:
        bits = []
        if entity:
            bits.append(f"entity={entity!r}")
        if officer:
            bits.append(f"officer={officer!r}")
        if jurisdiction:
            bits.append(f"jurisdiction={jurisdiction!r}")
        print(
            f"ICIJ: 0 matches for {', '.join(bits)}. "
            "The bulk database covers offshore leaks (Panama, Paradise, Pandora, "
            "Bahamas, Offshore Leaks). Most private US individuals are NOT in it.",
            file=sys.stderr,
        )
    return len(rows)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--entity", help="Search by entity name (substring, case-insensitive)")
    p.add_argument("--officer", help="Search by officer / individual name (substring, case-insensitive)")
    p.add_argument("--jurisdiction", help="Filter results by jurisdiction substring")
    p.add_argument("--limit", type=int, default=500)
    p.add_argument("--out", required=True)
    p.add_argument(
        "--cache-dir",
        type=Path,
        default=None,
        help="Override cache directory (default: $HERMES_OSINT_CACHE/icij or ~/.cache/hermes-osint/icij)",
    )
    p.add_argument(
        "--force-refresh",
        action="store_true",
        help="Re-download the bulk ZIP even if a recent cached copy exists.",
    )
    a = p.parse_args()
    if not (a.entity or a.officer or a.jurisdiction):
        p.error("must supply at least one of --entity / --officer / --jurisdiction")
    n = fetch(
        entity=a.entity,
        officer=a.officer,
        jurisdiction=a.jurisdiction,
        out_path=a.out,
        cache_dir=a.cache_dir or _cache_dir(),
        force_refresh=a.force_refresh,
        limit=a.limit,
    )
    print(f"Wrote {n} ICIJ Offshore Leaks rows to {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
