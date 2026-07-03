"""Ingest the California Data Broker Registry into broker records (coverage breadth).

The CA registry (CPPA, under the Delete Act) is the authoritative universe of data
brokers doing business with California residents -- ~545 businesses in 2025, each
required to publish a name, website, contact email, and a CCPA-rights/deletion URL.
This is the same universe commercial services (DeleteMe/Incogni/Optery) draw from,
plus the FCRA/GLBA-regulated and marketing/risk brokers most lists omit.

These are NOT people-search sites you scan with a name -- most have no per-person
lookup UI. They are worked through the LEGAL lane: the CA DROP portal
(privacy.ca.gov/drop) is a single request that deletes from ALL registered brokers
at once (CA residents), and per-broker CCPA deletion emails to the contact address
are the fallback / non-CA path. So registry records are kept in their own lane
(loaded only when asked) and never dumped into the people-search scan pipeline.

`parse()` is pure (CSV text in, records out) so it is tested offline; `fetch()` is
the only network call and can be bypassed by passing csv_text directly to refresh().
"""
from __future__ import annotations

import csv
import datetime
import io
import re
import urllib.request
from pathlib import Path

import storage

# CA CPPA registry CSVs are published per year (registry2024.csv, registry2025.csv, ...).
# 2025 is the latest COMPLETE dataset; the current year's file is empty until the Jan
# registration window closes. DEFAULT_URL is the known-good fallback; `ca_candidate_urls`
# probes newer years first so coverage auto-advances when the next year is published.
_CA_CSV = "https://cppa.ca.gov/data_broker_registry/registry{year}.csv"
_CA_FLOOR_YEAR = 2025
DEFAULT_URL = _CA_CSV.format(year=_CA_FLOOR_YEAR)
DROP_URL = "https://privacy.ca.gov/drop"
USER_AGENT = "Mozilla/5.0 (compatible; unbroker/1.0; data opt-out)"


def ca_candidate_urls(today: datetime.date | None = None) -> list[str]:
    """Newest-year-first CA registry URLs to try (auto-advances; never below the 2025 floor)."""
    year = (today or datetime.date.today()).year
    years = list(range(max(year, _CA_FLOOR_YEAR), _CA_FLOOR_YEAR - 1, -1))
    return [_CA_CSV.format(year=y) for y in years]

# Multi-source registry lane. Only California publishes a clean bulk CSV (with contact email +
# CCPA-rights URL per broker) AND offers a one-shot deletion portal (DROP). Vermont, Oregon, and
# Texas maintain registries too, but only as searchable PORTALS (no reliable bulk export) and with
# no DROP-equivalent -- and they overlap CA heavily (CA is effectively the superset). So they are
# wired as first-class portal sources (official URL surfaced to the operator) rather than scraped.
# Adding any state that later publishes a CSV is a one-line "format: csv" entry (the parser is
# column-detection based, not CA-specific).
SOURCES = {
    "ca": {"jurisdiction": "US-CA", "format": "csv", "url": DEFAULT_URL, "has_drop": True,
           "name": "California Data Broker Registry (CPPA)"},
    "vt": {"jurisdiction": "US-VT", "format": "portal", "has_drop": False,
           "url": "https://bizfilings.vermont.gov/online/DatabrokerInquire/",
           "name": "Vermont Data Broker Registry (Secretary of State)"},
    "or": {"jurisdiction": "US-OR", "format": "portal", "has_drop": False,
           "url": "https://dfr.oregon.gov/business/licensing/data-broker-registry/Pages/index.aspx",
           "name": "Oregon Data Broker Registry (DCBS)"},
    "tx": {"jurisdiction": "US-TX", "format": "portal", "has_drop": False,
           "url": "https://texas-sos.appianportalsgov.com/data-broker-registry",
           "name": "Texas Data Broker Registry (Secretary of State)"},
}


def portals() -> list[dict]:
    """Registry sources that are searchable portals (no bulk export) -- surfaced to the operator."""
    return [{"key": k, "jurisdiction": s["jurisdiction"], "name": s["name"], "url": s["url"]}
            for k, s in SOURCES.items() if s["format"] == "portal"]

# Field label -> substring to locate its column on the header row (robust to
# year-to-year column shifts; the registry re-orders/adds columns between years).
_LABELS = {
    "name": "data broker name:",
    "dba": "doing business as",
    "website": "data broker primary website:",
    "email": "primary contact email",
    "rights_url": "exercise their ca consumer privacy act rights",
    "fcra": "regulated by the federal fair credit reporting act (fcra):",
}


def _norm(s: str) -> str:
    """Registry CSVs use NBSPs and a BOM; normalize for matching + clean values."""
    return re.sub(r"\s+", " ", (s or "").replace("\ufeff", "").replace("\xa0", " ")).strip()


def slug(name: str, website: str = "") -> str:
    base = re.sub(r"\.(com|org|net|io|ai|inc|co|us|info|llc)\b", "", (name or "").strip(), flags=re.I)
    s = re.sub(r"[^a-z0-9]+", "", base.lower())
    if s:
        return s
    dom = re.sub(r"^https?://(www\.)?", "", (website or "").lower())
    return re.sub(r"[^a-z0-9]+", "", dom.split("/")[0]) or "broker"


def _domain(website: str) -> str:
    dom = re.sub(r"^https?://(www\.)?", "", (website or "").strip().lower())
    return dom.split("/")[0]


def _find_colmap(rows: list[list[str]]) -> tuple[int, dict[str, int]]:
    """Locate the label row (col0 == 'Data broker name:') and map fields to columns."""
    for i, row in enumerate(rows[:5]):
        if row and _norm(row[0]).lower().startswith("data broker name:"):
            colmap: dict[str, int] = {}
            for field, needle in _LABELS.items():
                for j, cell in enumerate(row):
                    c = _norm(cell).lower()
                    if needle in c and not c.startswith("if the data broker"):
                        colmap[field] = j
                        break
            return i, colmap
    raise ValueError("CA registry: could not locate the header row")


def _get(row: list[str], idx: int | None) -> str:
    return _norm(row[idx]) if idx is not None and idx < len(row) else ""


def _build(row: list[str], cm: dict[str, int], jurisdiction: str = "US-CA",
           has_drop: bool = True) -> dict | None:
    name = _get(row, cm.get("name"))
    website = _get(row, cm.get("website"))
    if not (name or website):
        return None
    email = _get(row, cm.get("email"))
    rights = _get(row, cm.get("rights_url"))
    dba = _get(row, cm.get("dba"))
    fcra = _get(row, cm.get("fcra")).lower().startswith("y")
    state = jurisdiction.split("-")[-1]

    method = "email" if email else ("web_form" if rights else "drop")
    if has_drop:
        notes = ("Registered CA data broker. One CA DROP request (privacy.ca.gov/drop) deletes from "
                 "this and every registered broker at once; or send a CCPA deletion request to the "
                 "contact email.")
    else:
        notes = (f"Registered {state} data broker (no one-shot delete portal in {state}). Send a "
                 "CCPA/state-law deletion request to the contact email.")
    if fcra:
        notes += (" FCRA-regulated: some data is credit-reporting data with separate rules -- deletion "
                  "may be limited; a consumer report dispute/security-freeze may apply instead.")
    return {
        "id": slug(name, website),
        "name": name or _domain(website),
        "dba": dba or None,
        "category": "data_broker",
        "priority": "long_tail",
        "jurisdictions": [jurisdiction],
        "search": {"method": "none", "url": website, "fetch": "none", "by": ["registry"]},
        "optout": {
            "method": method,
            "url": rights or website or None,
            "email": email or None,
            "requires": {"profile_url": False, "email_verification": False, "captcha": False,
                         "gov_id": False, "account": False, "phone_callback": False, "payment": False},
            "inputs": ["full_name", "contact_email"],
            "deletion": {
                "via": "drop" if has_drop else "email",
                "email": email or None,
                "url": rights or None,
                "kinds": ["ccpa", "generic"],
                "notes": ("Covered by the CA DROP one-shot (privacy.ca.gov/drop); CCPA email fallback."
                          if has_drop else "CCPA/state-law deletion email (no one-shot portal)."),
            },
            "fcra": fcra,
            "est_processing_days": 45,
            "notes": notes,
        },
        "source": f"{state}-registry",
        "confidence": "registry",
        "last_verified": None,
    }


def parse(csv_text: str, jurisdiction: str = "US-CA", has_drop: bool = True) -> list[dict]:
    """Parse a data-broker-registry CSV into broker records (deduped by id).

    Column detection is by header label, not fixed position, so any state that publishes a
    registry CSV with name/website/email/rights columns parses without new code.
    """
    rows = list(csv.reader(io.StringIO(csv_text)))
    if not rows:
        return []
    header_i, cm = _find_colmap(rows)
    out: list[dict] = []
    seen: dict[str, int] = {}
    for row in rows[header_i + 1:]:
        if not any(c.strip() for c in row):
            continue
        rec = _build(row, cm, jurisdiction, has_drop)
        if not rec:
            continue
        bid = rec["id"]
        if bid in seen:  # disambiguate id collisions by domain, then a counter
            dom = re.sub(r"[^a-z0-9]+", "", _domain(rec["search"]["url"]))
            cand = f"{bid}-{dom}" if dom and dom != bid else bid
            while cand in seen:
                seen[bid] += 1
                cand = f"{bid}-{seen[bid]}"
            rec["id"] = cand
        seen.setdefault(rec["id"], 0)
        seen.setdefault(bid, 0)
        out.append(rec)
    return out


MIN_EXPECTED_CA = 100  # CA registry has ~500+; far fewer => wrong/empty file, warn


def fetch(url: str = DEFAULT_URL, timeout: int = 60) -> str:
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310
        return resp.read().decode("utf-8", errors="replace")


def _fetch_ca_latest() -> tuple[str, list[dict]]:
    """Try newest CA registry year first; return (url, records) for the first non-empty."""
    last: tuple[str, list[dict]] = (DEFAULT_URL, [])
    for url in ca_candidate_urls():
        try:
            recs = parse(fetch(url), jurisdiction="US-CA", has_drop=True)
        except Exception:  # noqa: BLE001 - a missing year 404s; fall through to older years
            continue
        if recs:
            return url, recs
        last = (url, recs)
    return last


def refresh(cache_path: Path, url: str = DEFAULT_URL, csv_text: str | None = None) -> dict:
    """CA single-source refresh: fetch (or accept) the CA CSV and write the cache."""
    text = csv_text if csv_text is not None else fetch(url)
    records = parse(text)
    storage.write_json(cache_path, records)
    fcra = sum(1 for r in records if (r.get("optout") or {}).get("fcra"))
    return {"parsed": len(records), "fcra_regulated": fcra,
            "cache_path": str(cache_path), "source_url": url}


def refresh_all(cache_path: Path, fetched: dict[str, str] | None = None) -> dict:
    """Multi-source refresh: pull every CSV source, dedupe across states by domain, cache.

    `fetched` optionally supplies {source_key: csv_text} to bypass the network (tests). CSV
    sources are ingested as broker records; portal sources contribute their URL for the operator
    (no bulk export exists) but no records. CA is processed first so it wins domain collisions.
    """
    all_recs: list[dict] = []
    seen_domains: set[str] = set()
    per_source: dict[str, dict] = {}
    for key, src in SOURCES.items():
        if src["format"] != "csv":
            per_source[key] = {"jurisdiction": src["jurisdiction"], "format": "portal",
                               "url": src["url"], "records": 0,
                               "note": "searchable portal (no bulk export); operator/agent searches by name"}
            continue
        used_url = src["url"]
        try:
            if fetched is not None:
                text = fetched.get(key)
                if text is None:
                    raise RuntimeError("no CSV text supplied")
                recs = parse(text, jurisdiction=src["jurisdiction"], has_drop=src["has_drop"])
            elif key == "ca":
                used_url, recs = _fetch_ca_latest()   # newest-year-first with fallback
            else:
                recs = parse(fetch(src["url"]), jurisdiction=src["jurisdiction"], has_drop=src["has_drop"])
        except Exception as exc:  # noqa: BLE001 - one source failing must not sink the rest
            per_source[key] = {"jurisdiction": src["jurisdiction"], "format": "csv", "error": str(exc)}
            continue
        added = 0
        for r in recs:
            dom = _domain(r["search"]["url"])
            if dom and dom in seen_domains:
                continue
            if dom:
                seen_domains.add(dom)
            all_recs.append(r)
            added += 1
        entry = {"jurisdiction": src["jurisdiction"], "format": "csv", "url": used_url,
                 "parsed": len(recs), "added_after_dedupe": added,
                 "fcra": sum(1 for r in recs if (r.get("optout") or {}).get("fcra"))}
        if key == "ca" and len(recs) < MIN_EXPECTED_CA:
            entry["warning"] = (f"only {len(recs)} parsed (expected >{MIN_EXPECTED_CA}); the CA "
                                "registry file may be empty/moved - verify the source URL")
        per_source[key] = entry
    storage.write_json(cache_path, all_recs)
    return {"total": len(all_recs), "sources": per_source, "portals": portals(),
            "cache_path": str(cache_path)}
