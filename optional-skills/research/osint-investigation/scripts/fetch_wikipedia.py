#!/usr/bin/env python3
"""Search Wikipedia + Wikidata for an entity (person, company, place, concept).

Two free APIs:
  - Wikipedia OpenSearch + REST summary endpoint for narrative bio
  - Wikidata SPARQL endpoint for structured facts (birth, employer, awards, etc.)

Both are anonymous-access. Useful for resolving who-is-this-entity questions
and surfacing cross-references that other sources can join against.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import urllib.parse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from _http import get_json  # noqa: E402

WP_OPENSEARCH = "https://en.wikipedia.org/w/api.php"
WP_SUMMARY = "https://en.wikipedia.org/api/rest_v1/page/summary/"
WD_ACTION = "https://www.wikidata.org/w/api.php"

COLUMNS = [
    "source",
    "label",
    "description",
    "qid",
    "wikipedia_title",
    "wikipedia_url",
    "wikidata_url",
    "instance_of",
    "country",
    "occupation",
    "employer",
    "date_of_birth",
    "place_of_birth",
    "summary",
]


def _wp_search(query: str, limit: int) -> list[dict]:
    params = {
        "action": "opensearch",
        "search": query,
        "limit": str(min(limit, 20)),
        "format": "json",
    }
    url = f"{WP_OPENSEARCH}?{urllib.parse.urlencode(params)}"
    data = get_json(url)
    if not isinstance(data, list) or len(data) < 4:
        return []
    titles, descs, urls = data[1], data[2], data[3]
    out = []
    for i, title in enumerate(titles):
        out.append(
            {
                "title": title,
                "description": descs[i] if i < len(descs) else "",
                "url": urls[i] if i < len(urls) else "",
            }
        )
    return out


def _wp_summary(title: str) -> dict:
    """Pull the REST summary for a title — short bio, image, type."""
    url = f"{WP_SUMMARY}{urllib.parse.quote(title.replace(' ', '_'))}"
    try:
        return get_json(url)  # type: ignore[return-value]
    except Exception as e:  # noqa: BLE001
        print(f"Wikipedia summary lookup for {title!r} failed: {e}", file=sys.stderr)
        return {}


def _wd_lookup_by_qid(qid: str) -> dict:
    """Pull common facts for a QID via Wikidata's Action API (no SPARQL).

    The Action API is far more lenient on rate-limits than the SPARQL Query
    Service. We get claims as QIDs and then resolve labels in one batch call.
    """
    # Properties of interest. The Action API returns claims as QIDs or
    # typed literals, so the slot mapping is local-only.
    interesting = {
        "P31": "instance_of",
        "P17": "country",          # for orgs / places
        "P27": "country",          # for individuals (country of citizenship)
        "P106": "occupation",
        "P108": "employer",
        "P569": "date_of_birth",
        "P19": "place_of_birth",
    }
    params = {
        "action": "wbgetentities",
        "ids": qid,
        "props": "claims",
        "format": "json",
    }
    url = f"{WD_ACTION}?{urllib.parse.urlencode(params)}"
    try:
        data = get_json(url)
    except Exception as e:  # noqa: BLE001
        print(f"Wikidata wbgetentities for {qid} failed: {e}", file=sys.stderr)
        return {}
    if not isinstance(data, dict):
        return {}
    claims = (data.get("entities", {}).get(qid, {}) or {}).get("claims", {}) or {}

    # Collect raw values (QIDs or literals) and remember which slot each
    # came from. Date literals come back as ISO strings; QIDs need a label
    # resolution pass.
    qid_to_slots: dict[str, list[str]] = {}
    facts: dict[str, list[str]] = {}
    for prop_id, slot in interesting.items():
        for claim in claims.get(prop_id, []) or []:
            v = (claim.get("mainsnak", {}) or {}).get("datavalue", {}) or {}
            vtype = v.get("type")
            value = v.get("value")
            if vtype == "wikibase-entityid" and isinstance(value, dict):
                vqid = value.get("id", "")
                if vqid:
                    qid_to_slots.setdefault(vqid, [])
                    if slot not in qid_to_slots[vqid]:
                        qid_to_slots[vqid].append(slot)
            elif vtype == "time" and isinstance(value, dict):
                raw = value.get("time", "") or ""
                # +1955-10-28T00:00:00Z → 1955-10-28
                m = re.search(r"[+-]?(\d{4})-(\d{2})-(\d{2})", raw)
                if m:
                    facts.setdefault(slot, []).append(
                        f"{m.group(1)}-{m.group(2)}-{m.group(3)}"
                    )
            elif vtype == "string":
                facts.setdefault(slot, []).append(str(value))

    # Resolve labels for all referenced QIDs in one batch (up to 50 at a time).
    qids = list(qid_to_slots)
    for i in range(0, len(qids), 50):
        batch = qids[i : i + 50]
        params = {
            "action": "wbgetentities",
            "ids": "|".join(batch),
            "props": "labels",
            "languages": "en",
            "format": "json",
        }
        url = f"{WD_ACTION}?{urllib.parse.urlencode(params)}"
        try:
            data = get_json(url)
        except Exception as e:  # noqa: BLE001
            print(f"Wikidata label batch failed: {e}", file=sys.stderr)
            continue
        if not isinstance(data, dict):
            continue
        ents = data.get("entities", {}) or {}
        for vqid, ent in ents.items():
            label = (ent.get("labels", {}).get("en", {}) or {}).get("value", "") or vqid
            for slot in qid_to_slots.get(vqid, []):
                facts.setdefault(slot, []).append(label)

    # Deduplicate per slot, preserving order.
    deduped: dict[str, list[str]] = {}
    for slot, vals in facts.items():
        seen = set()
        out = []
        for v in vals:
            if v in seen:
                continue
            seen.add(v)
            out.append(v)
        deduped[slot] = out
    return deduped


def _wd_qid_for_title(title: str) -> str:
    """Get the Wikidata QID associated with a Wikipedia article title."""
    params = {
        "action": "query",
        "format": "json",
        "prop": "pageprops",
        "ppprop": "wikibase_item",
        "titles": title,
        "redirects": 1,
    }
    url = f"{WP_OPENSEARCH}?{urllib.parse.urlencode(params)}"
    try:
        data = get_json(url)
    except Exception:  # noqa: BLE001
        return ""
    if not isinstance(data, dict):
        return ""
    pages = data.get("query", {}).get("pages", {}) or {}
    for page in pages.values():
        qid = (page.get("pageprops") or {}).get("wikibase_item", "")
        if qid:
            return qid
    return ""


def fetch(query: str, limit: int, no_wikidata: bool, out_path: str) -> int:
    hits = _wp_search(query, limit)
    rows: list[dict[str, str]] = []
    for hit in hits[:limit]:
        title = hit.get("title", "")
        if not title:
            continue
        summary = _wp_summary(title)
        qid = _wd_qid_for_title(title) if not no_wikidata else ""
        facts: dict = {}
        if qid:
            facts = _wd_lookup_by_qid(qid)
        rows.append(
            {
                "source": "wikipedia+wikidata" if qid else "wikipedia",
                "label": title,
                "description": (summary.get("description") or hit.get("description") or "").strip(),
                "qid": qid,
                "wikipedia_title": title,
                "wikipedia_url": hit.get("url", ""),
                "wikidata_url": f"https://www.wikidata.org/wiki/{qid}" if qid else "",
                "instance_of": "; ".join(facts.get("instance_of", [])),
                "country": "; ".join(facts.get("country", [])),
                "occupation": "; ".join(facts.get("occupation", [])),
                "employer": "; ".join(facts.get("employer", [])),
                "date_of_birth": "; ".join(facts.get("date_of_birth", []))[:10] if facts.get("date_of_birth") else "",
                "place_of_birth": "; ".join(facts.get("place_of_birth", [])),
                "summary": (summary.get("extract") or "").replace("\n", " ")[:1000],
            }
        )

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=COLUMNS)
        w.writeheader()
        w.writerows(rows)
    if not rows:
        print(
            f"Wikipedia: 0 articles for query={query!r}. "
            "Private individuals not notable enough for a Wikipedia article "
            "won't appear here (the bar is real).",
            file=sys.stderr,
        )
    return len(rows)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--query", required=True, help="Entity name (person, company, place, concept)")
    p.add_argument("--limit", type=int, default=5)
    p.add_argument(
        "--no-wikidata",
        action="store_true",
        help="Skip the Wikidata SPARQL enrichment (faster, less detail)",
    )
    p.add_argument("--out", required=True)
    a = p.parse_args()
    n = fetch(query=a.query, limit=a.limit, no_wikidata=a.no_wikidata, out_path=a.out)
    print(f"Wrote {n} Wikipedia/Wikidata rows to {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
