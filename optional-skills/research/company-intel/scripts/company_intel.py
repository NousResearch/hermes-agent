#!/usr/bin/env python3
"""
Company Intelligence CLI — research companies via free public APIs.

Usage:
    python3 company_intel.py search "Tesla" --num 5
    python3 company_intel.py profile "Apple Inc."
    python3 company_intel.py wikidata "Apple Inc."
    python3 company_intel.py sec "Nvidia Corporation" --limit 5
    python3 company_intel.py news "OpenAI" --days 30
    python3 company_intel.py research "Microsoft Corporation"

No API keys required. Uses Wikipedia, Wikidata, SEC EDGAR, and HN Algolia APIs.
"""

import json
import os
import sys
import time
import urllib.parse
import urllib.request


USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
WIKIPEDIA_API = "https://en.wikipedia.org/api/rest_v1/page/summary"
WIKIDATA_SPARQL = "https://query.wikidata.org/sparql"
SEC_SEARCH = "https://efts.sec.gov/LATEST/search-index"
HN_ALGOLIA = "https://hn.algolia.com/api/v1/search"
WIKIPEDIA_SEARCH = "https://en.wikipedia.org/w/api.php"


def api_request(url: str, headers: dict = None) -> dict:
    """Make a GET request and parse JSON response."""
    if headers is None:
        headers = {"User-Agent": USER_AGENT}
    req = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(req, timeout=15) as resp:
        return json.loads(resp.read().decode())


def url_encode(params: dict) -> str:
    """Build URL-encoded query string."""
    return "&".join(
        f"{k}={urllib.parse.quote(str(v), safe='')}"
        for k, v in params.items()
        if v is not None
    )


# ─── WIKIPEDIA ───────────────────────────────────────────────────────────


def wikipedia_search(query: str, num: int = 5) -> dict:
    """Search Wikipedia for pages matching query."""
    params = {
        "action": "query",
        "list": "search",
        "srsearch": query,
        "format": "json",
        "srlimit": num,
    }
    url = f"{WIKIPEDIA_SEARCH}?{url_encode(params)}"
    data = api_request(url)
    results = []
    for r in data.get("query", {}).get("search", []):
        results.append({
            "title": r.get("title"),
            "page_id": r.get("pageid"),
            "snippet": r
            .get("snippet", "")
            .replace('<span class="searchmatch">', "")
            .replace("</span>", ""),
        })
    return {
        "query": query,
        "total": data.get("query", {}).get("searchinfo", {}).get("totalhits", 0),
        "results": results,
    }


def wikipedia_summary(title: str) -> dict:
    """Get Wikipedia summary for a page title."""
    safe_title = urllib.parse.quote(title.replace(" ", "_"))
    url = f"{WIKIPEDIA_API}/{safe_title}"
    try:
        data = api_request(url)
        return {
            "title": data.get("title"),
            "description": data.get("description"),
            "extract": data.get("extract", ""),
            "url": data.get("content_urls", {}).get("desktop", {}).get("page"),
            "thumbnail": data.get("thumbnail", {}).get("source"),
        }
    except Exception as e:
        return {"error": f"Wikipedia lookup failed: {str(e)}"}


# ─── WIKIDATA ────────────────────────────────────────────────────────────

PROPERTY_LABELS = {
    "P571": "founded",
    "P2139": "revenue",
    "P1128": "employees",
    "P414": "stock_exchange",
    "P159": "headquarters",
    "P856": "website",
    "P452": "industry",
    "P127": "owned_by",
    "P749": "parent_organization",
    "P1454": "legal_form",
    "P17": "country",
    "P276": "location",
}


def wikidata_resolve_uris(uris: list) -> dict:
    """Resolve Wikidata URIs to labels using the entity API."""
    if not uris:
        return {}
    ids = [
        u.split("/")[-1]
        for u in uris
        if u.startswith("http://www.wikidata.org/entity/")
    ]
    if not ids:
        return {}
    url = f"https://www.wikidata.org/w/api.php?action=wbgetentities&ids={'|'.join(ids)}&format=json&props=labels"
    try:
        data = api_request(url, {"User-Agent": USER_AGENT})
        resolved = {}
        for eid, entity in data.get("entities", {}).items():
            label = entity.get("labels", {}).get("en", {}).get("value", "")
            if label:
                resolved[eid] = label
        return resolved
    except Exception:
        return {}


def wikidata_query(company_name: str) -> dict:
    """Query Wikidata SPARQL for company structured data."""
    find_query = """
    SELECT ?item ?itemLabel WHERE {
      ?item wdt:P31/wdt:P279* wd:Q4830453 .
      ?item rdfs:label "%s"@en .
      SERVICE wikibase:label { bd:serviceParam wikibase:language "en". }
    }
    LIMIT 5
    """ % company_name.replace('"', '\\"')

    params = {"format": "json", "query": find_query}
    url = f"{WIKIDATA_SPARQL}?{url_encode(params)}"

    try:
        data = api_request(
            url, {"Accept": "application/json", "User-Agent": USER_AGENT}
        )
        bindings = data.get("results", {}).get("bindings", [])
        if not bindings:
            return {"error": f"No Wikidata entity found for '{company_name}'"}

        item_id = bindings[0].get("item", {}).get("value", "").split("/")[-1]
        if not item_id:
            return {"error": "Could not extract Wikidata ID"}

        # Query properties using VALUES (confirmed working)
        props = " ".join(f"wdt:{p}" for p in PROPERTY_LABELS)
        prop_query = """
        SELECT ?prop ?value WHERE {
          VALUES ?prop { %s }
          wd:%s ?prop ?value .
        }
        """ % (props, item_id)

        params2 = {"format": "json", "query": prop_query}
        url2 = f"{WIKIDATA_SPARQL}?{url_encode(params2)}"
        data2 = api_request(
            url2, {"Accept": "application/json", "User-Agent": USER_AGENT}
        )

        # Collect values and resolve URI labels
        uri_values = set()
        facts = {}
        for b in data2.get("results", {}).get("bindings", []):
            prop_uri = b.get("prop", {}).get("value", "")
            prop_id = prop_uri.split("/")[-1]
            label = PROPERTY_LABELS.get(prop_id, prop_id)
            val = b.get("value", {}).get("value", "")
            val_type = b.get("value", {}).get("type", "")

            if val_type == "uri" and "wikidata.org/entity/" in val:
                uri_values.add(val)

            if label not in facts:
                facts[label] = []
            facts[label].append({"value": val, "type": val_type})

        # Resolve URI values to human-readable labels
        if uri_values:
            resolved = wikidata_resolve_uris(list(uri_values))
            for label, items in facts.items():
                for item in items:
                    val = item["value"]
                    if val in resolved:
                        item["label"] = resolved[val]
                    eid = val.split("/")[-1]
                    if eid in resolved:
                        item["label"] = resolved[eid]

        return {
            "wikidata_id": item_id,
            "label": bindings[0].get("itemLabel", {}).get("value", ""),
            "facts": facts,
        }
    except Exception as e:
        return {"error": f"Wikidata query failed: {str(e)}"}


# ─── SEC EDGAR ───────────────────────────────────────────────────────────


def sec_search(query: str, limit: int = 10) -> dict:
    """Search SEC EDGAR for company filings and CIK.

    SEC requires a ``User-Agent`` header with contact info.
    Set ``SEC_USER_AGENT`` env var, e.g.::

        SEC_USER_AGENT="Research Name contact@example.com"
    """
    sec_ua = os.environ.get("SEC_USER_AGENT", "").strip()
    if not sec_ua:
        return {
            "error": "SEC EDGAR requires SEC_USER_AGENT env var with contact info. "
            "Set SEC_USER_AGENT='Your Name your@email'."
        }
    params = {"q": query, "itemsPerPage": limit}
    url = f"{SEC_SEARCH}?{url_encode(params)}"
    try:
        data = api_request(
            url,
            {
                "User-Agent": sec_ua,
                "Accept": "application/json",
            },
        )
        hits = data.get("hits", {}).get("hits", [])
        results = []
        for h in hits:
            source = h.get("_source", {})
            results.append({
                "cik": (source.get("ciks") or [None])[0],
                "company_name": (source.get("display_names") or [""])[0],
                "form_type": source.get("form"),
                "filing_date": source.get("file_date"),
                "description": (source.get("file_description") or "")[:200],
                "period_ending": source.get("period_ending"),
            })
        total = data.get("hits", {}).get("total", {}).get("value", 0)
        return {"query": query, "total_filings": total, "filings": results[:limit]}
    except Exception as e:
        return {"error": f"SEC EDGAR query failed: {str(e)}"}


# ─── HACKER NEWS ─────────────────────────────────────────────────────────


def hn_search(company: str, days: int = 7) -> dict:
    """Search Hacker News for company mentions."""
    # Calculate timestamp for date filter
    since_ts = int(time.time()) - (days * 86400)
    params = {
        "query": company,
        "tags": "story",
        "hitsPerPage": 10,
        "numericFilters": f"created_at_i>{since_ts}",
    }
    url = f"{HN_ALGOLIA}?{url_encode(params)}"
    try:
        data = api_request(url, {"User-Agent": USER_AGENT})
        hits = data.get("hits", [])
        results = []
        for h in hits:
            results.append({
                "title": h.get("title", ""),
                "url": h.get("url")
                or f"https://news.ycombinator.com/item?id={h.get('objectID')}",
                "points": h.get("points", 0),
                "author": h.get("author", ""),
                "created_at": h.get("created_at", ""),
                "num_comments": h.get("num_comments", 0),
            })
        return {
            "company": company,
            "days": days,
            "total_matches": data.get("nbHits", 0),
            "results": results,
        }
    except Exception as e:
        return {"error": f"HN Algolia query failed: {str(e)}"}


# ─── PROFILE (Wikipedia + Wikidata combo) ───────────────────────────────


def cmd_profile(args):
    """Company profile combining Wikipedia summary + Wikidata facts."""
    wiki = wikipedia_summary(args.company)
    wd = wikidata_query(args.company)

    output = {"company": args.company}
    if "error" not in wiki:
        output["wikipedia"] = wiki
    if "error" not in wd:
        output["wikidata"] = wd
    if "error" in wiki and "error" in wd:
        output["error"] = (
            f"Could not find company '{args.company}' in Wikipedia or Wikidata"
        )

    print(json.dumps(output, indent=2, ensure_ascii=False))


def cmd_search(args):
    """Search for companies by keyword."""
    result = wikipedia_search(args.query, args.num)
    print(json.dumps(result, indent=2, ensure_ascii=False))


def cmd_wikidata(args):
    """Query Wikidata for structured company data."""
    result = wikidata_query(args.company)
    print(json.dumps(result, indent=2, ensure_ascii=False))


def cmd_sec(args):
    """Search SEC EDGAR for company filings."""
    result = sec_search(args.company, args.limit)
    print(json.dumps(result, indent=2, ensure_ascii=False))


def cmd_news(args):
    """Search Hacker News for recent mentions."""
    result = hn_search(args.company, args.days)
    print(json.dumps(result, indent=2, ensure_ascii=False))


def cmd_research(args):
    """Full research report combining all sources."""
    company = args.company

    output = {
        "company": company,
        "sources": {},
    }

    # Wikipedia
    wiki = wikipedia_summary(company)
    if "error" not in wiki:
        output["sources"]["wikipedia"] = wiki

    # Wikidata
    wd = wikidata_query(company)
    if "error" not in wd:
        output["sources"]["wikidata"] = wd

    # SEC EDGAR
    sec = sec_search(company, 5)
    if "error" not in sec:
        output["sources"]["sec_edgar"] = sec

    # Hacker News
    hn = hn_search(company, 30)
    if "error" not in hn:
        output["sources"]["hacker_news"] = hn

    if not output["sources"]:
        output["error"] = f"No data found for '{company}' from any source"

    print(json.dumps(output, indent=2, ensure_ascii=False))


def print_usage():
    print(__doc__)


def main():
    if len(sys.argv) < 2:
        print_usage()
        sys.exit(1)

    command = sys.argv[1]

    if command == "profile":
        import argparse

        p = argparse.ArgumentParser(
            description="Get company profile (Wikipedia + Wikidata)"
        )
        p.add_argument("company", help="Company name (e.g. 'Apple Inc.')")
        args = p.parse_args(sys.argv[2:])
        cmd_profile(args)

    elif command == "search":
        import argparse

        p = argparse.ArgumentParser(description="Search for companies by keyword")
        p.add_argument("query", help="Search keyword")
        p.add_argument("--num", type=int, default=5, help="Number of results")
        args = p.parse_args(sys.argv[2:])
        cmd_search(args)

    elif command == "wikidata":
        import argparse

        p = argparse.ArgumentParser(description="Query Wikidata for company data")
        p.add_argument("company", help="Company name")
        args = p.parse_args(sys.argv[2:])
        cmd_wikidata(args)

    elif command == "sec":
        import argparse

        p = argparse.ArgumentParser(description="Search SEC EDGAR filings")
        p.add_argument("company", help="Company name")
        p.add_argument("--limit", type=int, default=10, help="Max filings to return")
        args = p.parse_args(sys.argv[2:])
        cmd_sec(args)

    elif command == "news":
        import argparse

        p = argparse.ArgumentParser(description="Search Hacker News mentions")
        p.add_argument("company", help="Company name")
        p.add_argument(
            "--days", type=int, default=7, help="How many days back to search"
        )
        args = p.parse_args(sys.argv[2:])
        cmd_news(args)

    elif command == "research":
        import argparse

        p = argparse.ArgumentParser(description="Full company research report")
        p.add_argument("company", help="Company name")
        args = p.parse_args(sys.argv[2:])
        cmd_research(args)

    elif command in ("--help", "-h"):
        print_usage()
    else:
        print(f"Unknown command: {command}", file=sys.stderr)
        print_usage()
        sys.exit(1)


if __name__ == "__main__":
    main()
