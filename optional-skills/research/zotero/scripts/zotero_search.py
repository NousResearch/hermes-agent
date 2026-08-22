#!/usr/bin/env python3
"""
zotero_search.py — Search and browse your Zotero library.

Usage:
  python zotero_search.py "query"
  python zotero_search.py "deep learning" --tag unread --type journalArticle
  python zotero_search.py --collection COLL_KEY
  python zotero_search.py --collections
  python zotero_search.py --item ITEM_KEY
  python zotero_search.py --stats
  python zotero_search.py --dupes
  python zotero_search.py --tag unread
  python zotero_search.py --since 2025-01-01
  python zotero_search.py --export bibtex COLL_KEY
  python zotero_search.py --export ris COLL_KEY
  python zotero_search.py --collection COLL_KEY --add-tag "sprint-2026"
  python zotero_search.py --remove-tag "old-tag"
"""

import argparse
import json
import os
import re
import sys
from collections import Counter, defaultdict
from datetime import datetime


def extract_year(date_str: str) -> str:
    """Extract a 4-digit year from strings like 'December 27, 2017' or '2017-12-27'."""
    match = re.search(r"\b(1[5-9]\d{2}|20\d{2})\b", date_str or "")
    return match.group(1) if match else (date_str[:4] if len(date_str) >= 4 else date_str)

try:
    import requests
except ImportError:
    sys.exit("requests is required: pip install requests")

API_BASE = "https://api.zotero.org"
HEADERS: dict = {}
USER_ID: str = ""


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------

def get_env() -> tuple[str, str]:
    api_key = os.environ.get("ZOTERO_API_KEY", "")
    user_id = os.environ.get("ZOTERO_USER_ID", "")
    if not api_key or not user_id:
        sys.exit(
            "Set ZOTERO_API_KEY and ZOTERO_USER_ID environment variables.\n"
            "  Get them at: https://www.zotero.org/settings/keys"
        )
    return api_key, user_id


def build_headers(api_key: str) -> dict:
    return {
        "Zotero-API-Version": "3",
        "Zotero-API-Key": api_key,
        "Content-Type": "application/json",
    }


# ---------------------------------------------------------------------------
# API helpers
# ---------------------------------------------------------------------------

def paginate(url: str, params: dict | None = None) -> list[dict]:
    """Fetch all pages of a paginated endpoint."""
    results = []
    params = dict(params or {})
    params.setdefault("limit", 100)
    start = 0
    while True:
        params["start"] = start
        resp = requests.get(url, headers=HEADERS, params=params)
        resp.raise_for_status()
        batch = resp.json()
        if not batch:
            break
        results.extend(batch)
        total = int(resp.headers.get("Total-Results", len(batch)))
        start += len(batch)
        if start >= total:
            break
    return results


def get_item(item_key: str) -> dict:
    resp = requests.get(f"{API_BASE}/users/{USER_ID}/items/{item_key}", headers=HEADERS)
    resp.raise_for_status()
    return resp.json()


def get_collections() -> list[dict]:
    return paginate(f"{API_BASE}/users/{USER_ID}/collections")


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------

def fmt_creators(creators: list[dict]) -> str:
    names = []
    for c in creators[:3]:
        last = c.get("lastName") or c.get("name", "")
        first = c.get("firstName", "")
        names.append(f"{last}, {first}".strip(", ") if first else last)
    result = "; ".join(names)
    if len(creators) > 3:
        result += f" et al. (+{len(creators) - 3})"
    return result


def fmt_item_line(item: dict, idx: int | None = None) -> str:
    data = item.get("data", {})
    key = data.get("key", "")
    title = data.get("title", "(no title)")[:80]
    creators = fmt_creators(data.get("creators", []))
    year = extract_year(data.get("date", "")) if data.get("date") else ""
    itype = data.get("itemType", "")
    tags = [t["tag"] for t in data.get("tags", []) if not t["tag"].startswith("_")]
    tag_str = f"  [{', '.join(tags[:3])}]" if tags else ""

    prefix = f"{idx:3}. " if idx is not None else "  "
    line = f"{prefix}[{key}] {title}"
    meta = f"      {creators} ({year}) · {itype}{tag_str}" if creators or year else f"      {itype}{tag_str}"
    return f"{line}\n{meta}"


def print_items(items: list[dict], limit: int = 50) -> None:
    shown = items[:limit]
    for i, item in enumerate(shown, 1):
        print(fmt_item_line(item, i))
        print()
    if len(items) > limit:
        print(f"  ... and {len(items) - limit} more (use --limit N to show more)")


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------

def cmd_search(query: str, args: argparse.Namespace) -> None:
    params: dict = {"q": query, "qmode": "everything"}
    if args.tag:
        params["tag"] = args.tag
    if args.type:
        params["itemType"] = args.type
    if args.collection:
        url = f"{API_BASE}/users/{USER_ID}/collections/{args.collection}/items/top"
    else:
        url = f"{API_BASE}/users/{USER_ID}/items/top"

    items = paginate(url, params)
    # Apply --since filter locally (API's `since` is by version, not date)
    if args.since:
        cutoff = args.since
        items = [
            it for it in items
            if it["data"].get("dateAdded", "") >= cutoff
        ]

    print(f"\n{len(items)} result(s) for: '{query}'\n")
    print_items(items, limit=args.limit)


def cmd_list_collection(collection_key: str, args: argparse.Namespace) -> None:
    params: dict = {}
    if args.tag:
        params["tag"] = args.tag
    if args.type:
        params["itemType"] = args.type
    url = f"{API_BASE}/users/{USER_ID}/collections/{collection_key}/items/top"
    items = paginate(url, params)
    if args.since:
        items = [it for it in items if it["data"].get("dateAdded", "") >= args.since]

    # Get collection name
    resp = requests.get(f"{API_BASE}/users/{USER_ID}/collections/{collection_key}", headers=HEADERS)
    col_name = resp.json().get("data", {}).get("name", collection_key) if resp.ok else collection_key

    print(f"\n{len(items)} items in '{col_name}':\n")
    print_items(items, limit=args.limit)


def cmd_collections() -> None:
    cols = get_collections()
    children: dict[str | bool, list] = defaultdict(list)
    for c in cols:
        parent = c["data"].get("parentCollection", False)
        children[parent].append(c)

    def print_level(parent_key: str | bool, indent: int = 0) -> None:
        for col in sorted(children[parent_key], key=lambda c: c["data"]["name"]):
            key = col["data"]["key"]
            name = col["data"]["name"]
            prefix = "  " * indent + ("└── " if indent > 0 else "")
            print(f"{prefix}{name}  [{key}]")
            print_level(key, indent + 1)

    print(f"\nYour Zotero collections ({len(cols)} total):\n")
    print_level(False)
    print()


def cmd_item(item_key: str) -> None:
    item = get_item(item_key)
    data = item.get("data", {})
    print(f"\n{'─' * 60}")
    print(f"Key:     {data.get('key')}")
    print(f"Type:    {data.get('itemType')}")
    print(f"Title:   {data.get('title', '')}")
    print(f"Authors: {fmt_creators(data.get('creators', []))}")
    print(f"Date:    {data.get('date', '')}")

    for field in ("publicationTitle", "publisher", "DOI", "ISBN", "url", "pages", "volume", "issue"):
        val = data.get(field, "")
        if val:
            print(f"{field.capitalize():8s} {val}")

    tags = [t["tag"] for t in data.get("tags", [])]
    if tags:
        print(f"Tags:    {', '.join(tags)}")

    collections = data.get("collections", [])
    if collections:
        print(f"Collections: {', '.join(collections)}")

    abstract = data.get("abstractNote", "")
    if abstract:
        print(f"\nAbstract:\n  {abstract[:500]}{'...' if len(abstract) > 500 else ''}")

    print(f"{'─' * 60}\n")

    # Show children summary
    children_resp = requests.get(
        f"{API_BASE}/users/{USER_ID}/items/{item_key}/children",
        headers=HEADERS
    )
    if children_resp.ok:
        children = children_resp.json()
        notes = [c for c in children if c["data"].get("itemType") == "note"]
        attachments = [c for c in children if c["data"].get("itemType") == "attachment"]
        if notes:
            print(f"Notes ({len(notes)}):")
            for n in notes:
                print(f"  [{n['data']['key']}] {n['data'].get('note', '')[:80]}")
            print()
        if attachments:
            print(f"Attachments ({len(attachments)}):")
            for a in attachments:
                print(f"  [{a['data']['key']}] {a['data'].get('filename', a['data'].get('title', ''))}")
            print()


def cmd_stats(args: argparse.Namespace) -> None:
    url = f"{API_BASE}/users/{USER_ID}/items/top"
    if args.collection:
        url = f"{API_BASE}/users/{USER_ID}/collections/{args.collection}/items/top"

    print("Fetching library stats (this may take a moment)...")
    items = paginate(url)

    type_counts: Counter = Counter()
    tag_counts: Counter = Counter()
    year_counts: Counter = Counter()

    for item in items:
        data = item.get("data", {})
        type_counts[data.get("itemType", "unknown")] += 1
        for t in data.get("tags", []):
            tag_counts[t["tag"]] += 1
        year = extract_year(data.get("date", ""))
        if year and year.isdigit():
            year_counts[year] += 1

    print(f"\n{'─' * 50}")
    print(f"Total items: {len(items)}")
    print(f"\nBy type:")
    for itype, count in type_counts.most_common():
        print(f"  {itype:30s} {count:4d}")
    print(f"\nTop tags:")
    for tag, count in tag_counts.most_common(15):
        print(f"  {tag:30s} {count:4d}")
    if year_counts:
        print(f"\nBy year (top 10):")
        for year, count in sorted(year_counts.items(), reverse=True)[:10]:
            print(f"  {year}  {count:4d}")
    print(f"{'─' * 50}\n")


def cmd_dupes() -> None:
    print("Scanning for duplicates (fetching all items)...")
    items = paginate(f"{API_BASE}/users/{USER_ID}/items/top")

    doi_map: dict[str, list] = defaultdict(list)
    isbn_map: dict[str, list] = defaultdict(list)
    title_map: dict[str, list] = defaultdict(list)

    for item in items:
        data = item.get("data", {})
        doi = data.get("DOI", "").strip().lower()
        isbn = re.sub(r"[-\s]", "", data.get("ISBN", "").strip())
        title = data.get("title", "").strip().lower()[:80]
        key = data.get("key", "")

        if doi:
            doi_map[doi].append(key)
        if isbn:
            isbn_map[isbn].append(key)
        if title:
            title_map[title].append(key)

    found = False
    for doi, keys in doi_map.items():
        if len(keys) > 1:
            print(f"\nDuplicate DOI: {doi}")
            for k in keys:
                print(f"  {k}")
            found = True

    for isbn, keys in isbn_map.items():
        if len(keys) > 1:
            print(f"\nDuplicate ISBN: {isbn}")
            for k in keys:
                print(f"  {k}")
            found = True

    for title, keys in title_map.items():
        if len(keys) > 1:
            print(f"\nDuplicate title: {title[:60]}")
            for k in keys:
                print(f"  {k}")
            found = True

    if not found:
        print("No duplicates found.")
    else:
        print(f"\nTip: delete duplicates with: python zotero_add.py (or use Zotero desktop Duplicate Items pane)")


def cmd_add_tag(collection_key: str, tag: str) -> None:
    items = paginate(f"{API_BASE}/users/{USER_ID}/collections/{collection_key}/items/top")
    print(f"Adding tag '{tag}' to {len(items)} items...")
    for item in items:
        data = item["data"]
        version = data["version"]
        tags = data.get("tags", [])
        if not any(t["tag"] == tag for t in tags):
            tags.append({"tag": tag})
            patch_headers = {**HEADERS, "If-Unmodified-Since-Version": str(version)}
            requests.patch(
                f"{API_BASE}/users/{USER_ID}/items/{data['key']}",
                headers=patch_headers,
                json={"tags": tags, "version": version},
            )
    print(f"Done.")


def cmd_remove_tag(tag: str) -> None:
    print(
        f"Tag deletion is disabled in this skill to prevent accidental data loss.\n"
        f"To remove tag '{tag}' from all items, open the Zotero desktop app → Tags panel → right-click and delete."
    )


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------

def item_to_bibtex(item: dict) -> str:
    data = item.get("data", {})
    key = data.get("key", "UNKNOWN")
    itype = data.get("itemType", "misc")
    type_map = {
        "journalArticle": "article",
        "book": "book",
        "conferencePaper": "inproceedings",
        "thesis": "phdthesis",
        "report": "techreport",
        "bookSection": "incollection",
        "preprint": "misc",
        "webpage": "misc",
    }
    bib_type = type_map.get(itype, "misc")

    creators = data.get("creators", [])
    author = " and ".join(
        f"{c.get('lastName', '')}, {c.get('firstName', '')}".strip(", ")
        for c in creators
    )
    title = data.get("title", "")
    year = extract_year(data.get("date", ""))

    fields = [
        ("author", author),
        ("title", f"{{{title}}}"),
        ("year", year),
    ]
    for field in ("publicationTitle", "journal", "volume", "number", "pages", "publisher", "DOI", "url"):
        val = data.get(field, "")
        if val:
            fields.append((field.lower(), val))

    body = ",\n  ".join(f"{k} = {{{v}}}" for k, v in fields if v)
    return f"@{bib_type}{{{key},\n  {body}\n}}\n"


def item_to_ris(item: dict) -> str:
    data = item.get("data", {})
    type_map = {
        "journalArticle": "JOUR",
        "book": "BOOK",
        "conferencePaper": "CONF",
        "thesis": "THES",
        "report": "RPRT",
        "preprint": "JOUR",
        "webpage": "ELEC",
    }
    ris_type = type_map.get(data.get("itemType", ""), "GEN")
    lines = [f"TY  - {ris_type}"]
    lines.append(f"TI  - {data.get('title', '')}")
    for c in data.get("creators", []):
        lines.append(f"AU  - {c.get('lastName', '')}, {c.get('firstName', '')}")
    if data.get("date"):
        lines.append(f"PY  - {extract_year(data['date'])}")
    if data.get("publicationTitle"):
        lines.append(f"JO  - {data['publicationTitle']}")
    if data.get("volume"):
        lines.append(f"VL  - {data['volume']}")
    if data.get("pages"):
        lines.append(f"SP  - {data['pages']}")
    if data.get("DOI"):
        lines.append(f"DO  - {data['DOI']}")
    if data.get("url"):
        lines.append(f"UR  - {data['url']}")
    if data.get("abstractNote"):
        lines.append(f"AB  - {data['abstractNote'][:500]}")
    lines.append("ER  - ")
    return "\n".join(lines) + "\n\n"


def cmd_export(fmt: str, collection_key: str) -> None:
    items = paginate(f"{API_BASE}/users/{USER_ID}/collections/{collection_key}/items/top")
    if fmt == "bibtex":
        for item in items:
            print(item_to_bibtex(item))
    elif fmt == "ris":
        for item in items:
            print(item_to_ris(item))
    else:
        sys.exit(f"Unknown export format: {fmt}. Use 'bibtex' or 'ris'.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Search and browse Zotero library")
    parser.add_argument("query", nargs="?", help="Keyword search query")
    parser.add_argument("--collection", metavar="KEY", help="Filter/list by collection key")
    parser.add_argument("--collections", action="store_true", help="List all collections")
    parser.add_argument("--item", metavar="KEY", help="Show full detail for one item")
    parser.add_argument("--tag", metavar="TAG", help="Filter by tag")
    parser.add_argument("--type", metavar="ITEMTYPE", help="Filter by item type")
    parser.add_argument("--since", metavar="DATE", help="Items added on or after date (YYYY-MM-DD)")
    parser.add_argument("--stats", action="store_true", help="Show library statistics")
    parser.add_argument("--dupes", action="store_true", help="Find duplicate items")
    parser.add_argument("--add-tag", metavar="TAG", help="Add tag to all items in --collection")
    parser.add_argument("--remove-tag", metavar="TAG", help="Remove tag from all items in library")
    parser.add_argument("--export", nargs=2, metavar=("FORMAT", "COLL_KEY"),
                        help="Export collection: bibtex|ris COLLECTION_KEY")
    parser.add_argument("--limit", type=int, default=50, help="Max items to display (default: 50)")
    args = parser.parse_args()

    api_key, user_id = get_env()
    global HEADERS, USER_ID
    HEADERS = build_headers(api_key)
    USER_ID = user_id

    if args.export:
        cmd_export(args.export[0], args.export[1])
    elif args.collections:
        cmd_collections()
    elif args.item:
        cmd_item(args.item)
    elif args.stats:
        cmd_stats(args)
    elif args.dupes:
        cmd_dupes()
    elif args.add_tag:
        if not args.collection:
            sys.exit("--add-tag requires --collection KEY")
        cmd_add_tag(args.collection, args.add_tag)
    elif args.remove_tag:
        cmd_remove_tag(args.remove_tag)
    elif args.query:
        if args.collection and not args.query:
            cmd_list_collection(args.collection, args)
        else:
            cmd_search(args.query, args)
    elif args.collection:
        cmd_list_collection(args.collection, args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
