#!/usr/bin/env python3
"""
zotero_add.py — Add items to your Zotero library.

Fetches metadata from external sources, creates a Zotero item,
and places it in the hermes-agent collection (or a specified one).

PDF reading is handled by zotero_read.py, which fetches PDFs directly
from the source URL (arXiv, open-access DOIs) — no upload needed.

Usage:
  python zotero_add.py --doi 10.1145/3290605.3300786
  python zotero_add.py --isbn 978-0-13-468599-1
  python zotero_add.py --arxiv 2301.07041
  python zotero_add.py --url https://example.com/article
  python zotero_add.py --bibtex refs.bib
  python zotero_add.py --doi 10.1234/x --collection ABCD1234
  python zotero_add.py --move ITEM_KEY --collection COLL_KEY
"""

import argparse
import json
import os
import re
import sys
import xml.etree.ElementTree as ET

try:
    import requests
except ImportError:
    sys.exit("requests is required: pip install requests")

API_BASE = "https://api.zotero.org"
CROSSREF_BASE = "https://api.crossref.org/works"
OPENLIBRARY_BASE = "https://openlibrary.org/api/books"
GBOOKS_BASE = "https://www.googleapis.com/books/v1/volumes"
ARXIV_BASE = "https://export.arxiv.org/api/query"
ATOM_NS = "http://www.w3.org/2005/Atom"
ARXIV_NS = "http://arxiv.org/schemas/atom"

HEADERS: dict = {}
USER_ID: str = ""


# ---------------------------------------------------------------------------
# Environment helpers
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
# Metadata fetchers
# ---------------------------------------------------------------------------

def fetch_doi(doi: str) -> dict:
    """Fetch metadata from CrossRef for a DOI."""
    doi_clean = doi.strip().lstrip("https://doi.org/").lstrip("http://dx.doi.org/")
    print(f"  Fetching DOI metadata from CrossRef: {doi_clean}")
    resp = requests.get(f"{CROSSREF_BASE}/{doi_clean}",
                        headers={"User-Agent": "hermes-agent/1.0 (mailto:user@example.com)"})
    if resp.status_code == 404:
        sys.exit(f"DOI not found in CrossRef: {doi_clean}")
    resp.raise_for_status()
    work = resp.json().get("message", {})

    authors = []
    for author in work.get("author", []):
        authors.append({
            "creatorType": "author",
            "firstName": author.get("given", ""),
            "lastName": author.get("family", ""),
        })

    # Map CrossRef type to Zotero itemType
    cr_type = work.get("type", "journal-article")
    type_map = {
        "journal-article": "journalArticle",
        "proceedings-article": "conferencePaper",
        "book": "book",
        "book-chapter": "bookSection",
        "monograph": "book",
        "report": "report",
        "dissertation": "thesis",
        "preprint": "preprint",
        "posted-content": "preprint",
    }
    item_type = type_map.get(cr_type, "journalArticle")

    date_parts = work.get("published", work.get("published-print", work.get("issued", {}))).get("date-parts", [[]])
    year = str(date_parts[0][0]) if date_parts and date_parts[0] else ""

    titles = work.get("title", [""])
    title = titles[0] if titles else ""

    item: dict = {
        "itemType": item_type,
        "title": title,
        "creators": authors,
        "DOI": work.get("DOI", doi_clean),
        "url": work.get("URL", f"https://doi.org/{doi_clean}"),
        "date": year,
        "abstractNote": work.get("abstract", ""),
    }

    if item_type == "journalArticle":
        containers = work.get("container-title", [])
        item["publicationTitle"] = containers[0] if containers else ""
        item["volume"] = work.get("volume", "")
        item["issue"] = work.get("issue", "")
        pages = work.get("page", "")
        item["pages"] = pages

    if item_type == "conferencePaper":
        containers = work.get("container-title", [])
        item["proceedingsTitle"] = containers[0] if containers else ""

    return item


def fetch_isbn(isbn: str) -> dict:
    """Fetch book metadata from Open Library (fallback: Google Books)."""
    isbn_clean = re.sub(r"[-\s]", "", isbn)
    print(f"  Fetching ISBN metadata from Open Library: {isbn_clean}")

    resp = requests.get(
        OPENLIBRARY_BASE,
        params={"bibkeys": f"ISBN:{isbn_clean}", "format": "json", "jscmd": "data"},
    )
    resp.raise_for_status()
    data = resp.json()
    book = data.get(f"ISBN:{isbn_clean}")

    if not book:
        print("  Not found in Open Library, trying Google Books...")
        return fetch_isbn_google(isbn_clean)

    authors = [
        {"creatorType": "author", "firstName": "", "lastName": a["name"]}
        for a in book.get("authors", [])
    ]
    # Split single "Firstname Lastname" strings
    split_authors = []
    for a in authors:
        parts = a["lastName"].rsplit(" ", 1)
        if len(parts) == 2:
            split_authors.append({"creatorType": "author", "firstName": parts[0], "lastName": parts[1]})
        else:
            split_authors.append(a)

    publishers = book.get("publishers", [{}])
    publisher = publishers[0].get("name", "") if publishers else ""

    return {
        "itemType": "book",
        "title": book.get("title", ""),
        "creators": split_authors,
        "publisher": publisher,
        "date": book.get("publish_date", ""),
        "numPages": str(book.get("number_of_pages", "")),
        "ISBN": isbn_clean,
        "url": book.get("url", ""),
        "abstractNote": book.get("description", {}).get("value", "") if isinstance(book.get("description"), dict) else book.get("description", ""),
    }


def fetch_isbn_google(isbn: str) -> dict:
    """Fallback: Google Books API for ISBN."""
    resp = requests.get(GBOOKS_BASE, params={"q": f"isbn:{isbn}"})
    if resp.status_code == 429:
        sys.exit(f"Google Books rate limit hit. Please wait a minute and retry.")
    resp.raise_for_status()
    items = resp.json().get("items", [])
    if not items:
        sys.exit(f"ISBN not found in Open Library or Google Books: {isbn}")

    info = items[0].get("volumeInfo", {})
    raw_authors = info.get("authors", [])
    authors = []
    for name in raw_authors:
        parts = name.rsplit(" ", 1)
        if len(parts) == 2:
            authors.append({"creatorType": "author", "firstName": parts[0], "lastName": parts[1]})
        else:
            authors.append({"creatorType": "author", "firstName": "", "lastName": name})

    return {
        "itemType": "book",
        "title": info.get("title", ""),
        "creators": authors,
        "publisher": info.get("publisher", ""),
        "date": info.get("publishedDate", "")[:4] if info.get("publishedDate") else "",
        "numPages": str(info.get("pageCount", "")),
        "ISBN": isbn,
        "abstractNote": info.get("description", ""),
    }


def fetch_arxiv(arxiv_id: str) -> dict:
    """Fetch preprint metadata from arXiv Atom API."""
    arxiv_id_clean = arxiv_id.strip().removeprefix("https://arxiv.org/abs/").removeprefix("arXiv:")
    print(f"  Fetching arXiv metadata: {arxiv_id_clean}")
    resp = requests.get(ARXIV_BASE, params={"id_list": arxiv_id_clean})
    resp.raise_for_status()

    root = ET.fromstring(resp.text)
    ns = {"a": ATOM_NS, "arxiv": ARXIV_NS}
    entry = root.find("a:entry", ns)
    if entry is None:
        sys.exit(f"arXiv ID not found: {arxiv_id_clean}")

    title = (entry.find("a:title", ns).text or "").strip().replace("\n", " ")
    summary = (entry.find("a:summary", ns).text or "").strip()
    published = (entry.find("a:published", ns).text or "")[:4]
    authors = [
        {
            "creatorType": "author",
            "firstName": " ".join(a.find("a:name", ns).text.split()[:-1]),
            "lastName": a.find("a:name", ns).text.split()[-1],
        }
        for a in entry.findall("a:author", ns)
        if a.find("a:name", ns) is not None
    ]

    cat_el = entry.find("arxiv:primary_category", ns)
    category = cat_el.get("term", "") if cat_el is not None else ""

    return {
        "itemType": "preprint",
        "title": title,
        "creators": authors,
        "abstractNote": summary,
        "date": published,
        "url": f"https://arxiv.org/abs/{arxiv_id_clean}",
        "archiveID": f"arXiv:{arxiv_id_clean}",
        "repository": "arXiv",
        "extra": f"arXiv: {arxiv_id_clean}\nPrimary category: {category}",
    }


def build_url_item(url: str) -> dict:
    """Build a minimal webpage item from a URL."""
    from datetime import date
    return {
        "itemType": "webpage",
        "title": url,
        "url": url,
        "accessDate": date.today().isoformat(),
    }


# ---------------------------------------------------------------------------
# Zotero write helpers
# ---------------------------------------------------------------------------

def get_hermes_collection_key() -> str | None:
    """Try to find the hermes-agent collection key."""
    resp = requests.get(
        f"{API_BASE}/users/{USER_ID}/collections/top",
        headers=HEADERS,
        params={"limit": 100},
    )
    resp.raise_for_status()
    for col in resp.json():
        if col.get("data", {}).get("name", "").lower() == "hermes-agent":
            return col["data"]["key"]
    return None


def post_items(items: list[dict]) -> list[str]:
    """Post items to Zotero (batch ≤50). Returns list of created keys."""
    created_keys = []
    for i in range(0, len(items), 50):
        batch = items[i:i + 50]
        resp = requests.post(
            f"{API_BASE}/users/{USER_ID}/items",
            headers=HEADERS,
            json=batch,
        )
        resp.raise_for_status()
        result = resp.json()
        for idx, item in result.get("successful", {}).items():
            key = item.get("data", {}).get("key") or item.get("key", "")
            created_keys.append(key)
        if result.get("failed"):
            for idx, err in result["failed"].items():
                print(f"  Warning: item {idx} failed: {err}", file=sys.stderr)
    return created_keys


def add_to_collection(item_key: str, collection_key: str) -> None:
    """Add an item to a collection."""
    # Fetch current item to get its collections
    resp = requests.get(f"{API_BASE}/users/{USER_ID}/items/{item_key}", headers=HEADERS)
    resp.raise_for_status()
    item = resp.json()
    version = item["data"]["version"]
    collections = item["data"].get("collections", [])
    if collection_key not in collections:
        collections.append(collection_key)
    patch_headers = {**HEADERS, "If-Unmodified-Since-Version": str(version)}
    resp2 = requests.patch(
        f"{API_BASE}/users/{USER_ID}/items/{item_key}",
        headers=patch_headers,
        json={"collections": collections, "version": version},
    )
    resp2.raise_for_status()


# ---------------------------------------------------------------------------
# BibTeX parser
# ---------------------------------------------------------------------------

def parse_bibtex(path: str) -> list[dict]:
    """Minimal BibTeX parser → list of Zotero item dicts."""
    with open(path, encoding="utf-8") as f:
        content = f.read()

    items = []
    entries = re.findall(r"@(\w+)\s*\{([^,]+),\s*(.*?)\n\}", content, re.DOTALL)

    type_map = {
        "article": "journalArticle",
        "book": "book",
        "inproceedings": "conferencePaper",
        "proceedings": "conferencePaper",
        "phdthesis": "thesis",
        "mastersthesis": "thesis",
        "misc": "webpage",
        "techreport": "report",
        "incollection": "bookSection",
        "unpublished": "manuscript",
    }

    for bib_type, citekey, body in entries:
        item_type = type_map.get(bib_type.lower(), "journalArticle")
        fields: dict = {}
        for match in re.finditer(r'(\w+)\s*=\s*[{"](.*?)[}"],?', body, re.DOTALL):
            fields[match.group(1).lower()] = match.group(2).strip()

        title = fields.get("title", "").replace("{", "").replace("}", "")
        year = fields.get("year", "")
        author_str = fields.get("author", "")
        authors = []
        for name in author_str.split(" and "):
            name = name.strip()
            if not name:
                continue
            if "," in name:
                last, first = name.split(",", 1)
                authors.append({"creatorType": "author", "firstName": first.strip(), "lastName": last.strip()})
            else:
                parts = name.rsplit(" ", 1)
                if len(parts) == 2:
                    authors.append({"creatorType": "author", "firstName": parts[0], "lastName": parts[1]})
                else:
                    authors.append({"creatorType": "author", "firstName": "", "lastName": name})

        item: dict = {
            "itemType": item_type,
            "title": title,
            "creators": authors,
            "date": year,
            "abstractNote": fields.get("abstract", ""),
            "extra": f"BibTeX key: {citekey}",
        }
        if fields.get("doi"):
            item["DOI"] = fields["doi"]
        if fields.get("url"):
            item["url"] = fields["url"]
        if fields.get("journal"):
            item["publicationTitle"] = fields["journal"]
        if fields.get("volume"):
            item["volume"] = fields["volume"]
        if fields.get("pages"):
            item["pages"] = fields["pages"]
        if fields.get("publisher"):
            item["publisher"] = fields["publisher"]
        if fields.get("isbn"):
            item["ISBN"] = fields["isbn"]
        if fields.get("booktitle"):
            item["proceedingsTitle"] = fields["booktitle"]

        items.append(item)

    return items


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Add items to Zotero library")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--doi", metavar="DOI")
    source.add_argument("--isbn", metavar="ISBN")
    source.add_argument("--arxiv", metavar="ARXIV_ID")
    source.add_argument("--url", metavar="URL")
    source.add_argument("--bibtex", metavar="FILE")
    source.add_argument("--move", metavar="ITEM_KEY", help="Move/copy existing item to a collection")
    parser.add_argument("--collection", metavar="COLLECTION_KEY",
                        help="Target collection key (default: hermes-agent root)")
    parser.add_argument("--tag", action="append", default=[], metavar="TAG",
                        help="Add tag(s) to the item (repeatable)")
    args = parser.parse_args()

    api_key, user_id = get_env()
    global HEADERS, USER_ID
    HEADERS = build_headers(api_key)
    USER_ID = user_id

    # Resolve target collection
    collection_key = args.collection
    if not collection_key and args.move is None:
        collection_key = get_hermes_collection_key()
        if collection_key:
            print(f"  Using hermes-agent collection: {collection_key}")
        else:
            print("  Warning: hermes-agent collection not found. Run zotero_setup.py first.")
            print("  Item will be added to library root.")

    # --move: just add to collection
    if args.move:
        if not args.collection:
            sys.exit("--move requires --collection COLLECTION_KEY")
        add_to_collection(args.move, args.collection)
        print(f"Added {args.move} to collection {args.collection}")
        return

    # Build item metadata
    if args.doi:
        item = fetch_doi(args.doi)
    elif args.isbn:
        item = fetch_isbn(args.isbn)
    elif args.arxiv:
        item = fetch_arxiv(args.arxiv)
    elif args.url:
        item = build_url_item(args.url)
    elif args.bibtex:
        items = parse_bibtex(args.bibtex)
        print(f"  Parsed {len(items)} entries from {args.bibtex}")
        for it in items:
            if collection_key:
                it["collections"] = [collection_key]
            if args.tag:
                it["tags"] = [{"tag": t} for t in args.tag]
            it.setdefault("tags", [{"tag": "unread"}])
        print(f"  Posting {len(items)} items to Zotero...")
        keys = post_items(items)
        print(f"\n✓ Created {len(keys)} items")
        for k in keys:
            print(f"  {k}")
        return

    # Apply collection and default tags
    if collection_key:
        item["collections"] = [collection_key]
    tags = [{"tag": t} for t in args.tag] if args.tag else [{"tag": "unread"}]
    item["tags"] = tags

    print(f"\n  Title: {item.get('title', '(unknown)')}")
    print(f"  Type:  {item['itemType']}")
    if item.get("creators"):
        first_author = item["creators"][0]
        print(f"  Author: {first_author.get('firstName', '')} {first_author.get('lastName', '')}")
    print(f"  Date:  {item.get('date', '')}")

    print("\n  Posting to Zotero...")
    keys = post_items([item])
    if keys:
        print(f"\n✓ Created item: {keys[0]}")
        print(f"  View: https://www.zotero.org/users/{USER_ID}/items/{keys[0]}")
    else:
        print("  No items created.")


if __name__ == "__main__":
    main()
