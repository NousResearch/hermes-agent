#!/usr/bin/env python3
"""
zotero_read.py — Read PDF content and metadata from Zotero items.

Tries methods in order (fastest first):
  1. Zotero fulltext index (GET /items/{key}/fulltext) — instant if indexed
  2. Download PDF via API + extract with pdfplumber
  3. Print metadata + abstract if no PDF found

Usage:
  python zotero_read.py ITEM_KEY
  python zotero_read.py ITEM_KEY --pages 10
  python zotero_read.py ITEM_KEY --out summary.txt
  python zotero_read.py ITEM_KEY --cite apa
  python zotero_read.py ITEM_KEY --cite mla
  python zotero_read.py ITEM_KEY --cite bibtex
  python zotero_read.py ITEM_KEY --metadata-only
"""

import argparse
import os
import sys
import tempfile
from pathlib import Path

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
    }


# ---------------------------------------------------------------------------
# Zotero fetch helpers
# ---------------------------------------------------------------------------

def get_item(item_key: str) -> dict:
    resp = requests.get(f"{API_BASE}/users/{USER_ID}/items/{item_key}", headers=HEADERS)
    resp.raise_for_status()
    return resp.json()


def get_children(item_key: str) -> list[dict]:
    resp = requests.get(f"{API_BASE}/users/{USER_ID}/items/{item_key}/children", headers=HEADERS)
    resp.raise_for_status()
    return resp.json()


def get_fulltext(attachment_key: str) -> dict | None:
    """Returns fulltext JSON or None if not indexed."""
    resp = requests.get(f"{API_BASE}/users/{USER_ID}/items/{attachment_key}/fulltext", headers=HEADERS)
    if resp.status_code == 404:
        return None
    if resp.status_code == 200:
        data = resp.json()
        if data.get("content"):
            return data
    return None


def download_pdf(attachment_key: str) -> bytes | None:
    """Download the PDF file bytes (follows redirect)."""
    resp = requests.get(
        f"{API_BASE}/users/{USER_ID}/items/{attachment_key}/file",
        headers=HEADERS,
        allow_redirects=True,
    )
    if resp.status_code == 200 and resp.content:
        return resp.content
    return None


# ---------------------------------------------------------------------------
# PDF extraction
# ---------------------------------------------------------------------------

def extract_pdf_text(pdf_bytes: bytes, max_pages: int | None = None) -> str:
    try:
        import pdfplumber
    except ImportError:
        sys.exit(
            "pdfplumber is needed to extract PDF text: pip install pdfplumber\n"
            "Alternatively, ensure the item is indexed in Zotero desktop first."
        )

    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as f:
        f.write(pdf_bytes)
        tmp_path = f.name

    try:
        import pdfplumber
        texts = []
        with pdfplumber.open(tmp_path) as pdf:
            pages = pdf.pages if max_pages is None else pdf.pages[:max_pages]
            for page in pages:
                text = page.extract_text()
                if text:
                    texts.append(text)
        return "\n\n".join(texts)
    finally:
        Path(tmp_path).unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# Citation formatters
# ---------------------------------------------------------------------------

def _derive_pdf_url(data: dict) -> str | None:
    """
    Try to derive a direct PDF URL from item metadata.
    Works for: arXiv preprints, open-access DOIs that redirect to PDFs.
    """
    import re as _re
    url = data.get("url", "")
    extra = data.get("extra", "")

    # arXiv abstract URL → PDF URL
    arxiv_match = _re.search(r"arxiv\.org/abs/([\w./v]+)", url)
    if arxiv_match:
        return f"https://arxiv.org/pdf/{arxiv_match.group(1)}"

    # arXiv ID in extra field (added by zotero_add.py)
    extra_match = _re.search(r"arXiv:\s*([\w./v]+)", extra)
    if extra_match:
        return f"https://arxiv.org/pdf/{extra_match.group(1)}"

    # archiveID field (set by fetch_arxiv)
    archive_id = data.get("archiveID", "")
    if archive_id.startswith("arXiv:"):
        return f"https://arxiv.org/pdf/{archive_id[6:]}"

    # DOI → try unpaywall for open-access PDF
    doi = data.get("DOI", "")
    if doi:
        try:
            unpaywall = requests.get(
                f"https://api.unpaywall.org/v2/{doi}",
                params={"email": "hermes-agent@example.com"},
                timeout=8,
            )
            if unpaywall.ok:
                oa = unpaywall.json().get("best_oa_location") or {}
                pdf_url = oa.get("url_for_pdf") or oa.get("url")
                if pdf_url and pdf_url.endswith(".pdf"):
                    return pdf_url
        except Exception:
            pass

    return None


def extract_year(date_str: str) -> str:
    """Extract a 4-digit year from a date string like 'December 27, 2017' or '2017-12-27'."""
    import re
    match = re.search(r"\b(1[5-9]\d{2}|20\d{2})\b", date_str or "")
    return match.group(1) if match else (date_str[:4] if len(date_str) >= 4 else date_str)


def fmt_creators_list(creators: list[dict]) -> list[str]:
    names = []
    for c in creators:
        last = c.get("lastName") or c.get("name", "")
        first = c.get("firstName", "")
        if first and last:
            names.append(f"{last}, {first}")
        elif last:
            names.append(last)
    return names


def fmt_author_apa(creators: list[dict]) -> str:
    names = fmt_creators_list(creators)
    if not names:
        return ""
    if len(names) == 1:
        return names[0]
    if len(names) <= 20:
        return ", ".join(names[:-1]) + ", & " + names[-1]
    return ", ".join(names[:19]) + ", ... " + names[-1]


def fmt_author_mla(creators: list[dict]) -> str:
    names = fmt_creators_list(creators)
    if not names:
        return ""
    if len(names) == 1:
        return names[0]
    return names[0] + ", et al."


def fmt_author_chicago(creators: list[dict]) -> str:
    names = fmt_creators_list(creators)
    if not names:
        return ""
    if len(names) <= 3:
        return ", ".join(names)
    return names[0] + ", et al."


def cite_apa(data: dict) -> str:
    creators = data.get("creators", [])
    author = fmt_author_apa(creators)
    year = extract_year(data.get("date", "")) or "n.d."
    title = data.get("title", "")
    itype = data.get("itemType", "")
    doi = data.get("DOI", "")
    url = data.get("url", "")

    if itype == "journalArticle":
        journal = data.get("publicationTitle", "")
        volume = data.get("volume", "")
        issue = data.get("issue", "")
        pages = data.get("pages", "")
        vol_str = f", *{volume}*" if volume else ""
        iss_str = f"({issue})" if issue else ""
        pg_str = f", {pages}" if pages else ""
        doi_str = f" https://doi.org/{doi}" if doi else (f" {url}" if url else "")
        return f"{author} ({year}). {title}. *{journal}*{vol_str}{iss_str}{pg_str}.{doi_str}"

    if itype == "book":
        publisher = data.get("publisher", "")
        return f"{author} ({year}). *{title}*. {publisher}."

    if itype == "conferencePaper":
        proc = data.get("proceedingsTitle", "")
        return f"{author} ({year}). {title}. In *{proc}*." + (f" https://doi.org/{doi}" if doi else "")

    if itype in ("preprint", "manuscript"):
        repo = data.get("repository", "Preprint")
        return f"{author} ({year}). *{title}*. {repo}." + (f" https://doi.org/{doi}" if doi else (f" {url}" if url else ""))

    # Generic fallback
    return f"{author} ({year}). {title}." + (f" https://doi.org/{doi}" if doi else "")


def cite_mla(data: dict) -> str:
    creators = data.get("creators", [])
    author = fmt_author_mla(creators)
    title = data.get("title", "")
    year = extract_year(data.get("date", "")) or "n.d."
    itype = data.get("itemType", "")
    doi = data.get("DOI", "")
    url = data.get("url", "")

    if itype == "journalArticle":
        journal = data.get("publicationTitle", "")
        volume = data.get("volume", "")
        issue = data.get("issue", "")
        pages = data.get("pages", "")
        loc = f"vol. {volume}" if volume else ""
        if issue:
            loc += f", no. {issue}"
        loc_str = f", {loc}" if loc else ""
        pg_str = f", pp. {pages}" if pages else ""
        doi_str = f", doi:{doi}" if doi else (f", {url}" if url else "")
        return f'{author}. "{title}." *{journal}*{loc_str}, {year}{pg_str}{doi_str}.'

    if itype == "book":
        publisher = data.get("publisher", "")
        return f"{author}. *{title}*. {publisher}, {year}."

    return f'{author}. "{title}." {year}.'


def cite_chicago(data: dict) -> str:
    creators = data.get("creators", [])
    author = fmt_author_chicago(creators)
    title = data.get("title", "")
    year = extract_year(data.get("date", "")) or "n.d."
    itype = data.get("itemType", "")
    doi = data.get("DOI", "")
    url = data.get("url", "")

    if itype == "journalArticle":
        journal = data.get("publicationTitle", "")
        volume = data.get("volume", "")
        issue = data.get("issue", "")
        pages = data.get("pages", "")
        vol_str = f" {volume}" if volume else ""
        iss_str = f", no. {issue}" if issue else ""
        pg_str = f": {pages}" if pages else ""
        doi_str = f" https://doi.org/{doi}." if doi else ""
        return f'{author}. "{title}." *{journal}*{vol_str}{iss_str} ({year}){pg_str}.{doi_str}'

    if itype == "book":
        publisher = data.get("publisher", "")
        return f"{author}. *{title}*. {publisher}, {year}."

    return f'{author}. "{title}." {year}.'


def cite_bibtex(data: dict) -> str:
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

    fields = [("author", author), ("title", f"{{{title}}}"), ("year", year)]
    for f in ("publicationTitle", "volume", "number", "pages", "publisher", "DOI", "url"):
        val = data.get(f, "")
        if val:
            fields.append((f.lower(), val))
    body = ",\n  ".join(f"{k} = {{{v}}}" for k, v in fields if v)
    return f"@{bib_type}{{{key},\n  {body}\n}}"


CITE_STYLES = {
    "apa": cite_apa,
    "mla": cite_mla,
    "chicago": cite_chicago,
    "bibtex": cite_bibtex,
}


# ---------------------------------------------------------------------------
# Print helpers
# ---------------------------------------------------------------------------

def print_metadata(data: dict) -> None:
    creators = data.get("creators", [])
    names = fmt_creators_list(creators)
    print(f"\n{'─' * 70}")
    print(f"Title:   {data.get('title', '')}")
    if names:
        print(f"Authors: {'; '.join(names[:5])}" + (" et al." if len(names) > 5 else ""))
    print(f"Type:    {data.get('itemType', '')}")
    print(f"Date:    {extract_year(data.get('date', ''))}")
    for field in ("publicationTitle", "publisher", "DOI", "ISBN", "url"):
        val = data.get(field, "")
        if val:
            print(f"{field.capitalize():8s} {val}")
    tags = [t["tag"] for t in data.get("tags", [])]
    if tags:
        print(f"Tags:    {', '.join(tags)}")
    abstract = data.get("abstractNote", "")
    if abstract:
        print(f"\nAbstract:\n{abstract[:600]}{'...' if len(abstract) > 600 else ''}")
    print(f"{'─' * 70}\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Read PDF content from a Zotero item")
    parser.add_argument("item_key", help="Zotero item key")
    parser.add_argument("--pages", type=int, metavar="N", help="Limit to first N pages")
    parser.add_argument("--out", metavar="FILE", help="Save extracted text to a file")
    parser.add_argument("--cite", choices=list(CITE_STYLES), metavar="STYLE",
                        help="Print formatted citation: apa, mla, chicago, bibtex")
    parser.add_argument("--metadata-only", action="store_true",
                        help="Show only metadata (no PDF extraction)")
    args = parser.parse_args()

    api_key, user_id = get_env()
    global HEADERS, USER_ID
    HEADERS = build_headers(api_key)
    USER_ID = user_id

    # Fetch item metadata
    print(f"Fetching item {args.item_key}...")
    item = get_item(args.item_key)
    data = item.get("data", {})
    print_metadata(data)

    # Citation mode
    if args.cite:
        cite_fn = CITE_STYLES[args.cite]
        print(f"Citation ({args.cite.upper()}):\n")
        print(cite_fn(data))
        print()
        return

    if args.metadata_only:
        return

    # Find PDF attachment
    print("Looking for PDF attachment...")
    children = get_children(args.item_key)
    pdf_attachments = [
        c for c in children
        if c["data"].get("itemType") == "attachment"
        and c["data"].get("contentType", "").startswith("application/pdf")
    ]

    # Also check if this item itself is an attachment
    if data.get("itemType") == "attachment" and data.get("contentType", "").startswith("application/pdf"):
        attachment_key = args.item_key
    elif pdf_attachments:
        attachment_key = pdf_attachments[0]["data"]["key"]
        print(f"Found attachment: {attachment_key}")
    else:
        attachment_key = None

    if attachment_key:
        # Try fulltext index first (fastest — populated by Zotero desktop app)
        print("Checking Zotero fulltext index...")
        fulltext = get_fulltext(attachment_key)
        if fulltext:
            content = fulltext.get("content", "")
            indexed = fulltext.get("indexedPages", "?")
            total = fulltext.get("totalPages", "?")
            print(f"✓ Fulltext index: {indexed}/{total} pages indexed\n")

            if args.pages:
                paragraphs = content.split("\n\n")
                approx_per_page = max(1, len(paragraphs) // max(1, int(indexed or 1)))
                content = "\n\n".join(paragraphs[:args.pages * approx_per_page])
                print(f"(Showing approximately first {args.pages} pages)\n")

            if args.out:
                Path(args.out).write_text(content, encoding="utf-8")
                print(f"Text saved to: {args.out}")
            else:
                print(content)
            return

        # Fallback: download from Zotero cloud storage
        print("Fulltext not indexed. Downloading from Zotero cloud...")
        pdf_bytes = download_pdf(attachment_key)
        if pdf_bytes:
            print(f"Downloaded {len(pdf_bytes):,} bytes. Extracting text...")
            text = extract_pdf_text(pdf_bytes, max_pages=args.pages)
            if text.strip():
                if args.out:
                    Path(args.out).write_text(text, encoding="utf-8")
                    print(f"Text saved to: {args.out}")
                else:
                    print(text)
                return
            print("No text extracted from Zotero-stored PDF (may be scanned). Trying source URL...")

    # No attachment, or extraction failed — try fetching directly from the source URL
    # (covers arXiv papers added via API without PDF upload, open-access DOIs, etc.)
    source_pdf_url = _derive_pdf_url(data)
    if source_pdf_url:
        print(f"Fetching PDF from source: {source_pdf_url}")
        try:
            resp = requests.get(source_pdf_url, headers={"User-Agent": "hermes-agent/1.0"},
                                allow_redirects=True, timeout=60)
            resp.raise_for_status()
            pdf_bytes = resp.content
            print(f"Downloaded {len(pdf_bytes):,} bytes. Extracting text...")
            text = extract_pdf_text(pdf_bytes, max_pages=args.pages)
            if text.strip():
                if args.out:
                    Path(args.out).write_text(text, encoding="utf-8")
                    print(f"Text saved to: {args.out}")
                else:
                    print(text)
                return
            print("No text extracted. The PDF may be scanned/image-based.")
        except Exception as e:
            print(f"Could not fetch PDF from source URL: {e}")

    # Nothing worked — show abstract
    print("\nNo PDF text available. Showing abstract only.")
    if data.get("abstractNote"):
        print("\nAbstract (full):")
        print(data["abstractNote"])


if __name__ == "__main__":
    main()
