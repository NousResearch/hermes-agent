#!/usr/bin/env python3
"""
Fetch arXiv paper metadata + blog posts and write a text report.

Equivalent to:
    curl -s "https://export.arxiv.org/api/query?id_list=ID"
but uses urllib (no curl dependency) so it works in any Python 3 env.

Usage:
    python3 fetch_arxiv_report.py            # writes arxiv_report.txt
    python3 fetch_arxiv_report.py -o out.txt # custom output path
"""
from __future__ import annotations

import argparse
import sys
import urllib.request
import urllib.error
import xml.etree.ElementTree as ET
from datetime import datetime
from typing import Optional

# Atom / arxiv namespace URIs (used as XML element-tag prefixes)
ATOM_NS = "{http://www.w3.org/2005/Atom}"
ARXIV_NS = "{http://arxiv.org/schemas/atom}"

ARXIV_PAPERS = [
    ("2307.03172", 'Liu et al., "Lost in the Middle"'),
    ("2309.17453", 'Xiao et al., "Efficient Streaming Language Models with Attention Sinks"'),
    ("2402.04617", "Li et al. (context-window extension paper)"),
    ("2310.01477", 'Peng et al., "YaRN: Efficient Context Window Extension of LLMs"'),
]

BLOG_URLS = [
    "https://huggingface.co/docs/transformers/en/model_doc/llama2",
    "https://huggingface.co/blog/llama2",
    "https://blog.salesforceairesearch.com/lost-in-the-middle/",
]

USER_AGENT = "arxiv-report-fetcher/1.0 (educational; mailto:none-given)"


def fetch_url(url: str, timeout: int = 30) -> tuple[int, str, str]:
    """Return (http_status, final_url, body). Body is utf-8 text."""
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read()
            charset = resp.headers.get_content_charset() or "utf-8"
            return resp.status, resp.geturl(), raw.decode(charset, errors="replace")
    except urllib.error.HTTPError as e:
        try:
            body = e.read().decode("utf-8", errors="replace")
        except Exception:
            body = ""
        return e.code, url, body
    except urllib.error.URLError as e:
        return 0, url, f"[URLError] {e.reason}"
    except Exception as e:  # pragma: no cover
        return 0, url, f"[Error] {type(e).__name__}: {e}"


def parse_arxiv_entry(xml_text: str) -> Optional[dict]:
    """Parse one Atom <entry> from the arxiv API response and extract fields."""
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError as e:
        return {"_parse_error": str(e)}

    entry = root.find(f"{ATOM_NS}entry")
    if entry is None:
        return {"_error": "no <entry> element found"}

    def text_of(tag: str) -> str:
        # Try plain Atom tag, then arxiv-namespaced variant
        for ns in (ATOM_NS, ARXIV_NS):
            el = entry.find(f"{ns}{tag}")
            if el is not None and el.text:
                return el.text.strip()
        return ""

    title = text_of("title").replace("\n", " ").replace("  ", " ").strip()

    authors = []
    for a in entry.findall(f"{ATOM_NS}author"):
        name_el = a.find(f"{ATOM_NS}name")
        if name_el is not None and name_el.text:
            authors.append(name_el.text.strip())

    abstract = text_of("summary").replace("\n", " ").strip()
    # arxiv returns summary as the abstract; some entries also have ARXIV_NS/abstract
    if not abstract:
        abstract = text_of("abstract")

    published = text_of("published")
    updated = text_of("updated")
    doi = text_of("doi")
    journal_ref = text_of("journal-ref")
    primary_category = ""
    cat_el = entry.find(f"{ARXIV_NS}primary_category")
    if cat_el is not None:
        primary_category = cat_el.attrib.get("term", "")

    return {
        "title": title,
        "authors": authors,
        "abstract": abstract,
        "published": published,
        "updated": updated,
        "doi": doi,
        "journal_ref": journal_ref,
        "primary_category": primary_category,
    }


def fmt_authors(authors: list[str]) -> str:
    if not authors:
        return "(none)"
    if len(authors) <= 6:
        return ", ".join(authors)
    return ", ".join(authors[:6]) + f", ... (+{len(authors) - 6} more, total {len(authors)})"


def fmt_date(iso: str) -> str:
    if not iso:
        return "(unknown)"
    try:
        return datetime.fromisoformat(iso.replace("Z", "+00:00")).strftime("%Y-%m-%d")
    except Exception:
        return iso


def section_arxiv(rid: str, hint: str, out: list[str]) -> None:
    url = f"https://export.arxiv.org/api/query?id_list={rid}"
    out.append("=" * 78)
    out.append(f"PAPER: arXiv:{rid}  (expected: {hint})")
    out.append(f"URL:   {url}")
    out.append("-" * 78)

    status, final, body = fetch_url(url)
    out.append(f"HTTP status: {status}    final URL: {final}")

    if status != 200 or not body:
        out.append(f"!! Fetch failed. Body: {body[:500]}")
        out.append("")
        return

    info = parse_arxiv_entry(body)
    if not info:
        out.append("!! Could not parse Atom feed")
        out.append(f"Raw (first 800 chars):\n{body[:800]}")
        out.append("")
        return

    if info.get("_error") or info.get("_parse_error"):
        out.append(f"!! Parser issue: {info}")
        out.append("")
        return

    out.append(f"Title:        {info['title']}")
    out.append(f"Authors:      {fmt_authors(info['authors'])}")
    out.append(f"Submitted:    {fmt_date(info['published'])} (published)")
    if info["updated"] and info["updated"] != info["published"]:
        out.append(f"Last updated: {fmt_date(info['updated'])}")
    if info["primary_category"]:
        out.append(f"Category:     {info['primary_category']}")
    if info["doi"]:
        out.append(f"DOI:          {info['doi']}")
    if info["journal_ref"]:
        out.append(f"Journal-ref:  {info['journal_ref']}")
    out.append("")
    out.append("Abstract:")
    abstract = info["abstract"] or "(empty)"
    # Wrap to ~74 chars
    line, lines = "", []
    for word in abstract.split():
        if len(line) + len(word) + 1 > 74:
            lines.append(line.rstrip())
            line = word
        else:
            line = (line + " " + word).strip()
    if line:
        lines.append(line)
    for ln in lines:
        out.append("  " + ln)
    out.append("")


def section_blog(url: str, out: list[str]) -> None:
    out.append("=" * 78)
    out.append(f"BLOG FETCH: {url}")
    out.append("-" * 78)
    status, final, body = fetch_url(url, timeout=30)
    out.append(f"HTTP status: {status}    final URL: {final}")
    if status == 0:
        out.append(f"Transport error: {body}")
        out.append("")
        return
    if status != 200:
        out.append(f"(non-200 response; showing first 600 chars of body if any)")
        out.append(body[:600])
        out.append("")
        return
    # Show first ~1500 chars, strip excessive whitespace
    snippet = " ".join(body.split())[:1500]
    out.append(snippet)
    out.append("")
    out.append(f"[total body length: {len(body)} chars]")
    out.append("")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("-o", "--output", default="arxiv_report.txt",
                    help="Output text file path (default: arxiv_report.txt)")
    args = ap.parse_args()

    out: list[str] = []
    out.append("arXiv + Blog Fetch Report")
    out.append(f"Generated: {datetime.utcnow().isoformat(timespec='seconds')}Z")
    out.append("=" * 78)
    out.append("")
    out.append("METHOD:")
    out.append("  - arxiv: direct GET https://export.arxiv.org/api/query?id_list=<ID>")
    out.append("           parsed with xml.etree.ElementTree (Atom namespace)")
    out.append("  - blogs: direct HTTP GET via urllib, raw snippet shown")
    out.append("  - user-agent: " + USER_AGENT)
    out.append("")

    out.append("#" * 78)
    out.append("# PART 1: arXiv paper metadata")
    out.append("#" * 78)
    out.append("")
    for rid, hint in ARXIV_PAPERS:
        section_arxiv(rid, hint, out)

    out.append("#" * 78)
    out.append("# PART 2: Blog / doc URL fetches")
    out.append("#" * 78)
    out.append("")
    for url in BLOG_URLS:
        section_blog(url, out)

    text = "\n".join(out)
    with open(args.output, "w", encoding="utf-8") as f:
        f.write(text)

    print(f"Wrote {args.output} ({len(text)} chars, {len(out)} lines)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
