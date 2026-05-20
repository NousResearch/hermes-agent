#!/usr/bin/env python3
"""Fetch arXiv metadata for kanban-paper-nexus workers (stdlib only)."""

from __future__ import annotations

import json
import re
import sys
import urllib.request
import xml.etree.ElementTree as ET

_NS = {"a": "http://www.w3.org/2005/Atom"}
_ARXIV_ID = re.compile(
    r"(?:arxiv\.org/(?:abs|pdf)/)?(\d{4}\.\d{4,5}(?:v\d+)?)",
    re.I,
)


def normalize_paper_id(raw: str) -> str:
    raw = (raw or "").strip()
    m = _ARXIV_ID.search(raw)
    if m:
        return m.group(1)
    m = re.match(r"^(\d{4}\.\d{4,5}(?:v\d+)?)$", raw)
    if m:
        return m.group(1)
    raise ValueError(f"Cannot parse arXiv id from: {raw!r}")


def fetch_entry(paper_id: str) -> dict:
    url = f"https://export.arxiv.org/api/query?id_list={paper_id}"
    with urllib.request.urlopen(url, timeout=30) as resp:
        root = ET.parse(resp).getroot()
    entry = root.find("a:entry", _NS)
    if entry is None:
        raise ValueError(f"No arXiv entry for {paper_id}")

    def _text(tag: str) -> str:
        el = entry.find(f"a:{tag}", _NS)
        return (el.text or "").strip() if el is not None else ""

    arxiv_id = _text("id").split("/abs/")[-1] or paper_id
    authors = [
        a.find("a:name", _NS).text.strip()
        for a in entry.findall("a:author", _NS)
        if a.find("a:name", _NS) is not None
    ]
    categories = [c.get("term", "") for c in entry.findall("a:category", _NS)]
    return {
        "paper_id": arxiv_id,
        "title": _text("title").replace("\n", " "),
        "summary": _text("summary"),
        "published": _text("published")[:10],
        "authors": authors,
        "categories": categories,
        "arxiv_abs": f"https://arxiv.org/abs/{arxiv_id}",
        "arxiv_pdf": f"https://arxiv.org/pdf/{arxiv_id}",
    }


def main() -> int:
    if len(sys.argv) < 2:
        print("Usage: paper_nexus_metadata.py <arxiv_id_or_url>", file=sys.stderr)
        return 2
    paper_id = normalize_paper_id(sys.argv[1])
    data = fetch_entry(paper_id)
    json.dump(data, sys.stdout, ensure_ascii=False, indent=2)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
