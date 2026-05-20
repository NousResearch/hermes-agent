#!/usr/bin/env python3
"""Fetch paper metadata for kanban-paper-nexus (arXiv + Semantic Scholar URLs)."""

from __future__ import annotations

import json
import os
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET

_NS = {"a": "http://www.w3.org/2005/Atom"}
_ARXIV_ID = re.compile(
    r"(?:arxiv\.org/(?:abs|pdf)/)?(\d{4}\.\d{4,5}(?:v\d+)?)",
    re.I,
)
_S2_PAPER = re.compile(
    r"semanticscholar\.org/paper/(?:[^/]+/)?([0-9a-f]{40})",
    re.I,
)
_S2_FIELDS = (
    "title,year,abstract,authors,externalIds,url,citationCount,"
    "influentialCitationCount,publicationVenue,openAccessPdf"
)


def _s2_headers() -> dict[str, str]:
    h = {"User-Agent": "hermes-paper-nexus/1.1"}
    key = os.environ.get("SEMANTIC_SCHOLAR_API_KEY", "").strip()
    if key:
        h["x-api-key"] = key
    return h


def _get_json(url: str, timeout: int = 45) -> dict:
    req = urllib.request.Request(url, headers=_s2_headers())
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        if exc.code == 429:
            for wait in (2, 5, 12):
                time.sleep(wait)
                try:
                    with urllib.request.urlopen(req, timeout=timeout) as resp:
                        return json.loads(resp.read().decode("utf-8"))
                except urllib.error.HTTPError as retry_exc:
                    if retry_exc.code != 429:
                        raise
            raise
        raise


def resolve_canonical_id(raw: str) -> str:
    """Parse canonical id locally (no network)."""
    raw = (raw or "").strip()
    arxiv = _arxiv_from_raw(raw)
    if arxiv:
        return re.sub(r"v\d+$", "", arxiv, flags=re.I)
    s2_corpus = _s2_corpus_from_raw(raw)
    if s2_corpus:
        return f"s2:{s2_corpus}"
    if re.match(r"^s2:([0-9a-f]{40})$", raw, re.I):
        return raw.lower()
    if re.match(r"^[0-9a-f]{40}$", raw, re.I):
        return f"s2:{raw.lower()}"
    raise ValueError(
        f"Cannot parse paper id from: {raw!r}. "
        "Use arXiv id/URL or https://www.semanticscholar.org/paper/<40-char-id>"
    )


def normalize_paper_id(raw: str) -> str:
    """Return canonical id: arXiv id (no vN) or ``s2:<40-char hash>`` (may call S2 API)."""
    meta = resolve_and_fetch(raw)
    return meta["canonical_id"]


def _arxiv_from_raw(raw: str) -> str | None:
    raw = (raw or "").strip()
    m = _ARXIV_ID.search(raw)
    if m:
        return m.group(1)
    m = re.match(r"^(\d{4}\.\d{4,5}(?:v\d+)?)$", raw)
    return m.group(1) if m else None


def _s2_corpus_from_raw(raw: str) -> str | None:
    m = _S2_PAPER.search(raw or "")
    return m.group(1).lower() if m else None


def fetch_s2_entry(corpus_id: str) -> dict:
    cid = corpus_id.strip().lower()
    url = (
        "https://api.semanticscholar.org/graph/v1/paper/"
        f"{urllib.parse.quote(cid)}?fields={_S2_FIELDS}"
    )
    item = _get_json(url)
    ext = item.get("externalIds") or {}
    arxiv = str(ext.get("ArXiv") or ext.get("arXiv") or "").strip()
    doi = str(ext.get("DOI") or "").strip()
    authors = [
        a.get("name", "") if isinstance(a, dict) else str(a)
        for a in (item.get("authors") or [])[:12]
    ]
    venue = item.get("publicationVenue") or {}
    venue_name = venue.get("name", "") if isinstance(venue, dict) else ""
    oa = item.get("openAccessPdf") or {}
    pdf_url = (oa.get("url") if isinstance(oa, dict) else "") or ""
    canonical = re.sub(r"v\d+$", "", arxiv, flags=re.I) if arxiv else f"s2:{cid}"
    published = str(item.get("year") or "")[:4]
    return {
        "paper_id": arxiv or f"s2:{cid}",
        "canonical_id": canonical,
        "source": "semantic_scholar",
        "s2_corpus_id": cid,
        "title": (item.get("title") or "").replace("\n", " ").strip(),
        "summary": (item.get("abstract") or "").strip(),
        "published": published,
        "authors": authors,
        "categories": [],
        "arxiv_abs": f"https://arxiv.org/abs/{arxiv}" if arxiv else "",
        "arxiv_pdf": f"https://arxiv.org/pdf/{arxiv}" if arxiv else pdf_url,
        "s2_url": item.get("url") or f"https://www.semanticscholar.org/paper/{cid}",
        "doi": doi,
        "venue": venue_name,
        "citation_count": int(item.get("citationCount") or 0),
        "influential_citation_count": int(item.get("influentialCitationCount") or 0),
    }


def fetch_entry(paper_id: str) -> dict:
    """Fetch arXiv Atom metadata."""
    url = f"https://export.arxiv.org/api/query?id_list={paper_id}"
    req = urllib.request.Request(url, headers={"User-Agent": "hermes-paper-nexus/1.1"})
    with urllib.request.urlopen(req, timeout=30) as resp:
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
    cid = re.sub(r"v\d+$", "", arxiv_id, flags=re.I)
    return {
        "paper_id": arxiv_id,
        "canonical_id": cid,
        "source": "arxiv",
        "title": _text("title").replace("\n", " "),
        "summary": _text("summary"),
        "published": _text("published")[:10],
        "authors": authors,
        "categories": categories,
        "arxiv_abs": f"https://arxiv.org/abs/{arxiv_id}",
        "arxiv_pdf": f"https://arxiv.org/pdf/{arxiv_id}",
        "s2_url": "",
        "doi": "",
        "venue": "",
    }


def resolve_and_fetch(raw: str) -> dict:
    """Resolve arXiv id, S2 URL/hash, or fail with clear error."""
    raw = (raw or "").strip()
    arxiv = _arxiv_from_raw(raw)
    if arxiv:
        meta = fetch_entry(arxiv)
        s2_corpus = _s2_corpus_from_raw(raw)
        if s2_corpus:
            meta["s2_url"] = f"https://www.semanticscholar.org/paper/{s2_corpus}"
        return meta

    s2_corpus = _s2_corpus_from_raw(raw)
    if s2_corpus:
        meta = fetch_s2_entry(s2_corpus)
        if meta.get("arxiv_abs"):
            try:
                arxiv_meta = fetch_entry(meta["paper_id"])
                meta.update({k: arxiv_meta[k] for k in ("summary", "categories") if arxiv_meta.get(k)})
                if not meta.get("summary"):
                    meta["summary"] = arxiv_meta.get("summary", "")
            except Exception:
                pass
        return meta

    if re.match(r"^[0-9a-f]{40}$", raw, re.I):
        return fetch_s2_entry(raw)

    raise ValueError(
        f"Cannot parse paper id from: {raw!r}. "
        "Use arXiv id/URL or https://www.semanticscholar.org/paper/<40-char-id>"
    )


def main() -> int:
    if len(sys.argv) < 2:
        print(
            "Usage: paper_nexus_metadata.py <arxiv_or_semanticscholar_url>",
            file=sys.stderr,
        )
        return 2
    data = resolve_and_fetch(sys.argv[1])
    json.dump(data, sys.stdout, ensure_ascii=False, indent=2)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
