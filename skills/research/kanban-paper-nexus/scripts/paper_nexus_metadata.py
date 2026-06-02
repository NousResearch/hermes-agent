#!/usr/bin/env python3
"""Fetch paper metadata for kanban-paper-nexus.

Supports:
- arXiv id / URL
- Semantic Scholar paper URL / paper id
- DOI raw string / doi.org URL
- OpenAlex work URL / id
"""

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
from html import unescape

_NS = {"a": "http://www.w3.org/2005/Atom"}
_ARXIV_ID = re.compile(
    r"(?:arxiv\.org/(?:abs|pdf)/)?(\d{4}\.\d{4,5}(?:v\d+)?)",
    re.I,
)
_S2_PAPER = re.compile(
    r"semanticscholar\.org/paper/(?:[^/]+/)?([0-9a-f]{40})",
    re.I,
)
_DOI = re.compile(
    r"^(?:https?://(?:dx\.)?doi\.org/)?(10\.\d{4,9}/[-._;()/:A-Z0-9]+)$",
    re.I,
)
_OPENALEX_WORK = re.compile(
    r"^(?:https?://openalex\.org/)?(W\d+)$",
    re.I,
)
_S2_FIELDS = (
    "paperId,title,year,abstract,authors,externalIds,url,citationCount,"
    "influentialCitationCount,publicationVenue,openAccessPdf"
)


def _s2_headers() -> dict[str, str]:
    h = {"User-Agent": "hermes-paper-nexus/1.1"}
    key = os.environ.get("SEMANTIC_SCHOLAR_API_KEY", "").strip()
    if key:
        h["x-api-key"] = key
    return h


def _get_json(url: str, timeout: int = 45, *, headers: dict[str, str] | None = None) -> dict:
    req_headers = dict(_s2_headers())
    if headers:
        req_headers.update(headers)
    req = urllib.request.Request(url, headers=req_headers)
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


def _doi_from_raw(raw: str) -> str | None:
    raw = (raw or "").strip()
    raw = raw.rstrip(").,;]")
    m = _DOI.match(raw)
    if not m:
        return None
    return m.group(1).lower()


def _openalex_from_raw(raw: str) -> str | None:
    raw = (raw or "").strip().rstrip("/").rstrip(").,;]")
    m = _OPENALEX_WORK.match(raw)
    if not m:
        return None
    return m.group(1).upper()


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
    doi = _doi_from_raw(raw)
    if doi:
        return f"doi:{doi}"
    openalex = _openalex_from_raw(raw)
    if openalex:
        return f"openalex:{openalex.lower()}"
    raise ValueError(
        f"Cannot parse paper id from: {raw!r}. "
        "Use arXiv id/URL, DOI, OpenAlex work URL/id, or https://www.semanticscholar.org/paper/<40-char-id>"
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


def _crossref_headers() -> dict[str, str]:
    mailto = os.environ.get("CROSSREF_MAILTO", "").strip() or "noreply@example.com"
    return {
        "User-Agent": f"hermes-paper-nexus/1.2 (mailto:{mailto})",
        "Accept": "application/json",
    }


def _crossref_url(doi: str) -> str:
    params = {}
    mailto = os.environ.get("CROSSREF_MAILTO", "").strip()
    if mailto:
        params["mailto"] = mailto
    q = urllib.parse.urlencode(params)
    base = f"https://api.crossref.org/works/{urllib.parse.quote(doi, safe='')}"
    return f"{base}?{q}" if q else base


def _openalex_url(path: str, *, params: dict[str, str] | None = None) -> str:
    query = dict(params or {})
    api_key = os.environ.get("OPENALEX_API_KEY", "").strip()
    mailto = (
        os.environ.get("OPENALEX_MAILTO", "").strip()
        or os.environ.get("CROSSREF_MAILTO", "").strip()
    )
    if api_key:
        query["api_key"] = api_key
    if mailto:
        query["mailto"] = mailto
    q = urllib.parse.urlencode(query)
    return f"https://api.openalex.org/{path}?{q}" if q else f"https://api.openalex.org/{path}"


def _clean_crossref_abstract(text: str) -> str:
    if not text:
        return ""
    cleaned = re.sub(r"<[^>]+>", " ", unescape(text))
    return " ".join(cleaned.split()).strip()


def _crossref_date(msg: dict) -> str:
    parts = (
        (msg.get("published-print") or {}).get("date-parts")
        or (msg.get("published-online") or {}).get("date-parts")
        or (msg.get("issued") or {}).get("date-parts")
        or []
    )
    if not parts or not parts[0]:
        return ""
    vals = [str(x) for x in parts[0][:3]]
    if len(vals) == 1:
        return vals[0]
    if len(vals) == 2:
        return f"{vals[0]}-{vals[1].zfill(2)}"
    return f"{vals[0]}-{vals[1].zfill(2)}-{vals[2].zfill(2)}"


def _openalex_abstract(inv: dict | None) -> str:
    if not isinstance(inv, dict) or not inv:
        return ""
    words: list[tuple[int, str]] = []
    for token, positions in inv.items():
        if not isinstance(positions, list):
            continue
        for pos in positions:
            if isinstance(pos, int):
                words.append((pos, token))
    words.sort()
    return " ".join(token for _, token in words).strip()


def _arxiv_from_openalex_ids(ids: dict | None) -> str:
    if not isinstance(ids, dict):
        return ""
    for key in ("arxiv", "ArXiv"):
        raw = ids.get(key)
        if raw:
            aid = _arxiv_from_raw(str(raw))
            if aid:
                return aid
    return ""


def fetch_crossref_entry(doi: str) -> dict:
    url = _crossref_url(doi)
    req = urllib.request.Request(url, headers=_crossref_headers())
    with urllib.request.urlopen(req, timeout=30) as resp:
        payload = json.loads(resp.read().decode("utf-8"))
    msg = payload.get("message") or {}
    titles = msg.get("title") or []
    title = str(titles[0] if titles else "").strip()
    authors = []
    for author in (msg.get("author") or [])[:12]:
        if not isinstance(author, dict):
            continue
        name = " ".join(
            part for part in (author.get("given", ""), author.get("family", "")) if part
        ).strip()
        if not name:
            name = str(author.get("name") or "").strip()
        if name:
            authors.append(name)
    venue = ""
    containers = msg.get("container-title") or []
    if containers:
        venue = str(containers[0]).strip()
    doi_url = f"https://doi.org/{doi}"
    return {
        "paper_id": f"doi:{doi}",
        "canonical_id": f"doi:{doi}",
        "source": "crossref",
        "title": title,
        "summary": _clean_crossref_abstract(str(msg.get("abstract") or "")),
        "published": _crossref_date(msg),
        "authors": authors,
        "categories": [],
        "arxiv_abs": "",
        "arxiv_pdf": "",
        "s2_url": "",
        "doi": doi,
        "doi_url": doi_url,
        "venue": venue,
        "citation_count": 0,
        "influential_citation_count": 0,
    }


def fetch_openalex_entry(work_ref: str) -> dict:
    wid = work_ref.strip()
    if wid.lower().startswith("openalex:"):
        wid = wid.split(":", 1)[1]
    if wid.lower().startswith("https://openalex.org/"):
        wid = wid.rsplit("/", 1)[-1]
    wid = wid.upper()
    url = _openalex_url(f"works/{urllib.parse.quote(wid, safe='')}")
    item = _get_json(url, headers={"User-Agent": "hermes-paper-nexus/1.2"})
    ids = item.get("ids") or {}
    arxiv = _arxiv_from_openalex_ids(ids)
    doi = _doi_from_raw(str(ids.get("doi") or ""))
    authors = []
    for authorship in (item.get("authorships") or [])[:12]:
        if not isinstance(authorship, dict):
            continue
        author = authorship.get("author") or {}
        name = str(author.get("display_name") or "").strip()
        if name:
            authors.append(name)
    best_oa = item.get("best_oa_location") or {}
    primary_loc = item.get("primary_location") or {}
    source = primary_loc.get("source") or {}
    pdf_url = str(best_oa.get("pdf_url") or primary_loc.get("pdf_url") or "").strip()
    landing = str(
        best_oa.get("landing_page_url")
        or primary_loc.get("landing_page_url")
        or item.get("id")
        or f"https://openalex.org/{wid}"
    ).strip()
    canonical = (
        re.sub(r"v\d+$", "", arxiv, flags=re.I)
        if arxiv
        else (f"doi:{doi}" if doi else f"openalex:{wid.lower()}")
    )
    paper_id = arxiv or (f"doi:{doi}" if doi else f"openalex:{wid.lower()}")
    venue = str(source.get("display_name") or item.get("host_venue", {}).get("display_name") or "").strip()
    return {
        "paper_id": paper_id,
        "canonical_id": canonical,
        "source": "openalex",
        "openalex_id": wid.lower(),
        "title": str(item.get("display_name") or "").strip(),
        "summary": _openalex_abstract(item.get("abstract_inverted_index")),
        "published": str(item.get("publication_year") or "")[:4],
        "authors": authors,
        "categories": [],
        "arxiv_abs": f"https://arxiv.org/abs/{arxiv}" if arxiv else "",
        "arxiv_pdf": f"https://arxiv.org/pdf/{arxiv}" if arxiv else pdf_url,
        "s2_url": "",
        "doi": doi or "",
        "doi_url": f"https://doi.org/{doi}" if doi else "",
        "venue": venue,
        "citation_count": int(item.get("cited_by_count") or 0),
        "influential_citation_count": 0,
        "url": landing,
    }


def fetch_s2_entry(paper_ref: str) -> dict:
    paper_ref = paper_ref.strip()
    cid = paper_ref.lower()
    url = (
        "https://api.semanticscholar.org/graph/v1/paper/"
        f"{urllib.parse.quote(paper_ref, safe='')}?fields={_S2_FIELDS}"
    )
    item = _get_json(url)
    ext = item.get("externalIds") or {}
    arxiv = str(ext.get("ArXiv") or ext.get("arXiv") or "").strip()
    doi = str(ext.get("DOI") or "").strip()
    paper_id = str(item.get("paperId") or "").strip().lower()
    authors = [
        a.get("name", "") if isinstance(a, dict) else str(a)
        for a in (item.get("authors") or [])[:12]
    ]
    venue = item.get("publicationVenue") or {}
    venue_name = venue.get("name", "") if isinstance(venue, dict) else ""
    oa = item.get("openAccessPdf") or {}
    pdf_url = (oa.get("url") if isinstance(oa, dict) else "") or ""
    s2_id = paper_id or (cid if re.match(r"^[0-9a-f]{40}$", cid, re.I) else "")
    canonical = re.sub(r"v\d+$", "", arxiv, flags=re.I) if arxiv else (
        f"s2:{s2_id}" if s2_id else (f"doi:{doi.lower()}" if doi else cid)
    )
    published = str(item.get("year") or "")[:4]
    return {
        "paper_id": arxiv or (f"s2:{s2_id}" if s2_id else canonical),
        "canonical_id": canonical,
        "source": "semantic_scholar",
        "s2_corpus_id": s2_id or cid,
        "title": (item.get("title") or "").replace("\n", " ").strip(),
        "summary": (item.get("abstract") or "").strip(),
        "published": published,
        "authors": authors,
        "categories": [],
        "arxiv_abs": f"https://arxiv.org/abs/{arxiv}" if arxiv else "",
        "arxiv_pdf": f"https://arxiv.org/pdf/{arxiv}" if arxiv else pdf_url,
        "s2_url": item.get("url") or (f"https://www.semanticscholar.org/paper/{s2_id}" if s2_id else ""),
        "doi": doi.lower(),
        "doi_url": f"https://doi.org/{doi.lower()}" if doi else "",
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
    """Resolve arXiv id, DOI, OpenAlex, S2 URL/hash, or fail with clear error."""
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

    doi = _doi_from_raw(raw)
    if doi:
        try:
            return fetch_s2_entry(f"DOI:{doi}")
        except Exception:
            return fetch_crossref_entry(doi)

    openalex = _openalex_from_raw(raw)
    if openalex:
        return fetch_openalex_entry(openalex)

    raise ValueError(
        f"Cannot parse paper id from: {raw!r}. "
        "Use arXiv id/URL, DOI, OpenAlex work URL/id, or https://www.semanticscholar.org/paper/<40-char-id>"
    )


def main() -> int:
    if len(sys.argv) < 2:
        print(
            "Usage: paper_nexus_metadata.py <arxiv|doi|openalex|semanticscholar>",
            file=sys.stderr,
        )
        return 2
    data = resolve_and_fetch(sys.argv[1])
    json.dump(data, sys.stdout, ensure_ascii=False, indent=2)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
