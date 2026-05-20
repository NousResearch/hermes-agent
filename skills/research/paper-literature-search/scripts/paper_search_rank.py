#!/usr/bin/env python3
"""Rank papers for a topic via Semantic Scholar (+ optional arXiv backfill)."""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from datetime import datetime, timezone

_S2_SEARCH = "https://api.semanticscholar.org/graph/v1/paper/search"
_ARXIV_NS = {"a": "http://www.w3.org/2005/Atom"}
_WEIGHTS = {"rel": 0.35, "cite": 0.30, "infl": 0.15, "rec": 0.15, "oa": 0.05}


def _get_json(url: str, timeout: int = 45, *, retries: int = 2) -> dict:
    headers = {"User-Agent": "hermes-paper-search/1.0"}
    key = os.environ.get("SEMANTIC_SCHOLAR_API_KEY", "").strip()
    if key:
        headers["x-api-key"] = key
    req = urllib.request.Request(url, headers=headers)
    last_err: Exception | None = None
    for attempt in range(retries + 1):
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            last_err = exc
            if exc.code == 429 and attempt < retries:
                time.sleep(1.5 * (attempt + 1))
                continue
            raise
    raise last_err  # type: ignore[misc]


def _log1p_norm(values: list[float]) -> list[float]:
    if not values:
        return []
    logs = [math.log1p(max(0.0, v)) for v in values]
    lo, hi = min(logs), max(logs)
    if hi <= lo:
        return [1.0 if v > 0 else 0.0 for v in values]
    return [(x - lo) / (hi - lo) for x in logs]


def _recency_score(year: int | None, *, profile: str) -> float:
    if not year or year < 1990:
        return 0.35
    now = datetime.now(timezone.utc).year
    age = now - int(year)
    if profile == "survey":
        if age <= 0:
            return 1.0
        if age == 1:
            return 0.95
        if age <= 4:
            return 0.85
        if age <= 10:
            return 0.65
        return 0.45
    # ml default
    if age <= 0:
        return 1.0
    if age == 1:
        return 0.92
    if age == 2:
        return 0.78
    if age == 3:
        return 0.55
    return 0.35


def _arxiv_id_from_external(ext: dict | None) -> str:
    if not ext:
        return ""
    arx = ext.get("ArXiv") or ext.get("arXiv") or ""
    return str(arx).strip()


def _keyword_rel(query: str, title: str, abstract: str) -> float:
    qtokens = {t.lower() for t in re.findall(r"[\w\u4e00-\u9fff]{2,}", query)}
    if not qtokens:
        return 0.5
    text = f"{title} {abstract}".lower()
    hit = sum(1 for t in qtokens if t in text)
    return min(1.0, hit / max(1, len(qtokens)))


def search_semantic_scholar(query: str, *, limit: int = 40) -> list[dict]:
    fields = ",".join([
        "title",
        "year",
        "abstract",
        "authors",
        "citationCount",
        "influentialCitationCount",
        "url",
        "externalIds",
        "openAccessPdf",
        "publicationVenue",
    ])
    params = urllib.parse.urlencode({
        "query": query,
        "limit": str(min(100, max(5, limit))),
        "fields": fields,
    })
    url = f"{_S2_SEARCH}?{params}"
    data = _get_json(url)
    papers = []
    for item in data.get("data") or []:
        ext = item.get("externalIds") or {}
        arxiv = _arxiv_id_from_external(ext)
        oa = item.get("openAccessPdf") or {}
        pdf_url = (oa.get("url") if isinstance(oa, dict) else "") or ""
        authors = item.get("authors") or []
        author_names = [
            a.get("name", "") if isinstance(a, dict) else str(a) for a in authors[:6]
        ]
        papers.append({
            "source": "semantic_scholar",
            "paper_id": item.get("paperId") or "",
            "title": (item.get("title") or "").strip(),
            "year": item.get("year"),
            "abstract": (item.get("abstract") or "")[:800],
            "citation_count": int(item.get("citationCount") or 0),
            "influential_citation_count": int(item.get("influentialCitationCount") or 0),
            "relevance": float(item.get("relevance") or 0.0),
            "url": item.get("url") or "",
            "arxiv_id": arxiv,
            "arxiv_abs": f"https://arxiv.org/abs/{arxiv}" if arxiv else "",
            "pdf_url": pdf_url,
            "venue": (item.get("publicationVenue") or {}).get("name", "")
            if isinstance(item.get("publicationVenue"), dict)
            else "",
            "authors": author_names,
        })
    return papers


def search_arxiv_backfill(query: str, *, limit: int = 15) -> list[dict]:
    q = urllib.parse.quote(f"all:{query}")
    url = (
        f"https://export.arxiv.org/api/query?search_query={q}"
        f"&start=0&max_results={limit}&sortBy=relevance&sortOrder=descending"
    )
    req = urllib.request.Request(url, headers={"User-Agent": "hermes-paper-search/1.0"})
    with urllib.request.urlopen(req, timeout=45) as resp:
        root = ET.parse(resp).getroot()
    out: list[dict] = []
    for entry in root.findall("a:entry", _ARXIV_NS):
        title = (entry.find("a:title", _ARXIV_NS).text or "").replace("\n", " ").strip()
        summary = (entry.find("a:summary", _ARXIV_NS).text or "").strip()
        published = (entry.find("a:published", _ARXIV_NS).text or "")[:4]
        pid = (entry.find("a:id", _ARXIV_NS).text or "").split("/abs/")[-1].strip()
        year = int(published) if published.isdigit() else None
        out.append({
            "source": "arxiv",
            "paper_id": f"arxiv:{pid}",
            "title": title,
            "year": year,
            "abstract": summary[:800],
            "citation_count": 0,
            "influential_citation_count": 0,
            "relevance": 0.0,
            "url": f"https://arxiv.org/abs/{pid}",
            "arxiv_id": pid,
            "arxiv_abs": f"https://arxiv.org/abs/{pid}",
            "pdf_url": f"https://arxiv.org/pdf/{pid}",
            "venue": "arXiv",
            "authors": [],
        })
    return out


def dedupe_papers(papers: list[dict]) -> list[dict]:
    seen_arxiv: set[str] = set()
    seen_title: set[str] = set()
    out: list[dict] = []
    for p in papers:
        aid = (p.get("arxiv_id") or "").lower()
        tit = re.sub(r"\s+", " ", (p.get("title") or "").lower())[:120]
        if aid and aid in seen_arxiv:
            continue
        if tit and tit in seen_title:
            continue
        if aid:
            seen_arxiv.add(aid)
        if tit:
            seen_title.add(tit)
        out.append(p)
    return out


def rank_papers(
    query: str,
    papers: list[dict],
    *,
    profile: str = "ml",
    boost_recency: bool = False,
    min_citations: int = 0,
    year_floor: int | None = None,
) -> list[dict]:
    w = dict(_WEIGHTS)
    if boost_recency:
        w["rec"] += 0.05
        w["cite"] -= 0.05

    filtered = []
    for p in papers:
        cc = int(p.get("citation_count") or 0)
        # arXiv backfill often has citation_count=0 (unknown); do not treat as below threshold
        if min_citations and cc > 0 and cc < min_citations:
            continue
        yr = p.get("year")
        if year_floor and yr and int(yr) < year_floor:
            continue
        filtered.append(p)

    cites = [float(p.get("citation_count") or 0) for p in filtered]
    infls = [float(p.get("influential_citation_count") or 0) for p in filtered]
    cite_n = _log1p_norm(cites)
    infl_n = _log1p_norm(infls)

    for i, p in enumerate(filtered):
        rel = float(p.get("relevance") or 0.0)
        if rel <= 0:
            rel = _keyword_rel(query, p.get("title", ""), p.get("abstract", ""))
        rec = _recency_score(p.get("year"), profile=profile)
        oa = 1.0 if (p.get("pdf_url") or p.get("arxiv_id")) else 0.0
        composite = (
            w["rel"] * rel
            + w["cite"] * cite_n[i]
            + w["infl"] * infl_n[i]
            + w["rec"] * rec
            + w["oa"] * oa
        )
        p["scores"] = {
            "rel": round(rel, 3),
            "cite_norm": round(cite_n[i], 3),
            "infl_norm": round(infl_n[i], 3),
            "rec": round(rec, 3),
            "oa": round(oa, 3),
            "composite": round(composite, 4),
            "display": int(round(composite * 100)),
        }

    filtered.sort(
        key=lambda x: (
            x["scores"]["composite"],
            x.get("citation_count") or 0,
            x.get("year") or 0,
        ),
        reverse=True,
    )
    return filtered


def run_search(
    query: str,
    *,
    candidate_limit: int = 40,
    top: int = 8,
    profile: str = "ml",
    boost_recency: bool = False,
    min_citations: int = 0,
    year_floor: int | None = None,
    arxiv_backfill: int = 12,
) -> dict:
    query = (query or "").strip()
    if not query:
        raise ValueError("empty query")

    papers: list[dict] = []
    try:
        papers = search_semantic_scholar(query, limit=candidate_limit)
    except urllib.error.HTTPError as exc:
        if exc.code != 429:
            raise
    papers = list(papers)
    if len(papers) < max(10, top):
        time.sleep(0.3)
        try:
            papers.extend(search_arxiv_backfill(query, limit=arxiv_backfill))
        except Exception:
            pass

    papers = [p for p in papers if p.get("title") and p.get("source") != "error"]
    papers = dedupe_papers(papers)
    ranked = rank_papers(
        query,
        papers,
        profile=profile,
        boost_recency=boost_recency,
        min_citations=min_citations,
        year_floor=year_floor,
    )
    top_papers = ranked[: max(1, top)]

    return {
        "query": query,
        "profile": profile,
        "weights": _WEIGHTS,
        "candidate_count": len(papers),
        "ranked_count": len(ranked),
        "top": top,
        "papers": top_papers,
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Rank literature for a search topic")
    ap.add_argument("query", help="search topic / category")
    ap.add_argument("--top", type=int, default=8)
    ap.add_argument("--candidate-limit", type=int, default=40)
    ap.add_argument("--profile", choices=("ml", "survey"), default="ml")
    ap.add_argument("--boost-recency", action="store_true")
    ap.add_argument("--min-citations", type=int, default=0)
    ap.add_argument("--year-floor", type=int, default=0)
    args = ap.parse_args()
    yf = args.year_floor if args.year_floor > 0 else None
    result = run_search(
        args.query,
        candidate_limit=args.candidate_limit,
        top=args.top,
        profile=args.profile,
        boost_recency=args.boost_recency,
        min_citations=args.min_citations,
        year_floor=yf,
    )
    json.dump(result, sys.stdout, ensure_ascii=False, indent=2)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
