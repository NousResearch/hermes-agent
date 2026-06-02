#!/usr/bin/env python3
"""Rank papers for a topic via Semantic Scholar + OpenAlex (+ arXiv backfill)."""

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
_OPENALEX_SEARCH = "https://api.openalex.org/works"
_ARXIV_NS = {"a": "http://www.w3.org/2005/Atom"}
_WEIGHTS = {"rel": 0.35, "cite": 0.30, "infl": 0.15, "rec": 0.15, "oa": 0.05}
_BOOL_TOKEN_RE = re.compile(r'"[^"]+"|\(|\)|\bAND\b|\bOR\b|\bNOT\b|[^\s()]+', re.I)
_QUERY_STOPWORDS = {"and", "or", "not"}


def _get_json(
    url: str,
    timeout: int = 45,
    *,
    retries: int = 2,
    headers: dict[str, str] | None = None,
) -> dict:
    req_headers = {"User-Agent": "hermes-paper-search/1.0"}
    if headers:
        req_headers.update(headers)
    req = urllib.request.Request(url, headers=req_headers)
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


def _arxiv_id_from_openalex_ids(ids: dict | None) -> str:
    if not isinstance(ids, dict):
        return ""
    for key in ("arxiv", "ArXiv"):
        raw = ids.get(key)
        if not raw:
            continue
        m = re.search(r"(?:arxiv\.org/(?:abs|pdf)/)?(\d{4}\.\d{4,5}(?:v\d+)?)", str(raw), re.I)
        if m:
            return m.group(1)
    return ""


def _openalex_search_url(query: str, *, limit: int) -> str:
    params = {
        "search": query,
        "per-page": str(min(50, max(5, limit))),
        "filter": "is_paratext:false",
    }
    api_key = os.environ.get("OPENALEX_API_KEY", "").strip()
    mailto = (
        os.environ.get("OPENALEX_MAILTO", "").strip()
        or os.environ.get("CROSSREF_MAILTO", "").strip()
    )
    if api_key:
        params["api_key"] = api_key
    if mailto:
        params["mailto"] = mailto
    return f"{_OPENALEX_SEARCH}?{urllib.parse.urlencode(params)}"


def _keyword_rel(query: str, title: str, abstract: str) -> float:
    qtokens = {t.lower() for t in re.findall(r"[\w\u4e00-\u9fff]{2,}", query)}
    if not qtokens:
        return 0.5
    text = f"{title} {abstract}".lower()
    hit = sum(1 for t in qtokens if t in text)
    return min(1.0, hit / max(1, len(qtokens)))


def _normalize_match_text(text: str) -> str:
    text = (text or "").lower()
    text = re.sub(r"[^a-z0-9\u4e00-\u9fff]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return f" {text} "


def _term_variants(term: str) -> list[str]:
    raw = (term or "").strip().strip('"').strip()
    if not raw:
        return []
    variants = {raw.lower()}
    simplified = re.sub(r"[^a-z0-9\u4e00-\u9fff]+", " ", raw.lower()).strip()
    if simplified:
        variants.add(simplified)
    if "." in raw:
        variants.add(raw.lower().replace(".", " "))
        variants.add(raw.lower().replace(".", ""))
    if "-" in raw:
        variants.add(raw.lower().replace("-", " "))
        variants.add(raw.lower().replace("-", ""))
    return [v for v in variants if v]


def _extract_clause_terms(clause: str) -> list[str]:
    terms: list[str] = []
    for tok in _BOOL_TOKEN_RE.findall(clause):
        up = tok.upper()
        if up in {"AND", "OR", "NOT", "(", ")"}:
            continue
        term = tok.strip().strip('"').strip()
        if term:
            terms.append(term)
    return terms


def parse_query_spec(query: str) -> dict:
    tokens = _BOOL_TOKEN_RE.findall(query or "")
    if not tokens:
        return {"positive_groups": [], "negative_terms": [], "search_query": (query or "").strip()}
    groups: list[tuple[bool, str]] = []
    current: list[str] = []
    depth = 0
    next_negated = False
    i = 0
    while i < len(tokens):
        tok = tokens[i]
        up = tok.upper()
        if tok == "(":
            depth += 1
            current.append(tok)
        elif tok == ")":
            depth = max(0, depth - 1)
            current.append(tok)
        elif depth == 0 and up == "AND":
            clause = " ".join(current).strip()
            if clause:
                groups.append((next_negated, clause))
            current = []
            next_negated = i + 1 < len(tokens) and tokens[i + 1].upper() == "NOT"
            if next_negated:
                i += 1
        else:
            current.append(tok)
        i += 1
    clause = " ".join(current).strip()
    if clause:
        groups.append((next_negated, clause))
    positive_groups: list[list[str]] = []
    negative_terms: list[str] = []
    for negated, clause_text in groups:
        terms = _extract_clause_terms(clause_text)
        if not terms:
            continue
        if negated:
            negative_terms.extend(terms)
        else:
            positive_groups.append(terms)
    if not positive_groups and not negative_terms:
        positive_groups = [_extract_clause_terms(query)]
    search_terms: list[str] = []
    for group in positive_groups:
        for term in group:
            low = term.lower()
            if low in _QUERY_STOPWORDS or low in search_terms:
                continue
            search_terms.append(term)
    return {
        "positive_groups": positive_groups,
        "negative_terms": negative_terms,
        "search_query": " ".join(search_terms).strip() or (query or "").strip(),
    }


def build_candidate_queries(spec: dict, *, max_queries: int = 6) -> list[str]:
    base = (spec.get("search_query") or "").strip()
    queries: list[str] = [base] if base else []
    groups = spec.get("positive_groups") or []
    if len(groups) < 2:
        return queries or [base]
    anchors = groups[0][:max_queries]
    context_terms: list[str] = []
    for group in groups[1:]:
        for term in group[:2]:
            if term not in context_terms:
                context_terms.append(term)
    context = " ".join(context_terms).strip()
    for anchor in anchors:
        q = " ".join(part for part in (anchor, context) if part).strip()
        if q and q not in queries:
            queries.append(q)
    return queries[: max_queries + 1]


def _paper_matches_query_spec(paper: dict, spec: dict) -> bool:
    text = _normalize_match_text(f"{paper.get('title', '')} {paper.get('abstract', '')}")
    for group in spec.get("positive_groups") or []:
        if group and not any(any(f" {v} " in text for v in _term_variants(term)) for term in group):
            return False
    for term in spec.get("negative_terms") or []:
        if any(f" {v} " in text for v in _term_variants(term)):
            return False
    return True


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
    headers = {}
    key = os.environ.get("SEMANTIC_SCHOLAR_API_KEY", "").strip()
    if key:
        headers["x-api-key"] = key
    data = _get_json(url, headers=headers)
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


def search_openalex(query: str, *, limit: int = 25) -> list[dict]:
    url = _openalex_search_url(query, limit=limit)
    data = _get_json(url, headers={"User-Agent": "hermes-paper-search/1.1"})
    papers = []
    for item in data.get("results") or []:
        ids = item.get("ids") or {}
        arxiv = _arxiv_id_from_openalex_ids(ids)
        authors = []
        for authorship in (item.get("authorships") or [])[:6]:
            if not isinstance(authorship, dict):
                continue
            author = authorship.get("author") or {}
            name = str(author.get("display_name") or "").strip()
            if name:
                authors.append(name)
        best_oa = item.get("best_oa_location") or {}
        primary_loc = item.get("primary_location") or {}
        pdf_url = str(best_oa.get("pdf_url") or primary_loc.get("pdf_url") or "").strip()
        landing_url = str(
            best_oa.get("landing_page_url")
            or primary_loc.get("landing_page_url")
            or item.get("id")
            or ""
        ).strip()
        venue = ""
        source = primary_loc.get("source") or {}
        if isinstance(source, dict):
            venue = str(source.get("display_name") or "").strip()
        papers.append({
            "source": "openalex",
            "paper_id": str(item.get("id") or "").rsplit("/", 1)[-1],
            "title": str(item.get("display_name") or "").strip(),
            "year": item.get("publication_year"),
            "abstract": _openalex_abstract(item.get("abstract_inverted_index"))[:800],
            "citation_count": int(item.get("cited_by_count") or 0),
            "influential_citation_count": 0,
            "relevance": 0.0,
            "url": landing_url,
            "arxiv_id": arxiv,
            "arxiv_abs": f"https://arxiv.org/abs/{arxiv}" if arxiv else "",
            "pdf_url": pdf_url,
            "venue": venue,
            "authors": authors,
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
    spec = parse_query_spec(query)
    search_query = spec["search_query"]
    search_queries = build_candidate_queries(spec)
    per_query_limit = min(
        candidate_limit,
        max(8, math.ceil(candidate_limit / max(1, len(search_queries))) + 2),
    )

    papers: list[dict] = []
    for provider_query in search_queries:
        try:
            papers.extend(search_semantic_scholar(provider_query, limit=per_query_limit))
        except urllib.error.HTTPError as exc:
            if exc.code != 429:
                raise
        if len(dedupe_papers(papers)) >= candidate_limit:
            break
    papers = list(papers)
    if len(papers) < max(10, top):
        for provider_query in search_queries:
            try:
                papers.extend(search_openalex(provider_query, limit=min(per_query_limit, 25)))
            except Exception:
                pass
            if len(dedupe_papers(papers)) >= candidate_limit:
                break
    if len(papers) < max(10, top):
        time.sleep(0.3)
        for provider_query in search_queries:
            try:
                papers.extend(search_arxiv_backfill(provider_query, limit=arxiv_backfill))
            except Exception:
                pass
            if len(dedupe_papers(papers)) >= candidate_limit:
                break

    papers = [p for p in papers if p.get("title") and p.get("source") != "error"]
    papers = dedupe_papers(papers)
    papers = [p for p in papers if _paper_matches_query_spec(p, spec)]
    ranked = rank_papers(
        search_query,
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
