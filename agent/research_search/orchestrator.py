"""Research-search orchestration for Hermes tools and CodeAct recipes."""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from typing import Any
from urllib.parse import urlparse

from .store import (
    ResearchSearchStore,
    ResearchSearchUnavailableError,
    canonicalize_url,
    document_id_for_url,
    resolve_db_path,
    utc_now_iso,
)


def _load_config() -> dict[str, Any]:
    try:
        from hermes_cli.config import load_config

        cfg = load_config()
        return cfg if isinstance(cfg, dict) else {}
    except Exception:
        return {}


def _research_cfg(config: dict[str, Any] | None = None) -> dict[str, Any]:
    cfg = config if isinstance(config, dict) else _load_config()
    raw = cfg.get("research_search") or {}
    if not isinstance(raw, dict):
        raw = {}
    return {
        "enabled": bool(raw.get("enabled", True)),
        "db_path": raw.get("db_path", ""),
        "default_depth": str(raw.get("default_depth") or "thorough"),
        "max_queries": int(raw.get("max_queries") or 12),
        "max_pages": int(raw.get("max_pages") or 10),
        "cache_ttl_hours": int(raw.get("cache_ttl_hours") or 24),
        "crawler_policy": str(raw.get("crawler_policy") or "power_user"),
        "browser_fallback": bool(raw.get("browser_fallback", True)),
        "auto_index_research_results": bool(
            raw.get("auto_index_research_results", True)
        ),
        "vector": {
            "enabled": bool((raw.get("vector") or {}).get("enabled", True)),
            "provider": str(
                (raw.get("vector") or {}).get("provider")
                or "sentence_transformers"
            ),
            "model": str(
                (raw.get("vector") or {}).get("model")
                or "BAAI/bge-small-en-v1.5"
            ),
            "batch_size": int((raw.get("vector") or {}).get("batch_size") or 32),
            "chunk_chars": int((raw.get("vector") or {}).get("chunk_chars") or 1800),
            "chunk_overlap_chars": int(
                (raw.get("vector") or {}).get("chunk_overlap_chars") or 250
            ),
            "bm25_weight": float((raw.get("vector") or {}).get("bm25_weight") or 0.7),
            "vector_weight": float(
                (raw.get("vector") or {}).get("vector_weight") or 0.3
            ),
            "candidate_limit": int(
                (raw.get("vector") or {}).get("candidate_limit") or 80
            ),
        },
    }


def _store(config: dict[str, Any] | None = None) -> ResearchSearchStore:
    cfg = config if isinstance(config, dict) else _load_config()
    return ResearchSearchStore(resolve_db_path(cfg))


def _safe_json(raw: Any) -> dict[str, Any]:
    if isinstance(raw, dict):
        return raw
    if not isinstance(raw, str):
        return {"error": f"Expected JSON string, got {type(raw).__name__}"}
    try:
        parsed = json.loads(raw)
        return parsed if isinstance(parsed, dict) else {"data": parsed}
    except Exception as exc:
        return {"error": f"Malformed JSON result: {exc}", "raw": raw}


def dispatch_tool(name: str, args: dict[str, Any]) -> str:
    """Dispatch another Hermes tool through the live registry."""
    from tools.registry import registry

    return registry.dispatch(name, args)


def classify_topic_type(question: str) -> str:
    """Deterministic topic classifier for selecting search profiles."""
    q = str(question or "").lower()
    if any(k in q for k in ("today", "latest", "current", "breaking", "news")):
        return "current_events"
    if any(k in q for k in ("api", "docs", "github", "package", "library", "bug")):
        return "technical"
    if any(k in q for k in ("paper", "arxiv", "study", "journal", "citation")):
        return "academic"
    if any(k in q for k in ("law", "regulation", "statute", "court", "legal")):
        return "legal_regulatory"
    if any(k in q for k in ("price", "pricing", "company", "market", "filing")):
        return "company_market"
    if any(k in q for k in ("best", "review", "buy", "recommend", "compare")):
        return "product"
    if any(k in q for k in ("near me", "local", "restaurant", "weather")):
        return "local"
    if any(k in q for k in ("twitter", "x.com", "reddit", "social sentiment")):
        return "social"
    if any(k in q for k in ("obscure", "hard to find", "exact phrase")):
        return "obscure_lookup"
    return "general"


def _query(query: str, kind: str, vertical: str = "web") -> dict[str, str]:
    return {"query": query, "kind": kind, "vertical": vertical}


def generate_query_plan(
    question: str,
    topic_type: str = "auto",
    freshness: str = "auto",
    depth: str = "thorough",
) -> dict[str, Any]:
    """Generate typed fan-out queries and source requirements."""
    topic = topic_type if topic_type != "auto" else classify_topic_type(question)
    now_year = datetime.now(timezone.utc).year
    base = str(question or "").strip()
    queries: list[dict[str, str]] = [_query(base, "base")]
    source_requirements = ["independent"]

    if topic in {"current_events", "company_market", "product"} or freshness in {
        "latest",
        "recent",
    }:
        queries.append(_query(f"{base} latest {now_year}", "recent"))
        source_requirements.append("current")

    if topic in {"technical", "legal_regulatory", "company_market", "product"}:
        queries.append(_query(f"{base} official", "official"))
        source_requirements.append("official")

    if topic == "technical":
        queries.extend(
            [
                _query(f"{base} documentation changelog", "technical_docs"),
                _query(f"{base} GitHub issues breaking change", "community"),
            ]
        )
    elif topic == "academic":
        queries.extend(
            [
                _query(f"{base} survey paper recent", "academic"),
                _query(f"{base} benchmark dataset replication", "contradiction"),
            ]
        )
        source_requirements.extend(["primary", "contradiction"])
    elif topic == "legal_regulatory":
        queries.extend(
            [
                _query(f"{base} regulator guidance statute", "primary"),
                _query(f"{base} enforcement action case", "contradiction"),
            ]
        )
        source_requirements.extend(["primary", "official"])
    elif topic == "product":
        queries.extend(
            [
                _query(f"{base} professional review", "review"),
                _query(f"{base} complaints problems failure", "adversarial"),
            ]
        )
        source_requirements.append("adversarial")
    elif topic == "obscure_lookup":
        queries.extend(
            [
                _query(f'"{base}"', "exact_phrase"),
                _query(f"{base} archive cached", "archive"),
            ]
        )
    else:
        queries.append(_query(f"{base} official source", "official"))

    if depth == "thorough":
        queries.append(_query(f"{base} false outdated incorrect", "adversarial"))
        if "adversarial" not in source_requirements:
            source_requirements.append("adversarial")

    deduped: list[dict[str, str]] = []
    seen: set[str] = set()
    for item in queries:
        key = item["query"].lower()
        if key in seen:
            continue
        seen.add(key)
        deduped.append(item)

    return {
        "topic_type": topic,
        "freshness": freshness,
        "depth": depth,
        "queries": deduped,
        "source_requirements": sorted(set(source_requirements)),
    }


def _source_type(url: str, query_kind: str = "") -> str:
    host = urlparse(str(url or "")).netloc.lower()
    if query_kind in {"official", "primary", "technical_docs"}:
        return "official"
    if any(host.endswith(suffix) for suffix in (".gov", ".edu")):
        return "primary"
    if any(k in host for k in ("github.com", "docs.", "developer.", "readthedocs")):
        return "docs"
    if any(k in host for k in ("reddit", "stackoverflow", "news.ycombinator")):
        return "community"
    if any(k in host for k in ("nytimes", "reuters", "apnews", "bbc", "espn")):
        return "news"
    if query_kind in {"adversarial", "contradiction"}:
        return "adversarial"
    return "unknown"


def _quality_score(source_type: str, status: str) -> float:
    base = {
        "primary": 0.95,
        "official": 0.9,
        "docs": 0.82,
        "news": 0.75,
        "community": 0.55,
        "adversarial": 0.6,
        "unknown": 0.45,
    }.get(source_type, 0.45)
    if status == "failed":
        return 0.0
    if status == "search_only":
        return max(0.2, base - 0.25)
    return base


def _results_from_search(parsed: dict[str, Any]) -> list[dict[str, Any]]:
    web = ((parsed.get("data") or {}).get("web") or []) if parsed else []
    return [item for item in web if isinstance(item, dict)]


def _results_from_extract(parsed: dict[str, Any]) -> list[dict[str, Any]]:
    results = parsed.get("results") if isinstance(parsed, dict) else None
    return [item for item in results or [] if isinstance(item, dict)]


def _excerpt(text: str, limit: int = 1500) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip()[:limit]


def _browser_fallback(url: str) -> dict[str, Any]:
    try:
        nav = _safe_json(dispatch_tool("browser_navigate", {"url": url}))
        if nav.get("error"):
            return {"error": nav.get("error") or "browser_navigate failed"}
        snap = _safe_json(dispatch_tool("browser_snapshot", {}))
        text = snap.get("snapshot") or snap.get("content") or snap.get("text") or ""
        if text:
            return {"url": url, "title": "", "content": str(text)}
        return {"error": "browser snapshot returned no text"}
    except Exception as exc:
        return {"error": str(exc)}


def _trim_sources(sources: list[dict[str, Any]], max_sources: int) -> list[dict[str, Any]]:
    return _rank_sources(sources)[:max_sources]


def _host(url: str) -> str:
    return urlparse(str(url or "")).netloc.lower()


def _rank_sources(sources: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Rank sources with extraction status, relevance, quality, and domain diversity."""
    domain_seen: set[str] = set()
    ranked = sorted(
        sources,
        key=lambda s: (
            1 if s.get("status") in {"extracted", "local_cache"} else 0,
            float(s.get("relevance_score") or 0),
            float(s.get("source_quality_score") or 0),
            1 if s.get("source_type") in {"official", "primary", "docs"} else 0,
            len(str(s.get("content") or s.get("excerpt") or "")),
        ),
        reverse=True,
    )
    diversified: list[dict[str, Any]] = []
    overflow: list[dict[str, Any]] = []
    for source in ranked:
        host = _host(source.get("url") or "")
        if host and host in domain_seen:
            overflow.append(source)
            continue
        if host:
            domain_seen.add(host)
        diversified.append(source)
    return diversified + overflow


def _analyze_gaps(
    question: str,
    sources: list[dict[str, Any]],
    plan: dict[str, Any] | None = None,
) -> dict[str, Any]:
    plan = plan or generate_query_plan(question)
    requirements = set(plan.get("source_requirements") or [])
    usable = [s for s in sources if s.get("status") != "failed"]
    extracted = [s for s in usable if s.get("status") in {"extracted", "local_cache"}]
    domains = {_host(s.get("url") or "") for s in usable if _host(s.get("url") or "")}
    source_types = {s.get("source_type") for s in usable}
    gaps: list[str] = []
    next_queries: list[dict[str, str]] = []

    if not usable:
        gaps.append("No sources were gathered.")
        next_queries.append(_query(str(question), "base_retry"))
    if len(extracted) < 2:
        gaps.append("Fewer than two extracted/cache-backed sources were gathered.")
    if len(domains) < 2 and len(usable) >= 2:
        gaps.append("Sources are not domain-diverse.")
        next_queries.append(_query(f"{question} independent analysis", "independent"))
    if "official" in requirements and not ({"official", "primary", "docs"} & source_types):
        gaps.append("No official or primary source was gathered.")
        next_queries.append(_query(f"{question} official primary source", "official"))
    if "adversarial" in requirements and "adversarial" not in source_types:
        gaps.append("No adversarial/contradiction source was gathered.")
        next_queries.append(_query(f"{question} problems criticism outdated incorrect", "adversarial"))
    if plan.get("topic_type") in {"current_events", "company_market"} and not any(
        s.get("query_kind") == "recent" for s in usable
    ):
        gaps.append("No explicitly recent source lane was gathered.")
        next_queries.append(_query(f"{question} latest {datetime.now(timezone.utc).year}", "recent"))

    failed = [s for s in sources if s.get("status") == "failed"]
    if failed and len(failed) >= max(2, len(sources) // 2):
        gaps.append("High extraction failure rate; browser fallback or alternate sources may be needed.")

    deduped_queries: list[dict[str, str]] = []
    seen: set[str] = set()
    for item in next_queries:
        key = item["query"].lower()
        if key not in seen:
            seen.add(key)
            deduped_queries.append(item)

    return {
        "gaps": gaps,
        "next_queries": deduped_queries,
        "metrics": {
            "sources": len(sources),
            "usable_sources": len(usable),
            "extracted_sources": len(extracted),
            "independent_domains": len(domains),
            "failed_sources": len(failed),
        },
    }


def _apply_gap_pass(bundle: dict[str, Any]) -> list[str]:
    return _analyze_gaps(
        str(bundle.get("question") or ""),
        list(bundle.get("sources") or []),
        bundle.get("plan") or {},
    )["gaps"]


def _web_backend_status() -> dict[str, Any]:
    try:
        from tools import web_tools

        backend = web_tools._get_backend()
        searxng_cfg = web_tools._searxng_config()
        return {
            "active_backend": backend,
            "searxng": {
                "enabled": searxng_cfg["enabled"],
                "base_url": searxng_cfg["base_url"],
                "reachable": web_tools._is_searxng_available(),
            },
        }
    except Exception as exc:
        return {"active_backend": "unknown", "error": str(exc)}


def _hybrid_rank_local_chunks(
    query: str,
    chunks: list[dict[str, Any]],
    store: ResearchSearchStore,
    cfg: dict[str, Any],
) -> list[dict[str, Any]]:
    if not chunks:
        return []
    vector_cfg = cfg.get("vector") or {}
    bm25_weight = float(vector_cfg.get("bm25_weight") or 0.7)
    vector_weight = float(vector_cfg.get("vector_weight") or 0.3)
    ranked = [dict(item) for item in chunks]

    try:
        from . import embeddings

        status = embeddings.vector_status(cfg)
        if not status.get("available"):
            raise RuntimeError(status.get("error") or "vectors unavailable")
        query_vec = embeddings.embed_texts([query], cfg)[0]
        chunk_ids = [str(item.get("chunk_id") or "") for item in ranked]
        stored = store.get_chunk_embeddings(
            chunk_ids,
            str(vector_cfg.get("provider") or "sentence_transformers"),
            str(vector_cfg.get("model") or "BAAI/bge-small-en-v1.5"),
        )
        for item in ranked:
            emb = stored.get(str(item.get("chunk_id") or ""))
            vector_score = 0.0
            if emb:
                vector_score = embeddings.cosine_similarity(
                    query_vec,
                    embeddings.blob_to_vector(emb["vector"]),
                )
            bm25 = float(item.get("score") or 0.0)
            item["bm25_score"] = bm25
            item["vector_score"] = vector_score
            item["score"] = (bm25_weight * bm25) + (vector_weight * vector_score)
    except Exception:
        for item in ranked:
            item["bm25_score"] = float(item.get("score") or 0.0)
            item["vector_score"] = None

    return sorted(ranked, key=lambda item: float(item.get("score") or 0.0), reverse=True)


def _index_document(
    store: ResearchSearchStore,
    source: dict[str, Any],
    query: str,
    cfg: dict[str, Any],
) -> dict[str, int]:
    counts = {"documents": 0, "chunks": 0, "embeddings": 0, "evidence": 0}
    content = str(source.get("content") or "")
    if not content:
        return counts
    doc = store.upsert_document(
        {
            "url": source.get("url") or "",
            "title": source.get("title") or "",
            "content": content,
            "vertical": source.get("vertical") or "web",
            "source_type": source.get("source_type") or "unknown",
            "status": source.get("status") or "extracted",
            "error": source.get("error") or "",
            "metadata": {
                "query": source.get("query") or query,
                "query_kind": source.get("query_kind") or "",
                "source_backend": source.get("source_backend") or "",
            },
        }
    )
    counts["documents"] = 1

    if hasattr(store, "upsert_chunks"):
        vector_cfg = cfg.get("vector") or {}
        chunks = store.upsert_chunks(
            doc["id"],
            content,
            chunk_chars=int(vector_cfg.get("chunk_chars") or 1800),
            overlap_chars=int(vector_cfg.get("chunk_overlap_chars") or 250),
            metadata={"url": doc["url"], "title": doc["title"]},
        )
        counts["chunks"] = len(chunks)
    else:
        chunks = []

    if hasattr(store, "upsert_evidence"):
        store.upsert_evidence(
            {
                "document_id": doc["id"],
                "chunk_id": chunks[0]["id"] if chunks else "",
                "query": query,
                "claim": source.get("title") or "",
                "excerpt": source.get("excerpt") or _excerpt(content),
                "relevance_score": source.get("relevance_score") or 0.0,
                "source_quality_score": source.get("source_quality_score") or 0.0,
                "confidence": 0.6 if source.get("status") == "extracted" else 0.35,
                "metadata": {"url": doc["url"]},
            }
        )
        counts["evidence"] = 1

    try:
        from . import embeddings

        if chunks and embeddings.vector_status(cfg).get("available"):
            texts = [chunk["text"] for chunk in chunks]
            vectors = embeddings.embed_texts(texts, cfg)
            provider = str((cfg.get("vector") or {}).get("provider") or "sentence_transformers")
            model = str((cfg.get("vector") or {}).get("model") or "BAAI/bge-small-en-v1.5")
            for chunk, vector in zip(chunks, vectors):
                blob, dim = embeddings.vector_to_blob(vector)
                store.upsert_embedding(chunk["id"], provider, model, blob, dim)
                counts["embeddings"] += 1
    except Exception:
        pass

    return counts


def research_local_search(
    query: str,
    vertical: str = "auto",
    limit: int = 10,
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    try:
        cfg = _research_cfg(config)
        store = _store(config)
        try:
            chunk_limit = min(
                max(int(limit or 10) * 3, int(limit or 10)),
                int((cfg.get("vector") or {}).get("candidate_limit") or 80),
            )
            chunk_results = store.search_chunks(
                query, limit=chunk_limit, vertical=vertical
            )
            results = _hybrid_rank_local_chunks(query, chunk_results, store, cfg)[
                : int(limit or 10)
            ]
        except ResearchSearchUnavailableError:
            results = store.search(query, limit=limit, vertical=vertical)
        return {"success": True, "results": results}
    except ResearchSearchUnavailableError as exc:
        return {"success": False, "error": str(exc), "results": []}


def research_index_url(
    urls: list[str],
    vertical: str = "web",
    force_refresh: bool = False,
    render_mode: str = "auto",
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    del force_refresh, render_mode  # Reserved for cache/renderer expansion.
    cfg = _research_cfg(config)
    store = _store(config)
    try:
        store.connect()
    except ResearchSearchUnavailableError as exc:
        return {"success": False, "error": str(exc), "indexed": []}

    raw = dispatch_tool("web_extract", {"urls": list(urls or [])[:5]})
    parsed = _safe_json(raw)
    if parsed.get("error"):
        return {"success": False, "error": parsed["error"], "indexed": []}

    indexed = []
    for result in _results_from_extract(parsed):
        url = result.get("url") or ""
        status = "failed" if result.get("error") else "extracted"
        source = {
            "url": url,
            "title": result.get("title") or "",
            "content": result.get("content") or "",
            "status": status,
            "source_type": _source_type(url),
            "vertical": vertical,
            "excerpt": _excerpt(result.get("content") or ""),
            "relevance_score": 1.0 if status == "extracted" else 0.0,
            "source_quality_score": _quality_score(_source_type(url), status),
            "error": result.get("error") or "",
        }
        counts = _index_document(store, source, url, cfg)
        indexed.append(
            {
                "url": url,
                "id": document_id_for_url(url),
                "status": status,
                "chunks": counts["chunks"],
                "embeddings": counts["embeddings"],
            }
        )
    return {"success": True, "indexed": indexed}


def research_status(config: dict[str, Any] | None = None) -> dict[str, Any]:
    cfg = _research_cfg(config)
    status = _store(config).status()
    try:
        from . import embeddings

        vector_status = embeddings.vector_status(cfg)
    except Exception as exc:
        vector_status = {
            "enabled": bool((cfg.get("vector") or {}).get("enabled", True)),
            "available": False,
            "provider": str((cfg.get("vector") or {}).get("provider") or ""),
            "model": str((cfg.get("vector") or {}).get("model") or ""),
            "error": str(exc),
        }
    status["enabled"] = cfg["enabled"]
    status["crawler_policy"] = cfg["crawler_policy"]
    status["auto_index_research_results"] = cfg["auto_index_research_results"]
    status["web"] = _web_backend_status()
    status["vector"] = vector_status
    return status


def research_plan(
    question: str,
    topic_type: str = "auto",
    freshness: str = "auto",
    depth: str = "thorough",
) -> dict[str, Any]:
    return {
        "success": True,
        "question": question,
        "plan": generate_query_plan(question, topic_type, freshness, depth),
        "workflow": [
            "Use local research memory first.",
            "Fan out typed discovery queries.",
            "Extract source pages before answering.",
            "Run gap analysis and one targeted gap pass for balanced/thorough depth.",
            "Synthesize final prose from the evidence bundle outside run_code.",
        ],
    }


def research_search_candidates(
    question: str = "",
    query: str = "",
    topic_type: str = "auto",
    freshness: str = "auto",
    depth: str = "balanced",
    max_queries: int | None = None,
    limit: int = 20,
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    cfg = _research_cfg(config)
    base = (query or question or "").strip()
    if not base:
        return {"success": False, "error": "query or question is required", "candidates": []}
    plan = generate_query_plan(base, topic_type, freshness, depth)
    queries = plan["queries"][: int(max_queries or min(cfg["max_queries"], 4))]
    candidates: dict[str, dict[str, Any]] = {}
    errors: list[str] = []
    local = research_local_search(base, limit=min(limit, 10), config=config)
    if local.get("success"):
        for item in local.get("results") or []:
            url = canonicalize_url(item.get("url") or "")
            if url:
                candidates[url] = {**item, "candidate_source": "local"}
    elif local.get("error"):
        errors.append(f"local_search: {local['error']}")

    for item in queries:
        raw = dispatch_tool("web_search", {"query": item["query"], "limit": min(limit, 20)})
        parsed = _safe_json(raw)
        if parsed.get("error"):
            errors.append(f"web_search({item['kind']}): {parsed['error']}")
            continue
        for result in _results_from_search(parsed):
            url = canonicalize_url(result.get("url") or "")
            if not url or url in candidates:
                continue
            candidates[url] = {
                **result,
                "url": url,
                "candidate_source": "web",
                "query": item["query"],
                "query_kind": item["kind"],
                "source_type": _source_type(url, item["kind"]),
            }
            if len(candidates) >= limit:
                break
        if len(candidates) >= limit:
            break
    return {
        "success": True,
        "question": question or query,
        "plan": plan,
        "candidates": list(candidates.values())[:limit],
        "errors": errors,
    }


def research_extract_evidence(
    urls: list[str],
    question: str = "",
    max_sources: int = 5,
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    del config
    raw = dispatch_tool("web_extract", {"urls": list(urls or [])[: int(max_sources or 5)]})
    parsed = _safe_json(raw)
    if parsed.get("error"):
        return {"success": False, "error": parsed["error"], "evidence": []}
    evidence: list[dict[str, Any]] = []
    for result in _results_from_extract(parsed):
        url = canonicalize_url(result.get("url") or "")
        status = "failed" if result.get("error") else "extracted"
        source_type = _source_type(url)
        content = result.get("content") or ""
        evidence.append(
            {
                "url": url,
                "title": result.get("title") or "",
                "status": status,
                "source_type": source_type,
                "excerpt": _excerpt(content),
                "relevance_score": 1.0 if status == "extracted" else 0.0,
                "source_quality_score": _quality_score(source_type, status),
                "error": result.get("error") or None,
                "question": question,
            }
        )
    return {"success": True, "evidence": _rank_sources(evidence)}


def research_rerank(
    question: str,
    sources: list[dict[str, Any]] | None = None,
    limit: int = 10,
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    del question, config
    ranked = _rank_sources([s for s in sources or [] if isinstance(s, dict)])
    return {"success": True, "sources": ranked[: int(limit or 10)]}


def research_gap_analyze(
    question: str,
    sources: list[dict[str, Any]] | None = None,
    plan: dict[str, Any] | None = None,
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    del config
    if plan is None:
        plan = generate_query_plan(question)
    result = _analyze_gaps(question, [s for s in sources or [] if isinstance(s, dict)], plan)
    return {"success": True, "plan": plan, **result}


def research_help(topic_type: str = "auto") -> dict[str, Any]:
    topic = topic_type if topic_type != "auto" else "general"
    return {
        "success": True,
        "recommended_default": "research_gather(question=...)",
        "recipe": "In CodeAct, prefer research_web(question=...) for source-grounded web tasks.",
        "tools": {
            "research_plan": "Inspect query lanes and source requirements without fetching.",
            "research_search_candidates": "Discovery only: local memory plus live web search.",
            "research_extract_evidence": "Extract page contents for selected URLs.",
            "research_rerank": "Rank already gathered candidate/source dicts.",
            "research_gap_analyze": "Check source diversity, extraction, official/primary, and adversarial gaps.",
            "research_status": "Inspect backend, DuckDB, FTS, vector, and auto-index status.",
        },
        "topic_type": topic,
    }


def research_gather(
    question: str,
    topic_type: str = "auto",
    freshness: str = "auto",
    depth: str = "thorough",
    max_queries: int | None = None,
    max_pages: int | None = None,
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    cfg = _research_cfg(config)
    if not cfg["enabled"]:
        return {"success": False, "error": "research_search is disabled"}

    max_queries = int(max_queries or cfg["max_queries"])
    max_pages = int(max_pages or cfg["max_pages"])
    depth = depth or cfg["default_depth"]
    plan = generate_query_plan(question, topic_type, freshness, depth)
    queries = plan["queries"][:max_queries]
    usage = {
        "search_calls": 0,
        "extract_calls": 0,
        "browser_fallbacks": 0,
        "indexed_documents": 0,
        "indexed_chunks": 0,
        "indexed_embeddings": 0,
        "indexed_evidence": 0,
        "gap_passes": 0,
    }
    errors: list[str] = []
    sources: list[dict[str, Any]] = []

    local = research_local_search(
        question,
        vertical="auto",
        limit=min(max_pages, 10),
        config=config,
    )
    if local.get("success"):
        for result in local.get("results") or []:
            source_type = result.get("source_type") or "unknown"
            sources.append(
                {
                    "title": result.get("title") or "",
                    "url": result.get("url") or "",
                    "status": "local_cache",
                    "source_type": source_type,
                    "excerpt": result.get("excerpt") or "",
                    "content": result.get("excerpt") or "",
                    "relevance_score": float(result.get("score") or 0.0),
                    "source_quality_score": _quality_score(source_type, "local_cache"),
                    "error": None,
                }
            )
    elif local.get("error"):
        errors.append(f"local_search: {local['error']}")

    discovered: dict[str, dict[str, Any]] = {}
    for query_item in queries:
        if len(discovered) >= max_pages * 2:
            break
        raw = dispatch_tool(
            "web_search",
            {"query": query_item["query"], "limit": min(10, max_pages * 2)},
        )
        usage["search_calls"] += 1
        parsed = _safe_json(raw)
        if parsed.get("error"):
            errors.append(f"web_search({query_item['kind']}): {parsed['error']}")
            continue
        for item in _results_from_search(parsed):
            url = canonicalize_url(item.get("url") or "")
            if not url or url in discovered:
                continue
            discovered[url] = {
                **item,
                "query": query_item["query"],
                "query_kind": query_item["kind"],
                "source_type": _source_type(url, query_item["kind"]),
            }

    urls_to_extract = list(discovered)[:max_pages]
    extract_results: list[dict[str, Any]] = []
    for idx in range(0, len(urls_to_extract), 5):
        batch = urls_to_extract[idx : idx + 5]
        raw = dispatch_tool("web_extract", {"urls": batch})
        usage["extract_calls"] += 1
        parsed = _safe_json(raw)
        if parsed.get("error"):
            errors.append(f"web_extract: {parsed['error']}")
            continue
        extract_results.extend(_results_from_extract(parsed))

    by_url = {canonicalize_url(r.get("url") or ""): r for r in extract_results}
    store = None
    store_ready = False
    if cfg["auto_index_research_results"]:
        store = _store(config)
        try:
            store.connect()
            store_ready = True
        except ResearchSearchUnavailableError:
            store_ready = False

    for url, meta in discovered.items():
        extracted = by_url.get(url)
        status = "search_only"
        title = meta.get("title") or ""
        content = meta.get("description") or ""
        error = None
        if extracted:
            title = extracted.get("title") or title
            content = extracted.get("content") or content
            error = extracted.get("error")
            status = "failed" if error else "extracted"

        if (
            status != "extracted"
            and cfg["browser_fallback"]
            and cfg["crawler_policy"] == "power_user"
        ):
            fallback = _browser_fallback(url)
            usage["browser_fallbacks"] += 1
            if fallback.get("content"):
                title = fallback.get("title") or title
                content = fallback.get("content") or content
                error = None
                status = "extracted"
            elif fallback.get("error"):
                error = error or fallback["error"]

        source_type = meta.get("source_type") or _source_type(url)
        source = {
            "title": title,
            "url": url,
            "status": status,
            "source_type": source_type,
            "excerpt": _excerpt(content),
            "content": content[:5000] if isinstance(content, str) else "",
            "relevance_score": 1.0 if status == "extracted" else 0.45,
            "source_quality_score": _quality_score(source_type, status),
            "query": meta.get("query") or "",
            "query_kind": meta.get("query_kind") or "",
            "source_backend": meta.get("source_backend") or "",
            "error": error,
        }
        sources.append(source)

        if store_ready and store is not None and content:
            try:
                counts = _index_document(
                    store,
                    {**source, "content": content if isinstance(content, str) else ""},
                    question,
                    cfg,
                )
                usage["indexed_documents"] += counts["documents"]
                usage["indexed_chunks"] += counts["chunks"]
                usage["indexed_embeddings"] += counts["embeddings"]
                usage["indexed_evidence"] += counts["evidence"]
            except Exception:
                store_ready = False

    initial_gap_report = _analyze_gaps(question, sources, plan)
    if (
        depth != "fast"
        and initial_gap_report["next_queries"]
        and usage["search_calls"] < max_queries
        and len(sources) < max_pages
    ):
        remaining_pages = max(max_pages - len(sources), 0)
        gap_discovered: dict[str, dict[str, Any]] = {}
        for query_item in initial_gap_report["next_queries"]:
            if usage["search_calls"] >= max_queries or len(gap_discovered) >= remaining_pages:
                break
            raw = dispatch_tool(
                "web_search",
                {"query": query_item["query"], "limit": min(10, max(remaining_pages, 1))},
            )
            usage["search_calls"] += 1
            parsed = _safe_json(raw)
            if parsed.get("error"):
                errors.append(f"web_search_gap({query_item['kind']}): {parsed['error']}")
                continue
            for item in _results_from_search(parsed):
                url = canonicalize_url(item.get("url") or "")
                if not url or url in discovered or url in gap_discovered:
                    continue
                gap_discovered[url] = {
                    **item,
                    "query": query_item["query"],
                    "query_kind": query_item["kind"],
                    "source_type": _source_type(url, query_item["kind"]),
                }
                if len(gap_discovered) >= remaining_pages:
                    break
        if gap_discovered:
            usage["gap_passes"] += 1
            gap_urls = list(gap_discovered)[:remaining_pages]
            gap_extract_results: list[dict[str, Any]] = []
            for idx in range(0, len(gap_urls), 5):
                batch = gap_urls[idx : idx + 5]
                raw = dispatch_tool("web_extract", {"urls": batch})
                usage["extract_calls"] += 1
                parsed = _safe_json(raw)
                if parsed.get("error"):
                    errors.append(f"web_extract_gap: {parsed['error']}")
                    continue
                gap_extract_results.extend(_results_from_extract(parsed))
            gap_by_url = {canonicalize_url(r.get("url") or ""): r for r in gap_extract_results}
            for url, meta in gap_discovered.items():
                extracted = gap_by_url.get(url)
                status = "search_only"
                title = meta.get("title") or ""
                content = meta.get("description") or ""
                error = None
                if extracted:
                    title = extracted.get("title") or title
                    content = extracted.get("content") or content
                    error = extracted.get("error")
                    status = "failed" if error else "extracted"
                source_type = meta.get("source_type") or _source_type(url)
                source = {
                    "title": title,
                    "url": url,
                    "status": status,
                    "source_type": source_type,
                    "excerpt": _excerpt(content),
                    "content": content[:5000] if isinstance(content, str) else "",
                    "relevance_score": 0.85 if status == "extracted" else 0.4,
                    "source_quality_score": _quality_score(source_type, status),
                    "query": meta.get("query") or "",
                    "query_kind": meta.get("query_kind") or "",
                    "source_backend": meta.get("source_backend") or "",
                    "error": error,
                }
                sources.append(source)
                if store_ready and store is not None and content:
                    try:
                        counts = _index_document(
                            store,
                            {**source, "content": content if isinstance(content, str) else ""},
                            question,
                            cfg,
                        )
                        usage["indexed_documents"] += counts["documents"]
                        usage["indexed_chunks"] += counts["chunks"]
                        usage["indexed_embeddings"] += counts["embeddings"]
                        usage["indexed_evidence"] += counts["evidence"]
                    except Exception:
                        store_ready = False

    sources = _trim_sources(sources, max_pages)
    bundle = {
        "success": True,
        "question": question,
        "topic_type": plan["topic_type"],
        "freshness": freshness,
        "depth": depth,
        "plan": plan,
        "sources": sources,
        "gaps": [],
        "conflicts": [],
        "errors": errors,
        "usage": usage,
    }
    bundle["gaps"] = _apply_gap_pass(bundle)

    if cfg["auto_index_research_results"] and store_ready and store is not None:
        try:
            store.record_run(
                {
                    "question": question,
                    "topic_type": bundle["topic_type"],
                    "freshness": freshness,
                    "depth": depth,
                    "plan": plan,
                    "gaps": bundle["gaps"],
                    "created_at": utc_now_iso(),
                }
            )
        except Exception:
            pass

    return bundle
