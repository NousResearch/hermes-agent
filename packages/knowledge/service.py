"""KnowledgeService — Hermes' orchestration layer over KnowledgeProvider (Step 4).

Owns: provider choice, fallback, merge, rerank, cache, retry, timeout, logging.
Owns NO reasoning and NO provider protocol details.
"""
from __future__ import annotations

import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeout
from typing import Any, Dict, Iterable, List, Optional

from .cache import TTLCache, cache_key
from .config import KnowledgeConfig
from .provider import KnowledgeProvider
from .providers import build_provider
import os

from .types import Chunk, Document, HealthStatus, IndexResult, SearchResult

logger = logging.getLogger("hermes.knowledge")

_RETRIEVAL_HINTS = (
    "what do i know", "what have i learned", "my notes", "obsidian", "vault",
    "architecture decision", "adr", "documentation", "docs", "readme",
    "meeting notes", "similar notes", "in my knowledge", "second brain",
    "according to my", "previous conversation", "we discussed", "project doc",
    "summarize all", "find every", "code snippet",
)


class KnowledgeService:
    def __init__(
        self,
        config: Optional[KnowledgeConfig] = None,
        provider: Optional[KnowledgeProvider] = None,
        providers: Optional[List[KnowledgeProvider]] = None,
    ):
        self.config = config or KnowledgeConfig.load()
        if providers:
            self._providers = list(providers)
        elif provider is not None:
            self._providers = [provider]
        else:
            self._providers = self._build_providers()
        self.cache = TTLCache(self.config.cache_size, self.config.cache_ttl)
        self._pool = ThreadPoolExecutor(max_workers=4,
                                        thread_name_prefix="knowledge")
        self._lock = threading.RLock()
        self.stats: Dict[str, int] = {"searches": 0, "cache_hits": 0,
                                      "retries": 0, "failures": 0, "timeouts": 0}

    # -- provider selection --------------------------------------------
    def _build_providers(self) -> List[KnowledgeProvider]:
        names = [self.config.provider] + list(self.config.fallback_providers)
        built: List[KnowledgeProvider] = []
        for n in names:
            try:
                built.append(build_provider(n, **self.config.options_for(n)))
            except Exception as exc:
                logger.warning("knowledge: provider %r unavailable: %s", n, exc)
        if not built:
            built.append(build_provider("local", **self.config.options_for("local")))
        return built

    @property
    def providers(self) -> List[KnowledgeProvider]:
        return list(self._providers)

    @property
    def primary(self) -> KnowledgeProvider:
        return self._providers[0]

    def set_provider(self, provider: KnowledgeProvider) -> None:
        """Hot-swap the backend. Hermes above this line is unaffected."""
        with self._lock:
            self._providers = [provider] + self._providers[1:]
            self.cache.clear()

    # -- resilience ----------------------------------------------------
    def _call(self, fn, *args, **kwargs):
        """Run *fn* with timeout + bounded retries + exponential backoff."""
        last: Optional[Exception] = None
        for attempt in range(self.config.retries + 1):
            fut = self._pool.submit(fn, *args, **kwargs)
            try:
                return fut.result(timeout=self.config.timeout)
            except FuturesTimeout as exc:
                self.stats["timeouts"] += 1
                last = exc
                fut.cancel()
                logger.warning("knowledge: timeout on attempt %d", attempt + 1)
            except Exception as exc:
                last = exc
                logger.warning("knowledge: error on attempt %d: %s", attempt + 1, exc)
            if attempt < self.config.retries:
                self.stats["retries"] += 1
                time.sleep(self.config.retry_backoff * (2 ** attempt))
        self.stats["failures"] += 1
        raise last if last else RuntimeError("knowledge call failed")

    # -- merge + rerank -------------------------------------------------
    @staticmethod
    def merge(results: Iterable[SearchResult], limit: int) -> List[Chunk]:
        seen: Dict[str, Chunk] = {}
        for res in results:
            for c in res.chunks:
                key = c.id or f"{c.document_id}:{hash(c.text)}"
                if key not in seen or c.score > seen[key].score:
                    seen[key] = c
        return sorted(seen.values(), key=lambda c: c.score, reverse=True)[:limit]

    def rerank(self, query: str, chunks: List[Chunk], limit: int) -> List[Chunk]:
        """Lexical + positional rerank on top of provider scores."""
        from .embeddings import keyword_overlap

        scored = []
        for c in chunks:
            lex = keyword_overlap(query, c.text)
            title = c.citation.title if c.citation else ""
            title_hit = keyword_overlap(query, title)
            final = 0.65 * c.score + 0.25 * lex + 0.10 * title_hit
            c.score = final
            if c.citation:
                c.citation.score = round(final, 4)
            scored.append(c)
        scored.sort(key=lambda c: c.score, reverse=True)
        return [c for c in scored if c.score >= self.config.min_score][:limit]

    # -- public API ------------------------------------------------------
    def search(self, query: str, limit: Optional[int] = None,
               workspace: Optional[str] = None,
               filters: Optional[Dict[str, Any]] = None,
               use_cache: bool = True) -> SearchResult:
        return self._run("search", query, limit, workspace, filters, use_cache)

    def retrieve(self, query: str, limit: Optional[int] = None,
                 workspace: Optional[str] = None,
                 filters: Optional[Dict[str, Any]] = None,
                 use_cache: bool = True) -> SearchResult:
        return self._run("retrieve", query, limit, workspace, filters, use_cache)

    def _run(self, mode: str, query: str, limit, workspace, filters,
             use_cache: bool) -> SearchResult:
        t0 = time.perf_counter()
        limit = int(limit or self.config.top_k)
        ws = workspace or self.config.workspace
        self.stats["searches"] += 1

        key = cache_key(mode, query, limit, ws, filters,
                        [p.name for p in self._providers])
        if use_cache:
            hit = self.cache.get(key)
            if hit is not None:
                self.stats["cache_hits"] += 1
                cached = SearchResult(**{**hit.__dict__})
                cached.cached = True
                cached.elapsed_ms = (time.perf_counter() - t0) * 1000
                return cached

        results: List[SearchResult] = []
        errors: List[str] = []
        for provider in self._providers:
            try:
                fn = getattr(provider, mode)
                res = self._call(fn, query, limit=limit, workspace=ws, filters=filters)
                results.append(res)
                if res.chunks:
                    break  # primary satisfied the query; skip fallbacks
            except Exception as exc:
                errors.append(f"{provider.name}: {exc}")
                logger.error("knowledge: provider %s failed: %s", provider.name, exc)

        if not results:
            return SearchResult(query=query, provider="none", workspace=ws,
                                elapsed_ms=(time.perf_counter() - t0) * 1000,
                                error="; ".join(errors) or "all providers failed")

        merged = self.rerank(query, self.merge(results, limit * 3), limit)
        answer = next((r.answer for r in results if r.answer), "")
        out = SearchResult(
            query=query, chunks=merged, answer=answer,
            provider="+".join(dict.fromkeys(r.provider for r in results)),
            workspace=ws,
            elapsed_ms=(time.perf_counter() - t0) * 1000,
            confidence=self._confidence(merged),
            error="; ".join(errors),
        )
        if use_cache and merged:
            self.cache.set(key, out)
        return out

    @staticmethod
    def _confidence(chunks: List[Chunk]) -> float:
        if not chunks:
            return 0.0
        top = chunks[0].score
        support = min(len(chunks), 3) / 3.0
        return round(min(1.0, 0.7 * top + 0.3 * support * top), 4)

    def retrieve_with_sources(self, query: str, limit: Optional[int] = None,
                              workspace: Optional[str] = None,
                              filters: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        res = self.retrieve(query, limit=limit, workspace=workspace, filters=filters)
        return res.to_dict()

    # camelCase aliases matching the spec
    retrieveWithSources = retrieve_with_sources

    def find_relevant_context(self, query: str, limit: Optional[int] = None,
                              workspace: Optional[str] = None,
                              max_chars: int = 6000) -> str:
        """Prompt-ready, citation-annotated context block (Step 6 injection)."""
        res = self.search(query, limit=limit, workspace=workspace)
        if not res.chunks:
            return ""
        lines = ["<knowledge_context>"]
        used = 0
        for i, c in enumerate(res.chunks, 1):
            cit = c.citation
            head = (f"[{i}] title={cit.title} | path={cit.path} | "
                    f"workspace={cit.workspace} | chunk={cit.chunk_id} | "
                    f"score={cit.score:.3f}") if cit else f"[{i}]"
            body = c.text.strip()
            if used + len(body) > max_chars:
                body = body[: max(0, max_chars - used)]
            lines.append(head)
            lines.append(body)
            used += len(body)
            if used >= max_chars:
                break
        lines.append("</knowledge_context>")
        lines.append("Cite these sources as [n] with their path in the answer.")
        return "\n".join(lines)

    findRelevantContext = find_relevant_context

    def find_similar(self, document_id: str, limit: Optional[int] = None,
                     workspace: Optional[str] = None) -> SearchResult:
        return self._call(self.primary.find_similar, document_id,
                          limit=int(limit or self.config.top_k),
                          workspace=workspace or self.config.workspace)

    # -- indexing --------------------------------------------------------
    def index(self, document: Document) -> IndexResult:
        self.cache.clear()
        return self._call(self.primary.index, document)

    def update(self, document: Document) -> IndexResult:
        self.cache.clear()
        return self._call(self.primary.update, document)

    def delete(self, document_id: str, workspace: Optional[str] = None) -> IndexResult:
        self.cache.clear()
        return self._call(self.primary.delete, document_id, workspace=workspace)

    def list_documents(self, workspace: Optional[str] = None) -> List[Dict[str, Any]]:
        return self.primary.list_documents(workspace or self.config.workspace)

    # -- ops -------------------------------------------------------------
    def freshness(self) -> Dict[str, Any]:
        """Report how current the index is, from the sync worker's state file.

        Lets Hermes answer "is my knowledge base up to date?" and warn the user
        when the continuous sync worker is down or lagging.
        """
        import json as _json

        path = os.path.join(os.path.dirname(self.config.db_path),
                            "sync_worker_state.json")
        try:
            with open(path) as fh:
                state = _json.load(fh)
        except Exception:
            return {"worker_running": False, "state_file": path,
                    "detail": "no sync worker state found; index may be stale"}
        stale = state.get("staleness_seconds")
        return {
            "worker_running": True,
            "worker_mode": state.get("mode"),
            "watching": state.get("watching", []),
            "seconds_since_last_sync": stale,
            "up_to_date": stale is not None and stale < 3600,
            "counts": state.get("counts", {}),
            "last_error": state.get("last_error", ""),
        }

    def health(self) -> Dict[str, Any]:
        checks: List[HealthStatus] = []
        for p in self._providers:
            try:
                checks.append(self._call(p.health))
            except Exception as exc:
                checks.append(HealthStatus(False, p.name, str(exc)))
        return {
            "healthy": any(c.healthy for c in checks),
            "providers": [c.to_dict() for c in checks],
            "cache": self.cache.stats(),
            "stats": dict(self.stats),
            "freshness": self.freshness(),
            "config": {"provider": self.config.provider,
                       "fallbacks": self.config.fallback_providers,
                       "workspace": self.config.workspace,
                       "top_k": self.config.top_k},
        }

    # -- Step 6 helper: does this question need external knowledge? -------
    @staticmethod
    def requires_retrieval(question: str) -> bool:
        q = (question or "").lower()
        if len(q.split()) < 3:
            return False
        return any(h in q for h in _RETRIEVAL_HINTS) or q.strip().endswith("?")


_SINGLETON: Optional[KnowledgeService] = None
_SINGLETON_LOCK = threading.Lock()


def get_knowledge_service(refresh: bool = False) -> KnowledgeService:
    global _SINGLETON
    with _SINGLETON_LOCK:
        if _SINGLETON is None or refresh:
            _SINGLETON = KnowledgeService()
        return _SINGLETON
