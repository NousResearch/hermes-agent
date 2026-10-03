"""skill-rag plugin entry point.

Uses dependency injection for all dependencies.
Reuses parse_frontmatter from agent/skill_utils.py.
"""
from __future__ import annotations

import logging
import threading
import time
from typing import Any, Dict, List, Optional, Set

from .config import LOG_PREFIX, DEFAULTS
from .indexer import Indexer
from . import retrieval

logger = logging.getLogger(__name__)

# --- Dependency injection container ---

_indexer: Optional[Indexer] = None
_initialized: bool = False
_init_lock: threading.Lock = threading.Lock()
_ctx: Optional[Any] = None


def _resolve_config() -> Dict[str, Any]:
    """Resolve plugin config from ctx.get_config() with defaults fallback."""
    cfg = dict(DEFAULTS)
    if _ctx is not None:
        try:
            for key in DEFAULTS:
                val = _ctx.get_config(key)
                if val is not None:
                    cfg[key] = val
            logger.debug("%s _resolve_config: resolved from ctx=%s", LOG_PREFIX, cfg)
        except Exception as e:
            logger.debug("%s _resolve_config: ctx.get_config failed (%s), using defaults", LOG_PREFIX, e)
    return cfg


def _ensure_init() -> None:
    """Lazy initialization with timing. Thread-safe via lock."""
    global _indexer, _initialized
    if _initialized:
        logger.debug("%s _ensure_init: already initialized, skipping", LOG_PREFIX)
        return
    with _init_lock:
        if _initialized:
            logger.debug("%s _ensure_init: already initialized (double-check), skipping", LOG_PREFIX)
            return
        logger.info("%s _ensure_init: starting initialization", LOG_PREFIX)
        start = time.monotonic()
        config = _resolve_config()
        try:
            _indexer = Indexer(config=config)
            logger.debug("%s _ensure_init: Indexer() created with config=%s", LOG_PREFIX, config)
        except Exception as e:
            logger.error("%s _ensure_init: Indexer() failed: %s", LOG_PREFIX, e, exc_info=True)
            return
        try:
            _indexer.sync_startup()
            logger.debug("%s _ensure_init: sync_startup() completed", LOG_PREFIX)
        except Exception as e:
            logger.error("%s _ensure_init: sync_startup() failed: %s", LOG_PREFIX, e, exc_info=True)
        elapsed = (time.monotonic() - start) * 1000
        skill_count = getattr(_indexer, '_skill_count', -1)
        logger.info(
            "%s initialized in %.0fms, %d skills indexed",
            LOG_PREFIX, elapsed, skill_count,
        )
        _initialized = True


def _get_skills_in_context(history: List[Any]) -> Set[str]:
    """Get all skills already in context from ALL sources:

    1. Skills loaded via skill_view (tool calls in history)
    2. Skills in <available_skills> blocks (system prompt + our injections)

    ALL are excluded from recommendations to avoid duplication.
    """
    result = retrieval.extract_context_skills(history)
    logger.debug("%s _get_skills_in_context: %d skills already in context: %s",
                 LOG_PREFIX, len(result), sorted(result) if result else "none")
    return result


def on_pre_llm_call(**kwargs: Any) -> Optional[Dict[str, str]]:
    """pre_llm_call hook — semantic skill retrieval.

    Returns {"context": "..."} for injection into user message.
    Context uses <available_skills> format for tracking.
    """
    hook_start = time.monotonic()
    logger.info("%s on_pre_llm_call: ENTERED, kwargs keys=%s", LOG_PREFIX, sorted(kwargs.keys()))
    try:
        _ensure_init()
        if _indexer is None:
            logger.warning("%s on_pre_llm_call: _indexer is None after init, returning None", LOG_PREFIX)
            return None

        config = _indexer.config
        user_message: str = kwargs.get("user_message") or ""
        history: List[Any] = kwargs.get("conversation_history") or []
        session_id: str = kwargs.get("session_id") or ""
        is_first_turn: bool = kwargs.get("is_first_turn", False)
        model: str = kwargs.get("model") or ""
        platform: str = kwargs.get("platform") or ""
        logger.info("%s on_pre_llm_call: user_message=%r (len=%d), history_len=%d, "
                    "session=%s, first_turn=%s, model=%s, platform=%s",
                    LOG_PREFIX, user_message[:120], len(user_message), len(history),
                    session_id[:40], is_first_turn, model, platform)

        if not user_message and not history:
            logger.info("%s on_pre_llm_call: empty user_message AND empty history, returning None", LOG_PREFIX)
            return None

        query = retrieval.build_query(history, user_message)
        logger.info("%s on_pre_llm_call: build_query returned %r (len=%d)", LOG_PREFIX, query[:120], len(query))
        if not query.strip():
            logger.info("%s on_pre_llm_call: query is blank, returning None", LOG_PREFIX)
            return None

        # Get skills already in context (loaded + in <available_skills> blocks)
        already = _get_skills_in_context(history)
        if already:
            logger.info("%s on_pre_llm_call: %d skills already in context (will exclude): %s",
                        LOG_PREFIX, len(already), sorted(already))

        # Vector search
        top_k = config.get("top_k", DEFAULTS["top_k"])
        threshold = config.get("threshold", DEFAULTS["threshold"])
        logger.info("%s on_pre_llm_call: starting vector search (top_k=%d, threshold=%.3f, exclude=%d)",
                    LOG_PREFIX, top_k, threshold, len(already))
        search_start = time.monotonic()
        try:
            results = retrieval.retrieve(
                _indexer, query,
                exclude_names=already,
                config=retrieval.RetrievalConfig(top_k=top_k, threshold=threshold),
            )
        except Exception as e:
            logger.error("%s on_pre_llm_call: vector search FAILED: %s", LOG_PREFIX, e, exc_info=True)
            raise
        search_ms = (time.monotonic() - search_start) * 1000
        logger.info("%s on_pre_llm_call: vector search returned %d results in %.1fms: %s",
                    LOG_PREFIX, len(results), search_ms,
                    [(r["name"], round(r["score"], 3)) for r in results] if results else "none")

        if not results:
            logger.info("%s on_pre_llm_call: no results from vector search, returning None", LOG_PREFIX)
            return None

        # Check freshness and reindex if needed
        result_names = [r["name"] for r in results]
        stale = _indexer.check_fresh(result_names)
        logger.info("%s on_pre_llm_call: check_fresh returned %d stale: %s",
                    LOG_PREFIX, len(stale), stale if stale else "none")
        if stale:
            logger.info("%s on_pre_llm_call: reindexing stale skills: %s", LOG_PREFIX, stale)
            reindex_start = time.monotonic()
            _indexer.reindex(stale)
            reindex_ms = (time.monotonic() - reindex_start) * 1000
            logger.info("%s on_pre_llm_call: reindex completed in %.1fms", LOG_PREFIX, reindex_ms)
            results = retrieval.retrieve(
                _indexer, query,
                exclude_names=already,
                config=retrieval.RetrievalConfig(top_k=top_k, threshold=threshold),
            )
            logger.info("%s on_pre_llm_call: re-search after reindex returned %d results: %s",
                        LOG_PREFIX, len(results),
                        [(r["name"], round(r["score"], 3)) for r in results] if results else "none")
            if not results:
                logger.info("%s on_pre_llm_call: no results after reindex, returning None", LOG_PREFIX)
                return None

            # BM25 fallback if vector still stale
            still_stale = _indexer.check_fresh([r["name"] for r in results])
            logger.info("%s on_pre_llm_call: still_stale check: %s", LOG_PREFIX, still_stale if still_stale else "none")
            if still_stale:
                logger.info("%s on_pre_llm_call: falling back to BM25", LOG_PREFIX)
                results = retrieval.retrieve_bm25(
                    _indexer, query,
                    exclude_names=already,
                    config=retrieval.RetrievalConfig(top_k=top_k),
                )
                logger.info("%s on_pre_llm_call: BM25 fallback returned %d results: %s",
                            LOG_PREFIX, len(results),
                            [(r["name"], round(r["score"], 3)) for r in results] if results else "none")
                if not results:
                    logger.info("%s on_pre_llm_call: no results from BM25, returning None", LOG_PREFIX)
                    return None

        # Build context in <available_skills> format (no marker needed)
        context = retrieval.build_context(results)
        context_len = len(context) if context else 0
        total_ms = (time.monotonic() - hook_start) * 1000
        if context:
            logger.info("%s on_pre_llm_call: SUCCESS — injecting %d skills (%d chars context, %.1fms total): %s",
                        LOG_PREFIX, len(results), context_len, total_ms,
                        ", ".join(r["name"] for r in results))
            logger.debug("%s on_pre_llm_call: context content:\n%s", LOG_PREFIX, context)
            return {"context": context}
        else:
            logger.info("%s on_pre_llm_call: build_context returned empty, returning None (%.1fms total)",
                        LOG_PREFIX, total_ms)
            return None

    except Exception as e:
        total_ms = (time.monotonic() - hook_start) * 1000
        logger.warning("%s on_pre_llm_call: EXCEPTION after %.1fms: %s: %s",
                       LOG_PREFIX, total_ms, type(e).__name__, e, exc_info=True)
        return None


def on_skill_lifecycle(**kwargs: Any) -> None:
    """on_skill_lifecycle hook — re-index on skill change."""
    logger.info("%s on_skill_lifecycle: ENTERED, kwargs=%s", LOG_PREFIX, {k: v for k, v in kwargs.items() if k != 'payload'})
    try:
        _ensure_init()
        if _indexer is None:
            logger.warning("%s on_skill_lifecycle: _indexer is None, skipping", LOG_PREFIX)
            return
        action: str = kwargs.get("action") or ""
        name: Optional[str] = kwargs.get("skill_name") or kwargs.get("name")
        logger.info("%s on_skill_lifecycle: action=%r skill_name=%r", LOG_PREFIX, action, name)
        if name:
            logger.info("%s on_skill_lifecycle: reindexing skill: %s", LOG_PREFIX, name)
            reindex_start = time.monotonic()
            _indexer.reindex([name])
            reindex_ms = (time.monotonic() - reindex_start) * 1000
            logger.info("%s on_skill_lifecycle: reindex of %s completed in %.1fms", LOG_PREFIX, name, reindex_ms)
        else:
            logger.info("%s on_skill_lifecycle: no skill name in kwargs, skipping reindex", LOG_PREFIX)
    except Exception as e:
        logger.warning("%s on_skill_lifecycle: EXCEPTION: %s: %s", LOG_PREFIX, type(e).__name__, e, exc_info=True)


def register(ctx: Any) -> None:
    """Plugin registration in Hermes PluginContext."""
    global _ctx
    _ctx = ctx
    logger.info("%s register: ENTERED, ctx=%s type=%s", LOG_PREFIX, ctx, type(ctx).__name__)
    try:
        ctx.register_hook("pre_llm_call", on_pre_llm_call)
        logger.info("%s register: pre_llm_call hook registered OK", LOG_PREFIX)
    except Exception as e:
        logger.error("%s register: pre_llm_call registration FAILED: %s", LOG_PREFIX, e, exc_info=True)
    try:
        ctx.register_hook("on_skill_lifecycle", on_skill_lifecycle)
        logger.info("%s register: on_skill_lifecycle hook registered OK", LOG_PREFIX)
    except Exception as e:
        logger.error("%s register: on_skill_lifecycle registration FAILED: %s", LOG_PREFIX, e, exc_info=True)
    logger.info("%s register: COMPLETED — hooks: pre_llm_call, on_skill_lifecycle", LOG_PREFIX)


def _cleanup() -> None:
    """Cleanup on process shutdown."""
    global _indexer
    if _indexer is not None:
        logger.info("%s _cleanup: closing indexer", LOG_PREFIX)
        try:
            _indexer.close()
            logger.info("%s _cleanup: indexer closed OK", LOG_PREFIX)
        except Exception as e:
            logger.warning("%s _cleanup: indexer close failed: %s", LOG_PREFIX, e)
        _indexer = None
    else:
        logger.debug("%s _cleanup: indexer already None", LOG_PREFIX)


import atexit
atexit.register(_cleanup)
logger.info("%s module loaded: __init__.py imported, atexit registered", LOG_PREFIX)
