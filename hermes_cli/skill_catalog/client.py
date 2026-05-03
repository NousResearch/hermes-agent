"""Fetch aggregated Skill Hub results with TTL cache (no DB)."""

from __future__ import annotations

import logging
import threading
import time
from typing import Any, Dict, List, Tuple

from tools.skills_hub import GitHubAuth, SkillMeta, create_source_router, parallel_search_sources

from hermes_cli.skill_catalog.normalize import dedupe_metas, meta_to_item
from hermes_cli.skill_catalog.pagination import filter_by_category, paginate, sort_items

logger = logging.getLogger(__name__)

_CACHE_LOCK = threading.Lock()
_CACHE: Dict[str, Tuple[float, List[Dict[str, Any]]]] = {}
_CACHE_TTL_SEC = 90.0

# Cap hub fan-out; Hermes index empty-query already slices per-source via ``limit``.
_PER_SOURCE_LIMIT_INDEX = 600
_PER_SOURCE_LIMIT_OFFICIAL = 400
_OVERALL_TIMEOUT = 45.0


def _cache_key(query: str, source_filter: str) -> str:
    return f"{source_filter}\x00{query.strip().lower()}"


def _parallel_fetch(query: str, source_filter: str) -> List[SkillMeta]:
    auth = GitHubAuth()
    sources = create_source_router(auth)
    per_limits = {
        "hermes-index": _PER_SOURCE_LIMIT_INDEX,
        "official": _PER_SOURCE_LIMIT_OFFICIAL,
        "github": 120,
        "skills-sh": 120,
        "clawhub": 120,
        "claude-marketplace": 120,
        "lobehub": 120,
        "well-known": 80,
    }
    raw, _, timed_out = parallel_search_sources(
        sources,
        query=query,
        per_source_limits=per_limits,
        source_filter=source_filter,
        overall_timeout=_OVERALL_TIMEOUT,
    )
    if timed_out:
        logger.debug("skill_catalog: parallel_search timed out sources: %s", timed_out)
    deduped = dedupe_metas(raw)
    items: List[Dict[str, Any]] = []
    for i, meta in enumerate(deduped):
        items.append(meta_to_item(meta, order=i))
    return items


def get_catalog_items(query: str, source_filter: str = "all") -> List[Dict[str, Any]]:
    """Return normalized items (no category filter / pagination)."""
    key = _cache_key(query, source_filter)
    now = time.monotonic()
    with _CACHE_LOCK:
        hit = _CACHE.get(key)
        if hit and now - hit[0] < _CACHE_TTL_SEC:
            return [dict(x) for x in hit[1]]

    items = _parallel_fetch(query, source_filter)

    with _CACHE_LOCK:
        _CACHE[key] = (now, [dict(x) for x in items])
        # Bound cache size
        if len(_CACHE) > 64:
            oldest = sorted(_CACHE.items(), key=lambda kv: kv[1][0])[:32]
            for k, _ in oldest:
                _CACHE.pop(k, None)

    return [dict(x) for x in items]


def build_catalog_page(
    *,
    category: str = "all",
    page: int = 1,
    page_size: int = 20,
    q: str = "",
    sort: str = "relevance",
    source_filter: str = "all",
) -> Dict[str, Any]:
    """Hub fetch → filter category → sort → paginate."""
    query = (q or "").strip()
    base = get_catalog_items(query, source_filter)
    filtered = filter_by_category(base, category)
    sorted_rows = sort_items(filtered, sort)
    rows, pagination = paginate(sorted_rows, page=page, page_size=page_size)
    return {"items": rows, "pagination": pagination}
