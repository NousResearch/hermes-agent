"""
Federated search provider — aggregate results from multiple search backends
with LLM-based relevance ranking.

Configuration in config.yaml::

    web:
      search_backend: federated
      federated:
        timeout: 10                  # seconds to wait for all backends (config item 1)
        max_results: 8               # top N results after ranking (config item 3)
        ranker:                      # LLM for relevance ranking (config item 2)
                                     # Note: LLM providers may have concurrency limits.
                                     # When the ranking LLM fails (timeout, rate limit, etc.)
                                     # the plugin automatically falls back to NONE mode
                                     # (keyword-based scoring, no LLM call).
                                     # For best results, use a non-main-model provider
                                     # or one with higher concurrency limits.
          provider: opencode-go
          model: deepseek-v4-flash
        backends:
          - name: tavily             # use an existing registered provider
          - name: minimax            # custom HTTP backend
            type: custom
            base_url: "https://api.minimaxi.com"
            api_key_env: MINIMAX_CN_API_KEY
            search_path: /v1/coding_plan/search
            query_param: "q"
"""

from __future__ import annotations

import concurrent.futures
import logging
import os
import time
from typing import Any, Dict, List, Optional

from agent.web_search_provider import WebSearchProvider

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_DEFAULT_TIMEOUT = 10
_DEFAULT_MAX_RESULTS = 8
_CUSTOM_BACKEND_TIMEOUT = 15
_CUSTOM_BACKEND_MAX_RESULTS = 10
_SKIP_RANK_IF = 3
_MAX_RANK_INPUT = 10
_RANK_TIMEOUT = 20

# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------


def _read_config() -> Optional[Dict[str, Any]]:
    try:
        from hermes_cli.config import load_config
        cfg = load_config()
        web = cfg.get("web", {})
        if not isinstance(web, dict):
            return None
        return web.get("federated")
    except Exception:
        return None


def _get_registered_provider(name: str) -> Optional[WebSearchProvider]:
    try:
        from agent.web_search_registry import get_provider
        return get_provider(name)
    except Exception:
        return None


# ---------------------------------------------------------------------------
# LLM ranking & keyword ranking
# ---------------------------------------------------------------------------


def _keyword_rank(
    query: str,
    results: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Rank results by keyword match frequency in title + description.

    Fast, no LLM needed. De-duplicates by URL.
    """
    seen_urls = set()
    deduped = []
    for r in results:
        url = r.get("url", "")
        if url and url not in seen_urls:
            seen_urls.add(url)
            deduped.append(r)

    terms = query.lower().split()
    def _score(r):
        title = (r.get("title", "") or "").lower()
        desc = (r.get("description", "") or "").lower()
        return sum(1 for t in terms if t in title or t in desc)

    deduped.sort(key=_score, reverse=True)
    return deduped


def _rank_results(
    query: str,
    results: List[Dict[str, Any]],
    ranker_config: Optional[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Rank search results by relevance to *query*.

    Ranking mode is selected by the ``provider`` field in *ranker_config*:

    - ``"none"`` / unset / empty → keyword match scoring (fast, no LLM).
    - ``"auto"`` or a real provider name → LLM-based ranking.
      On LLM failure (timeout, rate limit, concurrency limit) the function
      automatically falls back to keyword (``none``) ranking.
      **Recommendation**: use a non-main-model provider or one with higher
      concurrency limits to avoid ranking failures degrading search quality.

    Optimizations:
    - Skip LLM ranking when <= _SKIP_RANK_IF results.
    - Truncate titles to 60 chars, descriptions to 80 chars.
    - Hard timeout via _RANK_TIMEOUT.
    """
    if not results or len(results) <= _SKIP_RANK_IF:
        return results

    provider = (ranker_config or {}).get("provider") or ""
    model = (ranker_config or {}).get("model") or ""

    # "none" → keyword ranking, no LLM
    if not provider or provider == "none":
        return _keyword_rank(query, results)

    # ── LLM-based ranking ──
    try:
        from agent.auxiliary_client import call_llm

        lines = []
        for i, r in enumerate(results):
            title = (r.get("title", "") or "")[:60]
            desc = (r.get("description", "") or "")[:80]
            lines.append(f"[{i + 1}] {title}\n{desc}")
        results_text = "\n".join(lines)

        sys = (
            "You rank search results by relevance. Rules:\n"
            "- Return ONLY a JSON array of result indices ranked by relevance, e.g. [3,1,5,2,4]\n"
            "- Include ALL results. Most relevant first.\n"
            "- No other text."
        )
        user = f"Query: {query}\n\nResults:\n{results_text}\n\nRanked indices:"

        logger.info(
            "LLM ranking %d results (provider=%s, model=%s)",
            len(results), provider or "auto", model or "auto",
        )

        response = call_llm(
            task="web_extract",
            provider=provider or None,
            model=model or None,
            messages=[{"role": "system", "content": sys}, {"role": "user", "content": user}],
            temperature=0,
            max_tokens=128,
            timeout=_RANK_TIMEOUT,
        )

        raw = (response.choices[0].message.content or "").strip()
        import json, re

        match = re.search(r"\[[\d\s,,\]]+\]", raw)
        if match:
            indices = json.loads(match.group())
            if isinstance(indices, list):
                ranked, seen = [], set()
                for idx in indices:
                    pos = int(idx) - 1
                    if 0 <= pos < len(results) and pos not in seen:
                        ranked.append(results[pos])
                        seen.add(pos)
                for i, r in enumerate(results):
                    if i not in seen:
                        ranked.append(r)
                return ranked

        logger.warning("LLM ranking unparseable, falling back to keyword ranking")
    except Exception as exc:
        logger.warning("LLM ranking failed (%s), falling back to keyword ranking", exc)

    return _keyword_rank(query, results)


# ---------------------------------------------------------------------------
# Custom HTTP backend search
# ---------------------------------------------------------------------------


def _search_custom_backend(
    backend_config: Dict[str, Any],
    query: str,
    limit: int,
) -> List[Dict[str, Any]]:
    """Execute a search against a custom HTTP endpoint.

    Config keys: base_url, api_key_env, search_path, query_param (default ``q``),
    auth_style (``bearer`` or ``x-api-key``, default ``bearer``).
    """
    import httpx

    base_url = (backend_config.get("base_url") or "").rstrip("/")
    api_key_env = backend_config.get("api_key_env", "")
    search_path = backend_config.get("search_path", "/v1/coding_plan/search")
    query_param = backend_config.get("query_param", "q")
    auth_style = backend_config.get("auth_style", "bearer")
    api_key = os.environ.get(api_key_env, "") if api_key_env else ""

    if not base_url or not api_key:
        return []

    url = f"{base_url}{search_path}"
    if auth_style == "x-api-key":
        headers = {"Content-Type": "application/json", "x-api-key": api_key}
    else:
        headers = {"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"}

    payload = {query_param: query, "max_results": min(limit, _CUSTOM_BACKEND_MAX_RESULTS)}

    try:
        resp = httpx.post(url, json=payload, headers=headers, timeout=_CUSTOM_BACKEND_TIMEOUT)
        resp.raise_for_status()
        data = resp.json()
        results = _extract_custom_results(data)
        if not results:
            logger.warning("Custom backend returned no parseable results")
        return results
    except httpx.TimeoutException:
        logger.warning("Custom backend timed out: %s", url)
    except httpx.HTTPStatusError as exc:
        logger.warning("Custom backend HTTP error: %s (%s)", exc, exc.response.text[:200])
    except Exception as exc:
        logger.warning("Custom backend failed: %s", exc)
    return []


def _extract_custom_results(data: Any) -> List[Dict[str, Any]]:
    """Extract search results from various API response shapes."""
    if not isinstance(data, dict):
        return []

    results: List[Dict[str, Any]] = []

    # {organic: [{title, link, snippet, date}, ...]}
    organic = data.get("organic")
    if isinstance(organic, list):
        for item in organic:
            if isinstance(item, dict):
                results.append({
                    "title": str(item.get("title", "") or ""),
                    "url": str(item.get("link", "") or item.get("url", "") or ""),
                    "description": str(item.get("snippet", "") or item.get("content", "") or ""),
                    "position": len(results) + 1,
                })
        if results:
            return results

    # {data: {web: [...]}}
    web_data = data.get("data")
    if isinstance(web_data, dict):
        web_list = web_data.get("web")
        if isinstance(web_list, list):
            for item in web_list:
                if isinstance(item, dict):
                    results.append({
                        "title": str(item.get("title", "") or ""),
                        "url": str(item.get("url", "") or ""),
                        "description": str(item.get("description", "") or item.get("content", "") or ""),
                        "position": len(results) + 1,
                    })
            if results:
                return results

    # {results: [{...}]}
    raw = data.get("results")
    if isinstance(raw, list):
        for item in raw:
            if isinstance(item, dict):
                results.append({
                    "title": str(item.get("title", "") or ""),
                    "url": str(item.get("url", "") or item.get("link", "") or ""),
                    "description": str(item.get("content", "") or item.get("snippet", "") or ""),
                    "position": len(results) + 1,
                })
        if results:
            return results
    return results


# ---------------------------------------------------------------------------
# Per-backend worker (used by ThreadPoolExecutor)
# ---------------------------------------------------------------------------


def _search_one_backend(backend: Dict[str, Any], query: str, limit: int) -> List[Dict[str, Any]]:
    """Single-backend search worker for thread-pool execution."""
    name = str(backend.get("name", "?"))
    typ = str(backend.get("type", "") or "")

    try:
        if typ == "custom":
            results = _search_custom_backend(backend, query, limit)
        else:
            provider = _get_registered_provider(name)
            if provider is None:
                logger.warning("Backend '%s' not registered, skipping", name)
                return []
            resp = provider.search(query, limit=limit)
            if isinstance(resp, dict) and resp.get("success"):
                data = resp.get("data", {})
                items = data.get("web", []) if isinstance(data, dict) else []
                results = [
                    {"title": str(r.get("title", "") or ""),
                     "url": str(r.get("url", "") or ""),
                     "description": str(r.get("description", "") or ""),
                     "position": i + 1}
                    for i, r in enumerate(items) if isinstance(r, dict)
                ]
            else:
                err = resp.get("error", "unknown") if isinstance(resp, dict) else "unknown"
                logger.warning("Backend '%s' failed: %s", name, err)
                return []
        logger.info("Backend '%s' returned %d results", name, len(results))
        return results
    except Exception as exc:
        logger.warning("Backend '%s' error: %s", name, exc)
        return []


# ---------------------------------------------------------------------------
# Provider
# ---------------------------------------------------------------------------


class FederatedSearchProvider(WebSearchProvider):
    """Aggregated search provider that fans out to multiple sub-backends."""

    @property
    def name(self) -> str:
        return "federated"

    @property
    def display_name(self) -> str:
        return "Federated Search"

    def is_available(self) -> bool:
        config = _read_config()
        if not config:
            return False
        backends = config.get("backends")
        return isinstance(backends, list) and len(backends) > 0

    def supports_search(self) -> bool:
        return True

    def supports_extract(self) -> bool:
        return False

    def search(self, query: str, limit: int = 5) -> Dict[str, Any]:
        try:
            from tools.interrupt import is_interrupted
            if is_interrupted():
                return {"success": False, "error": "Interrupted"}

            config = _read_config()
            if not config:
                return {"success": False, "error": "federated search not configured"}

            backends = config.get("backends", [])
            if not isinstance(backends, list) or not backends:
                return {"success": False, "error": "no search backends configured"}

            timeout = int(config.get("timeout", _DEFAULT_TIMEOUT))
            max_results = int(config.get("max_results", _DEFAULT_MAX_RESULTS))
            ranker_config = config.get("ranker")

            logger.info(
                "Federated search: '%s' (%d backends, timeout=%ds, max_results=%d)",
                query, len(backends), timeout, max_results,
            )

            # ---------- parallel backend execution ----------
            all_results: List[Dict[str, Any]] = []
            errors: List[str] = []
            deadline = time.time() + timeout

            with concurrent.futures.ThreadPoolExecutor(max_workers=len(backends)) as pool:
                futures = {pool.submit(_search_one_backend, b, query, limit): b for b in backends}

                for future in concurrent.futures.as_completed(futures, timeout=timeout):
                    if time.time() >= deadline or is_interrupted():
                        # Cancel remaining futures
                        for f in futures:
                            f.cancel()
                        break
                    try:
                        results = future.result(timeout=2)
                        all_results.extend(results)
                    except concurrent.futures.TimeoutError:
                        b = futures[future]
                        errors.append(f"backend '{b.get('name','?')}' timed out")
                    except Exception as exc:
                        b = futures[future]
                        errors.append(f"backend '{b.get('name','?')}' failed: {exc}")

            if not all_results:
                if errors:
                    return {"success": False, "error": "All backends failed: " + "; ".join(errors)}
                return {"success": True, "data": {"web": []}}

            # ---------- ranking ----------
            # _rank_results handles all modes internally:
            # - provider: none / empty → keyword scoring (fast)
            # - provider: <real>      → LLM, falls back to keyword on failure
            rank_input = all_results[:_MAX_RANK_INPUT]
            ranked = _rank_results(query, rank_input, ranker_config)

            # Top N
            top = ranked[:max_results]
            for i, r in enumerate(top):
                r["position"] = i + 1

            logger.info(
                "Federated search: %d raw -> %d ranked (total %.1fs)",
                len(all_results), len(top), time.time() - (time.time() - 20),
            )

            return {
                "success": True,
                "data": {
                    "web": [
                        {"title": r.get("title", ""), "url": r.get("url", ""),
                         "description": r.get("description", ""), "position": r.get("position", i + 1)}
                        for i, r in enumerate(top)
                    ],
                },
            }

        except Exception as exc:
            logger.error("Federated search error: %s", exc)
            return {"success": False, "error": f"Federated search failed: {exc}"}

    def get_setup_schema(self) -> Dict[str, Any]:
        return {
            "name": "Federated Search",
            "badge": "advanced",
            "tag": (
                "Aggregate multiple search backends with LLM-based ranking. "
                "Configure backends under web.federated.backends in config.yaml."
            ),
            "env_vars": [],
        }
