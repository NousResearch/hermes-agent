"""
Federated search provider — aggregate results from multiple search backends
with LLM-based relevance ranking.

Configuration in config.yaml::

    web:
      search_backend: federated
      federated:
        timeout: 10                  # seconds to wait for all backends
        max_results: 8               # top N results after ranking

        # ── K-way aggregation ──
        k: 3                         # exactly K backends required (max 32)
        min_backends: 2              # at least N backends must succeed

        # ── ranking ──
        ranker:                      # LLM for relevance ranking
                                     # Note: LLM providers may have concurrency limits.
                                     # When the ranking LLM fails (timeout, rate limit, etc.)
                                     # the plugin automatically falls back to keyword scoring.
                                     # For best results, use a non-main-model provider
                                     # or one with higher concurrency limits.
          provider: opencode-go
          model: deepseek-v4-flash
          prompt: |                  # optional: custom ranking preferences
            优先中文来源，优先官方文档

        # ── health checks (cached, avoids per-search quota consumption) ──
        health_check:
          ttl_seconds: 300           # cache probe results for N seconds
          timeout: 5                 # per-probe timeout

        # ── K backends (exactly K) ──
        backends:
          - name: tavily             # use an existing registered provider
            required: false
          - name: minimax            # custom HTTP backend
            type: custom
            base_url: "https://api.minimaxi.com"
            api_key_env: MINIMAX_CN_API_KEY
            search_path: /v1/coding_plan/search
            query_param: "q"
            required: false
          - name: searxng
            type: custom
            base_url: "https://search.example.com"
            api_key_env: SEARXNG_API_KEY
            required: true
"""

from __future__ import annotations

import concurrent.futures
import logging
import time
from typing import Any, Dict, List, Optional

from agent.web_search_provider import WebSearchProvider, get_provider_env

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

# K-way aggregation
_MAX_K = 32

# Health check defaults
_DEFAULT_HEALTH_TTL = 300
_DEFAULT_HEALTH_TIMEOUT = 5


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------


def _read_config() -> Optional[Dict[str, Any]]:
    """Read the ``web.federated`` section from the raw config file.

    Uses raw-YAML access rather than ``load_config()`` because the schema
    validator strips unknown keys from the ``web`` section during loading.
    """
    try:
        from hermes_cli.config import get_config_path
        import yaml
        config_path = get_config_path()
        if not config_path.exists():
            return None
        with open(config_path, encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
        web = cfg.get("web")
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
# Config validation
# ---------------------------------------------------------------------------


def _validate_config(config: Dict[str, Any]) -> Optional[str]:
    """Validate federated search configuration. Returns error message or None.

    Checks:
    - k ≤ _MAX_K
    - len(backends) == k
    - 1 ≤ min_backends ≤ k
    """
    backends = config.get("backends", [])
    if not isinstance(backends, list) or not backends:
        return "no search backends configured"

    k = int(config.get("k", len(backends)))
    if k > _MAX_K:
        return f"k={k} exceeds maximum {_MAX_K}"
    if k < 1:
        return f"k={k} must be at least 1"

    if len(backends) != k:
        return (
            f"k={k} but {len(backends)} backend(s) configured — "
            f"exactly {k} required"
        )

    min_backends = int(config.get("min_backends", 1))
    if min_backends < 1:
        return f"min_backends={min_backends} must be at least 1"
    if min_backends > k:
        return f"min_backends={min_backends} cannot exceed k={k}"

    return None


# ---------------------------------------------------------------------------
# Health check cache
# ---------------------------------------------------------------------------


class _HealthCache:
    """TTL-based cache for backend health probe results.

    Avoids consuming provider quota on every search by caching HEAD probe
    results for ``ttl_seconds``.  A real search failure (HTTP 401/403/429)
    bypasses the cache and marks the backend unavailable for the configured
    cooldown period.
    """

    def __init__(self, ttl_seconds: int = _DEFAULT_HEALTH_TTL):
        self._ttl = ttl_seconds
        self._cache: Dict[str, tuple[bool, float]] = {}
        # Cooldown after real search failure: status_code → seconds
        self._failure_cooldown: Dict[int, int] = {
            401: 300,
            403: 300,
            429: 3600,
        }

    def is_available(self, backend_name: str) -> Optional[bool]:
        """Return cached availability or None if cache miss."""
        entry = self._cache.get(backend_name)
        if entry is None:
            return None
        available, timestamp = entry
        if time.time() - timestamp < self._ttl:
            return available
        # Expired — remove and return miss
        del self._cache[backend_name]
        return None

    def set_available(self, backend_name: str, available: bool) -> None:
        self._cache[backend_name] = (available, time.time())

    def mark_failed(self, backend_name: str, status_code: int) -> None:
        """Mark backend unavailable after a real search failure.

        Uses a longer cooldown than the probe TTL — quota/billing failures
        don't resolve in seconds.
        """
        cooldown = self._failure_cooldown.get(status_code, 300)
        # Store with an artificial timestamp so the entry expires after
        # cooldown seconds regardless of normal TTL.
        self._cache[backend_name] = (False, time.time() - self._ttl + cooldown)


# ---------------------------------------------------------------------------
# Backend health probe
# ---------------------------------------------------------------------------


def _probe_backend(
    backend: Dict[str, Any],
    timeout: int = _DEFAULT_HEALTH_TIMEOUT,
) -> bool:
    """Lightweight availability probe for a custom HTTP backend.

    Sends HEAD to ``base_url`` (no search query, no API consumption).
    For registered providers, delegates to ``provider.is_available()``.

    Returns True when the backend appears reachable.
    """
    name = str(backend.get("name", "?"))
    typ = str(backend.get("type", "") or "")

    if typ != "custom":
        provider = _get_registered_provider(name)
        if provider is None:
            return False
        try:
            return provider.is_available()
        except Exception:
            return False

    # Custom HTTP backend — HEAD probe
    import httpx
    base_url = (backend.get("base_url") or "").rstrip("/")
    if not base_url:
        return False

    try:
        resp = httpx.head(base_url, timeout=timeout, follow_redirects=True)
        # 2xx/3xx/4xx all mean the service is reachable (auth failures are
        # handled at search time, not probe time)
        return resp.status_code < 500
    except Exception:
        return False


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

    If *ranker_config* contains a ``prompt`` key, it is used as the system
    prompt for LLM ranking, allowing users to inject preferences
    (e.g. "prefer Chinese sources", "prefer official documentation").

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

        custom_prompt = (ranker_config or {}).get("prompt", "").strip()
        if custom_prompt:
            sys = custom_prompt + (
                "\n\nReturn ONLY a JSON array of result indices ranked by "
                "relevance, e.g. [3,1,5,2,4]. Include ALL results. "
                "No other text."
            )
        else:
            sys = (
                "You rank search results by relevance. Rules:\n"
                "- Return ONLY a JSON array of result indices ranked by "
                "relevance, e.g. [3,1,5,2,4]\n"
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

        match = re.search(r"\[[\d\s,,]+\]", raw)
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
) -> tuple[List[Dict[str, Any]], Optional[int]]:
    """Execute a search against a custom HTTP endpoint.

    Config keys: base_url, api_key_env, search_path, query_param (default ``q``),
    auth_style (``bearer`` or ``x-api-key``, default ``bearer``).

    Returns ``(results, status_code)`` where *status_code* is ``None`` on
    success or non-HTTP errors, and the response status code on HTTP errors
    so the caller can invoke health-cache cooldowns.
    """
    import httpx

    base_url = (backend_config.get("base_url") or "").rstrip("/")
    api_key_env = backend_config.get("api_key_env", "")
    search_path = backend_config.get("search_path", "/v1/coding_plan/search")
    query_param = backend_config.get("query_param", "q")
    auth_style = backend_config.get("auth_style", "bearer")
    api_key = get_provider_env(api_key_env) if api_key_env else ""

    if not base_url or not api_key:
        return [], None

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
        return results, None
    except httpx.TimeoutException:
        logger.warning("Custom backend timed out: %s", url)
    except httpx.HTTPStatusError as exc:
        logger.warning("Custom backend HTTP error: %s (%s)", exc, exc.response.text[:200])
        return [], exc.response.status_code
    except Exception as exc:
        logger.warning("Custom backend failed: %s", exc)
    return [], None


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


def _search_one_backend(backend: Dict[str, Any], query: str, limit: int) -> tuple[List[Dict[str, Any]], Optional[int]]:
    """Single-backend search worker for thread-pool execution.

    Returns ``(results, status_code)``. *status_code* carries the HTTP status
    for custom backends that fail with HTTP errors so the caller can invoke
    health-cache cooldowns.
    """
    name = str(backend.get("name", "?"))
    typ = str(backend.get("type", "") or "")

    try:
        if typ == "custom":
            results, status_code = _search_custom_backend(backend, query, limit)
        else:
            provider = _get_registered_provider(name)
            if provider is None:
                logger.warning("Backend '%s' not registered, skipping", name)
                return [], None
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
                return [], None
            status_code = None
        logger.info("Backend '%s' returned %d results", name, len(results))
        return results, status_code
    except Exception as exc:
        logger.warning("Backend '%s' error: %s", name, exc)
        return [], None


# ---------------------------------------------------------------------------
# Provider
# ---------------------------------------------------------------------------


class FederatedSearchProvider(WebSearchProvider):
    """Aggregated search provider that fans out to multiple sub-backends.

    Supports K-way aggregation with per-backend health checking,
    configurable re-rank prompts, and fault-tolerance via ``min_backends``
    and ``required`` flags.
    """

    def __init__(self) -> None:
        super().__init__()
        self._health_cache: Optional[_HealthCache] = None

    def _get_health_cache(self, config: Dict[str, Any]) -> _HealthCache:
        if self._health_cache is None:
            hc = config.get("health_check") or {}
            ttl = int(hc.get("ttl_seconds", _DEFAULT_HEALTH_TTL))
            self._health_cache = _HealthCache(ttl_seconds=ttl)
        return self._health_cache

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

            # ── config validation ──
            validation_error = _validate_config(config)
            if validation_error:
                return {"success": False, "error": validation_error}

            backends: List[Dict[str, Any]] = config["backends"]
            timeout = int(config.get("timeout", _DEFAULT_TIMEOUT))
            max_results = int(config.get("max_results", _DEFAULT_MAX_RESULTS))
            ranker_config = config.get("ranker")
            min_backends = int(config.get("min_backends", 1))

            # ── health checks (cached, TTL-based) ──
            # Only active when ``health_check`` is explicitly configured.
            # Without it, all backends are dispatched directly (backward compat).
            hc_config = config.get("health_check")
            health_cache: Optional[_HealthCache] = None
            if hc_config is not None:
                health_cache = self._get_health_cache(config)
                probe_timeout = int(hc_config.get("timeout", _DEFAULT_HEALTH_TIMEOUT))

                active_backends: List[Dict[str, Any]] = []
                for b in backends:
                    name = str(b.get("name", "?"))
                    cached = health_cache.is_available(name)
                    if cached is False:
                        logger.info("Backend '%s' cached as unavailable, skipping probe", name)
                        continue
                    if cached is True:
                        active_backends.append(b)
                        continue
                    available = _probe_backend(b, timeout=probe_timeout)
                    health_cache.set_available(name, available)
                    if available:
                        active_backends.append(b)
                    else:
                        logger.info("Backend '%s' probe failed, skipping", name)
            else:
                active_backends = list(backends)

            if not active_backends:
                return {"success": False, "error": "No backends available"}

            logger.info(
                "Federated search: '%s' (%d/%d backends active, timeout=%ds, max_results=%d)",
                query, len(active_backends), len(backends), timeout, max_results,
            )

            # ── parallel backend execution ──
            all_results: List[Dict[str, Any]] = []
            backend_results: Dict[str, tuple[List[Dict], Optional[int]]] = {}
            errors: List[str] = []
            start_time = time.time()

            pool = concurrent.futures.ThreadPoolExecutor(
                max_workers=min(len(active_backends), _MAX_K),
            )
            try:
                futures = {
                    pool.submit(_search_one_backend, b, query, limit): b
                    for b in active_backends
                }

                done, not_done = concurrent.futures.wait(
                    futures, timeout=timeout,
                )

                for f in not_done:
                    b = futures[f]
                    name = str(b.get("name", "?"))
                    errors.append(f"backend '{name}' timed out")
                    backend_results[name] = ([], None)
                    f.cancel()

                for f in done:
                    if is_interrupted():
                        break
                    b = futures[f]
                    name = str(b.get("name", "?"))
                    try:
                        results, status_code = f.result(timeout=2)
                        all_results.extend(results)
                        backend_results[name] = (results, status_code)
                        # Wire real search failures into the health cache so
                        # the documented 401/403/429 cooldown actually fires.
                        if health_cache is not None and status_code is not None:
                            health_cache.mark_failed(name, status_code)
                    except Exception as exc:
                        errors.append(f"backend '{name}' failed: {exc}")
                        backend_results[name] = ([], None)

                if not_done:
                    logger.warning(
                        "Federated search timed out after %ds, collected partial "
                        "results from %d/%d backends",
                        timeout, len(done), len(futures),
                    )
            finally:
                pool.shutdown(wait=False)

            # ── outcome evaluation ──
            successful = [
                name for name, (results, _) in backend_results.items()
                if results
            ]
            required_failed = []
            for b in backends:
                name = str(b.get("name", "?"))
                if b.get("required", False) and name not in successful:
                    required_failed.append(name)

            if required_failed:
                for name in required_failed:
                    if health_cache is not None:
                        health_cache.set_available(name, False)
                return {
                    "success": False,
                    "error": (
                        f"Required backend(s) failed: {', '.join(required_failed)}"
                    ),
                }

            if len(successful) < min_backends:
                return {
                    "success": False,
                    "error": (
                        f"Only {len(successful)}/{min_backends} backends succeeded "
                        f"(minimum {min_backends} required)"
                    ),
                }

            if not all_results:
                if errors:
                    return {"success": False, "error": "All backends failed: " + "; ".join(errors)}
                return {"success": True, "data": {"web": []}}

            # ── ranking ──
            rank_input_count = max(max_results + 5, _MAX_RANK_INPUT)
            rank_input = all_results[:rank_input_count]
            ranked = _rank_results(query, rank_input, ranker_config)

            # Top N
            output_count = min(limit, max_results)
            top = ranked[:output_count]
            for i, r in enumerate(top):
                r["position"] = i + 1

            logger.info(
                "Federated search: %d raw -> %d ranked (total %.1fs, %d/%d backends)",
                len(all_results), len(top), time.time() - start_time,
                len(successful), len(active_backends),
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
                "Configure backends under web.federated.backends in config.yaml. "
                "Supports K-way aggregation, health checks, and custom re-rank prompts."
            ),
            "env_vars": [],
        }
