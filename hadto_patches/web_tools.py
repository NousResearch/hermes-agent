#!/usr/bin/env python3
"""
Standalone Web Tools Module

This module provides generic web tools that work with multiple backend providers.
Backend is selected during ``hermes tools`` setup (web.backend in config.yaml).

Available tools:
- web_search_tool: Search the web for information
- web_extract_tool: Extract content from specific web pages
- web_crawl_tool: Crawl websites with specific instructions (Firecrawl only)

Backend compatibility:
- Firecrawl: https://docs.firecrawl.dev/introduction (search, extract, crawl)
- Parallel: https://docs.parallel.ai (search, extract)

LLM Processing:
- Uses OpenRouter API with Gemini 3 Flash Preview for intelligent content extraction
- Extracts key excerpts and creates markdown summaries to reduce token usage

Debug Mode:
- Set WEB_TOOLS_DEBUG=true to enable detailed logging
- Creates web_tools_debug_UUID.json in ./logs directory
- Captures all tool calls, results, and compression metrics

Usage:
    from web_tools import web_search_tool, web_extract_tool, web_crawl_tool
    
    # Search the web
    results = web_search_tool("Python machine learning libraries", limit=3)
    
    # Extract content from URLs  
    content = web_extract_tool(["https://example.com"], format="markdown")
    
    # Crawl a website
    crawl_data = web_crawl_tool("example.com", "Find contact information")
"""

import json
import logging
import os
import re
import asyncio
from typing import List, Dict, Any, Optional
from urllib.parse import unquote, urlsplit, urlunsplit
import httpx
from firecrawl import Firecrawl
from agent.auxiliary_client import async_call_llm, extract_content_or_reasoning
from tools.debug_helpers import DebugSession
from tools.url_safety import is_safe_url
from tools.website_policy import check_website_access

logger = logging.getLogger("tools.web_tools")

WEB_PROVIDER_ENV_KEYS = {
    "exa": ("EXA_API_KEY",),
    "parallel": ("PARALLEL_API_KEY",),
    "tavily": ("TAVILY_API_KEY",),
    "firecrawl": ("FIRECRAWL_API_KEY", "FIRECRAWL_API_URL"),
}


def _url_contains_embedded_secret(url: str) -> bool:
    """Return True when a URL appears to embed an API key or token."""
    from agent.redact import _PREFIX_RE

    decoded = unquote(url or "")
    return bool(_PREFIX_RE.search(url or "") or _PREFIX_RE.search(decoded))


# ─── Backend Selection ────────────────────────────────────────────────────────

def _has_env(name: str) -> bool:
    val = os.getenv(name)
    return bool(val and val.strip())

def _load_web_config() -> dict:
    """Load the ``web:`` section from ~/.hermes/config.yaml."""
    try:
        from hermes_cli.config import load_config
        return load_config().get("web", {})
    except (ImportError, Exception):
        return {}

def _get_backend() -> str:
    """Determine which web backend to use.

    Reads ``web.backend`` from config.yaml (set by ``hermes tools``).
    Falls back to whichever API key is present for users who configured
    keys manually without running setup.
    """
    configured = (_load_web_config().get("backend") or "").lower().strip()
    if configured in ("parallel", "firecrawl", "tavily", "exa"):
        return configured

    # Fallback for manual / legacy config — pick highest-priority backend
    # that has a key configured.  Order: firecrawl > parallel > tavily > exa.
    for backend, keys in [
        ("firecrawl", ("FIRECRAWL_API_KEY", "FIRECRAWL_API_URL")),
        ("parallel",  ("PARALLEL_API_KEY",)),
        ("tavily",    ("TAVILY_API_KEY",)),
        ("exa",       ("EXA_API_KEY",)),
    ]:
        if any(_has_env(k) for k in keys):
            return backend

    return "firecrawl"  # default (backward compat)


def _provider_is_available(provider: str) -> bool:
    keys = WEB_PROVIDER_ENV_KEYS.get(provider, ())
    return any(_has_env(key) for key in keys)


def get_web_provider_status() -> dict:
    configured_backend = (_load_web_config().get("backend") or "").lower().strip() or None
    available = []
    missing = []
    providers = {}
    for provider, keys in WEB_PROVIDER_ENV_KEYS.items():
        present_keys = [key for key in keys if _has_env(key)]
        entry = {
            "available": bool(present_keys),
            "configured": provider == configured_backend,
            "required_env_keys": list(keys),
            "present_env_keys": present_keys,
        }
        providers[provider] = entry
        if entry["available"]:
            available.append(provider)
        else:
            missing.append(provider)
    return {
        "configured_backend": configured_backend,
        "available_providers": available,
        "missing_providers": missing,
        "providers": providers,
    }


# ─── Firecrawl v2 option builders ───────────────────────────────────────────

_FIRECRAWL_SEARCH_SOURCES = {"web", "images", "news"}
_FIRECRAWL_SEARCH_CATEGORIES = {"github", "research", "pdf"}
_FIRECRAWL_FORMAT_ALIASES = {
    "markdown": "markdown",
    "md": "markdown",
    "html": "html",
    "raw": "rawHtml",
    "raw_html": "rawHtml",
    "rawHtml": "rawHtml",
    "rawhtml": "rawHtml",
    "links": "links",
    "images": "images",
    "screenshot": "screenshot",
    "summary": "summary",
}
_FIRECRAWL_PDF_PARSER_MODES = {"fast", "auto", "ocr"}


def _as_str_list(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, str):
        values = [value]
    elif isinstance(value, (list, tuple, set)):
        values = list(value)
    else:
        return []
    result = []
    for item in values:
        if isinstance(item, str):
            stripped = item.strip()
            if stripped:
                result.append(stripped)
    return result


def _unique_in_order(values: List[str]) -> List[str]:
    seen = set()
    result = []
    for value in values:
        if value not in seen:
            seen.add(value)
            result.append(value)
    return result


def _coerce_int(value: Any, *, minimum: int | None = None, maximum: int | None = None) -> Optional[int]:
    if value is None or isinstance(value, bool):
        return None
    try:
        number = int(value)
    except (TypeError, ValueError):
        return None
    if minimum is not None and number < minimum:
        return None
    if maximum is not None and number > maximum:
        number = maximum
    return number


def _coerce_float(value: Any, *, minimum: float | None = None) -> Optional[float]:
    if value is None or isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if minimum is not None and number < minimum:
        return None
    return number


def _normalize_firecrawl_domain(value: str) -> Optional[str]:
    if not isinstance(value, str):
        return None
    candidate = value.strip()
    if not candidate or _url_contains_embedded_secret(candidate):
        return None
    parsed = urlsplit(candidate if "://" in candidate else f"//{candidate}")
    host = (parsed.hostname or "").lower().strip(".")
    if not host:
        return None
    return host


def _normalize_firecrawl_domains(values: Any) -> List[str]:
    domains = []
    for value in _as_str_list(values):
        domain = _normalize_firecrawl_domain(value)
        if domain:
            domains.append(domain)
    return _unique_in_order(domains)


def _build_firecrawl_formats(
    format: Optional[str] = None,
    formats: Optional[List[str]] = None,
    *,
    default: Optional[List[str]] = None,
) -> List[str]:
    requested = _as_str_list(formats)
    if not requested and format:
        requested = [format]
    if not requested and default is not None:
        requested = default

    normalized = []
    for value in requested:
        mapped = _FIRECRAWL_FORMAT_ALIASES.get(value, _FIRECRAWL_FORMAT_ALIASES.get(value.lower()))
        if mapped:
            normalized.append(mapped)
    return _unique_in_order(normalized)




def _get_firecrawl_result_value(obj: Any, *names: str) -> Any:
    """Read a Firecrawl response field from dicts or SDK objects."""
    if obj is None:
        return None
    if hasattr(obj, "model_dump"):
        try:
            obj = obj.model_dump()
        except Exception:
            pass
    for name in names:
        if isinstance(obj, dict) and name in obj:
            return obj.get(name)
        if hasattr(obj, name):
            return getattr(obj, name)
    return None


def _compact_firecrawl_aux_content(format_name: str, value: Any) -> str:
    """Serialize non-text Firecrawl formats into compact, safe text content."""
    if value in (None, "", [], {}):
        return ""
    if isinstance(value, str):
        return clean_base64_images(f"## Firecrawl {format_name}\n\n{value}")
    try:
        serialized = json.dumps(value, ensure_ascii=False, indent=2)
    except TypeError:
        serialized = json.dumps(str(value), ensure_ascii=False)
    return clean_base64_images(f"## Firecrawl {format_name}\n\n```json\n{serialized}\n```")


def _choose_firecrawl_content(content_by_format: Dict[str, Any], preferred_formats: List[str]) -> str:
    """Pick requested Firecrawl content, including compact non-text formats."""
    if not preferred_formats:
        preferred_formats = ["markdown", "html"]
    for preferred in preferred_formats:
        value = content_by_format.get(preferred)
        if not value:
            continue
        if preferred in {"links", "images", "screenshot"}:
            compact = _compact_firecrawl_aux_content(preferred, value)
            if compact:
                return compact
        elif isinstance(value, str):
            return value
        else:
            compact = _compact_firecrawl_aux_content(preferred, value)
            if compact:
                return compact
    for fallback in ("markdown", "html", "rawHtml", "summary", "links", "images", "screenshot"):
        value = content_by_format.get(fallback)
        if value:
            return _compact_firecrawl_aux_content(fallback, value) if fallback in {"links", "images", "screenshot"} else value
    return ""

def _sanitize_firecrawl_actions(actions: Any) -> List[Dict[str, Any]]:
    """Allow low-risk browser setup actions, but not text entry, JS, or PDF generation."""
    if not isinstance(actions, list):
        return []

    sanitized: List[Dict[str, Any]] = []
    for action in actions:
        if not isinstance(action, dict):
            continue
        action_type = str(action.get("type", "")).strip()
        if action_type == "wait":
            milliseconds = _coerce_int(action.get("milliseconds"), minimum=1, maximum=30000)
            selector = action.get("selector") if isinstance(action.get("selector"), str) else None
            if milliseconds is not None:
                sanitized.append({"type": "wait", "milliseconds": milliseconds})
            elif selector:
                sanitized.append({"type": "wait", "selector": selector.strip()})
        elif action_type == "click":
            selector = action.get("selector")
            if isinstance(selector, str) and selector.strip():
                sanitized.append({"type": "click", "selector": selector.strip()})
        elif action_type == "press":
            key = action.get("key")
            if isinstance(key, str) and 0 < len(key.strip()) <= 40:
                sanitized.append({"type": "press", "key": key.strip()})
        elif action_type == "scroll":
            direction = action.get("direction", "down")
            if direction not in ("up", "down"):
                continue
            scroll_action = {"type": "scroll", "direction": direction}
            selector = action.get("selector")
            if isinstance(selector, str) and selector.strip():
                scroll_action["selector"] = selector.strip()
            sanitized.append(scroll_action)
        elif action_type == "screenshot":
            screenshot_action: Dict[str, Any] = {"type": "screenshot"}
            if isinstance(action.get("full_page"), bool):
                screenshot_action["full_page"] = action["full_page"]
            elif isinstance(action.get("fullPage"), bool):
                screenshot_action["full_page"] = action["fullPage"]
            quality = _coerce_int(action.get("quality"), minimum=1, maximum=100)
            if quality is not None:
                screenshot_action["quality"] = quality
            viewport = action.get("viewport")
            if isinstance(viewport, dict):
                width = _coerce_int(viewport.get("width"), minimum=1, maximum=5000)
                height = _coerce_int(viewport.get("height"), minimum=1, maximum=5000)
                if width is not None and height is not None:
                    screenshot_action["viewport"] = {"width": width, "height": height}
            sanitized.append(screenshot_action)
    return sanitized


def _build_firecrawl_scrape_kwargs(
    format: str = None,
    formats: Optional[List[str]] = None,
    only_main_content: Optional[bool] = None,
    only_clean_content: Optional[bool] = None,
    include_tags: Optional[List[str]] = None,
    exclude_tags: Optional[List[str]] = None,
    max_age: Optional[int] = None,
    min_age: Optional[int] = None,
    wait_for: Optional[int] = None,
    mobile: Optional[bool] = None,
    timeout: Optional[int] = None,
    pdf_parser_mode: Optional[str] = None,
    pdf_max_pages: Optional[int] = None,
    actions: Optional[List[Dict[str, Any]]] = None,
    location: Optional[str] = None,
    default_formats: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Build safe Firecrawl scrape kwargs.

    Firecrawl onlyMainContent defaults true. onlyCleanContent is an LLM cleaner
    beta and is not compatible with zero-data-retention. maxAge uses fresh-enough
    cache; minAge is cache-only and can return SCRAPE_NO_CACHED_DATA.
    """
    kwargs: Dict[str, Any] = {}

    built_formats = _build_firecrawl_formats(format, formats, default=default_formats)
    if built_formats:
        kwargs["formats"] = built_formats

    if only_main_content is not None:
        kwargs["only_main_content"] = bool(only_main_content)
    if only_clean_content is not None:
        kwargs["only_clean_content"] = bool(only_clean_content)

    tags = _as_str_list(include_tags)
    if tags:
        kwargs["include_tags"] = tags
    tags = _as_str_list(exclude_tags)
    if tags:
        kwargs["exclude_tags"] = tags

    int_value = _coerce_int(max_age, minimum=0)
    if int_value is not None:
        kwargs["max_age"] = int_value
    int_value = _coerce_int(min_age, minimum=1)
    if int_value is not None:
        kwargs["min_age"] = int_value
    int_value = _coerce_int(wait_for, minimum=0)
    if int_value is not None:
        kwargs["wait_for"] = int_value
    if mobile is not None:
        kwargs["mobile"] = bool(mobile)
    int_value = _coerce_int(timeout, minimum=1000, maximum=300000)
    if int_value is not None:
        kwargs["timeout"] = int_value

    parser: Dict[str, Any] = {"type": "pdf"}
    mode = pdf_parser_mode.strip().lower() if isinstance(pdf_parser_mode, str) else None
    if mode in _FIRECRAWL_PDF_PARSER_MODES:
        parser["mode"] = mode
    pages = _coerce_int(pdf_max_pages, minimum=1, maximum=10000)
    if pages is not None:
        parser["max_pages"] = pages
    if len(parser) > 1:
        kwargs["parsers"] = [parser]

    clean_actions = _sanitize_firecrawl_actions(actions)
    if clean_actions:
        kwargs["actions"] = clean_actions

    if isinstance(location, str) and location.strip():
        kwargs["location"] = location.strip()

    return kwargs


def _build_firecrawl_search_kwargs(
    query: str,
    limit: int = 5,
    sources: Optional[List[str]] = None,
    categories: Optional[List[str]] = None,
    include_domains: Optional[List[str]] = None,
    exclude_domains: Optional[List[str]] = None,
    tbs: Optional[str] = None,
    location: Optional[str] = None,
) -> Dict[str, Any]:
    kwargs: Dict[str, Any] = {"query": query}
    safe_limit = _coerce_int(limit, minimum=1, maximum=100)
    if safe_limit is not None:
        kwargs["limit"] = safe_limit

    safe_sources = [
        source for source in _as_str_list(sources)
        if source in _FIRECRAWL_SEARCH_SOURCES
    ]
    if safe_sources:
        kwargs["sources"] = _unique_in_order(safe_sources)

    safe_categories = [
        category for category in _as_str_list(categories)
        if category in _FIRECRAWL_SEARCH_CATEGORIES
    ]
    if safe_categories:
        kwargs["categories"] = _unique_in_order(safe_categories)

    includes = _normalize_firecrawl_domains(include_domains)
    excludes = _normalize_firecrawl_domains(exclude_domains)
    # Firecrawl documents includeDomains/excludeDomains as mutually exclusive.
    if includes:
        kwargs["include_domains"] = includes
    elif excludes:
        kwargs["exclude_domains"] = excludes

    if isinstance(tbs, str) and tbs.strip():
        kwargs["tbs"] = tbs.strip()
    if isinstance(location, str) and location.strip():
        kwargs["location"] = location.strip()

    return kwargs


def _build_firecrawl_crawl_kwargs(
    instructions: str = None,
    limit: int = 20,
    include_paths: Optional[List[str]] = None,
    exclude_paths: Optional[List[str]] = None,
    max_discovery_depth: Optional[int] = None,
    sitemap: Optional[str] = None,
    ignore_query_parameters: Optional[bool] = None,
    regex_on_full_url: Optional[bool] = None,
    crawl_entire_domain: Optional[bool] = None,
    allow_external_links: Optional[bool] = None,
    allow_subdomains: Optional[bool] = None,
    delay: Optional[float] = None,
    max_concurrency: Optional[int] = None,
    scrape_options: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    kwargs: Dict[str, Any] = {}
    if isinstance(instructions, str) and instructions.strip():
        kwargs["prompt"] = instructions.strip()

    safe_limit = _coerce_int(limit, minimum=1, maximum=10000)
    if safe_limit is not None:
        kwargs["limit"] = safe_limit

    paths = _as_str_list(include_paths)
    if paths:
        kwargs["include_paths"] = paths
    paths = _as_str_list(exclude_paths)
    if paths:
        kwargs["exclude_paths"] = paths

    int_value = _coerce_int(max_discovery_depth, minimum=0)
    if int_value is not None:
        kwargs["max_discovery_depth"] = int_value
    if sitemap in {"skip", "include", "only"}:
        kwargs["sitemap"] = sitemap

    if ignore_query_parameters is not None:
        kwargs["ignore_query_parameters"] = bool(ignore_query_parameters)
    if regex_on_full_url is not None:
        kwargs["regex_on_full_url"] = bool(regex_on_full_url)
    if crawl_entire_domain is not None:
        kwargs["crawl_entire_domain"] = bool(crawl_entire_domain)
    if allow_external_links is not None:
        kwargs["allow_external_links"] = bool(allow_external_links)
    if allow_subdomains is not None:
        kwargs["allow_subdomains"] = bool(allow_subdomains)

    delay_value = _coerce_float(delay, minimum=0)
    if delay_value is not None:
        kwargs["delay"] = delay_value
    int_value = _coerce_int(max_concurrency, minimum=1, maximum=100)
    if int_value is not None:
        kwargs["max_concurrency"] = int_value

    nested = scrape_options if isinstance(scrape_options, dict) else {}
    scrape_kwargs = _build_firecrawl_scrape_kwargs(
        format=nested.get("format"),
        formats=nested.get("formats"),
        only_main_content=nested.get("only_main_content"),
        only_clean_content=nested.get("only_clean_content"),
        include_tags=nested.get("include_tags"),
        exclude_tags=nested.get("exclude_tags"),
        max_age=nested.get("max_age"),
        min_age=nested.get("min_age"),
        wait_for=nested.get("wait_for"),
        mobile=nested.get("mobile"),
        timeout=nested.get("timeout"),
        pdf_parser_mode=nested.get("pdf_parser_mode"),
        pdf_max_pages=nested.get("pdf_max_pages"),
        actions=nested.get("actions"),
        location=nested.get("location"),
        default_formats=["markdown"],
    )
    kwargs["scrape_options"] = scrape_kwargs

    return kwargs

# ─── Firecrawl Client ────────────────────────────────────────────────────────

_firecrawl_client = None

def _get_firecrawl_client():
    """Get or create the Firecrawl client (lazy initialization).

    Uses the cloud API by default (requires FIRECRAWL_API_KEY).
    Set FIRECRAWL_API_URL to point at a self-hosted instance instead —
    in that case the API key is optional (set USE_DB_AUTHENTICATION=false
    on your Firecrawl server to disable auth entirely).
    """
    global _firecrawl_client
    if _firecrawl_client is None:
        api_key = os.getenv("FIRECRAWL_API_KEY")
        api_url = os.getenv("FIRECRAWL_API_URL")
        if not api_key and not api_url:
            logger.error("Firecrawl client initialization failed: missing configuration.")
            raise ValueError(
                "Firecrawl client not configured. "
                "Set FIRECRAWL_API_KEY (cloud) or FIRECRAWL_API_URL (self-hosted). "
                "This tool requires Firecrawl to be available."
            )
        kwargs = {}
        if api_key:
            kwargs["api_key"] = api_key
        if api_url:
            kwargs["api_url"] = api_url
        _firecrawl_client = Firecrawl(**kwargs)
    return _firecrawl_client

# ─── Parallel Client ─────────────────────────────────────────────────────────

_parallel_client = None
_async_parallel_client = None

def _get_parallel_client():
    """Get or create the Parallel sync client (lazy initialization).

    Requires PARALLEL_API_KEY environment variable.
    """
    from parallel import Parallel
    global _parallel_client
    if _parallel_client is None:
        api_key = os.getenv("PARALLEL_API_KEY")
        if not api_key:
            raise ValueError(
                "PARALLEL_API_KEY environment variable not set. "
                "Get your API key at https://parallel.ai"
            )
        _parallel_client = Parallel(api_key=api_key)
    return _parallel_client


def _get_async_parallel_client():
    """Get or create the Parallel async client (lazy initialization).

    Requires PARALLEL_API_KEY environment variable.
    """
    from parallel import AsyncParallel
    global _async_parallel_client
    if _async_parallel_client is None:
        api_key = os.getenv("PARALLEL_API_KEY")
        if not api_key:
            raise ValueError(
                "PARALLEL_API_KEY environment variable not set. "
                "Get your API key at https://parallel.ai"
            )
        _async_parallel_client = AsyncParallel(api_key=api_key)
    return _async_parallel_client

# ─── Tavily Client ───────────────────────────────────────────────────────────

_TAVILY_BASE_URL = "https://api.tavily.com"


def _tavily_request(endpoint: str, payload: dict) -> dict:
    """Send a POST request to the Tavily API.

    Auth is provided via ``api_key`` in the JSON body (no header-based auth).
    Raises ``ValueError`` if ``TAVILY_API_KEY`` is not set.
    """
    api_key = os.getenv("TAVILY_API_KEY")
    if not api_key:
        raise ValueError(
            "TAVILY_API_KEY environment variable not set. "
            "Get your API key at https://app.tavily.com/home"
        )
    payload["api_key"] = api_key
    url = f"{_TAVILY_BASE_URL}/{endpoint.lstrip('/')}"
    logger.info("Tavily %s request to %s", endpoint, url)
    response = httpx.post(url, json=payload, timeout=60)
    response.raise_for_status()
    return response.json()


def _tavily_search(query: str, limit: int = 5) -> dict:
    logger.info("Tavily search: '%s' (limit: %d)", query, limit)
    raw = _tavily_request(
        "search",
        {
            "query": query,
            "max_results": min(limit, 20),
            "include_raw_content": False,
            "include_images": False,
        },
    )
    return _normalize_tavily_search_results(raw)


def _normalize_tavily_search_results(response: dict) -> dict:
    """Normalize Tavily /search response to the standard web search format.

    Tavily returns ``{results: [{title, url, content, score, ...}]}``.
    We map to ``{success, data: {web: [{title, url, description, position}]}}``.
    """
    web_results = []
    for i, result in enumerate(response.get("results", [])):
        web_results.append({
            "title": result.get("title", ""),
            "url": result.get("url", ""),
            "description": result.get("content", ""),
            "position": i + 1,
        })
    return {"success": True, "data": {"web": web_results}}


def _normalize_tavily_documents(response: dict, fallback_url: str = "") -> List[Dict[str, Any]]:
    """Normalize Tavily /extract or /crawl response to the standard document format.

    Maps results to ``{url, title, content, raw_content, metadata}`` and
    includes any ``failed_results`` / ``failed_urls`` as error entries.
    """
    documents: List[Dict[str, Any]] = []
    for result in response.get("results", []):
        url = result.get("url", fallback_url)
        raw = result.get("raw_content", "") or result.get("content", "")
        documents.append({
            "url": url,
            "title": result.get("title", ""),
            "content": raw,
            "raw_content": raw,
            "metadata": {"sourceURL": url, "title": result.get("title", "")},
        })
    # Handle failed results
    for fail in response.get("failed_results", []):
        documents.append({
            "url": fail.get("url", fallback_url),
            "title": "",
            "content": "",
            "raw_content": "",
            "error": fail.get("error", "extraction failed"),
            "metadata": {"sourceURL": fail.get("url", fallback_url)},
        })
    for fail_url in response.get("failed_urls", []):
        url_str = fail_url if isinstance(fail_url, str) else str(fail_url)
        documents.append({
            "url": url_str,
            "title": "",
            "content": "",
            "raw_content": "",
            "error": "extraction failed",
            "metadata": {"sourceURL": url_str},
        })
    return documents


DEFAULT_MIN_LENGTH_FOR_SUMMARIZATION = 5000

# Allow per-task override via env var
DEFAULT_SUMMARIZER_MODEL = os.getenv("AUXILIARY_WEB_EXTRACT_MODEL", "").strip() or None

_debug = DebugSession("web_tools", env_var="WEB_TOOLS_DEBUG")


async def process_content_with_llm(
    content: str, 
    url: str = "", 
    title: str = "",
    model: str = DEFAULT_SUMMARIZER_MODEL,
    min_length: int = DEFAULT_MIN_LENGTH_FOR_SUMMARIZATION
) -> Optional[str]:
    """
    Process web content using LLM to create intelligent summaries with key excerpts.
    
    This function uses Gemini 3 Flash Preview (or specified model) via OpenRouter API 
    to intelligently extract key information and create markdown summaries,
    significantly reducing token usage while preserving all important information.
    
    For very large content (>500k chars), uses chunked processing with synthesis.
    For extremely large content (>2M chars), refuses to process entirely.
    
    Args:
        content (str): The raw content to process
        url (str): The source URL (for context, optional)
        title (str): The page title (for context, optional)
        model (str): The model to use for processing (default: google/gemini-3-flash-preview)
        min_length (int): Minimum content length to trigger processing (default: 5000)
        
    Returns:
        Optional[str]: Processed markdown content, or None if content too short or processing fails
    """
    # Size thresholds
    MAX_CONTENT_SIZE = 2_000_000  # 2M chars - refuse entirely above this
    CHUNK_THRESHOLD = 500_000     # 500k chars - use chunked processing above this
    CHUNK_SIZE = 100_000          # 100k chars per chunk
    MAX_OUTPUT_SIZE = 5000        # Hard cap on final output size
    
    try:
        content_len = len(content)
        
        # Refuse if content is absurdly large
        if content_len > MAX_CONTENT_SIZE:
            size_mb = content_len / 1_000_000
            logger.warning("Content too large (%.1fMB > 2MB limit). Refusing to process.", size_mb)
            return f"[Content too large to process: {size_mb:.1f}MB. Try using web_crawl with specific extraction instructions, or search for a more focused source.]"
        
        # Skip processing if content is too short
        if content_len < min_length:
            logger.debug("Content too short (%d < %d chars), skipping LLM processing", content_len, min_length)
            return None
        
        # Create context information
        context_info = []
        if title:
            context_info.append(f"Title: {title}")
        if url:
            context_info.append(f"Source: {url}")
        context_str = "\n".join(context_info) + "\n\n" if context_info else ""
        
        # Check if we need chunked processing
        if content_len > CHUNK_THRESHOLD:
            logger.info("Content large (%d chars). Using chunked processing...", content_len)
            return await _process_large_content_chunked(
                content, context_str, model, CHUNK_SIZE, MAX_OUTPUT_SIZE
            )
        
        # Standard single-pass processing for normal content
        logger.info("Processing content with LLM (%d characters)", content_len)
        
        processed_content = await _call_summarizer_llm(content, context_str, model)
        
        if processed_content:
            # Enforce output cap
            if len(processed_content) > MAX_OUTPUT_SIZE:
                processed_content = processed_content[:MAX_OUTPUT_SIZE] + "\n\n[... summary truncated for context management ...]"
            
            # Log compression metrics
            processed_length = len(processed_content)
            compression_ratio = processed_length / content_len if content_len > 0 else 1.0
            logger.info("Content processed: %d -> %d chars (%.1f%%)", content_len, processed_length, compression_ratio * 100)
        
        return processed_content
        
    except Exception as e:
        logger.debug("Error processing content with LLM: %s", e)
        return f"[Failed to process content: {str(e)[:100]}. Content size: {len(content):,} chars]"


async def _call_summarizer_llm(
    content: str, 
    context_str: str, 
    model: str, 
    max_tokens: int = 20000,
    is_chunk: bool = False,
    chunk_info: str = ""
) -> Optional[str]:
    """
    Make a single LLM call to summarize content.
    
    Args:
        content: The content to summarize
        context_str: Context information (title, URL)
        model: Model to use
        max_tokens: Maximum output tokens
        is_chunk: Whether this is a chunk of a larger document
        chunk_info: Information about chunk position (e.g., "Chunk 2/5")
        
    Returns:
        Summarized content or None on failure
    """
    if is_chunk:
        # Chunk-specific prompt - aware that this is partial content
        system_prompt = """You are an expert content analyst processing a SECTION of a larger document. Your job is to extract and summarize the key information from THIS SECTION ONLY.

Important guidelines for chunk processing:
1. Do NOT write introductions or conclusions - this is a partial document
2. Focus on extracting ALL key facts, figures, data points, and insights from this section
3. Preserve important quotes, code snippets, and specific details verbatim
4. Use bullet points and structured formatting for easy synthesis later
5. Note any references to other sections (e.g., "as mentioned earlier", "see below") without trying to resolve them

Your output will be combined with summaries of other sections, so focus on thorough extraction rather than narrative flow."""

        user_prompt = f"""Extract key information from this SECTION of a larger document:

{context_str}{chunk_info}

SECTION CONTENT:
{content}

Extract all important information from this section in a structured format. Focus on facts, data, insights, and key details. Do not add introductions or conclusions."""

    else:
        # Standard full-document prompt
        system_prompt = """You are an expert content analyst. Your job is to process web content and create a comprehensive yet concise summary that preserves all important information while dramatically reducing bulk.

Create a well-structured markdown summary that includes:
1. Key excerpts (quotes, code snippets, important facts) in their original format
2. Comprehensive summary of all other important information
3. Proper markdown formatting with headers, bullets, and emphasis

Your goal is to preserve ALL important information while reducing length. Never lose key facts, figures, insights, or actionable information. Make it scannable and well-organized."""

        user_prompt = f"""Please process this web content and create a comprehensive markdown summary:

{context_str}CONTENT TO PROCESS:
{content}

Create a markdown summary that captures all key information in a well-organized, scannable format. Include important quotes and code snippets in their original formatting. Focus on actionable information, specific details, and unique insights."""

    # Call the LLM with retry logic
    max_retries = 6
    retry_delay = 2
    last_error = None

    for attempt in range(max_retries):
        try:
            call_kwargs = {
                "task": "web_extract",
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                "temperature": 0.1,
                "max_tokens": max_tokens,
            }
            if model:
                call_kwargs["model"] = model
            response = await async_call_llm(**call_kwargs)
            content = extract_content_or_reasoning(response)
            if content:
                return content
            # Reasoning-only / empty response — let the retry loop handle it
            logger.warning("LLM returned empty content (attempt %d/%d), retrying", attempt + 1, max_retries)
            if attempt < max_retries - 1:
                await asyncio.sleep(retry_delay)
                retry_delay = min(retry_delay * 2, 60)
                continue
            return content  # Return whatever we got after exhausting retries
        except RuntimeError:
            logger.warning("No auxiliary model available for web content processing")
            return None
        except Exception as api_error:
            last_error = api_error
            if attempt < max_retries - 1:
                logger.warning("LLM API call failed (attempt %d/%d): %s", attempt + 1, max_retries, str(api_error)[:100])
                logger.warning("Retrying in %ds...", retry_delay)
                await asyncio.sleep(retry_delay)
                retry_delay = min(retry_delay * 2, 60)
            else:
                raise last_error
    
    return None


async def _process_large_content_chunked(
    content: str, 
    context_str: str, 
    model: str, 
    chunk_size: int,
    max_output_size: int
) -> Optional[str]:
    """
    Process large content by chunking, summarizing each chunk in parallel,
    then synthesizing the summaries.
    
    Args:
        content: The large content to process
        context_str: Context information
        model: Model to use
        chunk_size: Size of each chunk in characters
        max_output_size: Maximum final output size
        
    Returns:
        Synthesized summary or None on failure
    """
    # Split content into chunks
    chunks = []
    for i in range(0, len(content), chunk_size):
        chunk = content[i:i + chunk_size]
        chunks.append(chunk)
    
    logger.info("Split into %d chunks of ~%d chars each", len(chunks), chunk_size)
    
    # Summarize each chunk in parallel
    async def summarize_chunk(chunk_idx: int, chunk_content: str) -> tuple[int, Optional[str]]:
        """Summarize a single chunk."""
        try:
            chunk_info = f"[Processing chunk {chunk_idx + 1} of {len(chunks)}]"
            summary = await _call_summarizer_llm(
                chunk_content, 
                context_str, 
                model, 
                max_tokens=10000,
                is_chunk=True,
                chunk_info=chunk_info
            )
            if summary:
                logger.info("Chunk %d/%d summarized: %d -> %d chars", chunk_idx + 1, len(chunks), len(chunk_content), len(summary))
            return chunk_idx, summary
        except Exception as e:
            logger.warning("Chunk %d/%d failed: %s", chunk_idx + 1, len(chunks), str(e)[:50])
            return chunk_idx, None
    
    # Run all chunk summarizations in parallel
    tasks = [summarize_chunk(i, chunk) for i, chunk in enumerate(chunks)]
    results = await asyncio.gather(*tasks)
    
    # Collect successful summaries in order
    summaries = []
    for chunk_idx, summary in sorted(results, key=lambda x: x[0]):
        if summary:
            summaries.append(f"## Section {chunk_idx + 1}\n{summary}")
    
    if not summaries:
        logger.debug("All chunk summarizations failed")
        return "[Failed to process large content: all chunk summarizations failed]"
    
    logger.info("Got %d/%d chunk summaries", len(summaries), len(chunks))
    
    # If only one chunk succeeded, just return it (with cap)
    if len(summaries) == 1:
        result = summaries[0]
        if len(result) > max_output_size:
            result = result[:max_output_size] + "\n\n[... truncated ...]"
        return result
    
    # Synthesize the summaries into a final summary
    logger.info("Synthesizing %d summaries...", len(summaries))
    
    combined_summaries = "\n\n---\n\n".join(summaries)
    
    synthesis_prompt = f"""You have been given summaries of different sections of a large document. 
Synthesize these into ONE cohesive, comprehensive summary that:
1. Removes redundancy between sections
2. Preserves all key facts, figures, and actionable information
3. Is well-organized with clear structure
4. Is under {max_output_size} characters

{context_str}SECTION SUMMARIES:
{combined_summaries}

Create a single, unified markdown summary."""

    try:
        call_kwargs = {
            "task": "web_extract",
            "messages": [
                {"role": "system", "content": "You synthesize multiple summaries into one cohesive, comprehensive summary. Be thorough but concise."},
                {"role": "user", "content": synthesis_prompt}
            ],
            "temperature": 0.1,
            "max_tokens": 20000,
        }
        if model:
            call_kwargs["model"] = model
        response = await async_call_llm(**call_kwargs)
        final_summary = extract_content_or_reasoning(response)

        # Retry once on empty content (reasoning-only response)
        if not final_summary:
            logger.warning("Synthesis LLM returned empty content, retrying once")
            response = await async_call_llm(**call_kwargs)
            final_summary = extract_content_or_reasoning(response)

        # Enforce hard cap
        if len(final_summary) > max_output_size:
            final_summary = final_summary[:max_output_size] + "\n\n[... summary truncated for context management ...]"
        
        original_len = len(content)
        final_len = len(final_summary)
        compression = final_len / original_len if original_len > 0 else 1.0
        
        logger.info("Synthesis complete: %d -> %d chars (%.2f%%)", original_len, final_len, compression * 100)
        return final_summary
        
    except Exception as e:
        logger.warning("Synthesis failed: %s", str(e)[:100])
        # Fall back to concatenated summaries with truncation
        fallback = "\n\n".join(summaries)
        if len(fallback) > max_output_size:
            fallback = fallback[:max_output_size] + "\n\n[... truncated due to synthesis failure ...]"
        return fallback


def clean_base64_images(text: str) -> str:
    """
    Remove base64 encoded images from text to reduce token count and clutter.
    
    This function finds and removes base64 encoded images in various formats:
    - (data:image/png;base64,...)
    - (data:image/jpeg;base64,...)
    - (data:image/svg+xml;base64,...)
    - data:image/[type];base64,... (without parentheses)
    
    Args:
        text: The text content to clean
        
    Returns:
        Cleaned text with base64 images replaced with placeholders
    """
    # Pattern to match base64 encoded images wrapped in parentheses
    # Matches: (data:image/[type];base64,[base64-string])
    base64_with_parens_pattern = r'\(data:image/[^;]+;base64,[A-Za-z0-9+/=]+\)'
    
    # Pattern to match base64 encoded images without parentheses
    # Matches: data:image/[type];base64,[base64-string]
    base64_pattern = r'data:image/[^;]+;base64,[A-Za-z0-9+/=]+'
    
    # Replace parentheses-wrapped images first
    cleaned_text = re.sub(base64_with_parens_pattern, '[BASE64_IMAGE_REMOVED]', text)
    
    # Then replace any remaining non-parentheses images
    cleaned_text = re.sub(base64_pattern, '[BASE64_IMAGE_REMOVED]', cleaned_text)
    
    return cleaned_text


# ─── Exa Client ──────────────────────────────────────────────────────────────

_exa_client = None

def _get_exa_client():
    """Get or create the Exa client (lazy initialization).

    Requires EXA_API_KEY environment variable.
    """
    from exa_py import Exa
    global _exa_client
    if _exa_client is None:
        api_key = os.getenv("EXA_API_KEY")
        if not api_key:
            raise ValueError(
                "EXA_API_KEY environment variable not set. "
                "Get your API key at https://exa.ai"
            )
        _exa_client = Exa(api_key=api_key)
        _exa_client.headers["x-exa-integration"] = "hermes-agent"
    return _exa_client


# ─── Exa Search & Extract Helpers ─────────────────────────────────────────────

def _exa_search(query: str, limit: int = 10) -> dict:
    """Search using the Exa SDK and return results as a dict."""
    from tools.interrupt import is_interrupted
    if is_interrupted():
        return {"error": "Interrupted", "success": False}

    logger.info("Exa search: '%s' (limit=%d)", query, limit)
    response = _get_exa_client().search(
        query,
        num_results=limit,
        contents={
            "highlights": True,
        },
    )

    web_results = []
    for i, result in enumerate(response.results or []):
        highlights = result.highlights or []
        web_results.append({
            "url": result.url or "",
            "title": result.title or "",
            "description": " ".join(highlights) if highlights else "",
            "position": i + 1,
        })

    return {"success": True, "data": {"web": web_results}}


def _exa_extract(urls: List[str]) -> List[Dict[str, Any]]:
    """Extract content from URLs using the Exa SDK.

    Returns a list of result dicts matching the structure expected by the
    LLM post-processing pipeline (url, title, content, metadata).
    """
    from tools.interrupt import is_interrupted
    if is_interrupted():
        return [{"url": u, "error": "Interrupted", "title": ""} for u in urls]

    logger.info("Exa extract: %d URL(s)", len(urls))
    response = _get_exa_client().get_contents(
        urls,
        text=True,
    )

    results = []
    for result in response.results or []:
        content = result.text or ""
        url = result.url or ""
        title = result.title or ""
        results.append({
            "url": url,
            "title": title,
            "content": content,
            "raw_content": content,
            "metadata": {"sourceURL": url, "title": title},
        })

    return results


# ─── Parallel Search & Extract Helpers ────────────────────────────────────────

def _parallel_search(query: str, limit: int = 5) -> dict:
    """Search using the Parallel SDK and return results as a dict."""
    from tools.interrupt import is_interrupted
    if is_interrupted():
        return {"error": "Interrupted", "success": False}

    mode = os.getenv("PARALLEL_SEARCH_MODE", "agentic").lower().strip()
    if mode not in ("fast", "one-shot", "agentic"):
        mode = "agentic"

    logger.info("Parallel search: '%s' (mode=%s, limit=%d)", query, mode, limit)
    response = _get_parallel_client().beta.search(
        search_queries=[query],
        objective=query,
        mode=mode,
        max_results=min(limit, 20),
    )

    web_results = []
    for i, result in enumerate(response.results or []):
        excerpts = result.excerpts or []
        web_results.append({
            "url": result.url or "",
            "title": result.title or "",
            "description": " ".join(excerpts) if excerpts else "",
            "position": i + 1,
        })

    return {"success": True, "data": {"web": web_results}}


def _to_plain_dict(value: Any) -> Dict[str, Any]:
    if isinstance(value, dict):
        return dict(value)
    if hasattr(value, "model_dump"):
        return value.model_dump()
    if hasattr(value, "__dict__"):
        return dict(value.__dict__)
    return {}


def _firecrawl_scrape_payload(scrape_kwargs: Dict[str, Any]) -> Dict[str, Any]:
    payload: Dict[str, Any] = {}
    field_map = {
        "include_tags": "includeTags",
        "exclude_tags": "excludeTags",
        "only_main_content": "onlyMainContent",
        "only_clean_content": "onlyCleanContent",
        "max_age": "maxAge",
        "min_age": "minAge",
        "wait_for": "waitFor",
        "mobile": "mobile",
        "timeout": "timeout",
        "location": "location",
    }
    for key, value in scrape_kwargs.items():
        if key == "formats":
            payload["formats"] = value
        elif key == "parsers":
            converted = []
            for parser in value:
                if isinstance(parser, dict):
                    parser_data = dict(parser)
                    if "max_pages" in parser_data:
                        parser_data["maxPages"] = parser_data.pop("max_pages")
                    converted.append(parser_data)
                else:
                    converted.append(parser)
            payload["parsers"] = converted
        elif key == "actions":
            converted = []
            for action in value:
                if not isinstance(action, dict):
                    continue
                action_data = dict(action)
                if "full_page" in action_data:
                    action_data["fullPage"] = action_data.pop("full_page")
                converted.append(action_data)
            payload["actions"] = converted
        elif key in field_map:
            payload[field_map[key]] = value
    return payload


def _firecrawl_search_payload(search_kwargs: Dict[str, Any]) -> Dict[str, Any]:
    payload: Dict[str, Any] = {"query": search_kwargs["query"]}
    field_map = {
        "limit": "limit",
        "tbs": "tbs",
        "location": "location",
        "include_domains": "includeDomains",
        "exclude_domains": "excludeDomains",
    }
    for key, target in field_map.items():
        if key in search_kwargs:
            payload[target] = search_kwargs[key]
    if "sources" in search_kwargs:
        payload["sources"] = [{"type": source} for source in search_kwargs["sources"]]
    if "categories" in search_kwargs:
        payload["categories"] = [{"type": category} for category in search_kwargs["categories"]]
    return payload


def _firecrawl_crawl_payload(url: str, crawl_kwargs: Dict[str, Any]) -> Dict[str, Any]:
    payload: Dict[str, Any] = {"url": url}
    field_map = {
        "prompt": "prompt",
        "limit": "limit",
        "include_paths": "includePaths",
        "exclude_paths": "excludePaths",
        "max_discovery_depth": "maxDiscoveryDepth",
        "sitemap": "sitemap",
        "ignore_query_parameters": "ignoreQueryParameters",
        "regex_on_full_url": "regexOnFullURL",
        "crawl_entire_domain": "crawlEntireDomain",
        "allow_external_links": "allowExternalLinks",
        "allow_subdomains": "allowSubdomains",
        "delay": "delay",
        "max_concurrency": "maxConcurrency",
    }
    for key, target in field_map.items():
        if key in crawl_kwargs:
            payload[target] = crawl_kwargs[key]
    if "scrape_options" in crawl_kwargs:
        payload["scrapeOptions"] = _firecrawl_scrape_payload(crawl_kwargs["scrape_options"])
    return payload


def _firecrawl_http_client(client: Any):
    return getattr(getattr(client, "_v2_client", None), "http_client", None)


def _firecrawl_post_json(client: Any, endpoint: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    http_client = _firecrawl_http_client(client)
    if http_client is None:
        raise ValueError("Firecrawl SDK client does not expose a v2 HTTP client")
    response = http_client.post(endpoint, payload)
    if not getattr(response, "ok", False):
        response.raise_for_status()
    body = response.json()
    if isinstance(body, dict) and body.get("success") is False:
        raise Exception(body.get("error", "Unknown Firecrawl error"))
    return body if isinstance(body, dict) else {}


def _call_firecrawl_scrape(url: str, scrape_kwargs: Dict[str, Any]) -> Any:
    client = _get_firecrawl_client()
    # The installed SDK lags a few v2 docs fields on the high-level scrape()
    # signature, so use the v2 HTTP client when those fields are requested.
    if {"only_clean_content", "min_age"} & set(scrape_kwargs) and _firecrawl_http_client(client) is not None:
        body = _firecrawl_post_json(
            client,
            "/v2/scrape",
            {"url": url, **_firecrawl_scrape_payload(scrape_kwargs)},
        )
        return body.get("data", body)
    return client.scrape(url=url, **scrape_kwargs)


def _call_firecrawl_crawl(url: str, crawl_kwargs: Dict[str, Any]) -> Any:
    client = _get_firecrawl_client()
    scrape_options = crawl_kwargs.get("scrape_options", {})
    delay = crawl_kwargs.get("delay")
    needs_docs_payload = (
        bool({"only_clean_content", "min_age"} & set(scrape_options))
        or (isinstance(delay, float) and not delay.is_integer())
    )
    if needs_docs_payload and _firecrawl_http_client(client) is not None:
        body = _firecrawl_post_json(client, "/v2/crawl", _firecrawl_crawl_payload(url, crawl_kwargs))
        job_id = body.get("id")
        if not job_id:
            return body.get("data", body)
        from firecrawl.v2.methods.crawl import wait_for_crawl_completion

        return wait_for_crawl_completion(_firecrawl_http_client(client), job_id, poll_interval=2, timeout=None)
    return client.crawl(url=url, **crawl_kwargs)


def _normalize_firecrawl_search_response(response: Any) -> dict:
    response_dict = _to_plain_dict(response)
    data = response_dict.get("data", response_dict)
    normalized_data: Dict[str, List[Dict[str, Any]]] = {}

    for source in ("web", "images", "news"):
        raw_results = []
        if isinstance(data, dict):
            raw_results = data.get(source) or []
        elif hasattr(response, source):
            raw_results = getattr(response, source) or []
        results = []
        for item in raw_results:
            plain = _to_plain_dict(item)
            if plain:
                results.append(plain)
        if results or source == "web":
            normalized_data[source] = results

    return {"success": True, "data": normalized_data}


def _firecrawl_search(
    query: str,
    limit: int = 5,
    sources: Optional[List[str]] = None,
    categories: Optional[List[str]] = None,
    include_domains: Optional[List[str]] = None,
    exclude_domains: Optional[List[str]] = None,
    tbs: Optional[str] = None,
    location: Optional[str] = None,
) -> dict:
    logger.info("Searching the web for: '%s' (limit: %d)", query, limit)

    search_kwargs = _build_firecrawl_search_kwargs(
        query=query,
        limit=limit,
        sources=sources,
        categories=categories,
        include_domains=include_domains,
        exclude_domains=exclude_domains,
        tbs=tbs,
        location=location,
    )
    client = _get_firecrawl_client()

    if {"include_domains", "exclude_domains"} & set(search_kwargs) and _firecrawl_http_client(client) is not None:
        body = _firecrawl_post_json(client, "/v2/search", _firecrawl_search_payload(search_kwargs))
        response = body.get("data", body)
    else:
        sdk_kwargs = {
            key: value
            for key, value in search_kwargs.items()
            if key not in {"query", "include_domains", "exclude_domains"}
        }
        response = client.search(query=search_kwargs["query"], **sdk_kwargs)

    response_data = _normalize_firecrawl_search_response(response)
    results_count = sum(len(value) for value in response_data.get("data", {}).values())
    logger.info("Found %d search results", results_count)
    return response_data


def _canonicalize_result_url(url: str) -> str:
    if not url:
        return ""
    try:
        parts = urlsplit(url.strip())
    except Exception:
        return url.strip()
    path = parts.path.rstrip("/") or "/"
    return urlunsplit((parts.scheme.lower(), parts.netloc.lower(), path, parts.query, ""))


def _normalize_provider_results(provider: str, response_data: dict) -> list[dict]:
    results = response_data.get("data", {}).get("web", []) if isinstance(response_data, dict) else []
    normalized = []
    for index, result in enumerate(results):
        if not isinstance(result, dict):
            continue
        normalized.append(
            {
                "provider": provider,
                "url": result.get("url", ""),
                "title": result.get("title", ""),
                "description": result.get("description", ""),
                "position": int(result.get("position") or index + 1),
            }
        )
    return normalized


async def _parallel_extract(urls: List[str]) -> List[Dict[str, Any]]:
    """Extract content from URLs using the Parallel async SDK.

    Returns a list of result dicts matching the structure expected by the
    LLM post-processing pipeline (url, title, content, metadata).
    """
    from tools.interrupt import is_interrupted
    if is_interrupted():
        return [{"url": u, "error": "Interrupted", "title": ""} for u in urls]

    logger.info("Parallel extract: %d URL(s)", len(urls))
    response = await _get_async_parallel_client().beta.extract(
        urls=urls,
        full_content=True,
    )

    results = []
    for result in response.results or []:
        content = result.full_content or ""
        if not content:
            content = "\n\n".join(result.excerpts or [])
        url = result.url or ""
        title = result.title or ""
        results.append({
            "url": url,
            "title": title,
            "content": content,
            "raw_content": content,
            "metadata": {"sourceURL": url, "title": title},
        })

    for error in response.errors or []:
        results.append({
            "url": error.url or "",
            "title": "",
            "content": "",
            "error": error.content or error.error_type or "extraction failed",
            "metadata": {"sourceURL": error.url or ""},
        })

    return results


def web_search_tool(
    query: str,
    limit: int = 5,
    user_task: Optional[str] = None,
    sources: Optional[List[str]] = None,
    categories: Optional[List[str]] = None,
    include_domains: Optional[List[str]] = None,
    exclude_domains: Optional[List[str]] = None,
    tbs: Optional[str] = None,
    location: Optional[str] = None,
) -> str:
    """
    Search the web for information using available search API backend.

    This function provides a generic interface for web search that can work
    with multiple backends (Parallel or Firecrawl).

    Note: This function returns search result metadata only (URLs, titles, descriptions).
    Use web_extract_tool to get full content from specific URLs.
    
    Args:
        query (str): The search query to look up
        limit (int): Maximum number of results to return (default: 5)
    
    Returns:
        str: JSON string containing search results with the following structure:
             {
                 "success": bool,
                 "data": {
                     "web": [
                         {
                             "title": str,
                             "url": str,
                             "description": str,
                             "position": int
                         },
                         ...
                     ]
                 }
             }
    
    Raises:
        Exception: If search fails or API key is not set
    """
    debug_call_data = {
        "parameters": {
            "query": query,
            "limit": limit,
            "sources": sources,
            "categories": categories,
            "include_domains": include_domains,
            "exclude_domains": exclude_domains,
            "tbs": tbs,
            "location": location,
        },
        "error": None,
        "results_count": 0,
        "original_response_size": 0,
        "final_response_size": 0
    }
    
    try:
        from tools.interrupt import is_interrupted
        if is_interrupted():
            return json.dumps({"error": "Interrupted", "success": False})

        # Dispatch to the configured backend
        backend = _get_backend()
        if backend == "parallel":
            response_data = _parallel_search(query, limit)
            debug_call_data["results_count"] = len(response_data.get("data", {}).get("web", []))
            result_json = json.dumps(response_data, indent=2, ensure_ascii=False)
            debug_call_data["final_response_size"] = len(result_json)
            _debug.log_call("web_search_tool", debug_call_data)
            _debug.save()
            return result_json

        if backend == "exa":
            response_data = _exa_search(query, limit)
            debug_call_data["results_count"] = len(response_data.get("data", {}).get("web", []))
            result_json = json.dumps(response_data, indent=2, ensure_ascii=False)
            debug_call_data["final_response_size"] = len(result_json)
            _debug.log_call("web_search_tool", debug_call_data)
            _debug.save()
            return result_json

        if backend == "tavily":
            response_data = _tavily_search(query, limit)
            debug_call_data["results_count"] = len(response_data.get("data", {}).get("web", []))
            result_json = json.dumps(response_data, indent=2, ensure_ascii=False)
            debug_call_data["final_response_size"] = len(result_json)
            _debug.log_call("web_search_tool", debug_call_data)
            _debug.save()
            return result_json

        response_data = _firecrawl_search(
            query,
            limit,
            sources=sources,
            categories=categories,
            include_domains=include_domains,
            exclude_domains=exclude_domains,
            tbs=tbs,
            location=location,
        )
        results_count = len(response_data.get("data", {}).get("web", []))
        
        # Capture debug information
        debug_call_data["results_count"] = results_count
        
        # Convert to JSON
        result_json = json.dumps(response_data, indent=2, ensure_ascii=False)
        
        debug_call_data["final_response_size"] = len(result_json)
        
        # Log debug information
        _debug.log_call("web_search_tool", debug_call_data)
        _debug.save()
        
        return result_json
        
    except Exception as e:
        error_msg = f"Error searching web: {str(e)}"
        logger.debug("%s", error_msg)
        
        debug_call_data["error"] = error_msg
        _debug.log_call("web_search_tool", debug_call_data)
        _debug.save()
        
        return json.dumps({"error": error_msg}, ensure_ascii=False)


async def web_extract_tool(
    urls: List[str], 
    format: str = None, 
    formats: Optional[List[str]] = None,
    use_llm_processing: bool = True,
    model: str = DEFAULT_SUMMARIZER_MODEL,
    min_length: int = DEFAULT_MIN_LENGTH_FOR_SUMMARIZATION,
    only_main_content: Optional[bool] = None,
    only_clean_content: Optional[bool] = None,
    include_tags: Optional[List[str]] = None,
    exclude_tags: Optional[List[str]] = None,
    max_age: Optional[int] = None,
    min_age: Optional[int] = None,
    wait_for: Optional[int] = None,
    mobile: Optional[bool] = None,
    timeout: Optional[int] = None,
    pdf_parser_mode: Optional[str] = None,
    pdf_max_pages: Optional[int] = None,
    actions: Optional[List[Dict[str, Any]]] = None,
    location: Optional[str] = None,
) -> str:
    """
    Extract content from specific web pages using available extraction API backend.
    
    This function provides a generic interface for web content extraction that
    can work with multiple backends. Currently uses Firecrawl.
    
    Args:
        urls (List[str]): List of URLs to extract content from
        format (str): Desired output format ("markdown" or "html", optional)
        use_llm_processing (bool): Whether to process content with LLM for summarization (default: True)
        model (str): The model to use for LLM processing (default: google/gemini-3-flash-preview)
        min_length (int): Minimum content length to trigger LLM processing (default: 5000)
    
    Returns:
        str: JSON string containing extracted content. If LLM processing is enabled and successful,
             the 'content' field will contain the processed markdown summary instead of raw content.
    
    Raises:
        Exception: If extraction fails or API key is not set
    """
    debug_call_data = {
        "parameters": {
            "urls": urls,
            "format": format,
            "formats": formats,
            "use_llm_processing": use_llm_processing,
            "model": model,
            "min_length": min_length,
            "only_main_content": only_main_content,
            "only_clean_content": only_clean_content,
            "include_tags": include_tags,
            "exclude_tags": exclude_tags,
            "max_age": max_age,
            "min_age": min_age,
            "wait_for": wait_for,
            "mobile": mobile,
            "timeout": timeout,
            "pdf_parser_mode": pdf_parser_mode,
            "pdf_max_pages": pdf_max_pages,
            "actions": actions,
            "location": location,
        },
        "error": None,
        "pages_extracted": 0,
        "pages_processed_with_llm": 0,
        "original_response_size": 0,
        "final_response_size": 0,
        "compression_metrics": [],
        "processing_applied": []
    }
    
    try:
        logger.info("Extracting content from %d URL(s)", len(urls))

        for url in urls:
            if _url_contains_embedded_secret(url):
                return json.dumps({
                    "success": False,
                    "error": "Blocked: URL contains what appears to be an API key or token. Secrets must not be sent in URLs.",
                }, ensure_ascii=False)

        # ── SSRF protection — filter out private/internal URLs before any backend ──
        safe_urls = []
        ssrf_blocked: List[Dict[str, Any]] = []
        for url in urls:
            if not is_safe_url(url):
                ssrf_blocked.append({
                    "url": url, "title": "", "content": "",
                    "error": "Blocked: URL targets a private or internal network address",
                })
            else:
                safe_urls.append(url)

        # Dispatch only safe URLs to the configured backend
        if not safe_urls:
            results = []
        else:
            backend = _get_backend()

            if backend == "parallel":
                results = await _parallel_extract(safe_urls)
            elif backend == "exa":
                results = _exa_extract(safe_urls)
            elif backend == "tavily":
                logger.info("Tavily extract: %d URL(s)", len(safe_urls))
                raw = _tavily_request("extract", {
                    "urls": safe_urls,
                    "include_images": False,
                })
                results = _normalize_tavily_documents(raw, fallback_url=safe_urls[0] if safe_urls else "")
            else:
                # ── Firecrawl extraction ──
                scrape_kwargs = _build_firecrawl_scrape_kwargs(
                    format=format,
                    formats=formats,
                    only_main_content=only_main_content,
                    only_clean_content=only_clean_content,
                    include_tags=include_tags,
                    exclude_tags=exclude_tags,
                    max_age=max_age,
                    min_age=min_age,
                    wait_for=wait_for,
                    mobile=mobile,
                    timeout=timeout,
                    pdf_parser_mode=pdf_parser_mode,
                    pdf_max_pages=pdf_max_pages,
                    actions=actions,
                    location=location,
                    default_formats=["markdown", "html"],
                )

                # Always use individual scraping for simplicity and reliability
                # Batch scraping adds complexity without much benefit for small numbers of URLs
                results: List[Dict[str, Any]] = []

                from tools.interrupt import is_interrupted as _is_interrupted
                for url in safe_urls:
                    if _is_interrupted():
                        results.append({"url": url, "error": "Interrupted", "title": ""})
                        continue

                    # Website policy check — block before fetching
                    blocked = check_website_access(url)
                    if blocked:
                        logger.info("Blocked web_extract for %s by rule %s", blocked["host"], blocked["rule"])
                        results.append({
                            "url": url, "title": "", "content": "",
                            "error": blocked["message"],
                            "blocked_by_policy": {"host": blocked["host"], "rule": blocked["rule"], "source": blocked["source"]},
                        })
                        continue

                    try:
                        logger.info("Scraping: %s", url)
                        scrape_result = _call_firecrawl_scrape(url, scrape_kwargs)

                        # Process the result - properly handle object serialization
                        metadata = {}
                        title = ""
                        content_markdown = None
                        content_html = None
                        content_raw_html = None
                        content_summary = None
                        content_links = None
                        content_images = None
                        content_screenshot = None

                        # Extract data from the scrape result
                        if hasattr(scrape_result, 'model_dump'):
                            # Pydantic model - use model_dump to get dict
                            result_dict = scrape_result.model_dump()
                            content_markdown = result_dict.get('markdown')
                            content_html = result_dict.get('html')
                            content_raw_html = result_dict.get('rawHtml') or result_dict.get('raw_html')
                            content_summary = result_dict.get('summary')
                            content_links = result_dict.get('links')
                            content_images = result_dict.get('images')
                            content_screenshot = result_dict.get('screenshot')
                            metadata = result_dict.get('metadata', {})
                        elif hasattr(scrape_result, '__dict__'):
                            # Regular object with attributes
                            content_markdown = getattr(scrape_result, 'markdown', None)
                            content_html = getattr(scrape_result, 'html', None)
                            content_raw_html = getattr(scrape_result, 'rawHtml', None) or getattr(scrape_result, 'raw_html', None)
                            content_summary = getattr(scrape_result, 'summary', None)
                            content_links = getattr(scrape_result, 'links', None)
                            content_images = getattr(scrape_result, 'images', None)
                            content_screenshot = getattr(scrape_result, 'screenshot', None)

                            # Handle metadata - convert to dict if it's an object
                            metadata_obj = getattr(scrape_result, 'metadata', {})
                            if hasattr(metadata_obj, 'model_dump'):
                                metadata = metadata_obj.model_dump()
                            elif hasattr(metadata_obj, '__dict__'):
                                metadata = metadata_obj.__dict__
                            elif isinstance(metadata_obj, dict):
                                metadata = metadata_obj
                            else:
                                metadata = {}
                        elif isinstance(scrape_result, dict):
                            # Already a dictionary
                            content_markdown = scrape_result.get('markdown')
                            content_html = scrape_result.get('html')
                            content_raw_html = scrape_result.get('rawHtml') or scrape_result.get('raw_html')
                            content_summary = scrape_result.get('summary')
                            content_links = scrape_result.get('links')
                            content_images = scrape_result.get('images')
                            content_screenshot = scrape_result.get('screenshot')
                            metadata = scrape_result.get('metadata', {})

                        # Ensure metadata is a dict (not an object)
                        if not isinstance(metadata, dict):
                            if hasattr(metadata, 'model_dump'):
                                metadata = metadata.model_dump()
                            elif hasattr(metadata, '__dict__'):
                                metadata = metadata.__dict__
                            else:
                                metadata = {}

                        # Get title from metadata
                        title = metadata.get("title", "")

                        # Re-check final URL after redirect
                        final_url = metadata.get("sourceURL", url)
                        final_blocked = check_website_access(final_url)
                        if final_blocked:
                            logger.info("Blocked redirected web_extract for %s by rule %s", final_blocked["host"], final_blocked["rule"])
                            results.append({
                                "url": final_url, "title": title, "content": "", "raw_content": "",
                                "error": final_blocked["message"],
                                "blocked_by_policy": {"host": final_blocked["host"], "rule": final_blocked["rule"], "source": final_blocked["source"]},
                            })
                            continue

                        # Choose content based on requested format, falling back to markdown.
                        preferred_formats = _build_firecrawl_formats(format, formats)
                        if not preferred_formats:
                            preferred_formats = ["markdown", "html"]
                        content_by_format = {
                            "markdown": content_markdown,
                            "html": content_html,
                            "rawHtml": content_raw_html,
                            "summary": content_summary,
                            "links": content_links,
                            "images": content_images,
                            "screenshot": content_screenshot,
                        }
                        chosen_content = _choose_firecrawl_content(content_by_format, preferred_formats)

                        results.append({
                            "url": final_url,
                            "title": title,
                            "content": chosen_content,
                            "raw_content": chosen_content,
                            "metadata": metadata  # Now guaranteed to be a dict
                        })

                    except Exception as scrape_err:
                        logger.debug("Scrape failed for %s: %s", url, scrape_err)
                        results.append({
                            "url": url,
                            "title": "",
                            "content": "",
                            "raw_content": "",
                            "error": str(scrape_err)
                        })

        # Merge any SSRF-blocked results back in
        if ssrf_blocked:
            results = ssrf_blocked + results

        response = {"results": results}
        
        pages_extracted = len(response.get('results', []))
        logger.info("Extracted content from %d pages", pages_extracted)
        
        debug_call_data["pages_extracted"] = pages_extracted
        debug_call_data["original_response_size"] = len(json.dumps(response))
        
        # Process each result with LLM if enabled
        if use_llm_processing:
            logger.info("Processing extracted content with LLM (parallel)...")
            debug_call_data["processing_applied"].append("llm_processing")
            
            # Prepare tasks for parallel processing
            async def process_single_result(result):
                """Process a single result with LLM and return updated result with metrics."""
                url = result.get('url', 'Unknown URL')
                title = result.get('title', '')
                raw_content = result.get('raw_content', '') or result.get('content', '')
                
                if not raw_content:
                    return result, None, "no_content"
                
                original_size = len(raw_content)
                
                # Process content with LLM
                processed = await process_content_with_llm(
                    raw_content, url, title, model, min_length
                )
                
                if processed:
                    processed_size = len(processed)
                    compression_ratio = processed_size / original_size if original_size > 0 else 1.0
                    
                    # Update result with processed content
                    result['content'] = processed
                    result['raw_content'] = raw_content
                    
                    metrics = {
                        "url": url,
                        "original_size": original_size,
                        "processed_size": processed_size,
                        "compression_ratio": compression_ratio,
                        "model_used": model
                    }
                    return result, metrics, "processed"
                else:
                    metrics = {
                        "url": url,
                        "original_size": original_size,
                        "processed_size": original_size,
                        "compression_ratio": 1.0,
                        "model_used": None,
                        "reason": "content_too_short"
                    }
                    return result, metrics, "too_short"
            
            # Run all LLM processing in parallel
            results_list = response.get('results', [])
            tasks = [process_single_result(result) for result in results_list]
            processed_results = await asyncio.gather(*tasks)
            
            # Collect metrics and print results
            for result, metrics, status in processed_results:
                url = result.get('url', 'Unknown URL')
                if status == "processed":
                    debug_call_data["compression_metrics"].append(metrics)
                    debug_call_data["pages_processed_with_llm"] += 1
                    logger.info("%s (processed)", url)
                elif status == "too_short":
                    debug_call_data["compression_metrics"].append(metrics)
                    logger.info("%s (no processing - content too short)", url)
                else:
                    logger.warning("%s (no content to process)", url)
        else:
            # Print summary of extracted pages for debugging (original behavior)
            for result in response.get('results', []):
                url = result.get('url', 'Unknown URL')
                content_length = len(result.get('raw_content', ''))
                logger.info("%s (%d characters)", url, content_length)
        
        # Trim output to minimal fields per entry: title, content, error
        trimmed_results = [
            {
                "url": r.get("url", ""),
                "title": r.get("title", ""),
                "content": r.get("content", ""),
                "error": r.get("error"),
                **({  "blocked_by_policy": r["blocked_by_policy"]} if "blocked_by_policy" in r else {}),
            }
            for r in response.get("results", [])
        ]
        trimmed_response = {"results": trimmed_results}

        if trimmed_response.get("results") == []:
            result_json = json.dumps({"error": "Content was inaccessible or not found"}, ensure_ascii=False)

            cleaned_result = clean_base64_images(result_json)
        
        else:
            result_json = json.dumps(trimmed_response, indent=2, ensure_ascii=False)
            
            cleaned_result = clean_base64_images(result_json)
        
        debug_call_data["final_response_size"] = len(cleaned_result)
        debug_call_data["processing_applied"].append("base64_image_removal")
        
        # Log debug information
        _debug.log_call("web_extract_tool", debug_call_data)
        _debug.save()
        
        return cleaned_result
            
    except Exception as e:
        error_msg = f"Error extracting content: {str(e)}"
        logger.debug("%s", error_msg)
        
        debug_call_data["error"] = error_msg
        _debug.log_call("web_extract_tool", debug_call_data)
        _debug.save()
        
        return json.dumps({"error": error_msg}, ensure_ascii=False)


async def web_crawl_tool(
    url: str, 
    instructions: str = None, 
    depth: str = "basic", 
    use_llm_processing: bool = True,
    model: str = DEFAULT_SUMMARIZER_MODEL,
    min_length: int = DEFAULT_MIN_LENGTH_FOR_SUMMARIZATION,
    limit: int = 20,
    include_paths: Optional[List[str]] = None,
    exclude_paths: Optional[List[str]] = None,
    max_discovery_depth: Optional[int] = None,
    sitemap: Optional[str] = None,
    ignore_query_parameters: Optional[bool] = None,
    regex_on_full_url: Optional[bool] = None,
    crawl_entire_domain: Optional[bool] = None,
    allow_subdomains: Optional[bool] = None,
    allow_external_links: bool = False,
    delay: Optional[float] = None,
    max_concurrency: Optional[int] = None,
    scrape_options: Optional[Dict[str, Any]] = None,
) -> str:
    """
    Crawl a website with specific instructions using available crawling API backend.
    
    This function provides a generic interface for web crawling that can work
    with multiple backends. Currently uses Firecrawl.
    
    Args:
        url (str): The base URL to crawl (can include or exclude https://)
        instructions (str): Instructions for what to crawl/extract using LLM intelligence (optional)
        depth (str): Depth of extraction ("basic" or "advanced", default: "basic")
        use_llm_processing (bool): Whether to process content with LLM for summarization (default: True)
        model (str): The model to use for LLM processing (default: google/gemini-3-flash-preview)
        min_length (int): Minimum content length to trigger LLM processing (default: 5000)
    
    Returns:
        str: JSON string containing crawled content. If LLM processing is enabled and successful,
             the 'content' field will contain the processed markdown summary instead of raw content.
             Each page is processed individually.
    
    Raises:
        Exception: If crawling fails or API key is not set
    """
    debug_call_data = {
        "parameters": {
            "url": url,
            "instructions": instructions,
            "depth": depth,
            "use_llm_processing": use_llm_processing,
            "model": model,
            "min_length": min_length,
            "limit": limit,
            "include_paths": include_paths,
            "exclude_paths": exclude_paths,
            "max_discovery_depth": max_discovery_depth,
            "sitemap": sitemap,
            "ignore_query_parameters": ignore_query_parameters,
            "regex_on_full_url": regex_on_full_url,
            "crawl_entire_domain": crawl_entire_domain,
            "allow_subdomains": allow_subdomains,
            "allow_external_links": allow_external_links,
            "delay": delay,
            "max_concurrency": max_concurrency,
            "scrape_options": scrape_options,
        },
        "error": None,
        "pages_crawled": 0,
        "pages_processed_with_llm": 0,
        "original_response_size": 0,
        "final_response_size": 0,
        "compression_metrics": [],
        "processing_applied": []
    }
    
    try:
        backend = _get_backend()

        if _url_contains_embedded_secret(url):
            return json.dumps({
                "success": False,
                "error": "Blocked: URL contains what appears to be an API key or token. Secrets must not be sent in URLs.",
            }, ensure_ascii=False)

        # Tavily supports crawl via its /crawl endpoint
        if backend == "tavily":
            # Ensure URL has protocol
            if not url.startswith(('http://', 'https://')):
                url = f'https://{url}'

            # SSRF protection — block private/internal addresses
            if not is_safe_url(url):
                return json.dumps({"results": [{"url": url, "title": "", "content": "",
                    "error": "Blocked: URL targets a private or internal network address"}]}, ensure_ascii=False)

            # Website policy check
            blocked = check_website_access(url)
            if blocked:
                logger.info("Blocked web_crawl for %s by rule %s", blocked["host"], blocked["rule"])
                return json.dumps({"results": [{"url": url, "title": "", "content": "", "error": blocked["message"],
                    "blocked_by_policy": {"host": blocked["host"], "rule": blocked["rule"], "source": blocked["source"]}}]}, ensure_ascii=False)

            from tools.interrupt import is_interrupted as _is_int
            if _is_int():
                return json.dumps({"error": "Interrupted", "success": False})

            logger.info("Tavily crawl: %s", url)
            payload: Dict[str, Any] = {
                "url": url,
                "limit": 20,
                "extract_depth": depth,
            }
            if instructions:
                payload["instructions"] = instructions
            raw = _tavily_request("crawl", payload)
            results = _normalize_tavily_documents(raw, fallback_url=url)

            response = {"results": results}
            # Fall through to the shared LLM processing and trimming below
            # (skip the Firecrawl-specific crawl logic)
            pages_crawled = len(response.get('results', []))
            logger.info("Crawled %d pages", pages_crawled)
            debug_call_data["pages_crawled"] = pages_crawled
            debug_call_data["original_response_size"] = len(json.dumps(response))

            # Process each result with LLM if enabled
            if use_llm_processing:
                logger.info("Processing crawled content with LLM (parallel)...")
                debug_call_data["processing_applied"].append("llm_processing")

                async def _process_tavily_crawl(result):
                    page_url = result.get('url', 'Unknown URL')
                    title = result.get('title', '')
                    content = result.get('content', '')
                    if not content:
                        return result, None, "no_content"
                    original_size = len(content)
                    processed = await process_content_with_llm(content, page_url, title, model, min_length)
                    if processed:
                        result['raw_content'] = content
                        result['content'] = processed
                        metrics = {"url": page_url, "original_size": original_size, "processed_size": len(processed),
                                   "compression_ratio": len(processed) / original_size if original_size else 1.0, "model_used": model}
                        return result, metrics, "processed"
                    metrics = {"url": page_url, "original_size": original_size, "processed_size": original_size,
                               "compression_ratio": 1.0, "model_used": None, "reason": "content_too_short"}
                    return result, metrics, "too_short"

                tasks = [_process_tavily_crawl(r) for r in response.get('results', [])]
                processed_results = await asyncio.gather(*tasks)
                for result, metrics, status in processed_results:
                    if status == "processed":
                        debug_call_data["compression_metrics"].append(metrics)
                        debug_call_data["pages_processed_with_llm"] += 1

            trimmed_results = [{"url": r.get("url", ""), "title": r.get("title", ""), "content": r.get("content", ""), "error": r.get("error"),
                **({  "blocked_by_policy": r["blocked_by_policy"]} if "blocked_by_policy" in r else {})} for r in response.get("results", [])]
            result_json = json.dumps({"results": trimmed_results}, indent=2, ensure_ascii=False)
            cleaned_result = clean_base64_images(result_json)
            debug_call_data["final_response_size"] = len(cleaned_result)
            _debug.log_call("web_crawl_tool", debug_call_data)
            _debug.save()
            return cleaned_result

        # web_crawl requires Firecrawl — Parallel has no crawl API
        if not (os.getenv("FIRECRAWL_API_KEY") or os.getenv("FIRECRAWL_API_URL")):
            return json.dumps({
                "error": "web_crawl requires Firecrawl. Set FIRECRAWL_API_KEY, "
                         "or use web_search + web_extract instead.",
                "success": False,
            }, ensure_ascii=False)

        # Ensure URL has protocol
        if not url.startswith(('http://', 'https://')):
            url = f'https://{url}'
            logger.info("Added https:// prefix to URL: %s", url)
        
        instructions_text = f" with instructions: '{instructions}'" if instructions else ""
        logger.info("Crawling %s%s", url, instructions_text)
        
        # SSRF protection — block private/internal addresses
        if not is_safe_url(url):
            return json.dumps({"results": [{"url": url, "title": "", "content": "",
                "error": "Blocked: URL targets a private or internal network address"}]}, ensure_ascii=False)

        # Website policy check — block before crawling
        blocked = check_website_access(url)
        if blocked:
            logger.info("Blocked web_crawl for %s by rule %s", blocked["host"], blocked["rule"])
            return json.dumps({"results": [{"url": url, "title": "", "content": "", "error": blocked["message"],
                "blocked_by_policy": {"host": blocked["host"], "rule": blocked["rule"], "source": blocked["source"]}}]}, ensure_ascii=False)

        # Firecrawl v2 documents crawl prompt: explicit params override any
        # natural-language options generated from the prompt.
        crawl_params = _build_firecrawl_crawl_kwargs(
            instructions=instructions,
            limit=limit,
            include_paths=include_paths,
            exclude_paths=exclude_paths,
            max_discovery_depth=max_discovery_depth,
            sitemap=sitemap,
            ignore_query_parameters=ignore_query_parameters,
            regex_on_full_url=regex_on_full_url,
            crawl_entire_domain=crawl_entire_domain,
            allow_external_links=allow_external_links,
            allow_subdomains=allow_subdomains,
            delay=delay,
            max_concurrency=max_concurrency,
            scrape_options=scrape_options,
        )
        
        from tools.interrupt import is_interrupted as _is_int
        if _is_int():
            return json.dumps({"error": "Interrupted", "success": False})

        try:
            crawl_result = _call_firecrawl_crawl(url, crawl_params)
        except Exception as e:
            logger.debug("Crawl API call failed: %s", e)
            raise

        pages: List[Dict[str, Any]] = []
        
        # Process crawl results - the crawl method returns a CrawlJob object with data attribute
        data_list = []
        
        # The crawl_result is a CrawlJob object with a 'data' attribute containing list of Document objects
        if hasattr(crawl_result, 'data'):
            data_list = crawl_result.data if crawl_result.data else []
            logger.info("Status: %s", getattr(crawl_result, 'status', 'unknown'))
            logger.info("Retrieved %d pages", len(data_list))
            
            # Debug: Check other attributes if no data
            if not data_list:
                logger.debug("CrawlJob attributes: %s", [attr for attr in dir(crawl_result) if not attr.startswith('_')])
                logger.debug("Status: %s", getattr(crawl_result, 'status', 'N/A'))
                logger.debug("Total: %s", getattr(crawl_result, 'total', 'N/A'))
                logger.debug("Completed: %s", getattr(crawl_result, 'completed', 'N/A'))
                
        elif isinstance(crawl_result, dict) and 'data' in crawl_result:
            data_list = crawl_result.get("data", [])
        else:
            logger.warning("Unexpected crawl result type")
            logger.debug("Result type: %s", type(crawl_result))
            if hasattr(crawl_result, '__dict__'):
                logger.debug("Result attributes: %s", list(crawl_result.__dict__.keys()))
        
        for item in data_list:
            # Process each crawled page - properly handle object serialization
            page_url = "Unknown URL"
            title = ""
            content_markdown = None
            content_html = None
            content_raw_html = None
            content_summary = None
            content_links = None
            content_images = None
            content_screenshot = None
            metadata = {}
            
            # Extract data from the item
            if hasattr(item, 'model_dump'):
                # Pydantic model - use model_dump to get dict
                item_dict = item.model_dump()
                content_markdown = item_dict.get('markdown')
                content_html = item_dict.get('html')
                content_raw_html = item_dict.get('rawHtml') or item_dict.get('raw_html')
                content_summary = item_dict.get('summary')
                content_links = item_dict.get('links')
                content_images = item_dict.get('images')
                content_screenshot = item_dict.get('screenshot')
                metadata = item_dict.get('metadata', {})
            elif hasattr(item, '__dict__'):
                # Regular object with attributes
                content_markdown = getattr(item, 'markdown', None)
                content_html = getattr(item, 'html', None)
                content_raw_html = getattr(item, 'rawHtml', None) or getattr(item, 'raw_html', None)
                content_summary = getattr(item, 'summary', None)
                content_links = getattr(item, 'links', None)
                content_images = getattr(item, 'images', None)
                content_screenshot = getattr(item, 'screenshot', None)
                
                # Handle metadata - convert to dict if it's an object
                metadata_obj = getattr(item, 'metadata', {})
                if hasattr(metadata_obj, 'model_dump'):
                    metadata = metadata_obj.model_dump()
                elif hasattr(metadata_obj, '__dict__'):
                    metadata = metadata_obj.__dict__
                elif isinstance(metadata_obj, dict):
                    metadata = metadata_obj
                else:
                    metadata = {}
            elif isinstance(item, dict):
                # Already a dictionary
                content_markdown = item.get('markdown')
                content_html = item.get('html')
                content_raw_html = item.get('rawHtml') or item.get('raw_html')
                content_summary = item.get('summary')
                content_links = item.get('links')
                content_images = item.get('images')
                content_screenshot = item.get('screenshot')
                metadata = item.get('metadata', {})
            
            # Ensure metadata is a dict (not an object)
            if not isinstance(metadata, dict):
                if hasattr(metadata, 'model_dump'):
                    metadata = metadata.model_dump()
                elif hasattr(metadata, '__dict__'):
                    metadata = metadata.__dict__
                else:
                    metadata = {}
            
            # Extract URL and title from metadata
            page_url = metadata.get("sourceURL", metadata.get("url", "Unknown URL"))
            title = metadata.get("title", "")
            
            # Re-check crawled page URL against policy
            page_blocked = check_website_access(page_url)
            if page_blocked:
                logger.info("Blocked crawled page %s by rule %s", page_blocked["host"], page_blocked["rule"])
                pages.append({
                    "url": page_url, "title": title, "content": "", "raw_content": "",
                    "error": page_blocked["message"],
                    "blocked_by_policy": {"host": page_blocked["host"], "rule": page_blocked["rule"], "source": page_blocked["source"]},
                })
                continue

            # Choose content based on nested scrape options, falling back to markdown.
            crawl_preferred_formats = []
            if isinstance(crawl_params.get("scrape_options"), dict):
                crawl_preferred_formats = crawl_params["scrape_options"].get("formats", []) or []
            content = _choose_firecrawl_content({
                "markdown": content_markdown,
                "html": content_html,
                "rawHtml": content_raw_html,
                "summary": content_summary,
                "links": content_links,
                "images": content_images,
                "screenshot": content_screenshot,
            }, crawl_preferred_formats)
            
            pages.append({
                "url": page_url,
                "title": title,
                "content": content,
                "raw_content": content,
                "metadata": metadata  # Now guaranteed to be a dict
            })

        response = {"results": pages}
        
        pages_crawled = len(response.get('results', []))
        logger.info("Crawled %d pages", pages_crawled)
        
        debug_call_data["pages_crawled"] = pages_crawled
        debug_call_data["original_response_size"] = len(json.dumps(response))
        
        # Process each result with LLM if enabled
        if use_llm_processing:
            logger.info("Processing crawled content with LLM (parallel)...")
            debug_call_data["processing_applied"].append("llm_processing")
            
            # Prepare tasks for parallel processing
            async def process_single_crawl_result(result):
                """Process a single crawl result with LLM and return updated result with metrics."""
                page_url = result.get('url', 'Unknown URL')
                title = result.get('title', '')
                content = result.get('content', '')
                
                if not content:
                    return result, None, "no_content"
                
                original_size = len(content)
                
                # Process content with LLM
                processed = await process_content_with_llm(
                    content, page_url, title, model, min_length
                )
                
                if processed:
                    processed_size = len(processed)
                    compression_ratio = processed_size / original_size if original_size > 0 else 1.0
                    
                    # Update result with processed content
                    result['raw_content'] = content
                    result['content'] = processed
                    
                    metrics = {
                        "url": page_url,
                        "original_size": original_size,
                        "processed_size": processed_size,
                        "compression_ratio": compression_ratio,
                        "model_used": model
                    }
                    return result, metrics, "processed"
                else:
                    metrics = {
                        "url": page_url,
                        "original_size": original_size,
                        "processed_size": original_size,
                        "compression_ratio": 1.0,
                        "model_used": None,
                        "reason": "content_too_short"
                    }
                    return result, metrics, "too_short"
            
            # Run all LLM processing in parallel
            results_list = response.get('results', [])
            tasks = [process_single_crawl_result(result) for result in results_list]
            processed_results = await asyncio.gather(*tasks)
            
            # Collect metrics and print results
            for result, metrics, status in processed_results:
                page_url = result.get('url', 'Unknown URL')
                if status == "processed":
                    debug_call_data["compression_metrics"].append(metrics)
                    debug_call_data["pages_processed_with_llm"] += 1
                    logger.info("%s (processed)", page_url)
                elif status == "too_short":
                    debug_call_data["compression_metrics"].append(metrics)
                    logger.info("%s (no processing - content too short)", page_url)
                else:
                    logger.warning("%s (no content to process)", page_url)
        else:
            # Print summary of crawled pages for debugging (original behavior)
            for result in response.get('results', []):
                page_url = result.get('url', 'Unknown URL')
                content_length = len(result.get('content', ''))
                logger.info("%s (%d characters)", page_url, content_length)
        
        # Trim output to minimal fields per entry: title, content, error
        trimmed_results = [
            {
                "url": r.get("url", ""),
                "title": r.get("title", ""),
                "content": r.get("content", ""),
                "error": r.get("error"),
                **({  "blocked_by_policy": r["blocked_by_policy"]} if "blocked_by_policy" in r else {}),
            }
            for r in response.get("results", [])
        ]
        trimmed_response = {"results": trimmed_results}
        
        result_json = json.dumps(trimmed_response, indent=2, ensure_ascii=False)
        # Clean base64 images from crawled content
        cleaned_result = clean_base64_images(result_json)
        
        debug_call_data["final_response_size"] = len(cleaned_result)
        debug_call_data["processing_applied"].append("base64_image_removal")
        
        # Log debug information
        _debug.log_call("web_crawl_tool", debug_call_data)
        _debug.save()
        
        return cleaned_result
        
    except Exception as e:
        error_msg = f"Error crawling website: {str(e)}"
        logger.debug("%s", error_msg)
        
        debug_call_data["error"] = error_msg
        _debug.log_call("web_crawl_tool", debug_call_data)
        _debug.save()
        
        return json.dumps({"error": error_msg}, ensure_ascii=False)


# Convenience function to check if API key is available
def check_firecrawl_api_key() -> bool:
    """
    Check if the Firecrawl API key is available in environment variables.

    Returns:
        bool: True if API key is set, False otherwise
    """
    return bool(os.getenv("FIRECRAWL_API_KEY"))


def check_web_api_key() -> bool:
    """Check if any web backend API key is available (Exa, Parallel, Firecrawl, or Tavily)."""
    return bool(
        os.getenv("EXA_API_KEY")
        or os.getenv("PARALLEL_API_KEY")
        or os.getenv("FIRECRAWL_API_KEY")
        or os.getenv("FIRECRAWL_API_URL")
        or os.getenv("TAVILY_API_KEY")
    )


def check_auxiliary_model() -> bool:
    """Check if an auxiliary text model is available for LLM content processing."""
    try:
        from agent.auxiliary_client import resolve_provider_client
        for p in ("openrouter", "nous", "custom", "codex"):
            client, _ = resolve_provider_client(p)
            if client is not None:
                return True
        return False
    except Exception:
        return False


def get_debug_session_info() -> Dict[str, Any]:
    """Get information about the current debug session."""
    return _debug.get_session_info()


if __name__ == "__main__":
    """
    Simple test/demo when run directly
    """
    print("🌐 Standalone Web Tools Module")
    print("=" * 40)
    
    # Check if API keys are available
    web_available = check_web_api_key()
    nous_available = check_auxiliary_model()

    if web_available:
        backend = _get_backend()
        print(f"✅ Web backend: {backend}")
        if backend == "exa":
            print("   Using Exa API (https://exa.ai)")
        elif backend == "parallel":
            print("   Using Parallel API (https://parallel.ai)")
        elif backend == "tavily":
            print("   Using Tavily API (https://tavily.com)")
        else:
            print("   Using Firecrawl API (https://firecrawl.dev)")
    else:
        print("❌ No web search backend configured")
        print("Set EXA_API_KEY, PARALLEL_API_KEY, TAVILY_API_KEY, or FIRECRAWL_API_KEY")

    if not nous_available:
        print("❌ No auxiliary model available for LLM content processing")
        print("Set OPENROUTER_API_KEY, configure Nous Portal, or set OPENAI_BASE_URL + OPENAI_API_KEY")
        print("⚠️  Without an auxiliary model, LLM content processing will be disabled")
    else:
        print(f"✅ Auxiliary model available: {DEFAULT_SUMMARIZER_MODEL}")

    if not web_available:
        exit(1)

    print("🛠️  Web tools ready for use!")
    
    if nous_available:
        print(f"🧠 LLM content processing available with {DEFAULT_SUMMARIZER_MODEL}")
        print(f"   Default min length for processing: {DEFAULT_MIN_LENGTH_FOR_SUMMARIZATION} chars")
    
    # Show debug mode status
    if _debug.active:
        print(f"🐛 Debug mode ENABLED - Session ID: {_debug.session_id}")
        print(f"   Debug logs will be saved to: {_debug.log_dir}/web_tools_debug_{_debug.session_id}.json")
    else:
        print("🐛 Debug mode disabled (set WEB_TOOLS_DEBUG=true to enable)")
    
    print("\nBasic usage:")
    print("  from web_tools import web_search_tool, web_extract_tool, web_crawl_tool")
    print("  import asyncio")
    print("")
    print("  # Search (synchronous)")
    print("  results = web_search_tool('Python tutorials')")
    print("")
    print("  # Extract and crawl (asynchronous)")
    print("  async def main():")
    print("      content = await web_extract_tool(['https://example.com'])")
    print("      crawl_data = await web_crawl_tool('example.com', 'Find docs')")
    print("  asyncio.run(main())")
    
    if nous_available:
        print("\nLLM-enhanced usage:")
        print("  # Content automatically processed for pages >5000 chars (default)")
        print("  content = await web_extract_tool(['https://python.org/about/'])")
        print("")
        print("  # Customize processing parameters")
        print("  crawl_data = await web_crawl_tool(")
        print("      'docs.python.org',")
        print("      'Find key concepts',")
        print("      model='google/gemini-3-flash-preview',")
        print("      min_length=3000")
        print("  )")
        print("")
        print("  # Disable LLM processing")
        print("  raw_content = await web_extract_tool(['https://example.com'], use_llm_processing=False)")
    
    print("\nDebug mode:")
    print("  # Enable debug logging")
    print("  export WEB_TOOLS_DEBUG=true")
    print("  # Debug logs capture:")
    print("  # - All tool calls with parameters")
    print("  # - Original API responses")
    print("  # - LLM compression metrics")
    print("  # - Final processed results")
    print("  # Logs saved to: ./logs/web_tools_debug_UUID.json")
    
    print("\n📝 Run 'python test_web_tools_llm.py' to test LLM processing capabilities")


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------
from tools.registry import registry

WEB_SEARCH_SCHEMA = {
    "name": "web_search",
    "description": (
        "Search the web for information on any topic using a single backend. "
        "Firecrawl supports optional sources, categories, domain filters, tbs, and location; other backends ignore those filters."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "The search query to look up on the web"
            },
            "limit": {
                "type": "integer",
                "description": "Maximum number of results to return",
                "default": 5,
                "minimum": 1,
                "maximum": 20
            },
            "sources": {
                "type": "array",
                "items": {"type": "string", "enum": ["web", "images", "news"]},
                "description": "Firecrawl result sources to search"
            },
            "categories": {
                "type": "array",
                "items": {"type": "string", "enum": ["github", "research", "pdf"]},
                "description": "Firecrawl category filters"
            },
            "include_domains": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Firecrawl domains to restrict results to; hostnames only, no protocol needed"
            },
            "exclude_domains": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Firecrawl domains to exclude; ignored when include_domains is set"
            },
            "tbs": {
                "type": "string",
                "description": "Firecrawl time-based search filter such as qdr:w or sbd:1,qdr:w"
            },
            "location": {
                "type": "string",
                "description": "Firecrawl search location such as San Francisco,California,United States"
            }
        },
        "required": ["query"]
    }
}

WEB_EXTRACT_SCHEMA = {
    "name": "web_extract",
    "description": "Extract content from web page URLs. Returns page content in markdown format. Also works with PDF URLs (arxiv papers, documents, etc.) — pass the PDF link directly and it converts to markdown text. Pages under 5000 chars return full markdown; larger pages are LLM-summarized and capped at ~5000 chars per page. Pages over 2M chars are refused. If a URL fails or times out, use the browser tool to access it instead.",
    "parameters": {
        "type": "object",
        "properties": {
            "urls": {
                "type": "array",
                "items": {"type": "string"},
                "description": "List of URLs to extract content from (max 5 URLs per call)",
                "maxItems": 5
            },
            "format": {
                "type": "string",
                "enum": ["markdown", "html", "raw_html", "summary", "links", "images", "screenshot"],
                "description": "Single Firecrawl output format to request"
            },
            "formats": {
                "type": "array",
                "items": {"type": "string", "enum": ["markdown", "html", "raw_html", "summary", "links", "images", "screenshot"]},
                "description": "Multiple Firecrawl output formats to request; overrides format when provided"
            },
            "only_main_content": {
                "type": "boolean",
                "description": "Firecrawl onlyMainContent; defaults true in Firecrawl when omitted"
            },
            "only_clean_content": {
                "type": "boolean",
                "description": "Firecrawl beta LLM cleaner; not compatible with zero-data-retention"
            },
            "include_tags": {
                "type": "array",
                "items": {"type": "string"},
                "description": "HTML tags/selectors Firecrawl should include"
            },
            "exclude_tags": {
                "type": "array",
                "items": {"type": "string"},
                "description": "HTML tags/selectors Firecrawl should exclude"
            },
            "max_age": {
                "type": "integer",
                "description": "Firecrawl cache maxAge in milliseconds"
            },
            "min_age": {
                "type": "integer",
                "description": "Firecrawl cache-only minAge in milliseconds"
            },
            "wait_for": {
                "type": "integer",
                "description": "Extra Firecrawl wait time before extraction, in milliseconds"
            },
            "mobile": {
                "type": "boolean",
                "description": "Use Firecrawl mobile emulation"
            },
            "timeout": {
                "type": "integer",
                "description": "Firecrawl request timeout in milliseconds",
                "minimum": 1000,
                "maximum": 300000
            },
            "pdf_parser_mode": {
                "type": "string",
                "enum": ["fast", "auto", "ocr"],
                "description": "Firecrawl PDF parser mode"
            },
            "pdf_max_pages": {
                "type": "integer",
                "minimum": 1,
                "maximum": 10000,
                "description": "Maximum PDF pages Firecrawl should parse"
            },
            "actions": {
                "type": "array",
                "description": "Safe Firecrawl pre-scrape actions: wait, click, press, scroll, screenshot",
                "items": {"type": "object"}
            }
        },
        "required": ["urls"]
    }
}

WEB_CRAWL_SCHEMA = {
    "name": "web_crawl",
    "description": (
        "Crawl a website with Firecrawl. Uses instructions as Firecrawl crawl prompt; "
        "explicit crawl parameters override prompt-generated equivalents."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "url": {"type": "string", "description": "Base URL to crawl"},
            "instructions": {"type": "string", "description": "Natural-language Firecrawl crawl prompt"},
            "limit": {
                "type": "integer",
                "description": "Maximum pages to crawl",
                "default": 20,
                "minimum": 1,
                "maximum": 10000
            },
            "include_paths": {
                "type": "array",
                "items": {"type": "string"},
                "description": "URL pathname regexes to include"
            },
            "exclude_paths": {
                "type": "array",
                "items": {"type": "string"},
                "description": "URL pathname regexes to exclude"
            },
            "max_discovery_depth": {"type": "integer", "minimum": 0},
            "sitemap": {"type": "string", "enum": ["skip", "include", "only"]},
            "ignore_query_parameters": {"type": "boolean"},
            "regex_on_full_url": {"type": "boolean"},
            "crawl_entire_domain": {"type": "boolean"},
            "allow_subdomains": {"type": "boolean"},
            "allow_external_links": {
                "type": "boolean",
                "default": False,
                "description": "Allow external links; defaults false"
            },
            "delay": {"type": "number", "minimum": 0},
            "max_concurrency": {"type": "integer", "minimum": 1},
            "scrape_options": {
                "type": "object",
                "description": "Nested safe Firecrawl scrape options for each crawled page",
                "properties": {
                    "format": {"type": "string", "enum": ["markdown", "html", "raw_html", "summary", "links", "images", "screenshot"]},
                    "formats": {
                        "type": "array",
                        "items": {"type": "string", "enum": ["markdown", "html", "raw_html", "summary", "links", "images", "screenshot"]}
                    },
                    "only_main_content": {"type": "boolean"},
                    "only_clean_content": {"type": "boolean"},
                    "include_tags": {"type": "array", "items": {"type": "string"}},
                    "exclude_tags": {"type": "array", "items": {"type": "string"}},
                    "max_age": {"type": "integer"},
                    "min_age": {"type": "integer"},
                    "wait_for": {"type": "integer"},
                    "mobile": {"type": "boolean"},
                    "timeout": {"type": "integer", "minimum": 1000, "maximum": 300000},
                    "pdf_parser_mode": {"type": "string", "enum": ["fast", "auto", "ocr"]},
                    "pdf_max_pages": {"type": "integer", "minimum": 1, "maximum": 10000},
                    "actions": {"type": "array", "items": {"type": "object"}}
                }
            }
        },
        "required": ["url"]
    }
}

registry.register(
    name="web_search",
    toolset="web",
    schema=WEB_SEARCH_SCHEMA,
    handler=lambda args, **kw: web_search_tool(
        args.get("query", ""),
        limit=args.get("limit", 5),
        user_task=kw.get("user_task"),
        sources=args.get("sources"),
        categories=args.get("categories"),
        include_domains=args.get("include_domains"),
        exclude_domains=args.get("exclude_domains"),
        tbs=args.get("tbs"),
        location=args.get("location"),
    ),
    check_fn=check_web_api_key,
    requires_env=["EXA_API_KEY", "PARALLEL_API_KEY", "FIRECRAWL_API_KEY", "FIRECRAWL_API_URL", "TAVILY_API_KEY"],
    emoji="🔍",
)
registry.register(
    name="web_extract",
    toolset="web",
    schema=WEB_EXTRACT_SCHEMA,
    handler=lambda args, **kw: web_extract_tool(
        args.get("urls", [])[:5] if isinstance(args.get("urls"), list) else [],
        args.get("format", "markdown"),
        formats=args.get("formats"),
        only_main_content=args.get("only_main_content"),
        only_clean_content=args.get("only_clean_content"),
        include_tags=args.get("include_tags"),
        exclude_tags=args.get("exclude_tags"),
        max_age=args.get("max_age"),
        min_age=args.get("min_age"),
        wait_for=args.get("wait_for"),
        mobile=args.get("mobile"),
        timeout=args.get("timeout"),
        pdf_parser_mode=args.get("pdf_parser_mode"),
        pdf_max_pages=args.get("pdf_max_pages"),
        actions=args.get("actions"),
    ),
    check_fn=check_web_api_key,
    requires_env=["EXA_API_KEY", "PARALLEL_API_KEY", "FIRECRAWL_API_KEY", "FIRECRAWL_API_URL", "TAVILY_API_KEY"],
    is_async=True,
    emoji="📄",
)
registry.register(
    name="web_crawl",
    toolset="web",
    schema=WEB_CRAWL_SCHEMA,
    handler=lambda args, **kw: web_crawl_tool(
        args.get("url", ""),
        instructions=args.get("instructions"),
        limit=args.get("limit", 20),
        include_paths=args.get("include_paths"),
        exclude_paths=args.get("exclude_paths"),
        max_discovery_depth=args.get("max_discovery_depth"),
        sitemap=args.get("sitemap"),
        ignore_query_parameters=args.get("ignore_query_parameters"),
        regex_on_full_url=args.get("regex_on_full_url"),
        crawl_entire_domain=args.get("crawl_entire_domain"),
        allow_subdomains=args.get("allow_subdomains"),
        allow_external_links=args.get("allow_external_links", False),
        delay=args.get("delay"),
        max_concurrency=args.get("max_concurrency"),
        scrape_options=args.get("scrape_options"),
    ),
    check_fn=check_web_api_key,
    requires_env=["FIRECRAWL_API_KEY", "FIRECRAWL_API_URL"],
    is_async=True,
    emoji="🕸️",
)
