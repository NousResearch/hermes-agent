"""Fourteen native MrScraper tools (rendered-page tool lives under browser)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional

from plugins.mrscraper_client import (
    MrScraperClient,
    MrScraperError,
    compact_optional,
    encoded_path_segment,
)
from tools.registry import tool_error, tool_result

_PROMPTS_PATH = Path(__file__).with_name("structured_data_prompts.json")
STRUCTURED_DATA_PROMPTS: Dict[str, str] = json.loads(
    _PROMPTS_PATH.read_text(encoding="utf-8")
)

STRUCTURED_CATEGORIES = [
    "article",
    "forumThread",
    "hotel",
    "jobPosting",
    "post",
    "product",
    "property",
    "restaurant",
    "socialMediaProfile",
    "tourAttraction",
]


def _required_string(args: Mapping[str, Any], name: str) -> str:
    value = str(args.get(name) or "").strip()
    if not value:
        raise MrScraperError(f"{name} is required")
    return value


def _optional_string(args: Mapping[str, Any], name: str) -> Optional[str]:
    raw = args.get(name)
    if raw is None:
        return None
    value = str(raw).strip()
    return value or None


def _two_letter_code(args: Mapping[str, Any], name: str, default: str) -> str:
    value = str(args.get(name, default)).strip().lower()
    if len(value) != 2 or not value.isalpha() or not value.isascii():
        raise MrScraperError(f"{name} must be a two-letter code")
    return value


def _integer(args: Mapping[str, Any], name: str, default: int, minimum: int) -> int:
    raw = args.get(name, default)
    if isinstance(raw, bool):
        raise MrScraperError(f"{name} must be an integer")
    try:
        value = int(raw)
    except (TypeError, ValueError) as exc:
        raise MrScraperError(f"{name} must be an integer") from exc
    if value < minimum:
        raise MrScraperError(f"{name} must be at least {minimum}")
    return value


def _boolean(args: Mapping[str, Any], name: str, default: bool) -> bool:
    value = args.get(name, default)
    if not isinstance(value, bool):
        raise MrScraperError(f"{name} must be a boolean")
    return value


def _enum(
    args: Mapping[str, Any], name: str, default: str, allowed: Iterable[str]
) -> str:
    value = str(args.get(name, default))
    allowed_values = tuple(allowed)
    if value not in allowed_values:
        raise MrScraperError(f"{name} must be one of: {', '.join(allowed_values)}")
    return value


def _object(
    args: Mapping[str, Any], name: str, default: Optional[dict] = None
) -> Optional[dict]:
    raw = args.get(name, default)
    if raw is None:
        return None
    if not isinstance(raw, dict):
        raise MrScraperError(f"{name} must be an object")
    return raw


def _schema_message(
    prompt: Optional[str], schema: Optional[dict], label: str
) -> Optional[str]:
    if schema is None:
        return prompt
    suffix = f"{label}\n{json.dumps(schema, ensure_ascii=False, separators=(',', ':'))}"
    return f"{prompt}\n\n{suffix}" if prompt else suffix


def _crawl_payload(args: Mapping[str, Any]) -> Dict[str, Any]:
    return compact_optional({
        "graph": "map",
        "url": _required_string(args, "url"),
        "maxDepth": _integer(args, "max_depth", 2, 0),
        "maxPages": _integer(args, "max_pages", 50, 1),
        "limit": _integer(args, "limit", 50, 1),
        "includePatterns": _optional_string(args, "include_patterns"),
        "excludePatterns": _optional_string(args, "exclude_patterns"),
    })


def _prompt_payload(args: Mapping[str, Any]) -> Dict[str, Any]:
    prompt = _optional_string(args, "prompt")
    schema = _object(args, "output_schema")
    return compact_optional({
        "graph": "general",
        "url": _required_string(args, "url"),
        "message": _schema_message(
            prompt,
            schema,
            "Return the output as JSON matching this schema:",
        ),
        "mode": _enum(args, "mode", "Super", ("Super", "Cheap")),
        "proxyCountry": _optional_string(args, "proxy_country"),
    })


def _listing_payload(args: Mapping[str, Any]) -> Dict[str, Any]:
    prompt = _optional_string(args, "prompt")
    schema = _object(args, "output_schema")
    return compact_optional({
        "graph": "listing",
        "url": _required_string(args, "url"),
        "message": _schema_message(
            prompt,
            schema,
            "Return each item as JSON matching this schema:",
        ),
        "maxPages": _integer(args, "max_pages", 1, 1),
        "proxyCountry": _optional_string(args, "proxy_country"),
    })


def get_account_info(args: Mapping[str, Any]) -> Any:
    return MrScraperClient().primary_get("/api/v1/subscription-accounts")


def crawl_website_urls(args: Mapping[str, Any]) -> Any:
    return MrScraperClient().primary_post("/api/v1/scrapers-ai", _crawl_payload(args))


def search_google_serp(args: Mapping[str, Any]) -> Any:
    output_format = _enum(args, "format", "json", ("json", "html"))
    payload = {
        "query": _required_string(args, "query"),
        "region": _two_letter_code(args, "region", "us"),
        "language": _two_letter_code(args, "language", "en"),
        "page": _integer(args, "page", 1, 1),
        "format": output_format,
        "renderJs": _boolean(args, "render_js", False),
    }
    return MrScraperClient().serp_search(payload, html=output_format == "html")


def extract_page_by_prompt(args: Mapping[str, Any]) -> Any:
    return MrScraperClient().primary_post("/api/v1/scrapers-ai", _prompt_payload(args))


def extract_listings(args: Mapping[str, Any]) -> Any:
    return MrScraperClient().primary_post("/api/v1/scrapers-ai", _listing_payload(args))


def extract_structured_data(args: Mapping[str, Any]) -> Any:
    category = _enum(args, "category", "article", STRUCTURED_CATEGORIES)
    payload = compact_optional({
        "graph": "general",
        "url": _required_string(args, "url"),
        "message": STRUCTURED_DATA_PROMPTS[category],
        "mode": _enum(args, "mode", "Super", ("Super", "Cheap")),
        "proxyCountry": _optional_string(args, "proxy_country"),
    })
    return MrScraperClient().primary_post("/api/v1/scrapers-ai", payload)


def get_results(args: Mapping[str, Any]) -> Any:
    params = {
        "filters[scraperId]": _required_string(args, "scraper_id"),
        "page": _integer(args, "page", 1, 1),
        "pageSize": _integer(args, "page_size", 10, 1),
        "sort": _enum(args, "sort_by", "createdAt", ("createdAt",)),
        "sortOrder": _enum(args, "sort_order", "DESC", ("ASC", "DESC")),
    }
    return MrScraperClient().primary_get("/api/v1/results", params=params)


def get_latest_results(args: Mapping[str, Any]) -> Any:
    params = {
        "filters[scraperId]": _required_string(args, "scraper_id"),
        "page": 1,
        "pageSize": _integer(args, "count", 10, 1),
        "sort": "createdAt",
        "sortOrder": "DESC",
    }
    return MrScraperClient().primary_get("/api/v1/results", params=params)


def get_result_detail(args: Mapping[str, Any]) -> Any:
    result_id = _required_string(args, "result_id")
    return MrScraperClient().primary_get(
        f"/api/v1/results/{encoded_path_segment(result_id)}"
    )


def create_prompt_scraper(args: Mapping[str, Any]) -> Any:
    return extract_page_by_prompt(args)


def create_listing_scraper(args: Mapping[str, Any]) -> Any:
    return extract_listings(args)


def create_website_crawl_scraper(args: Mapping[str, Any]) -> Any:
    return crawl_website_urls(args)


_AI_ONLY = {
    "agent_type",
    "max_depth",
    "max_pages",
    "limit",
    "include_patterns",
    "exclude_patterns",
    "render_javascript",
    "return_cookies",
    "use_home_page",
    "wait_for_selector",
}
_MANUAL_ONLY = {
    "cookie_jar",
    "cookies",
    "home_page",
    "home_page_timeout",
    "paginator",
    "proxy",
    "record",
    "return_cookie",
    "token_cap",
}


def _reject_present(
    args: Mapping[str, Any], names: Iterable[str], context: str
) -> None:
    present = sorted(name for name in names if name in args)
    if present:
        raise MrScraperError(
            f"Parameters incompatible with {context}: {', '.join(present)}"
        )


def _normalize_urls(raw: Any) -> List[str]:
    if isinstance(raw, list):
        values = raw
    elif isinstance(raw, str):
        stripped = raw.strip()
        if not stripped:
            values = []
        else:
            try:
                decoded = json.loads(stripped)
            except json.JSONDecodeError:
                decoded = None
            if isinstance(decoded, list):
                values = decoded
            else:
                values = stripped.replace(",", "\n").splitlines()
    else:
        raise MrScraperError("urls must be an array of strings")
    if any(not isinstance(item, str) for item in values):
        raise MrScraperError("urls must be an array of strings")
    normalized = [str(item).strip() for item in values if str(item).strip()]
    if not normalized:
        raise MrScraperError("urls must contain at least one nonblank URL")
    return normalized


def run_existing_scraper(args: Mapping[str, Any]) -> Any:
    scraper_type = _enum(args, "scraper_type", "", ("ai", "manual"))
    common = {
        "scraperId": _required_string(args, "scraper_id"),
        "url": _required_string(args, "url"),
        "maxRetry": _integer(args, "max_retry", 3, 0),
        "proxyCountry": _optional_string(args, "proxy_country"),
    }
    if scraper_type == "manual":
        _reject_present(args, _AI_ONLY, "scraper_type='manual'")
        body = {
            **common,
            "bypassProxy": _boolean(args, "bypass_proxy", True),
            "cookieJar": _optional_string(args, "cookie_jar"),
            "cookies": args.get("cookies", []),
            "homePage": _boolean(args, "home_page", False),
            "homePageTimeout": _integer(args, "home_page_timeout", 10, 1),
            "html": _boolean(args, "html", False),
            "markdown": _boolean(args, "markdown", False),
            "paginator": _object(args, "paginator", {}) or {},
            "proxy": _optional_string(args, "proxy"),
            "record": _boolean(args, "record", False),
            "returnCookie": _boolean(args, "return_cookie", False),
            "screenshot": str(_boolean(args, "screenshot", False)).lower(),
            "stream": _boolean(args, "stream", False),
            "timeout": _integer(args, "timeout", 600, 1),
            "tokenCap": _integer(args, "token_cap", 0, 0),
        }
        if not isinstance(body["cookies"], list) or any(
            not isinstance(cookie, dict) for cookie in body["cookies"]
        ):
            raise MrScraperError("cookies must be an array of objects")
        return MrScraperClient().primary_post(
            "/api/v1/scrapers-manual-rerun", compact_optional(body)
        )

    _reject_present(args, _MANUAL_ONLY, "scraper_type='ai'")
    agent_type = _enum(args, "agent_type", "general", ("general", "listing", "map"))
    body: Dict[str, Any] = dict(common)
    if agent_type == "map":
        _reject_present(
            args,
            {
                "bypass_proxy",
                "html",
                "markdown",
                "render_javascript",
                "return_cookies",
                "screenshot",
                "stream",
                "timeout",
                "use_home_page",
                "wait_for_selector",
            },
            "agent_type='map'",
        )
        body.update(
            compact_optional({
                "maxDepth": _integer(args, "max_depth", 2, 0),
                "maxPages": _integer(args, "max_pages", 50, 1),
                "limit": _integer(args, "limit", 50, 1),
                "includePatterns": _optional_string(args, "include_patterns"),
                "excludePatterns": _optional_string(args, "exclude_patterns"),
            })
        )
    else:
        _reject_present(
            args,
            {"max_depth", "limit", "include_patterns", "exclude_patterns"},
            f"agent_type='{agent_type}'",
        )
        if agent_type == "general":
            _reject_present(
                args, {"max_pages", "timeout", "stream"}, "agent_type='general'"
            )
        else:
            body["maxPages"] = _integer(args, "max_pages", 5, 1)
            body["timeout"] = _integer(args, "timeout", 300, 1)
            body["stream"] = _boolean(args, "stream", False)
        body.update(
            compact_optional({
                "bypassProxy": _boolean(args, "bypass_proxy", False),
                "html": _boolean(args, "html", False),
                "markdown": _boolean(args, "markdown", False),
                "renderJavascript": _boolean(args, "render_javascript", False),
                "returnCookies": _boolean(args, "return_cookies", False),
                "screenshot": _boolean(args, "screenshot", False),
                "useHomePage": _boolean(args, "use_home_page", False),
                "waitForSelector": _optional_string(args, "wait_for_selector"),
            })
        )
    return MrScraperClient().primary_post(
        "/api/v1/scrapers-ai-rerun", compact_optional(body)
    )


def run_existing_scraper_batch(args: Mapping[str, Any]) -> Any:
    scraper_type = _enum(args, "scraper_type", "", ("ai", "manual"))
    suffix = "manual" if scraper_type == "manual" else "ai"
    body = {
        "scraperId": _required_string(args, "scraper_id"),
        "urls": _normalize_urls(args.get("urls")),
    }
    return MrScraperClient().primary_post(f"/api/v1/scrapers-{suffix}-rerun/bulk", body)


def _handler(operation: Callable[[Mapping[str, Any]], Any]) -> Callable:
    def handle(args: dict, **_kwargs: Any) -> str:
        try:
            return tool_result(operation(args))
        except MrScraperError as exc:
            extra = {}
            status_code = getattr(exc, "status_code", None)
            if status_code is not None:
                extra["status_code"] = status_code
            return tool_error(str(exc), **extra)
        except Exception as exc:  # noqa: BLE001 — plugin boundary
            return tool_error(f"MrScraper tool failed: {type(exc).__name__}: {exc}")

    operation_name = getattr(operation, "__name__", "mrscraper_operation")
    handle.__name__ = f"handle_{operation_name}"
    return handle


S = {"type": "string"}
B = {"type": "boolean"}
I = {"type": "integer"}
O = {"type": "object"}


def _tool_schema(
    name: str, description: str, properties: dict, required: List[str]
) -> dict:
    return {
        "name": name,
        "description": description,
        "parameters": {
            "type": "object",
            "properties": properties,
            "required": required,
            "additionalProperties": False,
        },
    }


CRAWL_PROPERTIES = {
    "url": {**S, "description": "Target website URL."},
    "max_depth": {**I, "minimum": 0, "default": 2},
    "max_pages": {**I, "minimum": 1, "default": 50},
    "limit": {**I, "minimum": 1, "default": 50},
    "include_patterns": {**S, "description": "Pipe-separated regular expressions."},
    "exclude_patterns": {**S, "description": "Pipe-separated regular expressions."},
}
PROMPT_PROPERTIES = {
    "url": S,
    "prompt": S,
    "output_schema": O,
    "mode": {**S, "enum": ["Super", "Cheap"], "default": "Super"},
    "proxy_country": S,
}
LISTING_PROPERTIES = {
    "url": S,
    "prompt": S,
    "output_schema": O,
    "max_pages": {**I, "minimum": 1, "default": 1},
    "proxy_country": S,
}

RUN_PROPERTIES = {
    "scraper_type": {**S, "enum": ["ai", "manual"]},
    "scraper_id": S,
    "url": S,
    "max_retry": {**I, "minimum": 0, "default": 3},
    "proxy_country": S,
    "agent_type": {
        **S,
        "enum": ["general", "listing", "map"],
        "default": "general",
        "description": "AI scrapers only; defaults to general.",
    },
    "bypass_proxy": {
        **B,
        "description": "Defaults to false for AI scrapers and true for manual scrapers.",
    },
    "html": {**B, "default": False},
    "markdown": {**B, "default": False},
    "render_javascript": {**B, "default": False},
    "return_cookies": {**B, "default": False},
    "screenshot": {**B, "default": False},
    "use_home_page": {**B, "default": False},
    "wait_for_selector": S,
    "max_pages": {
        **I,
        "minimum": 1,
        "description": "Defaults to 5 for AI listing and 50 for AI map scrapers.",
    },
    "timeout": {
        **I,
        "minimum": 1,
        "description": "Defaults to 300 seconds for AI listing and 600 for manual scrapers.",
    },
    "stream": {**B, "default": False},
    "max_depth": {**I, "minimum": 0},
    "limit": {**I, "minimum": 1},
    "include_patterns": S,
    "exclude_patterns": S,
    "cookie_jar": S,
    "cookies": {"type": "array", "items": O, "default": []},
    "home_page": {**B, "default": False},
    "home_page_timeout": {**I, "minimum": 1, "default": 10},
    "paginator": {**O, "default": {}},
    "proxy": S,
    "record": {**B, "default": False},
    "return_cookie": {**B, "default": False},
    "token_cap": {**I, "minimum": 0, "default": 0},
}

MRSCRAPER_TOOLS = (
    (
        "mrscraper_get_account_info",
        _tool_schema(
            "mrscraper_get_account_info",
            "Get MrScraper account details, token usage, and token limits.",
            {},
            [],
        ),
        _handler(get_account_info),
    ),
    (
        "mrscraper_crawl_website_urls",
        _tool_schema(
            "mrscraper_crawl_website_urls",
            "Create a map crawl that discovers website URLs.",
            CRAWL_PROPERTIES,
            ["url"],
        ),
        _handler(crawl_website_urls),
    ),
    (
        "mrscraper_search_google_serp",
        _tool_schema(
            "mrscraper_search_google_serp",
            "Search Google SERP through MrScraper and return JSON or HTML.",
            {
                "query": S,
                "region": {**S, "pattern": "^[A-Za-z]{2}$", "default": "us"},
                "language": {**S, "pattern": "^[A-Za-z]{2}$", "default": "en"},
                "page": {**I, "minimum": 1, "default": 1},
                "format": {**S, "enum": ["json", "html"], "default": "json"},
                "render_js": {**B, "default": False},
            },
            ["query"],
        ),
        _handler(search_google_serp),
    ),
    (
        "mrscraper_extract_page_by_prompt",
        _tool_schema(
            "mrscraper_extract_page_by_prompt",
            "Extract one page using natural-language instructions and an optional output schema.",
            PROMPT_PROPERTIES,
            ["url"],
        ),
        _handler(extract_page_by_prompt),
    ),
    (
        "mrscraper_extract_listings",
        _tool_schema(
            "mrscraper_extract_listings",
            "Extract listing or paginated content with an optional item schema.",
            LISTING_PROPERTIES,
            ["url"],
        ),
        _handler(extract_listings),
    ),
    (
        "mrscraper_extract_structured_data",
        _tool_schema(
            "mrscraper_extract_structured_data",
            "Extract a page using an exact bundled MrScraper structured-data preset.",
            {
                "url": S,
                "category": {**S, "enum": STRUCTURED_CATEGORIES, "default": "article"},
                "mode": {**S, "enum": ["Super", "Cheap"], "default": "Super"},
                "proxy_country": S,
            },
            ["url"],
        ),
        _handler(extract_structured_data),
    ),
    (
        "mrscraper_get_results",
        _tool_schema(
            "mrscraper_get_results",
            "Get paginated results for a scraper.",
            {
                "scraper_id": S,
                "page": {**I, "minimum": 1, "default": 1},
                "page_size": {**I, "minimum": 1, "default": 10},
                "sort_by": {**S, "enum": ["createdAt"], "default": "createdAt"},
                "sort_order": {**S, "enum": ["ASC", "DESC"], "default": "DESC"},
            },
            ["scraper_id"],
        ),
        _handler(get_results),
    ),
    (
        "mrscraper_get_latest_results",
        _tool_schema(
            "mrscraper_get_latest_results",
            "Get the newest results for a scraper.",
            {"scraper_id": S, "count": {**I, "minimum": 1, "default": 10}},
            ["scraper_id"],
        ),
        _handler(get_latest_results),
    ),
    (
        "mrscraper_get_result_detail",
        _tool_schema(
            "mrscraper_get_result_detail",
            "Get one scraper result by ID.",
            {"result_id": S},
            ["result_id"],
        ),
        _handler(get_result_detail),
    ),
    (
        "mrscraper_create_prompt_scraper",
        _tool_schema(
            "mrscraper_create_prompt_scraper",
            "Create a prompt-based scraper.",
            PROMPT_PROPERTIES,
            ["url"],
        ),
        _handler(create_prompt_scraper),
    ),
    (
        "mrscraper_create_listing_scraper",
        _tool_schema(
            "mrscraper_create_listing_scraper",
            "Create a listing scraper.",
            LISTING_PROPERTIES,
            ["url"],
        ),
        _handler(create_listing_scraper),
    ),
    (
        "mrscraper_create_website_crawl_scraper",
        _tool_schema(
            "mrscraper_create_website_crawl_scraper",
            "Create a website-crawl scraper.",
            CRAWL_PROPERTIES,
            ["url"],
        ),
        _handler(create_website_crawl_scraper),
    ),
    (
        "mrscraper_run_existing_scraper",
        _tool_schema(
            "mrscraper_run_existing_scraper",
            "Run an existing AI or manual scraper on one URL. Conditional parameters are validated for the selected scraper and agent type.",
            RUN_PROPERTIES,
            ["scraper_type", "scraper_id", "url"],
        ),
        _handler(run_existing_scraper),
    ),
    (
        "mrscraper_run_existing_scraper_batch",
        _tool_schema(
            "mrscraper_run_existing_scraper_batch",
            "Run an existing AI or manual scraper over a URL batch.",
            {
                "scraper_type": {**S, "enum": ["ai", "manual"]},
                "scraper_id": S,
                "urls": {"type": "array", "items": S, "minItems": 1},
            },
            ["scraper_type", "scraper_id", "urls"],
        ),
        _handler(run_existing_scraper_batch),
    ),
)
