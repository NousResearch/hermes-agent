"""Rendered-page operation for MrScraper.

MrScraper's public API renders individual pages but does not create persistent
CDP sessions, so this module deliberately registers a native tool rather than
pretending to implement :class:`agent.browser_provider.BrowserProvider`.
"""

from __future__ import annotations

from typing import Any, Dict

from plugins.mrscraper_client import MrScraperClient, MrScraperError
from tools.registry import tool_error, tool_result


def _integer(args: dict, name: str, default: int, minimum: int) -> int:
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


def _boolean(args: dict, name: str, default: bool) -> bool:
    value = args.get(name, default)
    if not isinstance(value, bool):
        raise MrScraperError(f"{name} must be a boolean")
    return value


def build_rendered_request(args: dict) -> tuple[Dict[str, Any], Dict[str, Any], int]:
    """Validate tool arguments and build query/body values for the API."""
    url = str(args.get("url") or "").strip()
    if not url:
        raise MrScraperError("url is required")

    timeout = _integer(args, "timeout", 300, 1)
    screenshot = _boolean(args, "screenshot", False)
    screenshot_mode = str(args.get("screenshot_mode", "full"))
    if screenshot_mode not in {"full", "top"}:
        raise MrScraperError("screenshot_mode must be one of: full, top")
    wait_until = str(args.get("wait_until", "domcontentloaded"))
    if wait_until not in {"domcontentloaded", "load", "networkidle"}:
        raise MrScraperError(
            "wait_until must be one of: domcontentloaded, load, networkidle"
        )

    wait_for_selector = args.get("wait_for_selector")
    if wait_for_selector is not None:
        wait_for_selector = str(wait_for_selector).strip() or None

    params: Dict[str, Any] = {
        "timeout": timeout,
        "geoCode": str(args.get("geo_code", "us")),
        "html": str(_boolean(args, "html", True)).lower(),
        "markdown": str(_boolean(args, "markdown", False)).lower(),
        "proxyCountry": str(args.get("proxy_country", "us")),
        "waitUntil": wait_until,
        "blockResources": str(_boolean(args, "block_resources", True)).lower(),
        "returnCookie": str(_boolean(args, "return_cookie", True)).lower(),
        "super": str(_boolean(args, "super_mode", True)).lower(),
    }
    if screenshot:
        params["screenshot"] = screenshot_mode
    if wait_for_selector is not None:
        params["waitForSelector"] = wait_for_selector

    body = {
        "url": url,
        "maxRetries": _integer(args, "max_retries", 3, 0),
        "tokenCap": _integer(args, "token_cap", 30, 1),
        "homePage": _boolean(args, "home_page", False),
    }
    return params, body, timeout


def fetch_rendered_html(args: dict) -> Any:
    params, body, timeout = build_rendered_request(args)
    return MrScraperClient().fetch_rendered(
        params=params,
        body=body,
        timeout=timeout,
    )


def handle_fetch_rendered_html(args: dict, **_kwargs: Any) -> str:
    try:
        return tool_result(fetch_rendered_html(args))
    except MrScraperError as exc:
        extra = {}
        status_code = getattr(exc, "status_code", None)
        if status_code is not None:
            extra["status_code"] = status_code
        return tool_error(str(exc), **extra)
    except Exception as exc:  # noqa: BLE001 — plugin boundary
        return tool_error(
            f"MrScraper rendered-page tool failed: {type(exc).__name__}: {exc}"
        )


MRSCRAPER_FETCH_RENDERED_HTML_SCHEMA = {
    "name": "mrscraper_fetch_rendered_html",
    "description": (
        "Fetch a browser-rendered page through MrScraper with optional HTML, "
        "Markdown, screenshot, proxy, cookie, and wait controls."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "url": {"type": "string", "description": "Target URL."},
            "max_retries": {"type": "integer", "minimum": 0, "default": 3},
            "timeout": {"type": "integer", "minimum": 1, "default": 300},
            "geo_code": {"type": "string", "default": "us"},
            "proxy_country": {"type": "string", "default": "us"},
            "screenshot": {"type": "boolean", "default": False},
            "screenshot_mode": {
                "type": "string",
                "enum": ["full", "top"],
                "default": "full",
            },
            "html": {"type": "boolean", "default": True},
            "markdown": {"type": "boolean", "default": False},
            "token_cap": {"type": "integer", "minimum": 1, "default": 30},
            "wait_for_selector": {"type": "string"},
            "wait_until": {
                "type": "string",
                "enum": ["domcontentloaded", "load", "networkidle"],
                "default": "domcontentloaded",
            },
            "block_resources": {"type": "boolean", "default": True},
            "home_page": {"type": "boolean", "default": False},
            "return_cookie": {"type": "boolean", "default": True},
            "super_mode": {"type": "boolean", "default": True},
        },
        "required": ["url"],
        "additionalProperties": False,
    },
}
