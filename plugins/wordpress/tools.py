"""Tool handlers for the WordPress plugin."""

from __future__ import annotations

from .auth import wordpress_requirements_met
from .client import WordPressClient
from .errors import WordPressAPIError, WordPressError
from tools.registry import tool_error, tool_result


def _wordpress_tool_error(exc: Exception) -> str:
    if isinstance(exc, WordPressAPIError):
        return tool_error(str(exc), status_code=exc.status_code)
    if isinstance(exc, WordPressError):
        return tool_error(str(exc))
    return tool_error(f"WordPress tool failed: {type(exc).__name__}: {exc}")


def handle_wp_site_info(args: dict, **kwargs) -> str:
    try:
        client = WordPressClient.from_env(base_url=args.get("base_url"))
        payload = client.get_site_info()
        return tool_result(payload)
    except Exception as exc:
        return _wordpress_tool_error(exc)


def _coerce_post_payload(args: dict) -> dict:
    payload = {}
    for field in (
        "title",
        "content",
        "excerpt",
        "slug",
        "status",
        "featured_media",
        "categories",
        "tags",
        "date",
        "date_gmt",
    ):
        value = args.get(field)
        if value is not None:
            payload[field] = value
    return payload


def handle_wp_post_list(args: dict, **kwargs) -> str:
    try:
        client = WordPressClient.from_env(base_url=args.get("base_url"))
        query = {
            "status": args.get("status"),
            "search": args.get("search"),
            "per_page": args.get("per_page"),
            "page": args.get("page"),
        }
        payload = client.list_posts(query=query)
        return tool_result(payload)
    except Exception as exc:
        return _wordpress_tool_error(exc)


def handle_wp_post_get(args: dict, **kwargs) -> str:
    if args.get("post_id") is None:
        return tool_error("post_id is required")
    try:
        client = WordPressClient.from_env(base_url=args.get("base_url"))
        payload = client.get_post(int(args["post_id"]), context=str(args.get("context") or "edit"))
        return tool_result(payload)
    except Exception as exc:
        return _wordpress_tool_error(exc)


def handle_wp_post_create(args: dict, **kwargs) -> str:
    if not any(str(args.get(field) or "").strip() for field in ("title", "content", "excerpt", "status")):
        return tool_error("Provide at least one of: title, content, excerpt, status")
    try:
        client = WordPressClient.from_env(base_url=args.get("base_url"))
        payload = client.create_post(_coerce_post_payload(args))
        return tool_result(payload)
    except Exception as exc:
        return _wordpress_tool_error(exc)


def handle_wp_post_update(args: dict, **kwargs) -> str:
    if args.get("post_id") is None:
        return tool_error("post_id is required")
    payload = _coerce_post_payload(args)
    if not payload:
        return tool_error("Provide at least one field to update")
    try:
        client = WordPressClient.from_env(base_url=args.get("base_url"))
        result = client.update_post(int(args["post_id"]), payload)
        return tool_result(result)
    except Exception as exc:
        return _wordpress_tool_error(exc)
