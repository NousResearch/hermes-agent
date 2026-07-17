from __future__ import annotations

from typing import Any

from tools.registry import tool_error, tool_result

from .client import XClient
from .oauth import load_tokens


def twitter_available() -> bool:
    return load_tokens() is not None


async def handle_bookmarks(args: dict, **kwargs) -> str:
    operation = str(args.get("operation") or "list").strip().lower()
    post_id = str(args.get("post_id") or "").strip()
    if operation not in {"list", "add", "remove"}:
        return tool_error("operation must be list, add, or remove")
    if operation != "list" and not post_id.isdigit():
        return tool_error("post_id must be a numeric X post ID")

    tokens = load_tokens()
    if tokens is None:
        return tool_error("Twitter OAuth is not configured")
    client = XClient(token=tokens.access_token)
    try:
        user_id = tokens.user_id
        if not user_id:
            user_id = str(((await client.identity()).get("data") or {}).get("id") or "")
        if not user_id:
            return tool_error("X did not return the authenticated user ID")
        return tool_result(
            await client.bookmarks(user_id, operation, post_id=post_id)
        )
    except Exception as exc:
        return tool_error(f"Twitter bookmarks failed: {exc}")
    finally:
        await client.close()


async def handle_post_metrics(args: dict, **kwargs) -> str:
    raw_ids: Any = args.get("post_ids")
    if not isinstance(raw_ids, list):
        return tool_error("post_ids must be an array of X post IDs")
    post_ids = [str(item).strip() for item in raw_ids]
    if not 1 <= len(post_ids) <= 100 or any(not item.isdigit() for item in post_ids):
        return tool_error("post_ids must contain 1 to 100 numeric X post IDs")

    tokens = load_tokens()
    if tokens is None:
        return tool_error("Twitter OAuth is not configured")
    client = XClient(token=tokens.access_token)
    try:
        return tool_result(await client.post_metrics(post_ids))
    except Exception as exc:
        return tool_error(f"Twitter post metrics failed: {exc}")
    finally:
        await client.close()


TWITTER_BOOKMARKS_SCHEMA = {
    "name": "twitter_bookmarks",
    "description": "List, add, or remove bookmarks for the authenticated X account.",
    "parameters": {
        "type": "object",
        "properties": {
            "operation": {"type": "string", "enum": ["list", "add", "remove"]},
            "post_id": {
                "type": "string",
                "description": "Required for add and remove; preserve the ID as a string.",
            },
        },
        "required": ["operation"],
    },
}


TWITTER_POST_METRICS_SCHEMA = {
    "name": "twitter_post_metrics",
    "description": "Read metrics for up to 100 X posts by ID.",
    "parameters": {
        "type": "object",
        "properties": {
            "post_ids": {
                "type": "array",
                "items": {"type": "string"},
                "minItems": 1,
                "maxItems": 100,
            }
        },
        "required": ["post_ids"],
    },
}


def register_tools(ctx) -> None:
    for name, schema, handler, emoji in (
        (
            "twitter_bookmarks",
            TWITTER_BOOKMARKS_SCHEMA,
            handle_bookmarks,
            "🔖",
        ),
        (
            "twitter_post_metrics",
            TWITTER_POST_METRICS_SCHEMA,
            handle_post_metrics,
            "📊",
        ),
    ):
        ctx.register_tool(
            name=name,
            toolset="twitter",
            schema=schema,
            handler=handler,
            check_fn=twitter_available,
            emoji=emoji,
        )
