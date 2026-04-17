"""Shared runtime helpers for direct oral send-message shortcuts."""

from __future__ import annotations

from typing import Any, Callable

from gateway.send_intents import (
    extract_send_confirmation_message,
    looks_like_send_confirmation,
    looks_like_send_query,
)
from gateway.send_requests import match_send_request


def match_admin_platform_send_request(
    *,
    source: Any,
    body: str,
    conversation_history: list[dict[str, Any]] | None,
    admin_ids_configured: bool,
    is_admin_user: bool,
    inline_extractor,
    history_target_extractor,
    direct_target_extractor,
    query_prompt_formatter,
) -> tuple[dict[str, Any] | None, str | None]:
    normalized_body = str(body or "").strip()
    if source is None or not normalized_body:
        return None, None
    if not admin_ids_configured or not is_admin_user:
        return None, None
    return match_send_request(
        source=source,
        body=normalized_body,
        conversation_history=conversation_history,
        inline_extractor=inline_extractor,
        history_target_extractor=history_target_extractor,
        direct_target_extractor=direct_target_extractor,
        looks_like_send_query=looks_like_send_query,
        looks_like_send_confirmation=looks_like_send_confirmation,
        extract_send_confirmation_message=extract_send_confirmation_message,
        query_prompt_formatter=query_prompt_formatter,
    )


def run_admin_send_shortcut(
    *,
    tool_args: dict[str, Any] | None,
    shortcut_error: str | None,
    tool_runner: Callable[[dict[str, Any]], dict[str, Any]],
    error_prefix: str,
    reply_formatter: Callable[[dict[str, Any]], str],
    logger,
) -> str | None:
    if shortcut_error:
        return shortcut_error
    if not tool_args:
        return None

    try:
        result = tool_runner(tool_args)
    except Exception as exc:
        logger.warning("%s: %s", error_prefix, exc)
        return f"{error_prefix}：{exc}"

    if result.get("error"):
        return str(result["error"])
    return reply_formatter(tool_args)
