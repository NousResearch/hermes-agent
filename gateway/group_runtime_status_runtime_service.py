"""Shared runtime helpers for oral group-runtime-status shortcuts."""

from __future__ import annotations

from typing import Any, Callable

from gateway.group_reply_formatters import format_group_runtime_status_reply
from gateway.group_runtime_status_requests import match_group_runtime_status_request


def try_handle_admin_platform_group_runtime_status(
    *,
    source: Any,
    body: str,
    conversation_history: list[dict[str, Any]] | None,
    admin_ids_configured: bool,
    is_admin_user: bool,
    looks_like_group_runtime_status_query,
    target_extractor,
    history_target_extractor,
    status_loader: Callable[[str], dict[str, Any]],
) -> str | None:
    normalized_body = str(body or "").strip()
    if source is None or not normalized_body:
        return None
    target = match_group_runtime_status_request(
        source=source,
        body=normalized_body,
        conversation_history=conversation_history,
        admin_ids_configured=admin_ids_configured,
        is_admin_user=is_admin_user,
        looks_like_group_runtime_status_query=looks_like_group_runtime_status_query,
        target_extractor=target_extractor,
        history_target_extractor=history_target_extractor,
    )
    if not target:
        return None
    return format_group_runtime_status_reply(**status_loader(target))
