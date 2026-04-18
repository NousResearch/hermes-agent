"""Pure request-matching helpers for group runtime status shortcuts."""

from __future__ import annotations

from typing import Any, Callable, Iterable

from gateway.group_control_intents import has_followup_group_reference


def match_group_runtime_status_request(
    *,
    source,
    body: str,
    conversation_history: Iterable[dict[str, Any]] | None,
    admin_ids_configured: bool,
    is_admin_user: bool,
    looks_like_group_runtime_status_query: Callable[[str], bool],
    target_extractor: Callable[[Any, str], str | None],
    history_target_extractor: Callable[[Any, Iterable[dict[str, Any]] | None], str],
) -> str | None:
    normalized_body = str(body or "").strip()
    if not admin_ids_configured:
        return None
    if not is_admin_user:
        return None
    if not looks_like_group_runtime_status_query(normalized_body):
        return None

    target = target_extractor(source, normalized_body)
    if target:
        return target
    if not has_followup_group_reference(normalized_body):
        return None
    return history_target_extractor(source, conversation_history) or None
