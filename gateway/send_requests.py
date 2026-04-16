"""Pure request-parsing helpers for oral send-message shortcuts."""

from __future__ import annotations

from typing import Any, Callable, Iterable


def match_send_request(
    *,
    source,
    body: str,
    conversation_history: Iterable[dict[str, Any]] | None,
    inline_extractor: Callable[[str], tuple[str, str]],
    history_target_extractor: Callable[[Any, Iterable[dict[str, Any]] | None], str],
    direct_target_extractor: Callable[[Any, str], str | None],
    looks_like_send_query: Callable[[str], bool],
    looks_like_send_confirmation: Callable[[str], bool],
    extract_send_confirmation_message: Callable[[str], str],
    query_prompt_formatter: Callable[[str], str],
) -> tuple[dict[str, str] | None, str | None]:
    normalized_body = str(body or "").strip()

    inline_target, inline_message = inline_extractor(normalized_body)
    if inline_target and inline_message:
        return {
            "target": inline_target,
            "message": inline_message,
        }, None

    if looks_like_send_confirmation(normalized_body):
        pending_target = history_target_extractor(source, conversation_history)
        confirmation_message = extract_send_confirmation_message(normalized_body)
        if pending_target and confirmation_message:
            return {
                "target": pending_target,
                "message": confirmation_message,
            }, None

    if looks_like_send_query(normalized_body):
        target = direct_target_extractor(source, normalized_body)
        if target:
            return None, query_prompt_formatter(str(target).replace("group:", "").strip())

    return None, None
