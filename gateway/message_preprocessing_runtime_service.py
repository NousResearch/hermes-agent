"""Shared runtime helpers for gateway message preprocessing."""

from __future__ import annotations

from typing import Any


def is_shared_thread_session(
    *,
    source: Any,
    thread_sessions_per_user: bool,
) -> bool:
    """Return True when a thread is shared across multiple participants."""

    return bool(
        getattr(source, "chat_type", None) != "dm"
        and getattr(source, "thread_id", None)
        and not thread_sessions_per_user
    )


def message_history_contains_reply_snippet(
    history: list[dict[str, Any]],
    *,
    reply_snippet: str,
    compare_chars: int = 200,
) -> bool:
    """Return True when the quoted reply text is already present in history."""

    probe = str(reply_snippet or "")[:compare_chars]
    if not probe:
        return False
    return any(
        probe in str(msg.get("content") or "")
        for msg in history
        if msg.get("role") in ("assistant", "user", "tool")
    )


def prepend_reply_context_if_missing(
    *,
    message_text: str,
    reply_to_text: str | None,
    reply_to_message_id: str | None,
    history: list[dict[str, Any]],
    reply_limit: int = 500,
) -> str:
    """Prepend quoted reply text when it is not already present in history."""

    if not reply_to_text or not reply_to_message_id:
        return message_text
    reply_snippet = str(reply_to_text)[:reply_limit]
    if message_history_contains_reply_snippet(history, reply_snippet=reply_snippet):
        return message_text
    return f'[Replying to: "{reply_snippet}"]\n\n{message_text}'


def prepend_shared_thread_sender(
    *,
    message_text: str,
    user_name: str | None,
    shared_thread: bool,
) -> str:
    """Prefix thread messages with the visible sender when needed."""

    if not shared_thread or not user_name or not str(message_text or "").strip():
        return message_text
    return f"[{user_name}] {message_text}"
