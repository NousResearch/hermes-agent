"""Shared runtime helpers for shared-group history preprocessing."""

from __future__ import annotations

from typing import Any

DEFAULT_SHARED_GROUP_VISIBLE_HISTORY_LIMIT = 24


def is_shared_group_internal_artifact(content: Any) -> bool:
    """Return True when a transcript entry is internal agent noise."""

    text = str(content or "").strip()
    if not text:
        return False
    return text.startswith(
        (
            "[CONTEXT COMPACTION]",
            "[Your active task list was preserved across context compression]",
        )
    )


def simplify_shared_group_history_for_agent(
    history: list[dict[str, Any]],
    *,
    visible_limit: int = DEFAULT_SHARED_GROUP_VISIBLE_HISTORY_LIMIT,
) -> list[dict[str, Any]]:
    """Reduce shared-group history to visible chat turns only."""

    visible_messages: list[dict[str, Any]] = []
    for msg in history:
        role = str(msg.get("role") or "").strip()
        if role not in ("user", "assistant"):
            continue

        content = msg.get("content")
        if not isinstance(content, str):
            continue
        content = content.strip()
        if not content or content == "[[NO_REPLY]]":
            continue
        if is_shared_group_internal_artifact(content):
            continue

        visible_messages.append({"role": role, "content": content})

    if visible_limit > 0 and len(visible_messages) > visible_limit:
        visible_messages = visible_messages[-visible_limit:]

    return visible_messages


def prepare_history_for_agent(
    history: list[dict[str, Any]],
    *,
    shared_session_kind: str | None,
    session_id: str = "",
    logger=None,
    visible_limit: int = DEFAULT_SHARED_GROUP_VISIBLE_HISTORY_LIMIT,
) -> list[dict[str, Any]]:
    """Return the transcript slice that should be replayed into the agent."""

    if shared_session_kind != "group":
        return history

    simplified = simplify_shared_group_history_for_agent(
        history,
        visible_limit=visible_limit,
    )
    if logger is not None and len(simplified) != len(history):
        logger.info(
            "Shared group history simplified for session %s: %d -> %d messages",
            session_id,
            len(history),
            len(simplified),
        )
    return simplified
