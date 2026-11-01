"""Destination-scoped recovery for strict MoA chat templates.

Private assistant observations can precede an assistant action. Destinations that
reject this adjacency get one merged retry; accepted request shapes stay intact.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


def destination_key(runtime: dict[str, Any]) -> tuple[str, str]:
    """``(route, model)`` identity of an aggregator destination: the base_url when the slot
    resolved one (two providers can share a model id), else the provider slug."""
    route = str(runtime.get("base_url") or runtime.get("provider") or "").strip().rstrip("/")
    return route, str(runtime.get("model") or "").strip()


def merge_same_role_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Merge adjacent users or plain assistant observations into the following action.

    Action fields, including tool calls, remain on the following row. Never merge
    an assistant row already carrying action metadata. Inputs remain unchanged.
    """
    from agent.agent_runtime_helpers import _UNMERGEABLE, _merge_user_content

    merged: list[dict[str, Any]] = []
    changed = False
    for message in messages:
        prev = merged[-1] if merged else None
        content: Any = _UNMERGEABLE
        if prev is not None and prev.get("role") == "user" and message.get("role") == "user":
            content = _merge_user_content(prev.get("content", ""), message.get("content", ""))
        assistant_observation = (prev is not None and prev.get("role") == "assistant"
                                 and message.get("role") == "assistant"
                                 and set(prev) <= {"role", "content"})
        if assistant_observation:
            action_content = message.get("content")
            content = _merge_user_content(prev.get("content", ""),
                                          "" if action_content is None else action_content)
        if content is _UNMERGEABLE:
            merged.append(message)
            continue
        merged[-1] = {**(message if assistant_observation else prev), "content": content}
        changed = True
    return merged if changed else messages


def is_role_alternation_rejection(exc: Exception, runtime: dict[str, Any]) -> bool:
    """True when the aggregator destination rejected the request for adjacent same-role messages."""
    from agent.error_classifier import FailoverReason, classify_api_error

    try:
        classified = classify_api_error(
            exc, provider=str(runtime.get("provider") or ""), model=str(runtime.get("model") or ""),
            base_url=str(runtime.get("base_url") or ""),
        )
    except Exception:  # pragma: no cover - classification must never mask the original error
        return False
    return classified.reason is FailoverReason.role_alternation
