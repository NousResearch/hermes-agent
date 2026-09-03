"""Helpers for durable deferred gateway clarify turns."""

from __future__ import annotations

import json
from typing import Optional

DEFERRED_CLARIFY_KIND = "hermes.deferred_clarify.v1"


def make_deferred_marker(interaction_id: str) -> str:
    """Return a provider-valid tool result that asks the loop to suspend."""
    return json.dumps(
        {
            "status": "deferred",
            "kind": DEFERRED_CLARIFY_KIND,
            "interaction_id": str(interaction_id),
            "message": "Clarify prompt sent to user; current turn is suspended.",
        },
        ensure_ascii=False,
        separators=(",", ":"),
    )


def parse_deferred_marker(content: object) -> Optional[str]:
    """Extract a deferred clarify interaction id from a tool result string."""
    if not isinstance(content, str):
        return None
    try:
        payload = json.loads(content)
    except Exception:
        return None
    if not isinstance(payload, dict):
        return None
    if payload.get("status") != "deferred" or payload.get("kind") != DEFERRED_CLARIFY_KIND:
        return None
    interaction_id = payload.get("interaction_id")
    if not isinstance(interaction_id, str) or not interaction_id:
        return None
    return interaction_id


def is_deferred_clarify_result(content: object) -> bool:
    return parse_deferred_marker(content) is not None


def find_deferred_clarify_interaction_id(
    messages: list[dict],
    tool_call_ids: set[str],
) -> Optional[str]:
    """Find a deferred marker in the real ``clarify_tool`` result envelope.

    ``clarify_tool`` wraps the callback value under ``user_response``.  Keep
    detection constrained to tool rows produced by the current assistant turn
    and explicitly named ``clarify`` so arbitrary tool output cannot suspend
    the conversation loop by mimicking the marker.
    """
    for message in reversed(messages):
        if not isinstance(message, dict):
            continue
        if message.get("role") == "assistant":
            break
        if message.get("role") != "tool":
            continue
        if message.get("tool_call_id") not in tool_call_ids:
            continue
        if (message.get("name") or message.get("tool_name")) != "clarify":
            continue
        content = message.get("content")
        if not isinstance(content, str):
            continue
        try:
            envelope = json.loads(content)
        except Exception:
            continue
        if not isinstance(envelope, dict):
            continue
        user_response = envelope.get("user_response")
        interaction_id = parse_deferred_marker(user_response)
        if interaction_id:
            return interaction_id
    return None


def build_recovery_prompt(*, question: str, answer: str) -> str:
    """Build the synthetic user turn that resumes a deferred clarify."""
    return (
        "The user answered a clarify prompt from the previous Hermes turn.\n\n"
        "Previous question:\n"
        f"{question}\n\n"
        "User answer:\n"
        f"{answer}\n\n"
        "Continue the previous task using this answer. Do not ask the same "
        "clarification again unless the answer is still ambiguous."
    )


__all__ = [
    "DEFERRED_CLARIFY_KIND",
    "make_deferred_marker",
    "parse_deferred_marker",
    "is_deferred_clarify_result",
    "find_deferred_clarify_interaction_id",
    "build_recovery_prompt",
]
