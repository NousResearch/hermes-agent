"""Action-stall continuation helpers.

This module is the tiny shared contract between the conversation loop that
inserts an internal corrective continuation and transports that can mechanically
force tool emission for that single retry. Keep the marker stable: it is a
wire-protocol string in in-memory message history, not user-facing prose.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

ACTION_STALL_CONTINUATION_PREFIX = "[System corrective continuation: tool execution required]"


def build_action_stall_continuation() -> str:
    """Return the internal user message used after an action-ack/no-tool stall."""
    return (
        f"{ACTION_STALL_CONTINUATION_PREFIX}\n"
        "Continue now. Execute the required tool calls and only send your "
        "final answer after completing the task."
    )


def latest_user_message_is_stall_continuation(messages: Sequence[Mapping[str, Any]] | None) -> bool:
    """True when the latest user message is the action-stall continuation.

    Defensive loop guard: walk backward from that latest user message to the
    nearest prior assistant turn. If it already had tool_calls, do not force
    another required-tool turn. The corrective continuation is meant only for
    the failure mode where the model narrated intent/progress but emitted zero
    tool calls.
    """
    if not messages:
        return False

    latest_user_idx: int | None = None
    latest_user: Mapping[str, Any] | None = None
    for idx in range(len(messages) - 1, -1, -1):
        msg = messages[idx]
        if isinstance(msg, Mapping) and msg.get("role") == "user":
            latest_user_idx = idx
            latest_user = msg
            break

    if latest_user_idx is None or latest_user is None:
        return False
    if latest_user_idx != len(messages) - 1:
        # The corrective marker only forces the immediate retry request. Once
        # an assistant tool-call turn or tool result has been appended after it,
        # the follow-up request must return to auto so the model can stop.
        return False

    content = latest_user.get("content")
    if isinstance(content, str):
        text = content.lstrip()
    elif isinstance(content, list):
        parts: list[str] = []
        for part in content:
            if isinstance(part, str):
                parts.append(part)
            elif isinstance(part, Mapping):
                part_text = part.get("text") or part.get("content")
                if isinstance(part_text, str):
                    parts.append(part_text)
        text = "\n".join(parts).lstrip()
    else:
        return False

    if not text.startswith(ACTION_STALL_CONTINUATION_PREFIX):
        return False

    saw_plain_assistant = False
    for idx in range(latest_user_idx - 1, -1, -1):
        msg = messages[idx]
        if not isinstance(msg, Mapping):
            continue
        role = msg.get("role")
        if role == "tool":
            return False
        if role == "assistant":
            if msg.get("tool_calls"):
                return False
            saw_plain_assistant = True
            continue
        if role == "user":
            # Force only when the current segment is user → assistant(no tools)
            # → corrective user marker, with no tool evidence in between.
            return saw_plain_assistant

    return False
