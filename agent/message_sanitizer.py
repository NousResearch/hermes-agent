"""Pure message-list hygiene helpers extracted from AIAgent (F-L1 step 1).

Every function here is deliberately free-standing — no ``self`` access, no
AIAgent imports — so these are easy to test in isolation and to call from
any module that builds a message list. The original definitions remain on
``AIAgent`` as thin ``@staticmethod`` wrappers for back-compat with code
that does ``AIAgent._method(...)`` (tests/test_agent_guardrails.py does this
directly for _sanitize_api_messages and _cap_delegate_task_calls).

Subsequent F-L1 steps will move more helpers here; resist the urge to pull
in ``self``-dependent methods — those need a different strategy (mixin or
an explicit agent-state argument) and belong to later PRs.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List

logger = logging.getLogger(__name__)


def get_tool_call_id(tc: Any) -> str:
    """Extract the id from a tool-call object regardless of shape.

    Accepts both dict form (OpenAI chat.completions) and object form (SDK
    ChoiceMessageToolCall). Returns "" when no id is present.
    """
    if isinstance(tc, dict):
        return tc.get("id", "") or ""
    return getattr(tc, "id", "") or ""


def sanitize_api_messages(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Fix orphaned tool_call / tool_result pairs before an LLM call.

    Runs unconditionally — not gated on whether the context compressor is
    present — so orphans from session loading or manual message
    manipulation are always caught.

    Guarantees, in order:
      1. Every ``tool`` message has a matching assistant ``tool_calls`` entry.
         Unmatched tool messages are DROPPED.
      2. Every assistant ``tool_calls`` entry has a matching ``tool``
         response. Missing responses are PATCHED with a placeholder.
    """
    surviving_call_ids: set = set()
    for msg in messages:
        if msg.get("role") == "assistant":
            for tc in msg.get("tool_calls") or []:
                cid = get_tool_call_id(tc)
                if cid:
                    surviving_call_ids.add(cid)

    result_call_ids: set = set()
    for msg in messages:
        if msg.get("role") == "tool":
            cid = msg.get("tool_call_id")
            if cid:
                result_call_ids.add(cid)

    # 1. Drop tool results with no matching assistant call
    orphaned_results = result_call_ids - surviving_call_ids
    if orphaned_results:
        messages = [
            m for m in messages
            if not (m.get("role") == "tool" and m.get("tool_call_id") in orphaned_results)
        ]
        logger.debug(
            "Pre-call sanitizer: removed %d orphaned tool result(s)",
            len(orphaned_results),
        )

    # 2. Inject stub results for calls whose result was dropped
    missing_results = surviving_call_ids - result_call_ids
    if missing_results:
        patched: List[Dict[str, Any]] = []
        for msg in messages:
            patched.append(msg)
            if msg.get("role") == "assistant":
                for tc in msg.get("tool_calls") or []:
                    cid = get_tool_call_id(tc)
                    if cid in missing_results:
                        patched.append({
                            "role": "tool",
                            "content": "[Result unavailable — see context summary above]",
                            "tool_call_id": cid,
                        })
        messages = patched
        logger.debug(
            "Pre-call sanitizer: added %d stub tool result(s)",
            len(missing_results),
        )
    return messages


def cap_delegate_task_calls(tool_calls: list) -> list:
    """Truncate excess delegate_task calls to MAX_CONCURRENT_CHILDREN.

    The delegate_tool caps the task list inside a single call, but the
    model can emit multiple separate delegate_task tool_calls in one
    turn. This truncates the excess, preserving all non-delegate calls.

    Returns the original list if no truncation was needed.
    """
    from tools.delegate_tool import MAX_CONCURRENT_CHILDREN
    delegate_count = sum(1 for tc in tool_calls if tc.function.name == "delegate_task")
    if delegate_count <= MAX_CONCURRENT_CHILDREN:
        return tool_calls
    kept_delegates = 0
    truncated = []
    for tc in tool_calls:
        if tc.function.name == "delegate_task":
            if kept_delegates < MAX_CONCURRENT_CHILDREN:
                truncated.append(tc)
                kept_delegates += 1
        else:
            truncated.append(tc)
    logger.warning(
        "Truncated %d excess delegate_task call(s) to enforce "
        "MAX_CONCURRENT_CHILDREN=%d limit",
        delegate_count - MAX_CONCURRENT_CHILDREN, MAX_CONCURRENT_CHILDREN,
    )
    return truncated


def deduplicate_tool_calls(tool_calls: list) -> list:
    """Remove duplicate (tool_name, arguments) pairs within a single turn.

    Only the first occurrence of each unique pair is kept.
    Returns the original list if no duplicates were found.
    """
    seen: set = set()
    unique: list = []
    for tc in tool_calls:
        key = (tc.function.name, tc.function.arguments)
        if key not in seen:
            seen.add(key)
            unique.append(tc)
        else:
            logger.warning("Removed duplicate tool call: %s", tc.function.name)
    return unique if len(unique) < len(tool_calls) else tool_calls
