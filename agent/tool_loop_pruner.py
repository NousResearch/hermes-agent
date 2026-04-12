"""Prune repeated tool call/response pairs from conversation history.

When a tool loop is detected, this module collapses the repeated pairs into
a single summary message. This directly reduces token-level anchoring by
removing the repeated occurrences of the wrong tool name from context.

The pruner keeps:
- All messages before the loop
- One summary message describing what happened
- The last call/response pair (model needs to see the most recent failure)
"""
from __future__ import annotations

import json
from typing import Optional


def prune_tool_loop(
    messages: list[dict],
    tool_name: str,
    streak: int,
    intended_tool: Optional[str] = None,
    detector: Optional[str] = None,
) -> list[dict]:
    """Replace repeated tool call/response pairs with a summary.

    Scans messages from the end, identifies the consecutive block of
    assistant(tool_call)+tool(result) pairs for tool_name, and collapses
    all but the last pair into a summary.

    Returns a new list -- does not mutate the input.
    """
    loop_pairs = _find_loop_pairs(messages, tool_name, streak)
    if len(loop_pairs) < 2:
        return list(messages)

    collapse = loop_pairs[:-1]
    collapse_indices = set()
    for assistant_idx, tool_idx in collapse:
        collapse_indices.add(assistant_idx)
        collapse_indices.add(tool_idx)

    summary_text = _build_summary(tool_name, len(collapse), intended_tool, detector)
    first_collapsed = min(collapse_indices)

    result = []
    summary_inserted = False
    for i, msg in enumerate(messages):
        if i in collapse_indices:
            if not summary_inserted and i >= first_collapsed:
                last_pair_assistant = messages[loop_pairs[-1][0]]
                call_id = _get_call_id(last_pair_assistant)
                result.append({
                    "role": "assistant",
                    "content": "",
                    "finish_reason": "tool_calls",
                    "tool_calls": [{
                        "id": call_id,
                        "call_id": call_id,
                        "type": "function",
                        "function": {
                            "name": tool_name,
                            "arguments": "{}",
                        },
                    }],
                })
                result.append({
                    "role": "tool",
                    "tool_call_id": call_id,
                    "content": summary_text,
                })
                summary_inserted = True
            continue
        result.append(msg)

    return result


def _find_loop_pairs(
    messages: list[dict], tool_name: str, max_pairs: int
) -> list[tuple[int, int]]:
    """Walk backward finding (assistant_idx, tool_idx) pairs for tool_name."""
    pairs = []
    i = len(messages) - 1
    while i >= 1 and len(pairs) < max_pairs:
        msg = messages[i]
        prev = messages[i - 1]
        if (
            msg.get("role") == "tool"
            and prev.get("role") == "assistant"
            and prev.get("tool_calls")
        ):
            tc_names = [
                tc["function"]["name"]
                for tc in prev["tool_calls"]
                if isinstance(tc, dict) and "function" in tc
            ]
            if tool_name in tc_names:
                pairs.append((i - 1, i))
                i -= 2
                continue
        i -= 1

    pairs.reverse()
    return pairs


def _get_call_id(assistant_msg: dict) -> str:
    tcs = assistant_msg.get("tool_calls", [])
    if tcs and isinstance(tcs[0], dict):
        return tcs[0].get("id", "loop_summary")
    return "loop_summary"


def _build_summary(
    tool_name: str,
    collapsed_count: int,
    intended_tool: Optional[str],
    detector: Optional[str],
) -> str:
    detector_desc = {
        "generic_repeat": "identical calls with identical arguments",
        "poll_no_progress": "repeated calls returning identical results",
        "ping_pong": "alternating between the same two actions without progress",
    }.get(detector, "repeated identical calls")

    summary = (
        f"[LOOP DETECTED] You called `{tool_name}` {collapsed_count + 1} times "
        f"({detector_desc}). The {collapsed_count} earlier attempts have been "
        f"removed from context to help you break free of this pattern. "
        f"The most recent attempt and its result are preserved below."
    )

    if intended_tool:
        summary += (
            f"\n\nYour reasoning mentioned `{intended_tool}` as the tool you "
            f"actually wanted to call, but you kept emitting `{tool_name}` instead. "
            f"This is a known token-anchoring issue -- the repeated appearances of "
            f"`{tool_name}` in context biased your generation. Those appearances "
            f"have now been pruned. Try calling `{intended_tool}` now."
        )
    else:
        summary += (
            f"\n\nTry a different approach. If you intended to call a different tool, "
            f"state its name explicitly in your response before making the call."
        )

    return summary
