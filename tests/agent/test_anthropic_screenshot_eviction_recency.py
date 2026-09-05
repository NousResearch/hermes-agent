"""Screenshot eviction must keep the three NEWEST screenshots.

``_evict_old_screenshots`` (agent/anthropic_message_convert.py) walks messages
newest -> oldest, replacing every image past the third with
``[screenshot removed to save context]``.  Its block-level loop used to walk the
blocks *inside* one message oldest -> newest while sharing the outer counter, so
recency was inconsistent as soon as one message held more than one screenshot.

That is the normal shape for parallel tool calls: sibling ``tool_result`` blocks
are appended into the same user message by
``_convert_tool_message_to_result``.  A batch of four screenshots therefore had
blocks 1-3 kept and block 4 — the most recent view of the screen — evicted.

These tests assert the docstring's contract directly: whichever way the
screenshots arrived, the surviving images are the last three in wire order.
"""

from typing import Any, Dict, List

from agent.anthropic_message_convert import convert_messages_to_anthropic

PLACEHOLDER = "[screenshot removed to save context]"


def _screenshot_call(call_id: str) -> Dict[str, Any]:
    return {
        "id": call_id,
        "type": "function",
        "function": {"name": "computer", "arguments": '{"action": "screenshot"}'},
    }


def _assistant_turn(call_ids: List[str]) -> Dict[str, Any]:
    return {"role": "assistant", "content": "", "tool_calls": [_screenshot_call(c) for c in call_ids]}


def _screenshot_result(call_id: str) -> Dict[str, Any]:
    """A tool message carrying one screenshot, tagged by ``call_id`` in its image URL."""
    return {
        "role": "tool",
        "tool_call_id": call_id,
        "content": [
            {"type": "text", "text": f"screenshot {call_id}"},
            {"type": "image_url", "image_url": {"url": f"https://shots.test/{call_id}.png"}},
        ],
    }


def _screenshot_tags(converted: List[Dict[str, Any]]) -> List[str]:
    """Call ids of the surviving images, in wire order."""
    tags = []
    for msg in converted:
        content = msg.get("content")
        for block in content if isinstance(content, list) else []:
            inner = block.get("content") if block.get("type") == "tool_result" else None
            for b in inner if isinstance(inner, list) else []:
                if b.get("type") == "image":
                    tags.append(b["source"]["url"].rsplit("/", 1)[-1].removesuffix(".png"))
    return tags


def _evicted_tags(converted: List[Dict[str, Any]]) -> List[str]:
    """Call ids of the tool_results whose image became a placeholder, in wire order."""
    tags = []
    for msg in converted:
        content = msg.get("content")
        for block in content if isinstance(content, list) else []:
            inner = block.get("content") if block.get("type") == "tool_result" else None
            blocks = inner if isinstance(inner, list) else []
            if any(b.get("type") == "text" and b.get("text") == PLACEHOLDER for b in blocks):
                tags.append(block["tool_use_id"])
    return tags


def test_parallel_screenshot_batch_keeps_the_newest_three():
    """Five screenshots in ONE merged user message: the last three survive."""
    call_ids = [f"call_{i}" for i in range(1, 6)]
    messages = [{"role": "user", "content": "look at the screen"}, _assistant_turn(call_ids)]
    messages += [_screenshot_result(c) for c in call_ids]

    _system, converted = convert_messages_to_anthropic(messages)

    # The batch really did merge into a single user turn — the case the bug needs.
    tool_result_msgs = [
        m for m in converted
        if m.get("role") == "user" and isinstance(m.get("content"), list)
        and any(b.get("type") == "tool_result" for b in m["content"])
    ]
    assert len(tool_result_msgs) == 1
    assert len(tool_result_msgs[0]["content"]) == len(call_ids)

    assert _screenshot_tags(converted) == call_ids[-3:]
    assert _evicted_tags(converted) == call_ids[:-3]


def test_screenshots_split_across_messages_keep_the_newest_three():
    """Screenshots arriving in separate turns AND in a batch share one recency order."""
    first, second, batch = ["call_1"], ["call_2"], ["call_3", "call_4", "call_5"]
    messages: List[Dict[str, Any]] = [{"role": "user", "content": "watch the screen"}]
    for group in (first, second, batch):
        messages.append(_assistant_turn(group))
        messages += [_screenshot_result(c) for c in group]

    _system, converted = convert_messages_to_anthropic(messages)

    assert _screenshot_tags(converted) == ["call_3", "call_4", "call_5"]
    assert _evicted_tags(converted) == ["call_1", "call_2"]
