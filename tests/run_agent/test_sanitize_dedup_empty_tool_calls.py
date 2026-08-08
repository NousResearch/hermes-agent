"""Regression test: the tool_call_id dedup pass must not re-create empty
tool_calls arrays.

When the pre-API sanitizer's dedup pass collapses an assistant message whose
calls are ALL duplicates of ids already seen earlier in the transcript, it
previously wrote ``tool_calls: []`` back onto the message. Strict
OpenAI-compatible providers (DeepSeek) reject an empty array outright with
HTTP 400 "Invalid 'messages[N].tool_calls': empty array. Expected an array
with minimum length 1, but got an empty array instead." — the same provider
class that motivated #58755. Duplicated tool-call blocks reach the sanitizer
from crash/resume glitches or a compression window that re-emits a tool
result. The dedup pass must drop the key entirely (semantics identical to
"no tool calls"), never emit an empty array.
"""

from agent.agent_runtime_helpers import sanitize_api_messages


def _assistant_with_calls(call_ids, content="x"):
    return {
        "role": "assistant",
        "content": content,
        "tool_calls": [
            {
                "id": cid,
                "type": "function",
                "function": {"name": "test_tool", "arguments": "{}"},
            }
            for cid in call_ids
        ],
    }


def _tool_result(call_id):
    return {"role": "tool", "tool_call_id": call_id, "content": "ok"}


def test_dedup_all_duplicates_drops_key_not_empty_array():
    """A replayed duplicate assistant tool-call block must not yield
    ``tool_calls: []`` — the exact empty array DeepSeek 400s on."""
    messages = [
        {"role": "user", "content": "hi"},
        _assistant_with_calls(["call_a", "call_b"]),
        _tool_result("call_a"),
        _tool_result("call_b"),
        _assistant_with_calls(["call_a", "call_b"]),  # duplicate re-emission
    ]

    out = sanitize_api_messages(messages)

    # No message may carry an empty tool_calls array.
    assert all(m.get("tool_calls") != [] for m in out)
    # The duplicate assistant block survives as a plain message (no key).
    dupe_block = [
        m for m in out if m["role"] == "assistant" and "tool_calls" not in m
    ]
    assert len(dupe_block) == 1
    assert dupe_block[0]["content"] == "x"


def test_dedup_unique_ids_preserved():
    """A healthy transcript is untouched."""
    messages = [
        {"role": "user", "content": "hi"},
        _assistant_with_calls(["call_1", "call_2"]),
        _tool_result("call_1"),
        _tool_result("call_2"),
    ]

    out = sanitize_api_messages(messages)

    assistant = next(m for m in out if m["role"] == "assistant")
    assert [tc["id"] for tc in assistant["tool_calls"]] == ["call_1", "call_2"]


def test_partial_duplicates_keep_first_occurrence():
    """When only SOME calls are duplicates, the surviving calls stay — the
    result is a non-empty array, never an empty one."""
    messages = [
        {"role": "user", "content": "hi"},
        _assistant_with_calls(["call_a", "call_b"]),
        _tool_result("call_a"),
        _tool_result("call_b"),
        _assistant_with_calls(["call_a", "call_c"]),
        _tool_result("call_c"),
    ]

    out = sanitize_api_messages(messages)

    assistant_blocks = [
        m for m in out if m["role"] == "assistant" and m.get("tool_calls")
    ]
    assert len(assistant_blocks) == 2
    assert [tc["id"] for tc in assistant_blocks[1]["tool_calls"]] == ["call_c"]
