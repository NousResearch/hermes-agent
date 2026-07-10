"""Regression: tool_result blocks must come FIRST in a user message's content.

Anthropic rejects requests where a user message carries a tool_result block
that is not at the front of the content array, when the preceding assistant
message ends with tool_use — the API reports:

    messages.N: `tool_use` ids were found without `tool_result` blocks
    immediately after

This happens in practice when context compression / session restore merges a
system-reminder text block IN FRONT of the tool_result inside the same user
message (observed 2026-07-05, session 20260705_124405_1849e5: the user msg had
content [text(system-reminder), tool_result] and Anthropic returned HTTP 400,
triggering an unnecessary fallback to gpt-5.5).

The adapter must reorder every user message so all tool_result blocks precede
non-tool_result blocks, preserving relative order within each group.
"""

from agent.anthropic_adapter import convert_messages_to_anthropic


def _mk_tool_use(tid):
    return {
        "type": "tool_use",
        "id": tid,
        "name": "mcp__hermes___patch",
        "input": {},
    }


def _mk_tool_result(tid, text="ok"):
    return {"type": "tool_result", "tool_use_id": tid, "content": text}


class TestToolResultBlockOrder:
    def test_text_before_tool_result_is_reordered(self):
        """The exact 2026-07-05 failure shape: [text, tool_result] user msg."""
        messages = [
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": "[PRIOR CONTEXT] ..."},
                    _mk_tool_use("toolu_01Mg8mqq"),
                ],
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "<system-reminder>persona...</system-reminder>"},
                    _mk_tool_result("toolu_01Mg8mqq"),
                ],
            },
        ]
        _, result = convert_messages_to_anthropic(messages)

        user_msg = result[1]
        assert user_msg["role"] == "user"
        types = [b["type"] for b in user_msg["content"]]
        # tool_result must be first; the tool_use must NOT have been stripped
        assert types[0] == "tool_result", types
        asst_types = [b["type"] for b in result[0]["content"]]
        assert "tool_use" in asst_types, (
            "tool_use was stripped instead of reordering the tool_result"
        )
        # the text block must survive, after the tool_result
        assert "text" in types[1:]

    def test_multiple_tool_results_keep_relative_order(self):
        messages = [
            {
                "role": "assistant",
                "content": [_mk_tool_use("t1"), _mk_tool_use("t2")],
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "reminder"},
                    _mk_tool_result("t1", "r1"),
                    {"type": "text", "text": "middle note"},
                    _mk_tool_result("t2", "r2"),
                ],
            },
        ]
        _, result = convert_messages_to_anthropic(messages)
        user_msg = result[1]
        blocks = user_msg["content"]
        assert [b["type"] for b in blocks[:2]] == ["tool_result", "tool_result"]
        assert blocks[0]["tool_use_id"] == "t1"
        assert blocks[1]["tool_use_id"] == "t2"
        texts = [b["text"] for b in blocks if b["type"] == "text"]
        assert texts == ["reminder", "middle note"]

    def test_already_ordered_message_untouched(self):
        messages = [
            {"role": "assistant", "content": [_mk_tool_use("t1")]},
            {
                "role": "user",
                "content": [
                    _mk_tool_result("t1"),
                    {"type": "text", "text": "follow-up"},
                ],
            },
        ]
        _, result = convert_messages_to_anthropic(messages)
        blocks = result[1]["content"]
        assert [b["type"] for b in blocks] == ["tool_result", "text"]

    def test_plain_user_text_message_unaffected(self):
        messages = [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "hi"},
            {"role": "user", "content": [{"type": "text", "text": "plain"}]},
        ]
        _, result = convert_messages_to_anthropic(messages)
        assert result[0]["content"] == "hello"
        assert result[2]["content"][0]["text"] == "plain"

    def test_merged_consecutive_user_messages_reordered(self):
        """Merge path: user text msg + user tool_result msg merged into one —
        the merged content [text, tool_result] must also be reordered."""
        messages = [
            {"role": "assistant", "content": [_mk_tool_use("t1")]},
            # a stray plain user message injected before the tool result
            {"role": "user", "content": [{"type": "text", "text": "injected"}]},
            {"role": "user", "content": [_mk_tool_result("t1")]},
        ]
        _, result = convert_messages_to_anthropic(messages)
        # after merge there is one user message; tool_result must lead
        user_msgs = [m for m in result if m["role"] == "user"]
        assert len(user_msgs) == 1
        types = [b["type"] for b in user_msgs[0]["content"]]
        assert types[0] == "tool_result", types
        # and the assistant tool_use must survive
        assert any(
            b.get("type") == "tool_use" for b in result[0]["content"]
        )
