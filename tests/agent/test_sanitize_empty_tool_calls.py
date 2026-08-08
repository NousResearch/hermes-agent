"""Regression tests for sanitize_api_messages() empty-content / empty-tool_calls
healing (PR #74906 + follow-up).

When dedup of duplicate tool_call_ids empties an assistant message's
``tool_calls`` list, the previous pre-call sanitizer pass
(``repair_empty_non_final_messages``, runs first in
``sanitize_api_messages``) saw the message as "has payload" and skipped
it. Dedup then stripped ``tool_calls``, leaving
``{role: assistant, content: null}`` on the wire — which strict
non-empty-content providers (Anthropic native, litellm/Bedrock) reject
with HTTP 400.

The dedup fix from PR #74906 drops ``tool_calls`` instead of writing an
empty array. The follow-up re-runs ``repair_empty_non_final_messages``
after dedup so the placeholder ("[response interrupted]" on upstream
main) is also injected into the post-dedup shape.
"""

from agent.agent_runtime_helpers import sanitize_api_messages


def _user(content="hi"):
    return {"role": "user", "content": content}


def _assistant_tool_call(call_id, *, content=""):
    return {
        "role": "assistant",
        "content": content,
        "tool_calls": [{
            "id": call_id, "type": "function",
            "function": {"name": "read_file", "arguments": "{}"},
        }],
    }


def _tool_result(call_id, content="ok"):
    return {"role": "tool", "tool_call_id": call_id, "content": content}


def _has_visible_content(msg):
    content = msg.get("content")
    return isinstance(content, str) and content.strip()


class TestSanitizeEmptyToolCalls:
    def test_cross_message_all_duplicates_healed(self):
        """Reviewer's exact scenario from the PR #74906 review.

        Transcript shape:
            [user,
             assistant(content:'', tool_calls:[call_X]),  # live
             tool(call_X),
             assistant(content:None, tool_calls:[call_X]),  # non-final, all dups
             user]

        After sanitize:
            - idx 1 keeps its tool_calls (the live call_X).
            - idx 3 has no tool_calls (dedup stripped them) AND must have
              visible content (post-dedup repair re-runs and substitutes the
              placeholder).
        """
        messages = [
            _user("first question"),
            _assistant_tool_call("call_X"),
            _tool_result("call_X"),
            _assistant_tool_call("call_X", content=None),  # dup + content-null
            _user("follow up"),
        ]
        out = sanitize_api_messages(messages)
        healed = out[3]
        assert "tool_calls" not in healed, (
            "dedup should strip empty tool_calls; instead got: "
            f"{healed.get('tool_calls')!r}"
        )
        assert _has_visible_content(healed), (
            "post-dedup repair should inject visible content; instead "
            f"got: {healed.get('content')!r}"
        )
        # Earlier call's assistant turn keeps its tool_calls intact.
        assert out[1]["tool_calls"][0]["id"] == "call_X"
        # Both user turns preserved verbatim.
        assert out[0]["role"] == "user" and out[0]["content"] == "first question"
        assert out[4]["role"] == "user" and out[4]["content"] == "follow up"

    def test_in_message_all_duplicates_healed(self):
        """Single non-final assistant turn whose two tool_calls share one id.

        After dedup, the message loses both tool_calls (the id was
        already-seen from a prior turn). Post-dedup repair then injects
        the placeholder content.
        """
        messages = [
            _user("q"),
            _assistant_tool_call("call_A"),
            _tool_result("call_A"),
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {"id": "call_A", "type": "function",
                     "function": {"name": "read_file", "arguments": "{}"}},
                    {"id": "call_A", "type": "function",
                     "function": {"name": "read_file", "arguments": "{}"}},
                ],
            },
            _user("next"),
        ]
        out = sanitize_api_messages(messages)
        healed = out[3]
        assert "tool_calls" not in healed, (
            "all tool_calls in the message were duplicates of an earlier "
            f"turn; expected stripped: {healed.get('tool_calls')!r}"
        )
        assert _has_visible_content(healed), (
            "post-dedup repair should inject visible content; instead "
            f"got: {healed.get('content')!r}"
        )

    def test_pre_existing_empty_tool_calls_with_null_content(self):
        """Dict already has ``tool_calls: []`` and ``content: null``.

        Pass 0 (drop empty tool_calls) and repair_empty_non_final_messages
        together produce a non-empty placeholder turn.
        """
        messages = [
            _user("q"),
            {"role": "assistant", "content": None, "tool_calls": []},
            _user("next"),
        ]
        out = sanitize_api_messages(messages)
        healed = out[1]
        assert "tool_calls" not in healed
        assert _has_visible_content(healed)

    def test_assistant_with_visible_content_unchanged(self):
        """Guard rail: messages with legitimate visible content are
        not touched by either pass.
        """
        messages = [
            _user("q"),
            {"role": "assistant", "content": "好的，我来处理。"},
            _user("next"),
        ]
        out = sanitize_api_messages(messages)
        assert out[1]["content"] == "好的，我来处理。"
        assert "tool_calls" not in out[1]

    def test_user_messages_with_content_unchanged(self):
        """Guard rail: user messages with content are never substituted.
        repair_empty_non_final_messages runs over (assistant, user) but
        only inserts a placeholder when the message has no payload; a
        user turn with text must stay untouched.
        """
        messages = [
            _user("middle"),
            {"role": "assistant", "content": "hi"},
            _user("end"),
        ]
        out = sanitize_api_messages(messages)
        assert out[0]["content"] == "middle"
        assert out[2]["content"] == "end"

    def test_final_message_exempt_from_repair(self):
        """The final assistant message is exempt from
        repair_empty_non_final_messages by design (an empty final
        assistant turn is legal). Our fix must preserve that.
        """
        messages = [
            _user("q"),
            {"role": "assistant", "content": None, "tool_calls": []},
        ]
        out = sanitize_api_messages(messages)
        assert out[-1]["role"] == "assistant"
        # No placeholder injected — final empty assistant is allowed.
        assert out[-1].get("content") is None
        assert "tool_calls" not in out[-1]