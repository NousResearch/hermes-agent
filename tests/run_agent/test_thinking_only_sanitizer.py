"""Tests for the thinking-only assistant message sanitizer.

Covers _is_thinking_only_assistant() + _drop_thinking_only_and_merge_users()
in run_agent.py. The sanitizer runs on the per-call api_messages copy and
drops assistant turns that contain only reasoning (no visible content, no
tool_calls). Adjacent user messages left behind are merged so role
alternation is preserved for the provider.

Claude Code uses this exact pattern (filterOrphanedThinkingOnlyMessages +
mergeAdjacentUserMessages in src/utils/messages.ts). See #16823 for the
backstory on why the alternative — fabricating "." stub text — was rejected.
"""

from run_agent import AIAgent


# ---------------------------------------------------------------------------
# _is_thinking_only_assistant — detection
# ---------------------------------------------------------------------------


class TestIsThinkingOnlyAssistant:

    def test_plain_assistant_reply_is_not_thinking_only(self):
        msg = {"role": "assistant", "content": "Hello there"}
        assert not AIAgent._is_thinking_only_assistant(msg)

















    def test_non_dict_returns_false(self):
        assert not AIAgent._is_thinking_only_assistant(None)
        assert not AIAgent._is_thinking_only_assistant("hello")

    def test_prefill_stub_detected_after_reasoning_stripped(self):
        # The per-call copy for a provider that doesn't echo reasoning back has
        # had reasoning_content/reasoning removed, leaving _thinking_prefill as
        # the only evidence the turn was ever a thinking-only stub.
        on_wire = {"role": "assistant", "content": "", "_thinking_prefill": True}
        assert AIAgent._is_thinking_only_assistant(on_wire)

    def test_healed_prefill_stub_is_still_detected(self):
        # repair_empty_non_final_messages runs before the drop pass and rewrites
        # a non-final stub's empty content to a placeholder. The marker has to
        # outrank that, or the healed stub survives and the request still ends
        # on a model turn.
        healed = {
            "role": "assistant",
            "content": "[response interrupted]",
            "_thinking_prefill": True,
        }
        assert AIAgent._is_thinking_only_assistant(healed)

    def test_prefill_marker_does_not_override_tool_calls(self):
        msg = {
            "role": "assistant",
            "content": "",
            "tool_calls": [{"id": "c1", "function": {"name": "t", "arguments": "{}"}}],
            "_thinking_prefill": True,
        }
        assert not AIAgent._is_thinking_only_assistant(msg)


# ---------------------------------------------------------------------------
# _drop_thinking_only_and_merge_users — the full pass
# ---------------------------------------------------------------------------


class TestDropThinkingOnlyAndMergeUsers:

    def test_empty_list_passthrough(self):
        assert AIAgent._drop_thinking_only_and_merge_users([]) == []

    def test_no_thinking_only_messages_is_noop_identity(self):
        msgs = [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hello"},
        ]
        out = AIAgent._drop_thinking_only_and_merge_users(msgs)
        # Should return the original list untouched (identity) when no changes.
        assert out is msgs


    def test_preserves_alternation_after_drop(self):
        msgs = [
            {"role": "user", "content": "u1"},
            {"role": "assistant", "content": "", "reasoning": "..."},
            {"role": "user", "content": "u2"},
            {"role": "assistant", "content": "real reply"},
        ]
        out = AIAgent._drop_thinking_only_and_merge_users(msgs)
        roles = [m["role"] for m in out]
        assert roles == ["user", "assistant"]
        assert out[0]["content"] == "u1\n\nu2"
        assert out[1]["content"] == "real reply"



    def test_does_not_touch_stored_messages_original_list_unmutated(self):
        original_first_user = {"role": "user", "content": "u1"}
        original_assistant = {"role": "assistant", "content": "", "reasoning": "..."}
        original_second_user = {"role": "user", "content": "u2"}
        msgs = [original_first_user, original_assistant, original_second_user]
        AIAgent._drop_thinking_only_and_merge_users(msgs)
        # Caller passes in a per-call copy already, but the sanitizer itself
        # must not rewrite the dicts it was handed on the drop path.
        # (It CAN mutate merged dicts — those come from the caller's copy.)
        assert original_first_user["content"] == "u1"
        assert original_second_user["content"] == "u2"

    def test_tool_result_between_user_and_thinking_preserved(self):
        # Tool results shouldn't block a drop — but they do block the merge
        # (user/tool are different roles). This scenario shouldn't happen in
        # practice because a thinking-only turn won't have tool_calls, but if
        # it did somehow, the surrounding tool result stays put.
        msgs = [
            {"role": "user", "content": "u1"},
            {"role": "assistant", "tool_calls": [{"id": "c1", "function": {"name": "t", "arguments": "{}"}}]},
            {"role": "tool", "tool_call_id": "c1", "content": "ok"},
            {"role": "assistant", "content": "", "reasoning": "..."},
            {"role": "user", "content": "u2"},
        ]
        out = AIAgent._drop_thinking_only_and_merge_users(msgs)
        assert [m["role"] for m in out] == ["user", "assistant", "tool", "user"]

    def test_merge_concatenates_list_content_user_messages(self):
        msgs = [
            {"role": "user", "content": [{"type": "text", "text": "first"}]},
            {"role": "assistant", "content": "", "reasoning": "..."},
            {"role": "user", "content": [{"type": "text", "text": "second"}]},
        ]
        out = AIAgent._drop_thinking_only_and_merge_users(msgs)
        assert len(out) == 1
        assert out[0]["content"] == [
            {"type": "text", "text": "first"},
            {"type": "text", "text": "second"},
        ]


    def test_trailing_prefill_stubs_dropped_so_request_ends_on_user(self):
        # Regression for the Gemini 400 "Requests ending with a model turn are
        # not supported". Two thinking-only responses in a row queue two prefill
        # stubs, and on a provider that doesn't echo reasoning back both arrive
        # here with their reasoning fields already stripped. Before the fix the
        # pass couldn't see them and the request went out ending on assistant.
        msgs = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "do the thing"},
            {"role": "assistant", "content": "", "_thinking_prefill": True},
            {"role": "assistant", "content": "", "_thinking_prefill": True},
        ]
        out = AIAgent._drop_thinking_only_and_merge_users(msgs)
        assert [m["role"] for m in out] == ["system", "user"]
        assert out[-1]["role"] != "assistant"

    def test_system_messages_ignored_by_pass(self):
        msgs = [
            {"role": "system", "content": "sys prompt"},
            {"role": "user", "content": "u1"},
            {"role": "assistant", "content": "", "reasoning": "..."},
            {"role": "user", "content": "u2"},
        ]
        out = AIAgent._drop_thinking_only_and_merge_users(msgs)
        assert len(out) == 2
        assert out[0]["role"] == "system"
        assert out[1]["role"] == "user"
        assert out[1]["content"] == "u1\n\nu2"
