"""Tests for agent/prompt_caching.py — Anthropic cache control injection."""


from agent.prompt_caching import (
    _apply_cache_marker,
    _build_marker,
    apply_anthropic_cache_control,
)


MARKER = {"type": "ephemeral"}


class TestApplyCacheMarker:
    def test_tool_message_gets_top_level_marker_on_native_anthropic(self):
        """Native Anthropic path: cache_control injected top-level (adapter moves it inside tool_result)."""
        msg = {"role": "tool", "content": "result"}
        _apply_cache_marker(msg, MARKER, native_anthropic=True)
        assert msg["cache_control"] == MARKER

    def test_tool_message_skips_marker_on_openrouter(self):
        """OpenRouter path: top-level cache_control on role:tool is invalid and causes silent hang."""
        msg = {"role": "tool", "content": "result"}
        _apply_cache_marker(msg, MARKER, native_anthropic=False)
        assert "cache_control" not in msg

    def test_none_content_gets_top_level_marker(self):
        msg = {"role": "assistant", "content": None}
        _apply_cache_marker(msg, MARKER)
        assert msg["cache_control"] == MARKER

    def test_empty_string_content_gets_top_level_marker(self):
        """Empty text blocks cannot have cache_control (Anthropic rejects them)."""
        msg = {"role": "assistant", "content": ""}
        _apply_cache_marker(msg, MARKER)
        assert msg["cache_control"] == MARKER
        # Must NOT wrap into [{"type": "text", "text": "", "cache_control": ...}]
        assert msg["content"] == ""

    def test_string_content_wrapped_in_list(self):
        msg = {"role": "user", "content": "Hello"}
        _apply_cache_marker(msg, MARKER)
        assert isinstance(msg["content"], list)
        assert len(msg["content"]) == 1
        assert msg["content"][0]["type"] == "text"
        assert msg["content"][0]["text"] == "Hello"
        assert msg["content"][0]["cache_control"] == MARKER

    def test_list_content_last_item_gets_marker(self):
        msg = {
            "role": "user",
            "content": [
                {"type": "text", "text": "First"},
                {"type": "text", "text": "Second"},
            ],
        }
        _apply_cache_marker(msg, MARKER)
        assert "cache_control" not in msg["content"][0]
        assert msg["content"][1]["cache_control"] == MARKER

    def test_empty_list_content_no_crash(self):
        msg = {"role": "user", "content": []}
        # Should not crash on empty list
        _apply_cache_marker(msg, MARKER)


class TestApplyAnthropicCacheControl:
    def test_empty_messages(self):
        result = apply_anthropic_cache_control([])
        assert result == []

    def test_returns_deep_copy(self):
        msgs = [{"role": "user", "content": "Hello"}]
        result = apply_anthropic_cache_control(msgs)
        assert result is not msgs
        assert result[0] is not msgs[0]
        # Original should be unmodified
        assert "cache_control" not in msgs[0].get("content", "")

    def test_caller_list_not_mutated_and_unmarked_msgs_shared(self):
        """Guard the shallow-copy change (was full deepcopy).

        The optimization returns ``list(api_messages)`` and deep-copies ONLY
        the <=4 messages that receive a cache_control marker. This test pins
        two invariants that a "deep-copies too little / too much" regression
        would break (prompt caching is sacred — the caller's history must
        never be mutated):

        1. The caller's original list and every message dict in it is left
           byte-identical after the call (no in-place marker leaks upstream).
        2. Un-marked messages in the middle are returned as the SAME object
           (shared reference) — proving we did not needlessly deep-copy the
           whole history — while marked messages are fresh copies.
        """
        import copy

        msgs = [
            {"role": "system", "content": "System"},
            {"role": "user", "content": "middle-unmarked-1"},
            {"role": "assistant", "content": "middle-unmarked-2"},
            {"role": "user", "content": "m3"},
            {"role": "assistant", "content": "m4"},
            {"role": "user", "content": "m5"},
        ]
        before = copy.deepcopy(msgs)
        result = apply_anthropic_cache_control(msgs, cache_ttl="5m")

        # (1) caller list + every element unchanged after the call.
        assert msgs == before, "apply_anthropic_cache_control mutated the caller's list"

        # System (0) + last 3 non-system (3,4,5) get markers => index 1 and 2
        # are un-marked and must be the SAME objects (shallow, not deep-copied).
        assert result[1] is msgs[1]
        assert result[2] is msgs[2]
        # Marked messages must be fresh copies (never the caller's objects).
        assert result[0] is not msgs[0]
        assert result[-1] is not msgs[-1]

        # Mutating a returned marked message must not bleed into the caller.
        result[0]["content"] = "TAMPERED"
        assert msgs[0]["content"] == "System"

    def test_output_equivalent_to_full_deepcopy_impl(self):
        """Byte-equivalence: shallow-copy output structurally matches what a
        naive full-deepcopy implementation would produce (same breakpoints,
        same TTL, same positions) for both native_anthropic modes."""
        import copy

        def _reference_full_deepcopy(api_messages, cache_ttl, native_anthropic):
            # Mirror of the pre-optimization implementation: deepcopy the whole
            # list, then apply markers to system + last (4 - used) non-system.
            messages = copy.deepcopy(api_messages)
            if not messages:
                return messages
            marker = _build_marker(cache_ttl)
            used = 0
            if messages[0].get("role") == "system":
                _apply_cache_marker(messages[0], marker, native_anthropic=native_anthropic)
                used += 1
            remaining = 4 - used
            non_sys = [i for i in range(len(messages)) if messages[i].get("role") != "system"]
            for idx in non_sys[-remaining:]:
                _apply_cache_marker(messages[idx], marker, native_anthropic=native_anthropic)
            return messages

        base = [
            {"role": "system", "content": "System"},
            {"role": "user", "content": "u1"},
            {"role": "assistant", "content": "a1"},
            {"role": "user", "content": "u2"},
            {"role": "assistant", "content": "a2"},
            {"role": "user", "content": "u3"},
        ]
        for native in (True, False):
            for ttl in ("5m", "1h"):
                got = apply_anthropic_cache_control(
                    copy.deepcopy(base), cache_ttl=ttl, native_anthropic=native
                )
                want = _reference_full_deepcopy(
                    copy.deepcopy(base), cache_ttl=ttl, native_anthropic=native
                )
                assert got == want, f"structural mismatch native={native} ttl={ttl}"

    def test_system_message_gets_marker(self):
        msgs = [
            {"role": "system", "content": "You are helpful"},
            {"role": "user", "content": "Hi"},
        ]
        result = apply_anthropic_cache_control(msgs)
        # System message should have cache_control
        sys_content = result[0]["content"]
        assert isinstance(sys_content, list)
        assert sys_content[0]["cache_control"]["type"] == "ephemeral"

    def test_last_3_non_system_get_markers(self):
        msgs = [
            {"role": "system", "content": "System"},
            {"role": "user", "content": "msg1"},
            {"role": "assistant", "content": "msg2"},
            {"role": "user", "content": "msg3"},
            {"role": "assistant", "content": "msg4"},
        ]
        result = apply_anthropic_cache_control(msgs)
        # System (index 0) + last 3 non-system (indices 2, 3, 4) = 4 breakpoints
        # Index 1 (msg1) should NOT have marker
        content_1 = result[1]["content"]
        if isinstance(content_1, str):
            assert True  # No marker applied (still a string)
        else:
            assert "cache_control" not in content_1[0]

    def test_no_system_message(self):
        msgs = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi"},
        ]
        result = apply_anthropic_cache_control(msgs)
        # Both should get markers (4 slots available, only 2 messages)
        assert len(result) == 2

    def test_1h_ttl(self):
        msgs = [{"role": "system", "content": "System prompt"}]
        result = apply_anthropic_cache_control(msgs, cache_ttl="1h")
        sys_content = result[0]["content"]
        assert isinstance(sys_content, list)
        assert sys_content[0]["cache_control"]["ttl"] == "1h"

    def test_max_4_breakpoints(self):
        msgs = [
            {"role": "system", "content": "System"},
        ] + [
            {"role": "user" if i % 2 == 0 else "assistant", "content": f"msg{i}"}
            for i in range(10)
        ]
        result = apply_anthropic_cache_control(msgs)
        # Count how many messages have cache_control
        count = 0
        for msg in result:
            content = msg.get("content")
            if isinstance(content, list):
                for item in content:
                    if isinstance(item, dict) and "cache_control" in item:
                        count += 1
            elif "cache_control" in msg:
                count += 1
        assert count <= 4
