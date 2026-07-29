"""Tests for agent/prompt_caching.py — Anthropic cache control injection."""


from agent.prompt_caching import (
    _apply_cache_marker,
    _build_marker,
    _can_carry_marker,
    apply_anthropic_cache_control,
    strip_anthropic_cache_control,
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

    def test_tool_message_wraps_non_empty_content_on_openrouter(self):
        """Non-empty tool content should be wrapped so the marker lands on a content part."""
        msg = {"role": "tool", "content": "tool result bytes"}
        _apply_cache_marker(msg, MARKER, native_anthropic=False)
        assert "cache_control" not in msg
        assert isinstance(msg["content"], list)
        assert msg["content"][0]["cache_control"] == MARKER

    def test_empty_assistant_message_skips_marker_on_openrouter(self):
        """OpenRouter path: empty assistant turns are pure tool_calls, marker would be ignored."""
        msg = {"role": "assistant", "content": ""}
        _apply_cache_marker(msg, MARKER, native_anthropic=False)
        assert "cache_control" not in msg

    def test_native_anthropic_empty_assistant_gets_top_level_marker(self):
        """Native Anthropic layout can still carry top-level marker on empty content."""
        msg = {"role": "assistant", "content": ""}
        _apply_cache_marker(msg, MARKER, native_anthropic=True)
        assert msg["cache_control"] == MARKER

    def test_none_content_skips_marker_on_openrouter(self):
        """OpenRouter path: None-content assistant turns are ignored."""
        msg = {"role": "assistant", "content": None}
        _apply_cache_marker(msg, MARKER, native_anthropic=False)
        assert "cache_control" not in msg

    def test_none_content_gets_top_level_marker_on_native_anthropic(self):
        """Native Anthropic path: None content still gets top-level marker."""
        msg = {"role": "assistant", "content": None}
        _apply_cache_marker(msg, MARKER, native_anthropic=True)
        assert msg["cache_control"] == MARKER

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


class TestCanCarryMarker:
    def test_native_anthropic_always_true(self):
        assert _can_carry_marker({"role": "assistant", "content": ""}, native_anthropic=True) is True
        assert _can_carry_marker({"role": "tool", "content": ""}, native_anthropic=True) is True

    def test_openrouter_content_parts_carry_marker(self):
        assert _can_carry_marker({"role": "user", "content": "text"}, native_anthropic=False) is True
        assert _can_carry_marker({"role": "user", "content": [{"type": "text", "text": "a"}]}, native_anthropic=False) is True

    def test_openrouter_empty_or_none_does_not_carry_marker(self):
        assert _can_carry_marker({"role": "assistant", "content": ""}, native_anthropic=False) is False
        assert _can_carry_marker({"role": "assistant", "content": None}, native_anthropic=False) is False
        assert _can_carry_marker({"role": "tool", "content": "result"}, native_anthropic=False) is True
        assert _can_carry_marker({"role": "tool", "content": ""}, native_anthropic=False) is False

    def test_openrouter_list_carrier_requires_last_part_dict(self):
        """Carrier predicate must agree with _apply_cache_marker, which only marks
        the LAST content part. A list whose last element isn't a dict cannot carry
        a marker and must not consume a breakpoint."""
        # Last part is a dict -> carrier.
        assert _can_carry_marker(
            {"role": "user", "content": [{"type": "text", "text": "a"}]},
            native_anthropic=False,
        ) is True
        # Last part is a non-dict (stray raw string) -> NOT a carrier, even though
        # an earlier part is a dict. Previously this passed the gate but got no
        # marker, wasting a breakpoint.
        assert _can_carry_marker(
            {"role": "user", "content": [{"type": "text", "text": "a"}, "trailing raw"]},
            native_anthropic=False,
        ) is False
        # Empty list -> not a carrier.
        assert _can_carry_marker({"role": "user", "content": []}, native_anthropic=False) is False


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

    def test_static_system_prefix_gets_its_own_marker(self):
        messages = [
            {"role": "system", "content": "stable prefix\n\nper-session context"},
            {"role": "user", "content": "old request"},
            {"role": "assistant", "content": "old response"},
            {"role": "user", "content": "new request"},
        ]

        result = apply_anthropic_cache_control(
            messages,
            static_system_prefix="stable prefix",
        )

        system_blocks = result[0]["content"]
        assert system_blocks == [
            {
                "type": "text",
                "text": "stable prefix",
                "cache_control": {"type": "ephemeral"},
            },
            {
                "type": "text",
                "text": "\n\nper-session context",
                "cache_control": {"type": "ephemeral"},
            },
        ]
        assert result[1]["content"] == "old request"
        assert result[2]["content"][0]["cache_control"] == {"type": "ephemeral"}
        assert result[3]["content"][0]["cache_control"] == {"type": "ephemeral"}

    def test_mismatched_static_prefix_uses_legacy_system_breakpoint(self):
        messages = [
            {"role": "system", "content": "current system prompt"},
            {"role": "user", "content": "old request"},
            {"role": "assistant", "content": "old response"},
            {"role": "user", "content": "new request"},
        ]

        result = apply_anthropic_cache_control(
            messages,
            static_system_prefix="stale system prompt",
        )

        assert len(result[0]["content"]) == 1
        assert result[1]["content"][0]["cache_control"] == {"type": "ephemeral"}
        assert result[2]["content"][0]["cache_control"] == {"type": "ephemeral"}
        assert result[3]["content"][0]["cache_control"] == {"type": "ephemeral"}

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

    def test_tool_loop_empty_assistant_and_tool_messages_do_not_consume_breakpoints(self):
        """Tool loops should keep breakpoints on messages that can carry markers."""
        msgs = [
            {"role": "system", "content": "You are helpful"},
            {"role": "user", "content": "run tool 1", "cache_control": MARKER},
            {"role": "assistant", "content": "", "tool_calls": [{"type": "function"}]},
            {"role": "tool", "content": "tool result 1"},
            {"role": "user", "content": "run tool 2", "cache_control": MARKER},
            {"role": "assistant", "content": "", "tool_calls": [{"type": "function"}]},
            {"role": "tool", "content": "tool result 2"},
        ]
        result = apply_anthropic_cache_control(msgs, native_anthropic=False)
        # Empty assistant/tool turns should not get broken markers
        assert "cache_control" not in result[2]
        assert "cache_control" not in result[3]
        assert "cache_control" not in result[5]
        assert "cache_control" not in result[6]

    def test_tool_message_marker_lands_on_content_part_on_openrouter(self):
        """Non-empty tool content should be wrapped so the marker lands on a content part."""
        msgs = [
            {"role": "user", "content": "hello"},
            {"role": "tool", "content": "tool output"},
        ]
        result = apply_anthropic_cache_control(msgs, native_anthropic=False)
        assert isinstance(result[1]["content"], list)
        assert result[1]["content"][0]["cache_control"] == {"type": "ephemeral"}
        assert "cache_control" not in result[1]


class TestNormalizationOrdering:
    """The conversation loop normalizes message text for prefix stability and
    injects cache breakpoints. Marking must happen AFTER normalization.

    ``_apply_cache_marker`` rewrites a plain-string ``content`` into a
    ``[{"type": "text", ...}]`` block. The loop's whitespace pass is guarded
    on ``isinstance(content, str)``, so anything marked first is skipped by
    it — and a message is only marked while it sits in the last-3 window.
    The same message would then be sent raw on one turn and stripped on the
    next, breaking the prefix match the breakpoints exist to protect.
    """

    def test_marking_a_string_hides_it_from_string_normalization(self):
        """The mechanism: marking changes content out of ``str`` shape."""
        msgs = [{"role": "user", "content": "hello  \n"}]
        marked = apply_anthropic_cache_control(msgs, native_anthropic=False)
        assert not isinstance(marked[0]["content"], str)
        # Raw whitespace survives, now unreachable by an isinstance(str) pass.
        assert marked[0]["content"][0]["text"] == "hello  \n"

    def test_normalized_then_marked_matches_the_unmarked_wire_text(self):
        """Normalize-then-mark keeps a message byte-identical across the
        turn where it rolls out of the cache window."""
        raw = "file1\nfile2\n"  # trailing newline: every shell tool result

        # Turn N+1, message has left the window: plain string, normalized.
        out_of_window = raw.strip()

        # Turn N, message is in the window: normalized first, then marked.
        marked = apply_anthropic_cache_control(
            [{"role": "tool", "content": raw.strip(), "tool_call_id": "t1"}],
            native_anthropic=False,
        )
        in_window = marked[0]["content"][0]["text"]

        assert in_window == out_of_window

    def test_cache_marking_runs_after_every_message_mutation(self):
        """Ordering invariant, locked against regression."""
        import inspect

        from agent import conversation_loop

        src = inspect.getsource(conversation_loop)
        # Anchor on the call-block decoration (before the retry loop), not the
        # mid-failover redecoration helper which also calls apply_*.
        anchor = src.index("Runs LAST, after every message mutation above")
        mark = src.index("apply_anthropic_cache_control(\n", anchor)
        for earlier in (
            'am["content"].strip()',              # whitespace normalization
            "_sanitize_api_messages(api_messages)",       # orphan sweep
            "_drop_thinking_only_and_merge_users(",       # drop / merge
            "_sanitize_messages_surrogates(api_messages)",
        ):
            assert src.index(earlier) < mark, (
                f"{earlier!r} must run before cache breakpoints are injected"
            )


class TestStripAnthropicCacheControl:
    """strip must undo decoration so failover can re-render for a new policy."""

    def test_removes_top_level_and_part_markers(self):
        messages = apply_anthropic_cache_control(
            [
                {"role": "system", "content": "sys"},
                {"role": "user", "content": "hi"},
                {"role": "assistant", "content": "yo"},
            ],
            native_anthropic=True,
        )
        assert any(
            "cache_control" in (m if isinstance(m.get("content"), str) else {})
            or (
                isinstance(m.get("content"), list)
                and any(
                    isinstance(p, dict) and "cache_control" in p for p in m["content"]
                )
            )
            or "cache_control" in m
            for m in messages
        )
        strip_anthropic_cache_control(messages)
        for msg in messages:
            assert "cache_control" not in msg
            content = msg.get("content")
            if isinstance(content, list):
                for part in content:
                    if isinstance(part, dict):
                        assert "cache_control" not in part

    def test_flattens_system_static_volatile_back_to_string(self):
        static = "You are helpful.\n"
        full = static + "Model: claude\nProvider: anthropic"
        messages = apply_anthropic_cache_control(
            [{"role": "system", "content": full}, {"role": "user", "content": "hi"}],
            native_anthropic=True,
            static_system_prefix=static,
        )
        assert isinstance(messages[0]["content"], list)
        strip_anthropic_cache_control(messages)
        assert messages[0]["content"] == full

    def test_preserves_multimodal_part_structure(self):
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "see", "cache_control": {"type": "ephemeral"}},
                    {"type": "image_url", "image_url": {"url": "data:image/png;base64,xx"}},
                ],
            }
        ]
        strip_anthropic_cache_control(messages)
        content = messages[0]["content"]
        assert isinstance(content, list) and len(content) == 2
        assert content[0] == {"type": "text", "text": "see"}
        assert content[1]["type"] == "image_url"

    def test_preserves_organic_multipart_text_lists(self):
        # Multi-part pure-text lists NOT produced by decoration (merged user
        # turns, imported transcripts) must keep their structure — a ""-join
        # would fuse "Hello"+"world" into "Helloworld" and change wire bytes
        # on the common no-failover path (redecoration runs every attempt).
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Hello"},
                    {"type": "text", "text": "world"},
                ],
            }
        ]
        strip_anthropic_cache_control(messages)
        content = messages[0]["content"]
        assert isinstance(content, list) and len(content) == 2
        assert [p["text"] for p in content] == ["Hello", "world"]

    def test_preserves_extra_part_keys_like_citations(self):
        # anthropic_adapter deliberately whitelists citations on text blocks;
        # strip must not destroy them by flattening the part away.
        messages = [
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "text",
                        "text": "answer",
                        "citations": [{"src": "doc"}],
                        "cache_control": {"type": "ephemeral"},
                    }
                ],
            }
        ]
        strip_anthropic_cache_control(messages)
        content = messages[0]["content"]
        assert isinstance(content, list)
        assert content[0]["citations"] == [{"src": "doc"}]
        assert "cache_control" not in content[0]

    def test_marker_removal_is_copy_on_write_for_part_dicts(self):
        # The per-call api_messages copy is SHALLOW (msg.copy()) — content
        # part dicts alias the persistent history. Stripping a marker must
        # never rewrite the stored transcript's part dicts in place.
        shared_part = {"type": "text", "text": "hi", "cache_control": {"type": "ephemeral"}}
        history = [{"role": "user", "content": [shared_part]}]
        api_messages = [m.copy() for m in history]
        strip_anthropic_cache_control(api_messages)
        assert "cache_control" in shared_part  # history untouched
        api_content = api_messages[0]["content"]
        if isinstance(api_content, list):
            assert all("cache_control" not in p for p in api_content)

