"""Tests for agent/prompt_caching.py — Anthropic cache control injection."""


from agent.prompt_caching import (
    _apply_cache_marker,
    _build_marker,
    _can_carry_marker,
    apply_anthropic_cache_control,
    build_prompt_cache_plan,
    effective_cache_ttl,
    strip_anthropic_cache_control,
    strip_anthropic_tool_cache_control,
)


MARKER = {"type": "ephemeral"}


def _native_marker_indexes(messages):
    return {
        index
        for index, message in enumerate(messages)
        if isinstance(message, dict)
        and (
            "cache_control" in message
            or any(
                isinstance(part, dict) and "cache_control" in part
                for part in (message.get("content") if isinstance(message.get("content"), list) else [])
            )
        )
    }


def _tool_heavy_native_history():
    return [
        {"role": "system", "content": "stable prefix\nvolatile suffix"},
        {"role": "user", "content": "first request"},
        {"role": "assistant", "content": "", "tool_calls": [{"id": "first", "function": {"name": "tool_00", "arguments": "{}"}}]},
        {"role": "tool", "tool_call_id": "first", "content": "first result"},
        {"role": "user", "content": "second request"},
        {"role": "assistant", "content": "", "tool_calls": [{"id": "second", "function": {"name": "tool_01", "arguments": "{}"}}]},
        {"role": "tool", "tool_call_id": "second", "content": "second result"},
    ]


def _tool_heavy_native_tools():
    return [
        {
            "type": "function",
            "function": {
                "name": f"tool_{index:02d}",
                "description": f"Deterministic tool {index}",
                "parameters": {"type": "object", "properties": {}},
            },
        }
        for index in range(28)
    ]


def test_t20880_tool_heavy_native_loop_reproduction():
    """A 28-tool native loop needs a tool marker and a retained transaction endpoint."""
    tools = _tool_heavy_native_tools()
    before_exchange = build_prompt_cache_plan(
        _tool_heavy_native_history(),
        tools,
        native_anthropic=True,
        static_system_prefix="stable prefix",
        direct_native_tool_cache=True,
    )
    after_exchange = build_prompt_cache_plan(
        _tool_heavy_native_history() + [
            {"role": "user", "content": "third request"},
            {"role": "assistant", "content": "", "tool_calls": [{"id": "third", "function": {"name": "tool_02", "arguments": "{}"}}]},
            {"role": "tool", "tool_call_id": "third", "content": "third result"},
        ],
        tools,
        native_anthropic=True,
        static_system_prefix="stable prefix",
        direct_native_tool_cache=True,
    )

    before_markers = _native_marker_indexes(before_exchange.messages)
    after_markers = _native_marker_indexes(after_exchange.messages)
    final_tool_marked = "cache_control" in after_exchange.tools[-1]
    shared_transaction_endpoint = bool((before_markers - {0}) & (after_markers - {0}))

    assert final_tool_marked
    assert shared_transaction_endpoint
    assert after_exchange.marker_count <= 4


class TestPromptCachePlan:
    def test_copies_sections_and_keeps_canonical_tools_plain(self):
        import copy

        messages = _tool_heavy_native_history()
        tools = _tool_heavy_native_tools()
        original_messages = copy.deepcopy(messages)
        original_tools = copy.deepcopy(tools)

        plan = build_prompt_cache_plan(
            messages,
            tools,
            native_anthropic=True,
            static_system_prefix="stable prefix",
            direct_native_tool_cache=True,
        )

        assert messages == original_messages
        assert tools == original_tools
        assert plan.messages is not messages
        assert plan.tools is not tools
        assert "cache_control" not in tools[-1]
        assert plan.tools[-1]["cache_control"] == MARKER
        assert plan.marker_count == 4

    def test_unmarkable_endpoint_does_not_consume_a_slot(self):
        messages = [
            {"role": "system", "content": "stable prefix\nvolatile"},
            {"role": "assistant", "content": "", "tool_calls": [{"id": "pending", "function": {"name": "tool_00", "arguments": "{}"}}]},
        ]
        plan = build_prompt_cache_plan(
            messages,
            _tool_heavy_native_tools(),
            native_anthropic=True,
            static_system_prefix="stable prefix",
            direct_native_tool_cache=True,
        )

        assert plan.marker_count == 2
        assert "cache_control" not in plan.messages[-1]

    def test_static_prefix_equal_to_whole_prompt_emits_no_empty_block(self):
        """Empty volatile suffix must not produce an empty text block.

        Anthropic rejects text blocks whose ``text`` is empty; when the
        stored system prompt IS the static prefix (no volatile tier), the
        plan must mark it as one whole block instead of a two-part split
        with a trailing ``{"type": "text", "text": ""}``.
        """
        messages = [
            {"role": "system", "content": "stable prefix"},
            {"role": "user", "content": "lookup"},
        ]
        plan = build_prompt_cache_plan(
            messages,
            _tool_heavy_native_tools(),
            native_anthropic=True,
            static_system_prefix="stable prefix",
            direct_native_tool_cache=True,
        )

        system_content = plan.messages[0]["content"]
        assert isinstance(system_content, list)
        for part in system_content:
            assert part.get("text"), "no empty text blocks on the wire"
        assert any("cache_control" in part for part in system_content)
        assert plan.tools[-1]["cache_control"] == MARKER

    def test_tool_strip_is_request_local(self):
        tools = _tool_heavy_native_tools()
        tools[-1]["cache_control"] = MARKER

        stripped = strip_anthropic_tool_cache_control(tools)

        assert "cache_control" in tools[-1]
        assert "cache_control" not in stripped[-1]


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






    def test_string_content_wrapped_in_list(self):
        msg = {"role": "user", "content": "Hello"}
        _apply_cache_marker(msg, MARKER)
        assert isinstance(msg["content"], list)
        assert len(msg["content"]) == 1
        assert msg["content"][0]["type"] == "text"
        assert msg["content"][0]["text"] == "Hello"
        assert msg["content"][0]["cache_control"] == MARKER




class TestCanCarryMarker:


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




    def test_1h_ttl(self):
        msgs = [{"role": "system", "content": "System prompt"}]
        result = apply_anthropic_cache_control(msgs, cache_ttl="1h")
        sys_content = result[0]["content"]
        assert isinstance(sys_content, list)
        assert sys_content[0]["cache_control"]["ttl"] == "1h"





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
        # Anchor on the call-block request plan, not the retry helper.
        anchor = src.index("Build the request-local cache sections")
        mark = src.index("build_prompt_cache_plan(\n", anchor)
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


class TestEffectiveCacheTtl:
    """#84733: Qwen/Alibaba routes document a 5-minute context cache only.

    ``effective_cache_ttl`` clamps a requested ``1h`` tier down to ``5m`` on
    those routes so the marker the provider would ignore/reject is never
    shipped and no false 1h-cache expectation survives.
    """

    def test_none_resolves_to_default_5m(self):
        assert effective_cache_ttl(None) == "5m"
        assert effective_cache_ttl(None, provider="anthropic", model="claude-x") == "5m"

    def test_5m_passthrough_everywhere(self):
        assert effective_cache_ttl("5m") == "5m"
        assert effective_cache_ttl("5m", provider="opencode", model="qwen3.6-plus") == "5m"

    def test_1h_preserved_on_non_qwen_routes(self):
        assert effective_cache_ttl("1h", provider="anthropic", model="claude-opus-4.8") == "1h"
        assert effective_cache_ttl("1h", provider="openrouter", model="claude-3-5-sonnet") == "1h"
        assert effective_cache_ttl("1h", provider="", model="") == "1h"

    def test_1h_clamped_for_qwen_model_on_any_route(self):
        assert effective_cache_ttl("1h", provider="openrouter", model="qwen3.6-plus") == "5m"
        assert effective_cache_ttl("1h", provider="anthropic", model="Qwen-Max") == "5m"

    def test_1h_clamped_for_alibaba_family_providers(self):
        for provider in ("opencode", "opencode-zen", "opencode-go", "alibaba"):
            assert effective_cache_ttl("1h", provider=provider, model="qwen-max") == "5m", provider
            assert effective_cache_ttl("1h", provider=provider.upper(), model="claude-x") == "5m", provider

    def test_marker_built_from_clamped_ttl_has_no_1h_key(self):
        marker = _build_marker(effective_cache_ttl("1h", provider="opencode", model="qwen3.6-plus"))
        assert marker == {"type": "ephemeral"}
        marker = _build_marker(effective_cache_ttl("1h", provider="anthropic", model="claude-x"))
        assert marker == {"type": "ephemeral", "ttl": "1h"}




