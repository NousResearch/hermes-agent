"""Tests for agent/prompt_caching.py — Anthropic cache control injection."""

import copy

from agent.prompt_caching import (
    _apply_cache_marker,
    _build_marker,
    _can_carry_marker,
    _count_cache_markers,
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

    def test_direct_tool_cache_with_no_tools_falls_back_safely(self):
        """direct_native_tool_cache=True with an empty tools list must fall
        back to the message-only layout: exactly one marker per message here
        (system + user + assistant), none on tools.
        """
        messages = [
            {"role": "system", "content": "System prompt"},
            {"role": "user", "content": "Question"},
            {"role": "assistant", "content": "Answer"},
        ]
        plan = build_prompt_cache_plan(
            messages,
            [],
            native_anthropic=True,
            direct_native_tool_cache=True,
        )
        assert plan.marker_count == 3
        assert len(plan.tools) == 0


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

    def test_native_anthropic_empty_assistant_still_carries_marker(self):
        """Native envelope can attach a top-level marker even to an empty
        assistant turn; only the OpenRouter-style layout must skip it.
        """
        assert _can_carry_marker({"role": "assistant", "content": None}, native_anthropic=True) is True

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
        # opencode-go is excluded: MEASURED to honour the 1h tier, see
        # test_1h_preserved_on_measured_opencode_go_route. The rest stay
        # clamped because they are unmeasured, not because they are known bad.
        for provider in ("opencode", "opencode-zen", "alibaba"):
            assert effective_cache_ttl("1h", provider=provider, model="qwen-max") == "5m", provider
            assert effective_cache_ttl("1h", provider=provider.upper(), model="claude-x") == "5m", provider

    def test_1h_preserved_on_measured_opencode_go_route(self):
        """opencode-go was MEASURED to honour the 1h tier (#84733 follow-up).

        Controlled run: identical request, only the ttl flag varying, read back
        after 11 minutes with no intervening call (a read renews the window and
        masks expiry).

            qwen3.8-max   ttl=1h -> cache_read 2122  SURVIVED
            qwen3.8-max   ttl=-  -> cache_read    0  EXPIRED   <- control
            glm-5.2       ttl=1h -> cache_read 2092  SURVIVED
            minimax-m2.5  ttl=1h -> cache_read    0  EXPIRED

        NB the provider labels every write ``ephemeral_5m_input_tokens``
        regardless of the ttl requested; that label is not evidence of the
        retention window.
        """
        assert effective_cache_ttl("1h", provider="opencode-go", model="qwen3.8-max") == "1h"
        assert effective_cache_ttl("1h", provider="opencode-go", model="glm-5.2") == "1h"

    def test_1h_clamped_for_measured_no_1h_model_even_on_allowed_route(self):
        assert effective_cache_ttl("1h", provider="opencode-go", model="minimax-m2.5") == "5m"
        # aggregator-prefixed spelling resolves to the same bare id
        assert effective_cache_ttl("1h", provider="opencode-go", model="vendor/MiniMax-M2.5") == "5m"

    def test_unmeasured_opencode_routes_stay_clamped(self):
        # Not "known bad" -- simply not measured. Do not widen without a run.
        assert effective_cache_ttl("1h", provider="opencode", model="qwen3.6-plus") == "5m"
        assert effective_cache_ttl("1h", provider="opencode-zen", model="qwen3.6-plus") == "5m"

    def test_ttl_allowlist_is_separate_from_cache_layout_optin(self):
        """Regression guard for the trap in the original shared-set design.

        ALIBABA_FAMILY_PROVIDERS drives the cache-marker-layout OPT-IN. Reusing
        it for the TTL clamp means narrowing the clamp DISABLES caching instead
        of extending its TTL.
        """
        from agent.prompt_caching import (
            ALIBABA_FAMILY_PROVIDERS,
            MEASURED_1H_PROVIDERS,
        )

        assert "opencode-go" in ALIBABA_FAMILY_PROVIDERS
        assert "opencode-go" in MEASURED_1H_PROVIDERS
        assert not (MEASURED_1H_PROVIDERS & {"alibaba"})

    def test_marker_built_from_clamped_ttl_has_no_1h_key(self):
        marker = _build_marker(effective_cache_ttl("1h", provider="opencode", model="qwen3.6-plus"))
        assert marker == {"type": "ephemeral"}
        marker = _build_marker(effective_cache_ttl("1h", provider="anthropic", model="claude-x"))
        assert marker == {"type": "ephemeral", "ttl": "1h"}


class TestApplyIdempotency:
    """apply_anthropic_cache_control on pre-decorated input (#90971).

    Before the idempotency fix, a second call on already-marked messages
    pushed the marker total to 5, reproducing the ``cache_control can only
    be specified up to 4 times`` HTTP 400.
    """

    @staticmethod
    def _fixture_messages():
        messages = [{"role": "system", "content": "STATIC_PREFIX rest of the prompt"}]
        for i in range(8):
            messages.append({"role": "user", "content": f"Hello {i}"})
            messages.append({"role": "assistant", "content": f"Hi {i}"})
        return messages

    def test_empty_messages_is_noop(self):
        assert apply_anthropic_cache_control([]) == []

    def test_repeated_apply_is_idempotent_and_keeps_exact_layout(self):
        """Repeated calls on the function's own output (no intervening
        strip_anthropic_cache_control) must converge to the exact same
        marker placement AND keep the intended four-breakpoint layout.
        Structural equality alone would still pass if a later round moved
        the breakpoints or dropped every marker; the exact count pins the
        layout.
        """
        messages = self._fixture_messages()

        round1 = apply_anthropic_cache_control(messages, static_system_prefix="STATIC_PREFIX")
        round2 = apply_anthropic_cache_control(round1, static_system_prefix="STATIC_PREFIX")
        round3 = apply_anthropic_cache_control(round2, static_system_prefix="STATIC_PREFIX")

        assert round1 == round2 == round3
        assert _count_cache_markers(round1, []) == 4

    def test_does_not_mutate_caller_messages_with_stale_top_level_markers(self):
        """A caller's live message list must never be mutated in place, even
        when it already carries stale cache_control markers (e.g. replayed
        history). The function's contract is copy-on-write.
        """
        caller_history = [
            {"role": "user", "content": f"u{i}", "cache_control": {"type": "ephemeral"}}
            for i in range(5)
        ]
        snapshot = copy.deepcopy(caller_history)

        apply_anthropic_cache_control(caller_history)

        assert caller_history == snapshot

    def test_does_not_mutate_caller_messages_with_stale_part_markers(self):
        """Same contract for the other detection branch: markers living on
        content parts (the shape decoration itself produces), where part-dict
        aliasing is the mutation risk.
        """
        caller_history = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": f"u{i}", "cache_control": {"type": "ephemeral"}},
                    {"type": "text", "text": "tail"},
                ],
            }
            for i in range(5)
        ]
        snapshot = copy.deepcopy(caller_history)

        result = apply_anthropic_cache_control(caller_history)

        assert caller_history == snapshot
        assert _count_cache_markers(result, []) <= 4




class TestOpenCodeGoOneHourPrecedence:
    """Precedence + eligibility guards for the opencode-go 1h allowance.

    The historical repair (payload ``d6b33faae1``, merged as ``a43fe4918d``)
    is not on the current upstream lineage, so ``effective_cache_ttl`` had
    regressed to evaluating the generic :func:`is_qwen_model` clamp *before*
    any route allowance. That ordering silently turned a configured
    ``prompt_caching.cache_ttl: 1h`` back into ``5m`` for every Qwen model on
    opencode-go.

    These tests pin the two things the historical suite left implicit:

    * the allowance must win over the generic Qwen clamp (ordering), and
    * restoring the 1h tier must not cost prompt-cache *eligibility* —
      the tempting "just drop opencode-go from ALIBABA_FAMILY_PROVIDERS"
      repair disables caching outright rather than extending its window,
      because that same set is the cache-marker-layout opt-in in
      ``agent_runtime_helpers.anthropic_prompt_cache_policy``.

    Evidence scope, stated honestly: the wire measurement behind the
    allowance was taken on ``qwen3.8-max`` and ``glm-5.2``. The rule is keyed
    on the *route*, not the model, so the currently deployed
    ``qwen3.7-plus`` is covered by it — but ``qwen3.7-plus`` itself has not
    been measured against a >5-minute delayed read. That claim stays open
    until field validation closes it.
    """

    DEPLOYED_MODEL = "qwen3.7-plus"

    # -- the deployed case ---------------------------------------------------

    def test_deployed_qwen37_plus_keeps_configured_1h(self):
        assert effective_cache_ttl(
            "1h", provider="opencode-go", model=self.DEPLOYED_MODEL
        ) == "1h"

    def test_deployed_qwen37_plus_marker_carries_ttl_1h(self):
        """M2/M6: the emitted marker must actually say ``ttl: "1h"``.

        A correct return value that never reaches the wire marker is the
        whole defect restated one layer down.
        """
        marker = _build_marker(
            effective_cache_ttl("1h", provider="opencode-go", model=self.DEPLOYED_MODEL)
        )
        assert marker == {"type": "ephemeral", "ttl": "1h"}

    def test_caching_remains_enabled_for_opencode_go_qwen(self):
        """M1': prompt caching must stay ON, not merely be re-tiered.

        Guards the dangerous naive repair. Dropping ``opencode-go`` from
        ``ALIBABA_FAMILY_PROVIDERS`` would make ``effective_cache_ttl``
        return ``1h`` while ``anthropic_prompt_cache_policy`` stops opting
        the route in at all — 5-minute caching quietly becomes *no* caching.
        Asserting the TTL alone cannot see that.
        """
        from agent.agent_runtime_helpers import (
            anthropic_prompt_cache_policy,
            blank_cache_policy_stub,
        )

        stub = blank_cache_policy_stub(cache_disabled=False)
        # Built-in route: keep the catalog/config fallback out of the test.
        stub._custom_providers = []

        should_cache, native_layout = anthropic_prompt_cache_policy(
            stub,
            provider="opencode-go",
            base_url="",
            api_mode="chat_completions",
            model=self.DEPLOYED_MODEL,
        )
        assert should_cache is True, "opencode-go/qwen lost prompt-cache eligibility"
        assert native_layout is False, "opencode-go takes the envelope layout"
        assert effective_cache_ttl(
            "1h", provider="opencode-go", model=self.DEPLOYED_MODEL
        ) == "1h"

    # -- precedence ----------------------------------------------------------

    def test_route_allowance_precedes_generic_qwen_clamp(self):
        """M8: kills the ordering mutation.

        The same model is clamped off-route and preserved on-route. That can
        only hold if the route allowance is evaluated *before*
        :func:`is_qwen_model`; hoisting the generic clamp back above it turns
        the second assertion red.
        """
        for model in ("qwen3.7-plus", "qwen3.8-max", "Qwen-Max"):
            assert effective_cache_ttl("1h", provider="openrouter", model=model) == "5m", model
            assert effective_cache_ttl("1h", provider="opencode-go", model=model) == "1h", model

    def test_measured_no_1h_model_still_beats_route_allowance(self):
        """Model-level denial wins inside the allowed route."""
        assert effective_cache_ttl("1h", provider="opencode-go", model="minimax-m2.5") == "5m"

    def test_no_1h_denial_does_not_leak_off_the_measured_route(self):
        """The denial is scoped to the route it was measured on.

        ``minimax-m2.5`` was observed ignoring the tier *on opencode-go*. That
        says nothing about MiniMax on its own Anthropic-compatible endpoint,
        which is a separate and genuinely cache-eligible route
        (``anthropic_prompt_cache_policy`` opts it in by provider id / host
        match). Consulting ``NO_1H_TIER_MODELS`` globally silently regressed
        that route's configured 1h to 5m off the back of an unrelated
        observation — an out-of-scope behaviour change this repair must not
        make.
        """
        for provider in ("minimax", "minimax-cn", "anthropic", "openrouter"):
            assert effective_cache_ttl("1h", provider=provider, model="MiniMax-M2.5") == "1h", provider

    def test_route_allowance_is_not_restricted_to_the_measured_models(self):
        """Records a real consequence of a route-keyed rule.

        Before this change ``opencode-go`` + a Claude model clamped to ``5m``
        via the family branch; it now keeps ``1h`` like everything else on the
        route. That is deliberate — the allowance is keyed on the route, not
        on the two models the delayed-read run happened to cover — but it is a
        behaviour change outside the measurement set, so it is pinned here
        rather than left as an uncovered side effect. opencode-go serves
        Claude over ``anthropic_messages``, so this route is reachable.
        """
        assert effective_cache_ttl("1h", provider="opencode-go", model="claude-sonnet-5") == "1h"
        assert effective_cache_ttl("1h", provider="OPENCODE-GO", model="claude-x") == "1h"

    # -- negative controls ---------------------------------------------------

    def test_generic_qwen_clamp_negative_control(self):
        """§9: 1h must NOT become global. Off-route Qwen stays at 5m."""
        for provider in ("openrouter", "anthropic", "together", ""):
            assert effective_cache_ttl("1h", provider=provider, model="qwen3.7-plus") == "5m", provider
        assert _build_marker(
            effective_cache_ttl("1h", provider="openrouter", model="qwen3.7-plus")
        ) == {"type": "ephemeral"}

    def test_1h_not_granted_to_the_rest_of_the_alibaba_family(self):
        """M3: the allowance is one measured route, not the whole family."""
        from agent.prompt_caching import ALIBABA_FAMILY_PROVIDERS, MEASURED_1H_PROVIDERS

        assert MEASURED_1H_PROVIDERS == frozenset({"opencode-go"})
        for provider in sorted(ALIBABA_FAMILY_PROVIDERS - MEASURED_1H_PROVIDERS):
            assert effective_cache_ttl("1h", provider=provider, model="qwen3.7-plus") == "5m", provider

    def test_allowance_is_provider_wide_not_pinned_to_one_model(self):
        """M5: repairing only the deployed model would leave siblings clamped."""
        for model in ("qwen3.7-plus", "qwen3.8-max", "qwen3.6-plus", "qwen-max", "glm-5.2"):
            assert effective_cache_ttl("1h", provider="opencode-go", model=model) == "1h", model

    def test_provider_spelling_variants_hit_the_allowance(self):
        """M7: alias/normalization must not route around the allow-list."""
        for spelling in ("opencode-go", "OpenCode-Go", "OPENCODE-GO"):
            assert effective_cache_ttl("1h", provider=spelling, model=self.DEPLOYED_MODEL) == "1h", spelling

    def test_lower_tiers_are_untouched_by_the_allowance(self):
        assert effective_cache_ttl("5m", provider="opencode-go", model=self.DEPLOYED_MODEL) == "5m"
        assert effective_cache_ttl(None, provider="opencode-go", model=self.DEPLOYED_MODEL) == "5m"

    # -- call-site coverage --------------------------------------------------

    def test_every_marker_emitting_call_site_goes_through_the_central_clamp(self):
        """M4: CLI and scheduled paths must not diverge.

        Enumerates every production call of the two marker-emitting entry
        points — :func:`build_prompt_cache_plan` and
        :func:`apply_anthropic_cache_control` — and requires each to receive
        its TTL from ``effective_cache_ttl(...)``.

        Stated as a *closed* set rather than a presence count: a new sender
        that hands a raw configured TTL straight to either entry point fails
        the membership assertion, and un-clamping an existing one fails the
        proximity assertion. ``auxiliary_client`` and the MoA aggregator plan
        are absent by design — they reach the wire through
        ``plan_cache_sections_for_destination``, which clamps internally, so
        the helper's own call site covers them.
        """
        import pathlib
        import re

        root = pathlib.Path(__file__).resolve().parents[2]
        entry_points = re.compile(r"\b(?:build_prompt_cache_plan|apply_anthropic_cache_control)\(")

        expected = {
            # (path, must be clamped at the call site)
            ("agent/agent_runtime_helpers.py", True),
            ("agent/conversation_loop.py", True),
            ("agent/moa_loop.py", True),
            # The planner forwarding an already-clamped ttl to the applier.
            ("agent/prompt_caching.py", False),
        }

        found = {}
        for path in sorted(root.glob("agent/*.py")) + [root / "run_agent.py"]:
            lines = path.read_text(encoding="utf-8").splitlines()
            for i, line in enumerate(lines):
                if not entry_points.search(line) or line.lstrip().startswith(("#", "def ", "*", '"')):
                    continue
                rel = path.relative_to(root).as_posix()
                window = "\n".join(lines[i : i + 14])
                found.setdefault(rel, []).append("cache_ttl=effective_cache_ttl(" in window)

        assert {(rel, all(flags)) for rel, flags in found.items()} == expected, (
            f"marker-emitting call sites changed: {found}"
        )
        # Pin the count too, so a second unclamped call inside an already
        # listed file cannot hide behind its clamped sibling.
        assert sum(len(v) for v in found.values()) == 5, found
