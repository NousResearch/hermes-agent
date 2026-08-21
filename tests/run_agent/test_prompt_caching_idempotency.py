"""Tests for Anthropic prompt caching idempotency and breakpoint bounds (#90971)."""

import copy

from agent.prompt_caching import (
    apply_anthropic_cache_control,
    build_prompt_cache_plan,
    _count_cache_markers,
    _can_carry_marker,
)


class TestPromptCachingIdempotency:
    def test_apply_anthropic_cache_control_empty_messages(self):
        """Empty messages list is a safe no-op."""
        assert apply_anthropic_cache_control([]) == []

    def test_apply_anthropic_cache_control_never_exceeds_four_markers(self):
        """Realistic conversations never push breakpoints_used past what
        _apply_system_cache_markers can return ({0, 1, 2}), so the marker
        total always stays within the 4-breakpoint API limit.
        """
        messages = [{"role": "system", "content": "STATIC_PREFIX rest of the prompt"}]
        for i in range(8):
            messages.append({"role": "user", "content": f"Hello {i}"})
            messages.append({"role": "assistant", "content": f"Hi {i}"})

        result = apply_anthropic_cache_control(messages, static_system_prefix="STATIC_PREFIX")
        assert _count_cache_markers(result, []) <= 4

    def test_apply_anthropic_cache_control_is_idempotent(self):
        """Calling apply_anthropic_cache_control repeatedly on its own output
        (no intervening strip_anthropic_cache_control) must converge to the
        exact same marker placement, not merely stay under budget: a test
        that only checks `<= 4` would still pass if a later round moved the
        breakpoints somewhere else, or dropped every marker. Before the
        idempotency fix, a second call on already-marked messages pushed the
        total to 5, reproducing the `cache_control can only be specified up
        to 4 times` HTTP 400 (#90971).
        """
        messages = [{"role": "system", "content": "STATIC_PREFIX rest of the prompt"}]
        for i in range(8):
            messages.append({"role": "user", "content": f"Hello {i}"})
            messages.append({"role": "assistant", "content": f"Hi {i}"})

        round1 = apply_anthropic_cache_control(messages, static_system_prefix="STATIC_PREFIX")
        round2 = apply_anthropic_cache_control(round1, static_system_prefix="STATIC_PREFIX")
        round3 = apply_anthropic_cache_control(round2, static_system_prefix="STATIC_PREFIX")

        assert round1 == round2 == round3
        assert _count_cache_markers(round1, []) <= 4

    def test_apply_anthropic_cache_control_does_not_mutate_caller_messages(self):
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

    def test_build_prompt_cache_plan_dynamic_tool_accounting(self):
        """build_prompt_cache_plan never exceeds 4 markers with tool-cache layout."""
        tools = [
            {"type": "function", "function": {"name": "tool_a"}},
            {"type": "function", "function": {"name": "tool_b"}},
            {"type": "function", "function": {"name": "tool_c"}},
        ]
        messages = [
            {"role": "system", "content": "PREFIX_STATIC System prompt"},
            {"role": "user", "content": "Run tool"},
            {"role": "assistant", "content": "Calling", "tool_calls": [{"name": "tool_a"}]},
            {"role": "tool", "content": "output", "tool_name": "tool_a"},
            {"role": "assistant", "content": "Done!"},
        ]

        plan = build_prompt_cache_plan(
            messages,
            tools,
            native_anthropic=True,
            direct_native_tool_cache=True,
            static_system_prefix="PREFIX_STATIC",
        )

        assert plan.marker_count <= 4
        # Exactly 1 tool marker on the last tool
        assert "cache_control" in plan.tools[-1]
        assert "cache_control" not in plan.tools[0]
        assert "cache_control" not in plan.tools[1]

    def test_build_prompt_cache_plan_direct_tool_cache_with_no_tools(self):
        """When direct_native_tool_cache=True but tools is empty, falls back safely."""
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
        assert plan.marker_count <= 4
        assert len(plan.tools) == 0

    def test_can_carry_marker_envelope_vs_native(self):
        """_can_carry_marker properly filters empty turns on non-native layouts."""
        empty_assistant = {"role": "assistant", "content": None}
        assert _can_carry_marker(empty_assistant, native_anthropic=False) is False
        assert _can_carry_marker(empty_assistant, native_anthropic=True) is True

        normal_user = {"role": "user", "content": "Hello"}
        assert _can_carry_marker(normal_user, native_anthropic=False) is True
        assert _can_carry_marker(normal_user, native_anthropic=True) is True
