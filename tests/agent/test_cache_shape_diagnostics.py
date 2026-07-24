"""Tests for agent.cache_shape — prompt-cache prefix-shape diagnostics (#68489).

Hermetic: pure-function tests over synthetic message lists; no network, no
agent instance, no config.
"""

from __future__ import annotations

from agent.cache_shape import (
    LOW_HIT_RATE_PCT,
    capture_prefix_shape,
    diagnose_cache_miss,
    prefix_changes,
)


def _messages(*contents: str, system: str = "You are Hermes."):
    msgs = [{"role": "system", "content": system}]
    for i, content in enumerate(contents):
        role = "user" if i % 2 == 0 else "assistant"
        msgs.append({"role": role, "content": content})
    return msgs


_TOOLS = [
    {"type": "function", "function": {"name": "read_file", "parameters": {}}},
    {"type": "function", "function": {"name": "web_search", "parameters": {}}},
]


class TestCapturePrefixShape:
    def test_system_message_hashed_separately_from_body(self):
        shape = capture_prefix_shape(_messages("hi", "hello"), _TOOLS)
        assert shape.system_hash
        assert len(shape.message_hashes) == 2
        assert shape.tool_count == 2

    def test_no_system_message(self):
        shape = capture_prefix_shape(
            [{"role": "user", "content": "hi"}], None
        )
        assert shape.system_hash == ""
        assert shape.tools_hash == ""
        assert shape.tool_count == 0
        assert len(shape.message_hashes) == 1

    def test_dict_key_order_does_not_change_hash(self):
        a = capture_prefix_shape(
            [{"role": "user", "content": "hi"}], None
        )
        b = capture_prefix_shape(
            [{"content": "hi", "role": "user"}], None
        )
        assert a.message_hashes == b.message_hashes

    def test_tool_list_order_changes_hash(self):
        # Tool schema *order* is part of the wire bytes — reordering must
        # register as a change even when the set of tools is identical.
        a = capture_prefix_shape(_messages("hi"), _TOOLS)
        b = capture_prefix_shape(_messages("hi"), list(reversed(_TOOLS)))
        assert a.tools_hash != b.tools_hash

    def test_unserializable_content_does_not_raise(self):
        shape = capture_prefix_shape(
            [{"role": "user", "content": object()}], None
        )
        assert shape.message_hashes


class TestPrefixChanges:
    def test_append_only_growth_reports_no_changes(self):
        prev = capture_prefix_shape(_messages("hi", "hello"), _TOOLS)
        cur = capture_prefix_shape(
            _messages("hi", "hello", "next question"), _TOOLS
        )
        assert prefix_changes(prev, cur) == []

    def test_system_prompt_change_detected(self):
        prev = capture_prefix_shape(_messages("hi"), _TOOLS)
        cur = capture_prefix_shape(
            _messages("hi", system="You are someone else."), _TOOLS
        )
        assert "system prompt changed" in prefix_changes(prev, cur)

    def test_tool_count_change_reported_with_counts(self):
        prev = capture_prefix_shape(_messages("hi"), _TOOLS)
        cur = capture_prefix_shape(_messages("hi"), _TOOLS[:1])
        changes = prefix_changes(prev, cur)
        assert any("2 → 1 tools" in c for c in changes)

    def test_history_rewrite_reports_first_divergent_message(self):
        prev = capture_prefix_shape(_messages("hi", "hello", "more"), _TOOLS)
        cur = capture_prefix_shape(
            _messages("hi", "REWRITTEN", "more"), _TOOLS
        )
        changes = prefix_changes(prev, cur)
        assert any("rewritten at message #2 of 3" in c for c in changes)

    def test_history_shrink_without_rewrite_reported(self):
        prev = capture_prefix_shape(_messages("a", "b", "c", "d"), _TOOLS)
        cur = capture_prefix_shape(_messages("a", "b"), _TOOLS)
        changes = prefix_changes(prev, cur)
        assert any("shrank (4 → 2 messages" in c for c in changes)


class TestDiagnoseCacheMiss:
    def test_none_when_no_previous_shape(self):
        cur = capture_prefix_shape(_messages("hi"), _TOOLS)
        assert (
            diagnose_cache_miss(
                None, cur, cache_read_tokens=0, prompt_tokens=1000
            )
            is None
        )

    def test_none_when_prompt_tokens_zero(self):
        shape = capture_prefix_shape(_messages("hi"), _TOOLS)
        assert (
            diagnose_cache_miss(
                shape, shape, cache_read_tokens=0, prompt_tokens=0
            )
            is None
        )

    def test_shape_change_reported_on_low_hit_rate(self):
        prev = capture_prefix_shape(_messages("hi"), _TOOLS)
        cur = capture_prefix_shape(
            _messages("hi", system="Different prompt."), _TOOLS
        )
        reason = diagnose_cache_miss(
            prev, cur, cache_read_tokens=0, prompt_tokens=10_000
        )
        assert reason is not None
        assert "system prompt changed" in reason

    def test_shape_change_suppressed_on_healthy_hit_rate(self):
        # A big appended tool result plus a marker shuffle can change shape
        # while the provider still serves most of the prefix from cache —
        # nothing to warn about.
        prev = capture_prefix_shape(_messages("hi"), _TOOLS)
        cur = capture_prefix_shape(
            _messages("hi", system="Different prompt."), _TOOLS
        )
        healthy = int(10_000 * (LOW_HIT_RATE_PCT + 10) / 100)
        assert (
            diagnose_cache_miss(
                prev, cur, cache_read_tokens=healthy, prompt_tokens=10_000
            )
            is None
        )

    def test_stable_prefix_with_zero_hits_flags_provider_side(self):
        prev = capture_prefix_shape(_messages("hi", "hello"), _TOOLS)
        cur = capture_prefix_shape(
            _messages("hi", "hello", "next"), _TOOLS
        )
        reason = diagnose_cache_miss(
            prev, cur, cache_read_tokens=0, prompt_tokens=10_000
        )
        assert reason is not None
        assert "provider-side" in reason

    def test_stable_prefix_with_partial_hits_stays_quiet(self):
        # Normal append-only growth: the new suffix is uncached, the prefix
        # hits. Any non-zero hit count with a stable shape is healthy.
        prev = capture_prefix_shape(_messages("hi", "hello"), _TOOLS)
        cur = capture_prefix_shape(
            _messages("hi", "hello", "next"), _TOOLS
        )
        assert (
            diagnose_cache_miss(
                prev, cur, cache_read_tokens=100, prompt_tokens=10_000
            )
            is None
        )
