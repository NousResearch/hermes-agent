"""Tests for agent.cache_shape — prompt-cache prefix-shape diagnostics (#68489).

Hermetic: pure-function tests over synthetic message lists; no network, no
agent instance, no config.
"""

from __future__ import annotations

from agent.cache_shape import (
    LOW_HIT_RATE_PCT,
    cache_scope,
    capture_prefix_shape,
    capture_request_shape,
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


class TestCacheScope:
    """Prompt caches are keyed per provider/model/endpoint (#68489 review).

    A fallback activation, a /model switch, or a credential-pool rotation to
    a different endpoint all move the request to a different cache namespace.
    The previous shape then describes a prefix the new backend never saw, so
    neither a prefix diff nor a "provider evicted it" conclusion is sound.
    """

    def test_scope_is_case_and_whitespace_insensitive(self):
        assert cache_scope("OpenAI", "GPT-5 ", "https://X/") == cache_scope(
            "openai", " gpt-5", "https://x/"
        )

    def test_scope_distinguishes_model_and_endpoint(self):
        base = cache_scope("openai", "gpt-5", "https://a/")
        assert base != cache_scope("openai", "gpt-4", "https://a/")
        assert base != cache_scope("openai", "gpt-5", "https://b/")
        assert base != cache_scope("anthropic", "gpt-5", "https://a/")

    def test_backend_change_reported_instead_of_provider_side_eviction(self):
        """The regression: identical prefix, zero hits, but a fallback fired.

        Before the scope check this returned "the miss is provider-side
        (cache TTL or eviction)", which is simply wrong — the new backend
        never held this prefix.
        """
        msgs = _messages("hi", "hello")
        prev = capture_prefix_shape(msgs, _TOOLS, scope=cache_scope("openai", "gpt-5", "https://a/"))
        cur = capture_prefix_shape(msgs, _TOOLS, scope=cache_scope("deepseek", "deepseek-chat", "https://b/"))

        reason = diagnose_cache_miss(
            prev, cur, cache_read_tokens=0, prompt_tokens=10_000
        )
        assert reason is not None
        assert "different backend" in reason
        assert "provider-side" not in reason
        assert "TTL" not in reason

    def test_backend_change_stays_quiet_when_hit_rate_is_healthy(self):
        msgs = _messages("hi", "hello")
        prev = capture_prefix_shape(msgs, _TOOLS, scope=cache_scope("openai", "gpt-5", ""))
        cur = capture_prefix_shape(msgs, _TOOLS, scope=cache_scope("deepseek", "deepseek-chat", ""))
        assert (
            diagnose_cache_miss(
                prev, cur, cache_read_tokens=9_000, prompt_tokens=10_000
            )
            is None
        )

    def test_backend_change_takes_priority_over_prefix_diff(self):
        """A rewritten prefix AND a routing change: blame the routing.

        Attributing it to compaction would send the user chasing a
        client-side cause for a cold cache that switching backends explains.
        """
        prev = capture_prefix_shape(
            _messages("hi", "hello"), _TOOLS, scope=cache_scope("openai", "gpt-5", "")
        )
        cur = capture_prefix_shape(
            _messages("different", "history"), None, scope=cache_scope("kimi", "k2", "")
        )
        reason = diagnose_cache_miss(
            prev, cur, cache_read_tokens=0, prompt_tokens=10_000
        )
        assert reason is not None
        assert "different backend" in reason

    def test_same_backend_still_diagnoses_provider_side_eviction(self):
        """The scope check must not swallow the original TTL diagnosis."""
        scope = cache_scope("openai", "gpt-5", "https://a/")
        prev = capture_prefix_shape(_messages("hi", "hello"), _TOOLS, scope=scope)
        cur = capture_prefix_shape(_messages("hi", "hello", "next"), _TOOLS, scope=scope)
        reason = diagnose_cache_miss(
            prev, cur, cache_read_tokens=0, prompt_tokens=10_000
        )
        assert reason is not None
        assert "provider-side" in reason


class TestCaptureRequestShape:
    """The shape must describe the *effective* payload (#68489 review).

    The loop rewrites the request after the old capture point — reasoning
    echo-back re-application, prompt-cache re-decoration, _build_api_kwargs,
    the codex_responses preflight, and request middleware all run first.
    capture_request_shape() therefore reads the finalized api_kwargs.
    """

    def test_reads_chat_completions_payload(self):
        api_kwargs = {
            "model": "gpt-5",
            "messages": _messages("hi", "hello"),
            "tools": _TOOLS,
        }
        shape = capture_request_shape(
            api_kwargs, provider="openai", model="gpt-5", base_url="https://a/"
        )
        assert shape.system_hash
        assert len(shape.message_hashes) == 2
        assert shape.tool_count == 2
        assert shape.scope == cache_scope("openai", "gpt-5", "https://a/")

    def test_reads_codex_responses_payload(self):
        """Codex Responses uses `input` + a top-level `instructions` string."""
        api_kwargs = {
            "model": "gpt-5-codex",
            "instructions": "You are Hermes.",
            "input": [{"role": "user", "content": "hi"}],
            "tools": _TOOLS,
        }
        shape = capture_request_shape(api_kwargs, provider="openai-codex", model="gpt-5-codex")
        assert shape.system_hash, "instructions must fold into system_hash"
        assert len(shape.message_hashes) == 1

    def test_instructions_change_is_attributed_to_the_system_prompt(self):
        def _shape(instructions: str):
            return capture_request_shape(
                {"instructions": instructions, "input": [{"role": "user", "content": "hi"}]},
                provider="openai-codex",
                model="gpt-5-codex",
            )

        assert prefix_changes(_shape("A"), _shape("B")) == ["system prompt changed"]

    def test_middleware_rewrite_of_the_payload_is_visible(self):
        """A middleware that mutates messages must change the fingerprint.

        This is the whole point of capturing after middleware rather than
        before it.
        """
        base = {"messages": _messages("hi"), "tools": _TOOLS}
        rewritten = {"messages": _messages("hi", system="Rewritten by middleware."), "tools": _TOOLS}
        before = capture_request_shape(base, provider="openai", model="gpt-5")
        after = capture_request_shape(rewritten, provider="openai", model="gpt-5")
        assert prefix_changes(before, after) == ["system prompt changed"]

    def test_missing_and_malformed_payload_keys_do_not_raise(self):
        assert capture_request_shape({}, provider="p", model="m").message_hashes == ()
        odd = capture_request_shape(
            {"messages": "not-a-list", "tools": "not-a-list"}, provider="p", model="m"
        )
        assert odd.message_hashes == ()
        assert odd.tool_count == 0
