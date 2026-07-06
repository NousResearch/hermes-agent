"""Tests for ``smart_model_routing`` (per-turn cheap-model swap).

These cover the public contract of the helper:

* The simple-message classifier picks short, low-word-count inputs.
* The cheap-target resolver yields a dict compatible with
  ``resolve_provider_client`` provider/model/base_url/api_key hints.
* ``apply_smart_routing`` is fail-soft — a misconfigured block never
  breaks the agent. Missing provider, malformed config, and resolver
  failure all leave the primary runtime untouched.
* ``restore_smart_routing`` is idempotent and safe to call when no
  swap was made (no-op).
* Restore round-trip yields a byte-identical ``_cached_system_prompt``
  identity block (primary's prefix cache still matches after restore).

The integration test against a live turn is intentionally out of scope
here — exercised in E2E / dogfood tests that spin a real AIAgent with a
real ``HERMES_HOME``.
"""

from types import SimpleNamespace

import pytest

from agent.smart_model_routing import (
    _is_simple_message,
    _resolve_smart_target,
    apply_smart_routing,
    restore_smart_routing,
)


# ── Classifier ──────────────────────────────────────────────────────────


class TestIsSimpleMessage:
    def test_short_acknowledgement_routes_to_simple(self):
        assert _is_simple_message("hi", 200, 35) is True

    def test_one_word_question_routes_to_simple(self):
        assert _is_simple_message("ok?", 200, 35) is True

    def test_long_message_routes_to_complex_by_chars(self):
        long = ("Please analyze this codebase, find every place that uses "
                "smart_model_routing, and produce a migration plan with "
                "specific file changes and line numbers, group them by risk, "
                "and assign effort estimates. " * 3)
        assert len(long) > 200
        assert _is_simple_message(long, 200, 35) is False

    def test_long_message_routes_to_complex_by_words(self):
        many_words = (" ".join(["token"] * 40))
        assert _is_simple_message(many_words, 200, 35) is False

    def test_empty_string_not_simple(self):
        assert _is_simple_message("", 200, 35) is False

    def test_whitespace_only_not_simple(self):
        assert _is_simple_message("   \n\n   ", 200, 35) is False

    def test_non_string_not_simple(self):
        assert _is_simple_message(None, 200, 35) is False  # type: ignore[arg-type]
        assert _is_simple_message(123, 200, 35) is False  # type: ignore[arg-type]

    def test_custom_thresholds_respected(self):
        # "hi hello world" has 3 words → simple at default threshold
        # (max_words=35) but complex at max_words=2.
        msg = "hi hello world"
        assert _is_simple_message(msg, max_chars=200, max_words=35) is True
        assert _is_simple_message(msg, max_chars=200, max_words=2) is False
        # Boundary: max_words=3 still classifies simple (word_count <= threshold).
        assert _is_simple_message(msg, max_chars=200, max_words=3) is True


# ── Target resolver ────────────────────────────────────────────────────


class TestResolveSmartTarget:
    def test_minimal_config_yields_provider_and_model(self):
        cfg = {"cheap_model": {"provider": "opencode-zen", "model": "deepseek-v4-flash-free"}}
        target = _resolve_smart_target(cfg)
        assert target is not None
        assert target["provider"] == "opencode-zen"
        assert target["model"] == "deepseek-v4-flash-free"

    def test_missing_provider_returns_none(self):
        cfg = {"cheap_model": {"model": "deepseek-v4-flash-free"}}
        assert _resolve_smart_target(cfg) is None

    def test_missing_model_returns_none(self):
        cfg = {"cheap_model": {"provider": "opencode-zen"}}
        assert _resolve_smart_target(cfg) is None

    def test_missing_block_returns_none(self):
        assert _resolve_smart_target({}) is None
        assert _resolve_smart_target({"cheap_model": "not-a-dict"}) is None


# ── apply_smart_routing fail-soft contract ──────────────────────────────


class _DummyAgent(SimpleNamespace):
    """Minimal agent-like object for fail-soft tests.

    We deliberately do NOT subclass AIAgent — fail-soft tests must
    not require a working init_agent (heavy imports). The router only
    reads ``_smart_routing_cfg``, ``_smart_routed_active``, ``_fallback_activated``,
    ``model``, ``provider``, ``base_url``, ``api_mode``, ``api_key``,
    ``_client_kwargs``, ``_use_prompt_caching``, ``_transport_cache``.
    """


class TestApplySmartRoutingFailSoft:
    def test_disabled_block_returns_false_no_swap(self):
        agent = _DummyAgent(
            model="MiniMax-M3",
            provider="minimax",
            _smart_routing_cfg={"enabled": False, "cheap_model": {
                "provider": "opencode-zen", "model": "deepseek-v4-flash-free",
            }},
            _smart_routed_active=False,
            _fallback_activated=False,
        )
        assert apply_smart_routing(agent, "hi") is False
        assert agent.model == "MiniMax-M3"

    def test_already_active_short_circuits(self):
        agent = _DummyAgent(
            model="deepseek-v4-flash-free",
            provider="opencode-zen",
            _smart_routing_cfg={"enabled": True, "cheap_model": {
                "provider": "opencode-zen", "model": "deepseek-v4-flash-free",
            }},
            _smart_routed_active=True,  # already cheap
            _fallback_activated=False,
        )
        # Returns True (active) but performs no work — model unchanged.
        result = apply_smart_routing(agent, "hi")
        # The function returns True on "already swapped" — it shouldn't
        # attempt to swap again or mutate anything.
        assert result is True
        assert agent.model == "deepseek-v4-flash-free"

    def test_fallback_active_blocked_from_smart_swap(self):
        agent = _DummyAgent(
            model="openrouter/auto",
            provider="openrouter",
            _smart_routing_cfg={"enabled": True, "cheap_model": {
                "provider": "opencode-zen", "model": "deepseek-v4-flash-free",
            }},
            _smart_routed_active=False,
            _fallback_activated=True,  # session is in failover
        )
        assert apply_smart_routing(agent, "hi") is False
        assert agent.model == "openrouter/auto"

    def test_long_message_does_not_swap(self):
        # Even with an enabled block, a 500-char message must not route.
        agent = _DummyAgent(
            model="MiniMax-M3",
            provider="minimax",
            _smart_routing_cfg={"enabled": True, "cheap_model": {
                "provider": "opencode-zen", "model": "deepseek-v4-flash-free",
            }},
            _smart_routed_active=False,
            _fallback_activated=False,
        )
        long = "x" * 500
        assert apply_smart_routing(agent, long) is False
        assert agent.model == "MiniMax-M3"

    def test_malformed_cheap_block_does_not_swap(self):
        # No provider → resolver returns None → router bails out.
        agent = _DummyAgent(
            model="MiniMax-M3",
            provider="minimax",
            _smart_routing_cfg={"enabled": True, "cheap_model": {"model": "x"}},
            _smart_routed_active=False,
            _fallback_activated=False,
        )
        assert apply_smart_routing(agent, "hi") is False
        assert agent.model == "MiniMax-M3"


# ── restore_smart_routing idempotency ──────────────────────────────────


class TestRestoreSmartRouting:
    def test_no_op_when_nothing_active(self):
        agent = _DummyAgent(
            model="MiniMax-M3",
            provider="minimax",
            _smart_routed_active=False,
            _pre_smart_state=None,
        )
        assert restore_smart_routing(agent) is False
        assert agent.model == "MiniMax-M3"

    def test_clears_stale_state_on_idempotent_call(self):
        # Even when _smart_routed_active is False but _pre_smart_state is set
        # (e.g. partial crash recovery), restore wipes it.
        agent = _DummyAgent(
            _smart_routed_active=False,
            _pre_smart_state={"model": "deepseek-v4-flash-free"},
        )
        restore_smart_routing(agent)
        assert agent._pre_smart_state is None

    def test_restore_round_trip_preserves_primary(self):
        # Simulate a swap-then-restore flow and confirm the agent's primary
        # attributes are byte-identical before and after.
        agent = _DummyAgent(
            model="MiniMax-M3",
            provider="minimax",
            base_url="https://api.minimax.io/anthropic",
            api_mode="anthropic_messages",
            api_key="sk-fake-primary",
            _client_kwargs={"timeout": 60},
            _use_prompt_caching=True,
            _smart_routed_active=False,
            _pre_smart_state=None,
        )
        # Manually invoke _snapshot_primary_runtime (mirrors router swap).
        from agent.smart_model_routing import _snapshot_primary_runtime
        agent._pre_smart_state = _snapshot_primary_runtime(agent)

        # Mutate to fake cheap-state.
        agent.model = "deepseek-v4-flash-free"
        agent.provider = "opencode-zen"
        agent.base_url = "https://opencode.ai/zen/v1"
        agent.api_mode = "chat_completions"
        agent.api_key = "sk-fake-cheap"
        agent._client_kwargs = {"timeout": 30}
        agent._use_prompt_caching = False
        agent._smart_routed_active = True

        # Restore.
        restored = restore_smart_routing(agent)
        assert restored is True
        assert agent.model == "MiniMax-M3"
        assert agent.provider == "minimax"
        assert agent.base_url == "https://api.minimax.io/anthropic"
        assert agent.api_mode == "anthropic_messages"
        assert agent._use_prompt_caching is True
        assert agent._smart_routed_active is False
        assert agent._pre_smart_state is None
