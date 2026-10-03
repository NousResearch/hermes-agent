"""Context-cache model-switch guard.

A mid-session model switch abandons the provider prompt cache, so the first call after the switch
re-reads the whole conversation at full input price. The guard asks for confirmation only when the
live session exceeds a configurable token threshold.
"""

from unittest.mock import patch

from hermes_cli.model_selection_guards import (
    DEFAULT_CONTEXT_CACHE_SWITCH_THRESHOLD,
    SelectionContext,
    _context_cache_guard,
    selection_context_for_agent,
    selection_warnings,
)


def _no_config(*_a, **_k):
    raise FileNotFoundError("no config in tests")


def _guard(model, ctx, cfg=_no_config):
    with patch("hermes_cli.config.load_config", cfg):
        return _context_cache_guard(model, "openrouter", None, None, None, ctx)


class TestContextCacheGuard:
    def test_silent_without_context_or_below_threshold(self):
        assert _guard("new/model", None) is None
        assert _guard("new/model", SelectionContext(context_tokens=5_000, current_model="old/model")) is None

    def test_fires_above_default_threshold(self):
        tokens = DEFAULT_CONTEXT_CACHE_SWITCH_THRESHOLD + 1
        warning = _guard("new/model", SelectionContext(context_tokens=tokens, current_model="old/model"))
        assert warning is not None
        assert warning.kind == "context_cache"
        assert f"{tokens:,}" in warning.message

    def test_same_model_reselect_stays_silent(self):
        ctx = SelectionContext(context_tokens=DEFAULT_CONTEXT_CACHE_SWITCH_THRESHOLD * 2, current_model="same/model")
        assert _guard("same/model", ctx) is None

    def test_config_threshold_override_and_zero_disables(self):
        ctx = SelectionContext(context_tokens=20_000, current_model="old/model")
        assert _guard("new/model", ctx, lambda: {"model": {"switch_context_confirm_tokens": 10_000}}) is not None
        huge = SelectionContext(context_tokens=10**9, current_model="old/model")
        assert _guard("new/model", huge, lambda: {"model": {"switch_context_confirm_tokens": 0}}) is None

    def test_registry_threads_selection_context(self):
        ctx = SelectionContext(context_tokens=DEFAULT_CONTEXT_CACHE_SWITCH_THRESHOLD + 1, current_model="old/model")
        with patch("hermes_cli.config.load_config", _no_config):
            with_ctx = selection_warnings("new/model", provider="openrouter", selection_context=ctx)
            without = selection_warnings("new/model", provider="openrouter")
        assert any(w.kind == "context_cache" for w in with_ctx)
        assert not any(w.kind == "context_cache" for w in without)


    def test_a_weaker_figure_is_not_quoted_as_the_context_size(self):
        """R3: the confirmation quotes the evidence the summary would. A session total is a total and
        a display seed is a local estimate — neither is the size of what the next reply re-reads, so
        neither may be worded as the context it holds."""
        total = SelectionContext(
            context_tokens=DEFAULT_CONTEXT_CACHE_SWITCH_THRESHOLD * 2,
            context_tokens_source="counter", current_model="old/model")
        seed = SelectionContext(
            context_tokens=DEFAULT_CONTEXT_CACHE_SWITCH_THRESHOLD * 2,
            context_tokens_source="estimate", current_model="old/model")

        for ctx, phrase in ((total, "prompt tokens in total"), (seed, "local estimate")):
            warning = _guard("new/model", ctx)
            assert warning is not None
            assert phrase in warning.message
            assert "This session holds ~" not in warning.message

    def test_the_move_is_stated_as_a_condition_not_a_cache_or_billing_promise(self):
        """R3: deployment inequality is neither cache-residency evidence nor a billing mode, so the
        cost line keeps the condition the summary states instead of guaranteeing an uncached,
        full-price re-read."""
        warning = _guard("new/model", SelectionContext(
            context_tokens=DEFAULT_CONTEXT_CACHE_SWITCH_THRESHOLD + 1, current_model="old/model"))

        assert "if new/model has not served it there is no warm prefix cache" in warning.message
        assert "full-price" not in warning.message


class TestSelectionContextForAgent:
    def test_measured_tokens_then_session_counter_fallback(self):
        class _CC:
            last_real_prompt_tokens = 123_456
            last_prompt_tokens = 123_456

        class _Measured:
            context_compressor = _CC()
            model = "current/model"

        class _Fallback:
            context_compressor = None
            session_prompt_tokens = 42_000
            model = "current/model"

        ctx = selection_context_for_agent(_Measured())
        assert (ctx.context_tokens, ctx.context_tokens_source) == (123_456, "measured")
        assert ctx.current_model == "current/model"
        fallback = selection_context_for_agent(_Fallback())
        assert (fallback.context_tokens, fallback.context_tokens_source) == (42_000, "counter")

    def test_a_display_seed_is_an_estimate_not_a_measurement(self):
        """``update_model`` clears the provider reading, so a seed written afterwards
        (``maybe_seed_preflight_display_tokens``) is the compressor's only figure and it is a local
        estimate — the class the summary quotes it with, and now the confirmation too."""

        class _CC:
            last_real_prompt_tokens = 0
            last_prompt_tokens = 200_000

        class _Seeded:
            context_compressor = _CC()
            model = "current/model"

        ctx = selection_context_for_agent(_Seeded())

        assert (ctx.context_tokens, ctx.context_tokens_source) == (200_000, "estimate")

    def test_no_agent_or_empty_session_returns_none(self):
        class _Empty:
            context_compressor = None
            session_prompt_tokens = 0
            model = "current/model"

        assert selection_context_for_agent(None) is None
        assert selection_context_for_agent(_Empty()) is None


def test_route_facts_ride_along_with_the_measured_size():
    """The confirmation decides a transition, so the context it is handed carries the whole route —
    a model string alone cannot tell a reselect from an endpoint move."""

    class _Agent:
        context_compressor = None
        session_prompt_tokens = 42_000
        model = "current/model"
        provider = "openrouter"
        base_url = "https://a.example/v1"

    ctx = selection_context_for_agent(_Agent())

    assert (ctx.current_model, ctx.current_provider, ctx.current_base_url) == (
        "current/model", "openrouter", "https://a.example/v1")
