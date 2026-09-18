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
        assert "uncached" in warning.message
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


class TestSelectionContextForAgent:
    def test_measured_tokens_then_session_counter_fallback(self):
        class _CC:
            last_prompt_tokens = 123_456

        class _Measured:
            context_compressor = _CC()
            model = "current/model"

        class _Fallback:
            context_compressor = None
            session_prompt_tokens = 42_000
            model = "current/model"

        ctx = selection_context_for_agent(_Measured())
        assert (ctx.context_tokens, ctx.current_model) == (123_456, "current/model")
        assert selection_context_for_agent(_Fallback()).context_tokens == 42_000

    def test_no_agent_or_empty_session_returns_none(self):
        class _Empty:
            context_compressor = None
            session_prompt_tokens = 0
            model = "current/model"

        assert selection_context_for_agent(None) is None
        assert selection_context_for_agent(_Empty()) is None
