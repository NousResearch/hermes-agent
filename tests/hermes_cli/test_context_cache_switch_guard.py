"""Tests for the context-cache model-switch guard.

Ported from langchain-ai/deepagents#5829 ("confirm model switches with large
context"): a mid-session model switch abandons the provider prompt cache, so
the first call after the switch re-reads the whole conversation at full input
price. The guard asks for confirmation when the live session exceeds a
configurable token threshold.
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


def _guard(model, ctx, provider="openrouter"):
    with patch("hermes_cli.config.load_config", _no_config):
        return _context_cache_guard(model, provider, None, None, None, ctx)


class TestContextCacheGuard:
    def test_silent_without_selection_context(self):
        assert _guard("new/model", None) is None

    def test_silent_below_threshold(self):
        ctx = SelectionContext(context_tokens=5_000, current_model="old/model")
        assert _guard("new/model", ctx) is None

    def test_fires_above_default_threshold(self):
        ctx = SelectionContext(
            context_tokens=DEFAULT_CONTEXT_CACHE_SWITCH_THRESHOLD + 1,
            current_model="old/model",
        )
        warning = _guard("new/model", ctx)
        assert warning is not None
        assert warning.kind == "context_cache"
        assert "uncached" in warning.message
        assert f"{DEFAULT_CONTEXT_CACHE_SWITCH_THRESHOLD + 1:,}" in warning.message

    def test_same_model_reselect_stays_silent(self):
        ctx = SelectionContext(
            context_tokens=DEFAULT_CONTEXT_CACHE_SWITCH_THRESHOLD * 2,
            current_model="same/model",
        )
        assert _guard("same/model", ctx) is None

    def test_config_threshold_override(self):
        def _cfg():
            return {"model": {"switch_context_confirm_tokens": 10_000}}

        ctx = SelectionContext(context_tokens=20_000, current_model="old/model")
        with patch("hermes_cli.config.load_config", _cfg):
            warning = _context_cache_guard(
                "new/model", "openrouter", None, None, None, ctx
            )
        assert warning is not None

    def test_config_zero_disables(self):
        def _cfg():
            return {"model": {"switch_context_confirm_tokens": 0}}

        ctx = SelectionContext(context_tokens=10**9, current_model="old/model")
        with patch("hermes_cli.config.load_config", _cfg):
            assert (
                _context_cache_guard("new/model", "openrouter", None, None, None, ctx)
                is None
            )

    def test_registry_threads_selection_context(self):
        ctx = SelectionContext(
            context_tokens=DEFAULT_CONTEXT_CACHE_SWITCH_THRESHOLD + 1,
            current_model="old/model",
        )
        with patch("hermes_cli.config.load_config", _no_config):
            warnings = selection_warnings(
                "new/model", provider="openrouter", selection_context=ctx
            )
        assert any(w.kind == "context_cache" for w in warnings)

    def test_registry_silent_without_context(self):
        with patch("hermes_cli.config.load_config", _no_config):
            warnings = selection_warnings("new/model", provider="openrouter")
        assert not any(w.kind == "context_cache" for w in warnings)

    def test_legacy_five_arg_guard_still_supported(self):
        # Externally patched guards with the pre-context 5-arg signature must
        # not break the registry (back-compat TypeError fallback).
        def _old_style(model, provider, base_url, api_key, model_info):
            from hermes_cli.model_selection_guards import SelectionWarning

            return SelectionWarning("cost", "t", model, provider or "", "OLD")

        with patch(
            "hermes_cli.model_selection_guards._GUARDS", (_old_style,)
        ):
            warnings = selection_warnings("m", provider="p")
        assert [w.message for w in warnings] == ["OLD"]


class TestSelectionContextForAgent:
    def test_none_agent(self):
        assert selection_context_for_agent(None) is None

    def test_uses_compressor_measured_tokens(self):
        class _CC:
            last_prompt_tokens = 123_456

        class _Agent:
            context_compressor = _CC()
            model = "current/model"

        ctx = selection_context_for_agent(_Agent())
        assert ctx is not None
        assert ctx.context_tokens == 123_456
        assert ctx.current_model == "current/model"

    def test_falls_back_to_session_prompt_tokens(self):
        class _Agent:
            context_compressor = None
            session_prompt_tokens = 42_000
            model = "current/model"

        ctx = selection_context_for_agent(_Agent())
        assert ctx is not None
        assert ctx.context_tokens == 42_000

    def test_empty_session_returns_none(self):
        class _Agent:
            context_compressor = None
            session_prompt_tokens = 0
            model = "current/model"

        assert selection_context_for_agent(_Agent()) is None
