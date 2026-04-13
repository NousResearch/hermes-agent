"""Tests for MiniMax provider hardening — context lengths, thinking guard, catalog."""


class TestMinimaxContextLengths:
    """Verify per-model context length entries for MiniMax models."""

    def test_m1_variants_have_1m_context(self):
        from agent.model_metadata import DEFAULT_CONTEXT_LENGTHS
        # Keys are lowercase because the lookup lowercases model names
        for model in ("minimax-m1", "minimax-m1-40k", "minimax-m1-80k",
                       "minimax-m1-128k", "minimax-m1-256k"):
            assert model in DEFAULT_CONTEXT_LENGTHS, f"{model} missing from context lengths"
            assert DEFAULT_CONTEXT_LENGTHS[model] == 1_000_000, f"{model} expected 1M"

    def test_m2_variants_have_1m_context(self):
        from agent.model_metadata import DEFAULT_CONTEXT_LENGTHS
        # Keys are lowercase because the lookup lowercases model names
        for model in ("minimax-m2.5", "minimax-m2.7"):
            assert model in DEFAULT_CONTEXT_LENGTHS, f"{model} missing from context lengths"
            assert DEFAULT_CONTEXT_LENGTHS[model] == 1_048_576, f"{model} expected 1048576"

    def test_minimax_prefix_fallback(self):
        from agent.model_metadata import DEFAULT_CONTEXT_LENGTHS
        # The generic "minimax" prefix entry should be 1M for unknown models
        assert DEFAULT_CONTEXT_LENGTHS["minimax"] == 1_048_576



class TestMinimaxThinkingGuard:
    """Verify that build_anthropic_kwargs does NOT add thinking params for MiniMax models."""

    def test_no_thinking_for_minimax_m27(self):
        from agent.anthropic_adapter import build_anthropic_kwargs
        kwargs = build_anthropic_kwargs(
            model="MiniMax-M2.7",
            messages=[{"role": "user", "content": "hello"}],
            tools=None,
            max_tokens=4096,
            reasoning_config={"enabled": True, "effort": "medium"},
        )
        assert "thinking" not in kwargs
        assert "output_config" not in kwargs

    def test_no_thinking_for_minimax_m1(self):
        from agent.anthropic_adapter import build_anthropic_kwargs
        kwargs = build_anthropic_kwargs(
            model="MiniMax-M1-128k",
            messages=[{"role": "user", "content": "hello"}],
            tools=None,
            max_tokens=4096,
            reasoning_config={"enabled": True, "effort": "high"},
        )
        assert "thinking" not in kwargs

    def test_thinking_still_works_for_claude(self):
        from agent.anthropic_adapter import build_anthropic_kwargs
        kwargs = build_anthropic_kwargs(
            model="claude-sonnet-4-20250514",
            messages=[{"role": "user", "content": "hello"}],
            tools=None,
            max_tokens=4096,
            reasoning_config={"enabled": True, "effort": "medium"},
        )
        assert "thinking" in kwargs


class TestMinimaxAuxModel:
    """Verify auxiliary model is standard (not highspeed)."""

    def test_minimax_aux_is_standard(self):
        from agent.auxiliary_client import _API_KEY_PROVIDER_AUX_MODELS
        assert _API_KEY_PROVIDER_AUX_MODELS["minimax"] == "MiniMax-M2.7"
        assert _API_KEY_PROVIDER_AUX_MODELS["minimax-cn"] == "MiniMax-M2.7"

    def test_minimax_aux_not_highspeed(self):
        from agent.auxiliary_client import _API_KEY_PROVIDER_AUX_MODELS
        assert "highspeed" not in _API_KEY_PROVIDER_AUX_MODELS["minimax"]
        assert "highspeed" not in _API_KEY_PROVIDER_AUX_MODELS["minimax-cn"]


class TestMinimaxModelCatalog:
    """Verify the model catalog includes M1 family and excludes deprecated models."""

    def test_catalog_includes_m1_family(self):
        from hermes_cli.models import _PROVIDER_MODELS
        for provider in ("minimax", "minimax-cn"):
            models = _PROVIDER_MODELS[provider]
            assert "MiniMax-M1" in models
            assert "MiniMax-M1-40k" in models
            assert "MiniMax-M1-80k" in models
            assert "MiniMax-M1-128k" in models
            assert "MiniMax-M1-256k" in models

    def test_catalog_excludes_deprecated(self):
        from hermes_cli.models import _PROVIDER_MODELS
        for provider in ("minimax", "minimax-cn"):
            models = _PROVIDER_MODELS[provider]
            assert "MiniMax-M2.1" not in models

    def test_catalog_excludes_highspeed(self):
        from hermes_cli.models import _PROVIDER_MODELS
        for provider in ("minimax", "minimax-cn"):
            models = _PROVIDER_MODELS[provider]
            assert "MiniMax-M2.7-highspeed" not in models
            assert "MiniMax-M2.5-highspeed" not in models
