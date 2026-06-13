"""Tests for ``hermes_cli/fallback_config.py`` — fallback chain resolution."""

from hermes_cli.fallback_config import get_fallback_chain


class TestGetFallbackChain:
    """``get_fallback_chain`` reads fallback entries from config."""

    def test_top_level_fallback_providers(self):
        """Top-level ``fallback_providers`` is still honored (legacy)."""
        config = {
            "fallback_providers": [
                {"provider": "openrouter", "model": "gpt-4o-mini"},
            ],
        }
        chain = get_fallback_chain(config)
        assert len(chain) == 1
        assert chain[0]["provider"] == "openrouter"
        assert chain[0]["model"] == "gpt-4o-mini"

    def test_nested_under_model_is_honored(self):
        """``fallback_providers`` nested under ``model:`` is now read (#45309)."""
        config = {
            "model": {
                "default": "gpt-5.5",
                "provider": "openai-codex",
                "fallback_providers": [
                    {"provider": "openrouter", "model": "gpt-4o-mini"},
                ],
            },
        }
        chain = get_fallback_chain(config)
        assert len(chain) == 1
        assert chain[0]["provider"] == "openrouter"

    def test_both_layers_merged(self):
        """Entries from top-level and ``model:`` are merged; duplicates deduped."""
        config = {
            "fallback_providers": [
                {"provider": "openrouter", "model": "gpt-4o-mini"},
            ],
            "model": {
                "fallback_providers": [
                    {"provider": "anthropic", "model": "claude-sonnet-4"},
                ],
            },
        }
        chain = get_fallback_chain(config)
        assert len(chain) == 2
        providers = [e["provider"] for e in chain]
        assert "openrouter" in providers
        assert "anthropic" in providers

    def test_nested_duplicate_deduped_across_layers(self):
        """Same entry in both layers is only included once."""
        config = {
            "fallback_providers": [
                {"provider": "openrouter", "model": "gpt-4o-mini"},
            ],
            "model": {
                "fallback_providers": [
                    {"provider": "openrouter", "model": "gpt-4o-mini"},
                ],
            },
        }
        chain = get_fallback_chain(config)
        assert len(chain) == 1

    def test_nested_fallback_model(self):
        """``fallback_model`` nested under ``model:`` is also read."""
        config = {
            "model": {
                "fallback_model": {"provider": "deepseek", "model": "deepseek-chat"},
            },
        }
        chain = get_fallback_chain(config)
        assert len(chain) == 1
        assert chain[0]["provider"] == "deepseek"

    def test_empty_config_returns_empty(self):
        """No fallback config → empty chain."""
        assert get_fallback_chain({}) == []
        assert get_fallback_chain(None) == []

    def test_nested_with_missing_model_section(self):
        """No ``model:`` key at all → only top-level is read (no crash)."""
        config = {"fallback_providers": [{"provider": "openrouter", "model": "gpt-4o"}]}
        chain = get_fallback_chain(config)
        assert len(chain) == 1
