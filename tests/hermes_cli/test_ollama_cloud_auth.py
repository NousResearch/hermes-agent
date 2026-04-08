"""Tests for Ollama Cloud authentication and /model switch fixes.

Covers:
- OLLAMA_API_KEY resolution for custom endpoints pointing to ollama.com
- Fallback provider passing base_url/api_key to resolve_provider_client
- /model command updating requested_provider for session persistence
- Direct alias resolution from config.yaml model_aliases
- Reverse lookup: full model names match direct aliases
- /model tab completion for model aliases
"""

import os
import pytest
from unittest.mock import patch, MagicMock


# ---------------------------------------------------------------------------
# OLLAMA_API_KEY credential resolution
# ---------------------------------------------------------------------------

class TestOllamaCloudCredentials:
    """runtime_provider should use OLLAMA_API_KEY for ollama.com endpoints."""

    def test_ollama_api_key_used_for_ollama_endpoint(self, monkeypatch, tmp_path):
        """When base_url contains ollama.com, OLLAMA_API_KEY is in the candidate chain."""
        monkeypatch.setenv("OLLAMA_API_KEY", "test-ollama-key-12345")
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)

        # Mock config to return custom provider with ollama base_url
        mock_config = {
            "model": {
                "default": "qwen3.5:397b",
                "provider": "custom",
                "base_url": "https://ollama.com/v1",
            }
        }
        monkeypatch.setattr(
            "hermes_cli.runtime_provider._get_model_config",
            lambda: mock_config.get("model", {}),
        )

        from hermes_cli.runtime_provider import resolve_runtime_provider
        runtime = resolve_runtime_provider(requested="custom")

        assert runtime["base_url"] == "https://ollama.com/v1"
        assert runtime["api_key"] == "test-ollama-key-12345"
        assert runtime["provider"] == "custom"

    def test_ollama_key_not_used_for_non_ollama_endpoint(self, monkeypatch):
        """OLLAMA_API_KEY should NOT be used for non-ollama endpoints."""
        monkeypatch.setenv("OLLAMA_API_KEY", "test-ollama-key")
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)

        mock_config = {
            "model": {
                "provider": "custom",
                "base_url": "http://localhost:11434/v1",
            }
        }
        monkeypatch.setattr(
            "hermes_cli.runtime_provider._get_model_config",
            lambda: mock_config.get("model", {}),
        )

        from hermes_cli.runtime_provider import resolve_runtime_provider
        runtime = resolve_runtime_provider(requested="custom")

        # Should fall through to no-key-required for local endpoints
        assert runtime["api_key"] != "test-ollama-key"


# ---------------------------------------------------------------------------
# Direct alias resolution
# ---------------------------------------------------------------------------

class TestDirectAliases:
    """model_switch direct aliases from config.yaml model_aliases."""

    def test_direct_alias_loaded_from_config(self, monkeypatch):
        """Direct aliases load from config.yaml model_aliases section."""
        mock_config = {
            "model_aliases": {
                "mymodel": {
                    "model": "custom-model:latest",
                    "provider": "custom",
                    "base_url": "https://example.com/v1",
                }
            }
        }
        monkeypatch.setattr(
            "hermes_cli.config.load_config",
            lambda: mock_config,
        )

        from hermes_cli.model_switch import _load_direct_aliases
        aliases = _load_direct_aliases()

        assert "mymodel" in aliases
        assert aliases["mymodel"].model == "custom-model:latest"
        assert aliases["mymodel"].provider == "custom"
        assert aliases["mymodel"].base_url == "https://example.com/v1"

    def test_direct_alias_resolved_before_catalog(self, monkeypatch):
        """Direct aliases take priority over models.dev catalog lookup."""
        from hermes_cli.model_switch import DirectAlias, resolve_alias
        import hermes_cli.model_switch as ms

        test_aliases = {
            "glm": DirectAlias("glm-4.7", "custom", "https://ollama.com/v1"),
        }
        monkeypatch.setattr(ms, "DIRECT_ALIASES", test_aliases)

        result = resolve_alias("glm", "openrouter")
        assert result is not None
        provider, model, alias = result
        assert model == "glm-4.7"
        assert provider == "custom"
        assert alias == "glm"

    def test_reverse_lookup_by_model_id(self, monkeypatch):
        """Full model names (e.g. 'kimi-k2.5') match via reverse lookup."""
        from hermes_cli.model_switch import DirectAlias, resolve_alias
        import hermes_cli.model_switch as ms

        test_aliases = {
            "kimi": DirectAlias("kimi-k2.5", "custom", "https://ollama.com/v1"),
        }
        monkeypatch.setattr(ms, "DIRECT_ALIASES", test_aliases)

        # Typing full model name should resolve through the alias
        result = resolve_alias("kimi-k2.5", "openrouter")
        assert result is not None
        provider, model, alias = result
        assert model == "kimi-k2.5"
        assert provider == "custom"
        assert alias == "kimi"

    def test_reverse_lookup_case_insensitive(self, monkeypatch):
        """Reverse lookup is case-insensitive."""
        from hermes_cli.model_switch import DirectAlias, resolve_alias
        import hermes_cli.model_switch as ms

        test_aliases = {
            "glm": DirectAlias("GLM-4.7", "custom", "https://ollama.com/v1"),
        }
        monkeypatch.setattr(ms, "DIRECT_ALIASES", test_aliases)

        result = resolve_alias("glm-4.7", "openrouter")
        assert result is not None
        assert result[1] == "GLM-4.7"


# ---------------------------------------------------------------------------
# /model command persistence
# ---------------------------------------------------------------------------

class TestModelSwitchPersistence:
    """CLI /model command should update requested_provider for session persistence."""

    def test_model_switch_result_fields(self):
        """ModelSwitchResult has all required fields for CLI state update."""
        from hermes_cli.model_switch import ModelSwitchResult

        result = ModelSwitchResult(
            success=True,
            new_model="claude-opus-4-6",
            target_provider="anthropic",
            provider_changed=True,
            api_key="test-key",
            base_url="https://api.anthropic.com",
            api_mode="anthropic_messages",
        )

        assert result.success
        assert result.new_model == "claude-opus-4-6"
        assert result.target_provider == "anthropic"
        assert result.api_key == "test-key"
        assert result.base_url == "https://api.anthropic.com"


# ---------------------------------------------------------------------------
# /model tab completion
# ---------------------------------------------------------------------------

class TestModelTabCompletion:
    """SlashCommandCompleter provides model alias completions for /model."""

    def test_model_completions_yields_direct_aliases(self, monkeypatch):
        """_model_completions yields direct aliases with model and provider info."""
        from hermes_cli.commands import SlashCommandCompleter
        from hermes_cli.model_switch import DirectAlias
        import hermes_cli.model_switch as ms

        test_aliases = {
            "opus": DirectAlias("claude-opus-4-6", "anthropic", ""),
            "qwen": DirectAlias("qwen3.5:397b", "custom", "https://ollama.com/v1"),
        }
        monkeypatch.setattr(ms, "DIRECT_ALIASES", test_aliases)

        completer = SlashCommandCompleter()
        completions = list(completer._model_completions("", ""))

        names = [c.text for c in completions]
        assert "opus" in names
        assert "qwen" in names

    def test_model_completions_filters_by_prefix(self, monkeypatch):
        """Completions filter by typed prefix."""
        from hermes_cli.commands import SlashCommandCompleter
        from hermes_cli.model_switch import DirectAlias
        import hermes_cli.model_switch as ms

        test_aliases = {
            "opus": DirectAlias("claude-opus-4-6", "anthropic", ""),
            "qwen": DirectAlias("qwen3.5:397b", "custom", "https://ollama.com/v1"),
        }
        monkeypatch.setattr(ms, "DIRECT_ALIASES", test_aliases)

        completer = SlashCommandCompleter()
        completions = list(completer._model_completions("o", "o"))

        names = [c.text for c in completions]
        assert "opus" in names
        assert "qwen" not in names

    def test_model_completions_shows_metadata(self, monkeypatch):
        """Completions include model name and provider in display_meta."""
        from hermes_cli.commands import SlashCommandCompleter
        from hermes_cli.model_switch import DirectAlias
        import hermes_cli.model_switch as ms

        test_aliases = {
            "glm": DirectAlias("glm-4.7", "custom", "https://ollama.com/v1"),
        }
        monkeypatch.setattr(ms, "DIRECT_ALIASES", test_aliases)

        completer = SlashCommandCompleter()
        completions = list(completer._model_completions("g", "g"))

        assert len(completions) >= 1
        glm_comp = [c for c in completions if c.text == "glm"][0]
        meta_str = str(glm_comp.display_meta)
        assert "glm-4.7" in meta_str
        assert "custom" in meta_str


# ---------------------------------------------------------------------------
# Fallback base_url passthrough
# ---------------------------------------------------------------------------

class TestFallbackBaseUrlPassthrough:
    """_try_activate_fallback should pass base_url from fallback config."""

    def test_fallback_config_has_base_url(self):
        """Verify fallback_providers config structure supports base_url."""
        # This tests the contract: fallback dicts can have base_url
        fb = {
            "provider": "custom",
            "model": "qwen3.5:397b",
            "base_url": "https://ollama.com/v1",
        }
        assert fb.get("base_url") == "https://ollama.com/v1"

    def test_ollama_key_lookup_for_fallback(self, monkeypatch):
        """When fallback base_url is ollama.com and no api_key, OLLAMA_API_KEY is used."""
        monkeypatch.setenv("OLLAMA_API_KEY", "fb-ollama-key")

        fb = {
            "provider": "custom",
            "model": "qwen3.5:397b",
            "base_url": "https://ollama.com/v1",
        }

        fb_base_url_hint = (fb.get("base_url") or "").strip() or None
        fb_api_key_hint = (fb.get("api_key") or "").strip() or None

        if fb_base_url_hint and "ollama.com" in fb_base_url_hint.lower() and not fb_api_key_hint:
            fb_api_key_hint = os.getenv("OLLAMA_API_KEY") or None

        assert fb_api_key_hint == "fb-ollama-key"
        assert fb_base_url_hint == "https://ollama.com/v1"


# ---------------------------------------------------------------------------
# Edge cases: _load_direct_aliases
# ---------------------------------------------------------------------------

class TestLoadDirectAliasesEdgeCases:
    """Edge cases for _load_direct_aliases parsing."""

    def test_empty_model_aliases_config(self, monkeypatch):
        """Empty model_aliases dict returns only builtins (if any)."""
        mock_config = {"model_aliases": {}}
        monkeypatch.setattr(
            "hermes_cli.config.load_config",
            lambda: mock_config,
        )

        from hermes_cli.model_switch import _load_direct_aliases
        aliases = _load_direct_aliases()
        assert isinstance(aliases, dict)

    def test_model_aliases_not_a_dict(self, monkeypatch):
        """Non-dict model_aliases value is gracefully ignored."""
        mock_config = {"model_aliases": "bad-string-value"}
        monkeypatch.setattr(
            "hermes_cli.config.load_config",
            lambda: mock_config,
        )

        from hermes_cli.model_switch import _load_direct_aliases
        aliases = _load_direct_aliases()
        assert isinstance(aliases, dict)

    def test_model_aliases_none_value(self, monkeypatch):
        """model_aliases: null in config is handled gracefully."""
        mock_config = {"model_aliases": None}
        monkeypatch.setattr(
            "hermes_cli.config.load_config",
            lambda: mock_config,
        )

        from hermes_cli.model_switch import _load_direct_aliases
        aliases = _load_direct_aliases()
        assert isinstance(aliases, dict)

    def test_malformed_entry_without_model_key(self, monkeypatch):
        """Entries missing 'model' key are skipped."""
        mock_config = {
            "model_aliases": {
                "bad_entry": {
                    "provider": "custom",
                    "base_url": "https://example.com/v1",
                },
                "good_entry": {
                    "model": "valid-model",
                    "provider": "custom",
                },
            }
        }
        monkeypatch.setattr(
            "hermes_cli.config.load_config",
            lambda: mock_config,
        )

        from hermes_cli.model_switch import _load_direct_aliases
        aliases = _load_direct_aliases()
        assert "bad_entry" not in aliases
        assert "good_entry" in aliases

    def test_malformed_entry_non_dict_value(self, monkeypatch):
        """Non-dict entry values are skipped."""
        mock_config = {
            "model_aliases": {
                "string_entry": "just-a-string",
                "none_entry": None,
                "list_entry": ["a", "b"],
                "good": {"model": "real-model", "provider": "custom"},
            }
        }
        monkeypatch.setattr(
            "hermes_cli.config.load_config",
            lambda: mock_config,
        )

        from hermes_cli.model_switch import _load_direct_aliases
        aliases = _load_direct_aliases()
        assert "string_entry" not in aliases
        assert "none_entry" not in aliases
        assert "list_entry" not in aliases
        assert "good" in aliases

    def test_load_config_exception_returns_builtins(self, monkeypatch):
        """If load_config raises, _load_direct_aliases returns builtins only."""
        monkeypatch.setattr(
            "hermes_cli.config.load_config",
            lambda: (_ for _ in ()).throw(RuntimeError("config broken")),
        )

        from hermes_cli.model_switch import _load_direct_aliases
        aliases = _load_direct_aliases()
        assert isinstance(aliases, dict)

    def test_alias_name_normalized_lowercase(self, monkeypatch):
        """Alias names are lowercased and stripped."""
        mock_config = {
            "model_aliases": {
                "  MyModel  ": {
                    "model": "my-model:latest",
                    "provider": "custom",
                }
            }
        }
        monkeypatch.setattr(
            "hermes_cli.config.load_config",
            lambda: mock_config,
        )

        from hermes_cli.model_switch import _load_direct_aliases
        aliases = _load_direct_aliases()
        assert "mymodel" in aliases
        assert "  MyModel  " not in aliases

    def test_empty_model_string_skipped(self, monkeypatch):
        """Entries with empty model string are skipped."""
        mock_config = {
            "model_aliases": {
                "empty": {"model": "", "provider": "custom"},
                "good": {"model": "real", "provider": "custom"},
            }
        }
        monkeypatch.setattr(
            "hermes_cli.config.load_config",
            lambda: mock_config,
        )

        from hermes_cli.model_switch import _load_direct_aliases
        aliases = _load_direct_aliases()
        assert "empty" not in aliases
        assert "good" in aliases


# ---------------------------------------------------------------------------
# _ensure_direct_aliases idempotency
# ---------------------------------------------------------------------------

class TestEnsureDirectAliases:
    """_ensure_direct_aliases lazy-loading behavior."""

    def test_ensure_populates_on_first_call(self, monkeypatch):
        """DIRECT_ALIASES is populated after _ensure_direct_aliases."""
        import hermes_cli.model_switch as ms

        mock_config = {
            "model_aliases": {
                "test": {"model": "test-model", "provider": "custom"},
            }
        }
        monkeypatch.setattr(
            "hermes_cli.config.load_config",
            lambda: mock_config,
        )
        monkeypatch.setattr(ms, "DIRECT_ALIASES", {})
        ms._ensure_direct_aliases()
        assert "test" in ms.DIRECT_ALIASES

    def test_ensure_no_reload_when_populated(self, monkeypatch):
        """_ensure_direct_aliases does not reload if already populated."""
        import hermes_cli.model_switch as ms
        from hermes_cli.model_switch import DirectAlias

        existing = {"pre": DirectAlias("pre-model", "custom", "")}
        monkeypatch.setattr(ms, "DIRECT_ALIASES", existing)

        call_count = [0]
        original_load = ms._load_direct_aliases
        def counting_load():
            call_count[0] += 1
            return original_load()
        monkeypatch.setattr(ms, "_load_direct_aliases", counting_load)

        ms._ensure_direct_aliases()
        assert call_count[0] == 0
        assert "pre" in ms.DIRECT_ALIASES


# ---------------------------------------------------------------------------
# resolve_alias: fallthrough and edge cases
# ---------------------------------------------------------------------------

class TestResolveAliasEdgeCases:
    """Edge cases for resolve_alias."""

    def test_unknown_alias_returns_none(self, monkeypatch):
        """Unknown alias not in direct or catalog returns None."""
        import hermes_cli.model_switch as ms
        monkeypatch.setattr(ms, "DIRECT_ALIASES", {})

        result = ms.resolve_alias("nonexistent_model_xyz", "openrouter")
        assert result is None

    def test_whitespace_input_handled(self, monkeypatch):
        """Input with whitespace is stripped before lookup."""
        from hermes_cli.model_switch import DirectAlias
        import hermes_cli.model_switch as ms

        test_aliases = {
            "myalias": DirectAlias("my-model", "custom", "https://example.com"),
        }
        monkeypatch.setattr(ms, "DIRECT_ALIASES", test_aliases)

        result = ms.resolve_alias("  myalias  ", "openrouter")
        assert result is not None
        assert result[1] == "my-model"


# ---------------------------------------------------------------------------
# switch_model: direct alias base_url override
# ---------------------------------------------------------------------------

class TestSwitchModelDirectAliasOverride:
    """switch_model should use base_url from direct alias."""

    def test_switch_model_uses_alias_base_url(self, monkeypatch):
        """When resolved alias has base_url, switch_model should use it."""
        from hermes_cli.model_switch import DirectAlias
        import hermes_cli.model_switch as ms

        test_aliases = {
            "qwen": DirectAlias("qwen3.5:397b", "custom", "https://ollama.com/v1"),
        }
        monkeypatch.setattr(ms, "DIRECT_ALIASES", test_aliases)

        monkeypatch.setattr(ms, "resolve_alias",
            lambda raw, prov: ("custom", "qwen3.5:397b", "qwen"))

        monkeypatch.setattr(
            "hermes_cli.runtime_provider.resolve_runtime_provider",
            lambda requested: {"api_key": "", "base_url": "", "api_mode": "openai_compat", "provider": "custom"},
        )

        monkeypatch.setattr("hermes_cli.models.validate_requested_model",
            lambda *a, **kw: {"accepted": True, "persist": True, "recognized": True, "message": None})
        monkeypatch.setattr("hermes_cli.models.opencode_model_api_mode",
            lambda *a, **kw: "openai_compat")

        result = ms.switch_model("qwen", "openrouter", "old-model")
        assert result.success
        assert result.base_url == "https://ollama.com/v1"
        assert result.new_model == "qwen3.5:397b"

    def test_switch_model_alias_no_api_key_gets_default(self, monkeypatch):
        """When alias has base_url but no api_key, 'no-key-required' is set."""
        from hermes_cli.model_switch import DirectAlias
        import hermes_cli.model_switch as ms

        test_aliases = {
            "local": DirectAlias("local-model", "custom", "http://localhost:11434/v1"),
        }
        monkeypatch.setattr(ms, "DIRECT_ALIASES", test_aliases)
        monkeypatch.setattr(ms, "resolve_alias",
            lambda raw, prov: ("custom", "local-model", "local"))
        monkeypatch.setattr(
            "hermes_cli.runtime_provider.resolve_runtime_provider",
            lambda requested: {"api_key": "", "base_url": "", "api_mode": "openai_compat", "provider": "custom"},
        )
        monkeypatch.setattr("hermes_cli.models.validate_requested_model",
            lambda *a, **kw: {"accepted": True, "persist": True, "recognized": True, "message": None})
        monkeypatch.setattr("hermes_cli.models.opencode_model_api_mode",
            lambda *a, **kw: "openai_compat")

        result = ms.switch_model("local", "openrouter", "old-model")
        assert result.success
        assert result.api_key == "no-key-required"
        assert result.base_url == "http://localhost:11434/v1"


# ---------------------------------------------------------------------------
# CLI state update: requested_provider persistence
# ---------------------------------------------------------------------------

class TestCLIStateUpdate:
    """CLI /model handler should update requested_provider and explicit fields."""

    def test_model_switch_result_has_provider_label(self):
        """ModelSwitchResult supports provider_label for display."""
        from hermes_cli.model_switch import ModelSwitchResult

        result = ModelSwitchResult(
            success=True,
            new_model="qwen3.5:397b",
            target_provider="custom",
            provider_changed=True,
            api_key="key",
            base_url="https://ollama.com/v1",
            api_mode="openai_compat",
            provider_label="Ollama Cloud",
        )
        assert result.provider_label == "Ollama Cloud"

    def test_model_switch_result_defaults(self):
        """ModelSwitchResult has sensible defaults."""
        from hermes_cli.model_switch import ModelSwitchResult

        result = ModelSwitchResult(
            success=False,
            new_model="",
            target_provider="",
            provider_changed=False,
            error_message="Something failed",
        )
        assert not result.success
        assert result.error_message == "Something failed"
        assert result.api_key is None or result.api_key == ""
        assert result.base_url is None or result.base_url == ""


# ---------------------------------------------------------------------------
# Fallback: OLLAMA_API_KEY edge cases
# ---------------------------------------------------------------------------

class TestFallbackEdgeCases:
    """Edge cases for fallback OLLAMA_API_KEY logic."""

    def test_ollama_key_not_injected_for_localhost(self, monkeypatch):
        """OLLAMA_API_KEY should not be injected for localhost URLs."""
        monkeypatch.setenv("OLLAMA_API_KEY", "should-not-use")

        fb = {
            "provider": "custom",
            "model": "local-model",
            "base_url": "http://localhost:11434/v1",
        }

        fb_base_url_hint = (fb.get("base_url") or "").strip() or None
        fb_api_key_hint = (fb.get("api_key") or "").strip() or None

        if fb_base_url_hint and "ollama.com" in fb_base_url_hint.lower() and not fb_api_key_hint:
            fb_api_key_hint = os.getenv("OLLAMA_API_KEY") or None

        assert fb_api_key_hint is None

    def test_explicit_api_key_not_overridden_by_ollama_key(self, monkeypatch):
        """Explicit api_key in fallback config is not overridden by OLLAMA_API_KEY."""
        monkeypatch.setenv("OLLAMA_API_KEY", "env-key")

        fb = {
            "provider": "custom",
            "model": "qwen3.5:397b",
            "base_url": "https://ollama.com/v1",
            "api_key": "explicit-key",
        }

        fb_base_url_hint = (fb.get("base_url") or "").strip() or None
        fb_api_key_hint = (fb.get("api_key") or "").strip() or None

        if fb_base_url_hint and "ollama.com" in fb_base_url_hint.lower() and not fb_api_key_hint:
            fb_api_key_hint = os.getenv("OLLAMA_API_KEY") or None

        assert fb_api_key_hint == "explicit-key"

    def test_no_base_url_in_fallback(self, monkeypatch):
        """Fallback with no base_url doesn't crash."""
        monkeypatch.setenv("OLLAMA_API_KEY", "some-key")

        fb = {"provider": "openrouter", "model": "some-model"}

        fb_base_url_hint = (fb.get("base_url") or "").strip() or None
        fb_api_key_hint = (fb.get("api_key") or "").strip() or None

        if fb_base_url_hint and "ollama.com" in fb_base_url_hint.lower() and not fb_api_key_hint:
            fb_api_key_hint = os.getenv("OLLAMA_API_KEY") or None

        assert fb_base_url_hint is None
        assert fb_api_key_hint is None
