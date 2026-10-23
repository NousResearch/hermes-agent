"""Tests for Ollama Cloud authentication and /model switch fixes.

Covers:
- OLLAMA_API_KEY resolution for custom endpoints pointing to ollama.com
- Fallback provider passing base_url/api_key to resolve_provider_client
- /model command updating requested_provider for session persistence
- Direct alias resolution from config.yaml model_aliases
- Reverse lookup: full model names match direct aliases
- /model tab completion for model aliases
"""


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


# ---------------------------------------------------------------------------
# Edge cases: _load_direct_aliases
# ---------------------------------------------------------------------------

class TestLoadDirectAliasesEdgeCases:
    """Edge cases for _load_direct_aliases parsing."""


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
# resolve_alias: fallthrough and edge cases
# ---------------------------------------------------------------------------

class TestResolveAliasEdgeCases:
    """Edge cases for resolve_alias."""


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
# resolve_alias: sort-key date-stamp handling
# ---------------------------------------------------------------------------

class TestResolveAliasSorting:
    """Aliases matching multiple catalog models must NOT silently pick one —
    resolve_alias raises AmbiguousAliasError with candidates sorted
    best-guess-first (dated snapshots demoted below real point versions)."""

    def test_anthropic_opus_ambiguous_lists_candidates(self, monkeypatch):
        """Multiple family matches surface a choice instead of auto-picking;
        the display ordering demotes date-stamped snapshots."""
        import pytest

        import hermes_cli.model_switch as ms

        monkeypatch.setattr("hermes_cli.models._PROVIDER_MODELS", {})
        monkeypatch.setattr(ms, "_ensure_direct_aliases", lambda: None)
        monkeypatch.setattr(ms, "DIRECT_ALIASES", {})
        monkeypatch.setattr(ms, "list_provider_models",
                            lambda p: ["claude-opus-4-1", "claude-opus-4-7",
                                       "claude-opus-4-8",
                                       "claude-opus-4-20250514"])
        with pytest.raises(ms.AmbiguousAliasError) as exc:
            ms.resolve_alias("opus", "anthropic")
        assert exc.value.candidates[0] == "claude-opus-4-8"
        assert set(exc.value.candidates) == {
            "claude-opus-4-1", "claude-opus-4-7",
            "claude-opus-4-8", "claude-opus-4-20250514",
        }

    def test_unsynced_new_model_sorts_first(self, monkeypatch):
        """A just-released model missing from models.dev still ranks above
        older, dated siblings in the candidate ordering."""
        import pytest

        import hermes_cli.model_switch as ms

        monkeypatch.setattr("hermes_cli.models._PROVIDER_MODELS", {})
        monkeypatch.setattr(ms, "_ensure_direct_aliases", lambda: None)
        monkeypatch.setattr(ms, "DIRECT_ALIASES", {})
        monkeypatch.setattr(ms, "list_provider_models",
                            lambda p: ["claude-opus-4-7", "claude-opus-4-8",
                                       "claude-opus-4-20250514",
                                       "claude-opus-4-9"])
        with pytest.raises(ms.AmbiguousAliasError) as exc:
            ms.resolve_alias("opus", "anthropic")
        assert exc.value.candidates[0] == "claude-opus-4-9"

    def test_single_match_resolves_without_error(self, monkeypatch):
        """Exactly one family match still resolves automatically."""
        import hermes_cli.model_switch as ms

        monkeypatch.setattr("hermes_cli.models._PROVIDER_MODELS", {})
        monkeypatch.setattr(ms, "_ensure_direct_aliases", lambda: None)
        monkeypatch.setattr(ms, "DIRECT_ALIASES", {})
        monkeypatch.setattr(ms, "list_provider_models",
                            lambda p: ["claude-opus-4-8", "claude-sonnet-4-6"])
        result = ms.resolve_alias("opus", "anthropic")
        assert result is not None and result[1] == "claude-opus-4-8"

    def test_switch_model_surfaces_ambiguity_message(self, monkeypatch):
        """switch_model returns a failure result listing the candidates
        instead of switching to a heuristic guess."""
        import hermes_cli.model_switch as ms

        monkeypatch.setattr("hermes_cli.models._PROVIDER_MODELS", {})
        monkeypatch.setattr(ms, "_ensure_direct_aliases", lambda: None)
        monkeypatch.setattr(ms, "DIRECT_ALIASES", {})
        monkeypatch.setattr(ms, "list_provider_models",
                            lambda p: ["claude-opus-4-8",
                                       "claude-opus-4-20250514"])
        result = ms.switch_model(
            "opus",
            current_provider="anthropic",
            current_model="claude-sonnet-4-6",
        )
        assert result.success is False
        assert "claude-opus-4-8" in result.error_message
        assert "claude-opus-4-20250514" in result.error_message


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
            lambda raw, prov, *_: ("custom", "qwen3.5:397b", "qwen"))

        monkeypatch.setattr(
            "hermes_cli.runtime_provider.resolve_runtime_provider",
            lambda **kwargs: {"api_key": "", "base_url": "", "api_mode": "openai_compat", "provider": "custom"},
        )

        monkeypatch.setattr("hermes_cli.models_validate.validate_requested_model",
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
            lambda raw, prov, *_: ("custom", "local-model", "local"))
        monkeypatch.setattr(
            "hermes_cli.runtime_provider.resolve_runtime_provider",
            lambda **kwargs: {"api_key": "", "base_url": "", "api_mode": "openai_compat", "provider": "custom"},
        )
        monkeypatch.setattr("hermes_cli.models_validate.validate_requested_model",
            lambda *a, **kw: {"accepted": True, "persist": True, "recognized": True, "message": None})
        monkeypatch.setattr("hermes_cli.models.opencode_model_api_mode",
            lambda *a, **kw: "openai_compat")

        result = ms.switch_model("local", "openrouter", "old-model")
        assert result.success
        assert result.api_key == "no-key-required"
        assert result.base_url == "http://localhost:11434/v1"

    @staticmethod
    def _explicit_switch_to_provider_b(monkeypatch, aliases, explicit="provider-b", extra_cfg=""):
        """``/model shared-model --provider provider-b`` against a real config.yaml, with the
        given direct aliases loaded. Only model validation is stubbed (no network)."""
        import os
        from pathlib import Path

        import hermes_cli.model_switch as ms
        from hermes_cli.config import load_config

        monkeypatch.setenv("PROVIDER_B_KEY", "sk-provider-b")
        (Path(os.environ["HERMES_HOME"]) / "config.yaml").write_text(
            "model:\n  provider: provider-a\n  default: old-model\n"
            "providers:\n"
            "  provider-a:\n    base_url: https://api-a.example.com/v1\n"
            "  provider-b:\n    base_url: https://api-b.example.com/v1\n    key_env: PROVIDER_B_KEY\n"
            + extra_cfg)
        monkeypatch.setattr(ms, "DIRECT_ALIASES", aliases)
        monkeypatch.setattr("hermes_cli.models_validate.validate_requested_model",
            lambda *a, **kw: {"accepted": True, "persist": True, "recognized": True, "message": None})
        cfg = load_config()
        return ms.switch_model(
            "shared-model", "provider-a", "old-model",
            current_base_url="https://api-a.example.com/v1", current_api_key="sk-provider-a",
            explicit_provider=explicit, user_providers=cfg["providers"],
            custom_providers=cfg.get("custom_providers"))

    def test_explicit_provider_never_adopts_alias_bound_to_another_provider(self, monkeypatch):
        """An alias on another provider's endpoint that targets the same model id must not
        outrank --provider: the turn and the credential stay on the provider the user named."""
        from hermes_cli.model_switch import DirectAlias

        result = self._explicit_switch_to_provider_b(monkeypatch, {
            "a-alias": DirectAlias("shared-model", "custom", "https://alias-host.example.com/v1",
                                   api_key="sk-alias-host"),
        })

        assert result.success, result.error_message
        assert result.target_provider == "provider-b"
        assert result.base_url == "https://api-b.example.com/v1"
        assert result.api_key == "sk-provider-b"
        assert result.resolved_via_alias == ""

    def test_explicit_provider_prefers_its_own_alias_for_a_shared_model(self, monkeypatch):
        """Several aliases expose one model id: the one owned by the named provider wins,
        whatever the mapping order (provider spelling is normalized)."""
        from hermes_cli.model_switch import DirectAlias

        result = self._explicit_switch_to_provider_b(monkeypatch, {
            "a-alias": DirectAlias("shared-model", "custom", "https://alias-host.example.com/v1",
                                   api_key="sk-alias-host"),
            "b-alias": DirectAlias("shared-model", "Provider-B", "https://api-b.example.com/v2"),
        })

        assert result.success, result.error_message
        assert result.resolved_via_alias == "b-alias"
        assert result.base_url == "https://api-b.example.com/v2"
        assert result.api_key != "sk-alias-host"

    def test_explicit_provider_keeps_alias_owned_by_legacy_custom_provider(self, monkeypatch):
        """A legacy ``custom_providers`` entry resolves to ``custom:<name>``; an alias that names
        it by its bare name is still that provider's alias and keeps its own endpoint and key."""
        from hermes_cli.model_switch import DirectAlias

        result = self._explicit_switch_to_provider_b(monkeypatch, {
            "a-alias": DirectAlias("shared-model", "provider-a", "https://alias-host.example.com/v1",
                                   api_key="sk-alias-host"),
            "corp-alias": DirectAlias("shared-model", "corp-llm", "https://corp.example.com/v2",
                                      api_key="sk-corp-alias"),
        }, explicit="corp-llm", extra_cfg=(
            "custom_providers:\n  - name: corp-llm\n"
            "    base_url: https://corp.example.com/v1\n    api_key: sk-corp\n"))

        assert result.success, result.error_message
        assert result.target_provider == "custom:corp-llm"
        assert result.resolved_via_alias == "corp-alias"
        assert result.base_url == "https://corp.example.com/v2"
        assert result.api_key == "sk-corp-alias"

    def test_implicit_switch_prefers_alias_of_current_legacy_custom_provider(self, monkeypatch):
        """Without --provider the current provider still owns a shared model id: on
        ``custom:corp-llm`` the alias naming ``corp-llm`` wins over another provider's alias."""
        import os
        from pathlib import Path

        import hermes_cli.model_switch as ms
        from hermes_cli.config import load_config
        from hermes_cli.model_switch import DirectAlias

        (Path(os.environ["HERMES_HOME"]) / "config.yaml").write_text(
            "model:\n  provider: custom:corp-llm\n  default: old-model\n"
            "providers:\n  provider-a:\n    base_url: https://api-a.example.com/v1\n"
            "custom_providers:\n  - name: corp-llm\n"
            "    base_url: https://corp.example.com/v1\n    api_key: sk-corp\n")
        monkeypatch.setattr(ms, "DIRECT_ALIASES", {
            "a-alias": DirectAlias("shared-model", "provider-a", "https://alias-host.example.com/v1",
                                   api_key="sk-alias-host"),
            "corp-alias": DirectAlias("shared-model", "corp-llm", "https://corp.example.com/v2",
                                      api_key="sk-corp-alias"),
        })
        monkeypatch.setattr("hermes_cli.models_validate.validate_requested_model",
            lambda *a, **kw: {"accepted": True, "persist": True, "recognized": True, "message": None})
        cfg = load_config()

        result = ms.switch_model(
            "shared-model", "custom:corp-llm", "old-model",
            current_base_url="https://corp.example.com/v1", current_api_key="sk-corp",
            user_providers=cfg["providers"], custom_providers=cfg.get("custom_providers"))

        assert result.success, result.error_message
        assert result.resolved_via_alias == "corp-alias"
        assert result.base_url == "https://corp.example.com/v2"
        assert result.api_key == "sk-corp-alias"
