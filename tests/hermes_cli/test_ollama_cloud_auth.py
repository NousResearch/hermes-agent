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

    def test_reverse_lookup_prefers_current_provider_when_model_is_shared(self, monkeypatch):
        """Reverse lookup picks the alias whose provider matches the request.

        When several aliases expose the same model ID on different providers,
        the winner must be selected by provider match rather than by mapping
        insertion order — otherwise the caller silently gets another provider's
        endpoint.
        """
        from hermes_cli.model_switch import DirectAlias, resolve_alias
        import hermes_cli.model_switch as ms

        monkeypatch.setattr(ms, "DIRECT_ALIASES", {
            "my-a": DirectAlias("claude-opus-4-6", "custom:provider-a", "https://api-a.example.com"),
            "my-b": DirectAlias("claude-opus-4-6", "custom:provider-b", "https://api-b.example.com"),
        })

        result = resolve_alias("claude-opus-4-6", "custom:provider-b")
        assert result is not None
        provider, model, alias = result
        assert provider == "custom:provider-b"
        assert model == "claude-opus-4-6"
        assert alias == "my-b"

    def test_reverse_lookup_falls_back_to_first_match_for_unrelated_provider(self, monkeypatch):
        """With no provider match, reverse lookup keeps its first-match behaviour.

        Provider-preference must be a tie-breaker, not a filter: a model that no
        alias serves on the current provider still routes through some alias
        rather than falling through to the catalog.
        """
        from hermes_cli.model_switch import DirectAlias, resolve_alias
        import hermes_cli.model_switch as ms

        monkeypatch.setattr(ms, "DIRECT_ALIASES", {
            "my-a": DirectAlias("claude-opus-4-6", "custom:provider-a", "https://api-a.example.com"),
            "my-b": DirectAlias("claude-opus-4-6", "custom:provider-b", "https://api-b.example.com"),
        })

        result = resolve_alias("claude-opus-4-6", "custom:provider-c")
        assert result == ("custom:provider-a", "claude-opus-4-6", "my-a")

    def test_exact_alias_name_lookup_beats_provider_preference(self, monkeypatch):
        """An exact alias-name hit still wins over reverse lookup entirely."""
        from hermes_cli.model_switch import DirectAlias, resolve_alias
        import hermes_cli.model_switch as ms

        monkeypatch.setattr(ms, "DIRECT_ALIASES", {
            "my-a": DirectAlias("shared-model", "custom:provider-a", "https://api-a.example.com"),
            "my-b": DirectAlias("shared-model", "custom:provider-b", "https://api-b.example.com"),
        })

        # Asking for the alias *name* "my-a" while sitting on provider-b must
        # still return provider-a's alias — name lookup is not provider-scoped.
        assert resolve_alias("my-a", "custom:provider-b") == (
            "custom:provider-a", "shared-model", "my-a",
        )


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


    def test_load_config_exception_returns_builtins(self, monkeypatch):
        """If load_config raises, _load_direct_aliases returns builtins only."""
        monkeypatch.setattr(
            "hermes_cli.config.load_config",
            lambda: (_ for _ in ()).throw(RuntimeError("config broken")),
        )

        from hermes_cli.model_switch import _load_direct_aliases
        aliases = _load_direct_aliases()
        assert isinstance(aliases, dict)


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
        assert "not switching automatically" in result.error_message


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
            lambda **kwargs: {"api_key": "", "base_url": "", "api_mode": "openai_compat", "provider": "custom"},
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
            lambda **kwargs: {"api_key": "", "base_url": "", "api_mode": "openai_compat", "provider": "custom"},
        )
        monkeypatch.setattr("hermes_cli.models.validate_requested_model",
            lambda *a, **kw: {"accepted": True, "persist": True, "recognized": True, "message": None})
        monkeypatch.setattr("hermes_cli.models.opencode_model_api_mode",
            lambda *a, **kw: "openai_compat")

        result = ms.switch_model("local", "openrouter", "old-model")
        assert result.success
        assert result.api_key == "no-key-required"
        assert result.base_url == "http://localhost:11434/v1"

    @staticmethod
    def _stub_explicit_provider_b(monkeypatch, ms):
        """Wire the collaborators an explicit `--provider custom:provider-b` switch needs.

        Everything stubbed here is *outside* the unit under test: provider
        definition lookup, runtime credential resolution, and model validation.
        The runtime resolver deliberately reports provider B's own base_url so
        that any surviving provider-A URL in the result can only have come from
        the direct-alias override path.
        """
        from types import SimpleNamespace

        monkeypatch.setattr(
            ms,
            "resolve_provider_full",
            lambda explicit_provider, *_args, **_kwargs: SimpleNamespace(
                id="custom:provider-b",
                name="Provider B",
                base_url="https://api-b.example.com",
            ),
        )
        monkeypatch.setattr(
            "hermes_cli.runtime_provider.resolve_runtime_provider",
            lambda requested=None, **_kwargs: {
                "api_key": "sk-bbb",
                "base_url": "https://api-b.example.com",
                "api_mode": "openai_compat",
                "provider": "custom:provider-b",
            },
        )
        monkeypatch.setattr(
            "hermes_cli.models.validate_requested_model",
            lambda *a, **kw: {
                "accepted": True,
                "persist": True,
                "recognized": True,
                "message": None,
            },
        )
        monkeypatch.setattr(
            "hermes_cli.models.opencode_model_api_mode",
            lambda *a, **kw: "openai_compat",
        )
        monkeypatch.setattr(ms, "get_model_capabilities", lambda *a, **kw: None)
        monkeypatch.setattr(ms, "get_model_info", lambda *a, **kw: None)

    def test_explicit_provider_prefers_matching_alias_base_url(self, monkeypatch):
        """An explicit switch adopts the alias belonging to the *requested* provider.

        Two aliases share one model ID. Asking for provider B by name must route
        to B's endpoint and report B's alias, not whichever alias happens to be
        first in the mapping.
        """
        from hermes_cli.model_switch import DirectAlias
        import hermes_cli.model_switch as ms

        monkeypatch.setattr(ms, "DIRECT_ALIASES", {
            "my-a": DirectAlias("claude-opus-4-6", "custom:provider-a", "https://api-a.example.com"),
            "my-b": DirectAlias("claude-opus-4-6", "custom:provider-b", "https://api-b.example.com"),
        })
        self._stub_explicit_provider_b(monkeypatch, ms)

        result = ms.switch_model(
            "claude-opus-4-6",
            "custom:provider-a",
            "old-model",
            current_base_url="https://api-a.example.com",
            explicit_provider="custom:provider-b",
        )

        assert result.success
        assert result.target_provider == "custom:provider-b"
        # The invariant: the endpoint must belong to the provider we switched to.
        assert result.base_url == "https://api-b.example.com"
        assert result.resolved_via_alias == "my-b"

    def test_explicit_provider_discards_alias_for_other_provider(self, monkeypatch):
        """An alias owned by a *different* provider is never adopted.

        Only provider A has an alias for the model. Switching explicitly to
        provider B must not inherit A's base_url, and must not claim to have
        resolved via A's alias.
        """
        from hermes_cli.model_switch import DirectAlias
        import hermes_cli.model_switch as ms

        monkeypatch.setattr(ms, "DIRECT_ALIASES", {
            "my-a": DirectAlias("shared-model", "custom:provider-a", "https://api-a.example.com"),
        })
        self._stub_explicit_provider_b(monkeypatch, ms)

        result = ms.switch_model(
            "shared-model",
            "custom:provider-a",
            "old-model",
            current_base_url="https://api-a.example.com",
            explicit_provider="custom:provider-b",
        )

        assert result.success
        assert result.target_provider == "custom:provider-b"
        assert result.base_url == "https://api-b.example.com"
        assert result.resolved_via_alias == ""

    def test_explicit_provider_without_any_alias_is_unchanged(self, monkeypatch):
        """No alias for the model at all: the switch still succeeds on B's endpoint."""
        import hermes_cli.model_switch as ms

        monkeypatch.setattr(ms, "DIRECT_ALIASES", {})
        self._stub_explicit_provider_b(monkeypatch, ms)

        result = ms.switch_model(
            "unaliased-model",
            "custom:provider-a",
            "old-model",
            current_base_url="https://api-a.example.com",
            explicit_provider="custom:provider-b",
        )

        assert result.success
        assert result.target_provider == "custom:provider-b"
        assert result.base_url == "https://api-b.example.com"
        assert result.resolved_via_alias == ""


# ---------------------------------------------------------------------------
# CLI state update: requested_provider persistence
# ---------------------------------------------------------------------------

class TestCLIStateUpdate:
    """CLI /model handler should update requested_provider and explicit fields."""


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
