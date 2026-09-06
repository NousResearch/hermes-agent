"""Regression tests for vendor-prefix model routing and dict model.aliases (#87189).

``--model nous/deepseek-v4-pro`` / ``--model ollama/qwen3.5:4b`` used to fall
through provider auto-detection and be sent to the configured default provider
(api.anthropic.com) with the prefixed name intact, producing HTTP 404. Dict
entries under ``model.aliases`` (``localqwen: {model: ..., provider: ...}``)
were silently dropped because only string values were parsed.
"""

import hermes_cli.models as models
import hermes_cli.model_switch as model_switch


class TestVendorPrefixRouting:
    """detect_provider_for_model honors a ``vendor/model`` prefix for
    providers the user actually configured in their ``providers:`` block."""

    def test_configured_provider_prefix_routes_to_provider(self, monkeypatch):
        monkeypatch.setattr(models, "_find_openrouter_slug", lambda _name: None)
        monkeypatch.setattr(models, "_configured_provider_ids", lambda: {"nous"})
        detected = models.detect_provider_for_model("nous/deepseek-v4-pro", "anthropic")
        assert detected == ("nous", "deepseek-v4-pro")

    def test_local_provider_prefix_routes_to_provider(self, monkeypatch):
        monkeypatch.setattr(models, "_find_openrouter_slug", lambda _name: None)
        monkeypatch.setattr(models, "_configured_provider_ids", lambda: {"ollama"})
        detected = models.detect_provider_for_model("ollama/qwen3.5:4b", "anthropic")
        assert detected == ("ollama", "qwen3.5:4b")

    def test_unconfigured_builtin_vendor_prefix_not_rerouted(self, monkeypatch):
        """Built-in vendor slugs keep catalog/default routing.

        ``google/gemini-2.5-flash`` is aggregator-native: the web config
        field expects it to switch to OpenRouter, not to the Gemini provider
        (``TestDenormalizeProviderSwitch`` in test_web_server.py). With no
        user-configured provider for the vendor, prefix routing must stay
        out of the way even when the models.dev catalog is unavailable.
        """
        monkeypatch.setattr(models, "_find_openrouter_slug", lambda _name: None)
        monkeypatch.setattr(models, "_configured_provider_ids", lambda: set())
        detected = models.detect_provider_for_model(
            "google/gemini-2.5-flash", "ollama-local"
        )
        assert detected is None

    def test_configured_provider_wins_over_alias_canonicalization(self, monkeypatch):
        """A user-named ``ollama`` block must not be rewritten to ``custom``."""
        monkeypatch.setattr(models, "_find_openrouter_slug", lambda _name: None)
        monkeypatch.setattr(models, "_configured_provider_ids", lambda: {"ollama"})
        assert models._PROVIDER_ALIASES.get("ollama") == "custom"  # precondition
        detected = models.detect_provider_for_model("ollama/qwen3.5:4b", "anthropic")
        assert detected == ("ollama", "qwen3.5:4b")

    def test_provider_alias_prefix_canonicalized_when_configured(self, monkeypatch):
        monkeypatch.setattr(models, "_find_openrouter_slug", lambda _name: None)
        monkeypatch.setattr(models, "_configured_provider_ids", lambda: {"zai"})
        detected = models.detect_provider_for_model("glm/glm-4.7", "anthropic")
        assert detected == ("zai", "glm-4.7")

    def test_unknown_vendor_prefix_still_unmatched(self, monkeypatch):
        monkeypatch.setattr(models, "_find_openrouter_slug", lambda _name: None)
        monkeypatch.setattr(models, "_configured_provider_ids", lambda: {"ollama"})
        assert models.detect_provider_for_model("notaprovider/foo-model", "anthropic") is None

    def test_openrouter_slug_still_wins_over_prefix_routing(self, monkeypatch):
        """Aggregator-native slugs keep their existing OpenRouter routing."""
        monkeypatch.setattr(
            models, "_find_openrouter_slug", lambda _name: "deepseek/deepseek-chat"
        )
        monkeypatch.setattr(models, "_configured_provider_ids", lambda: {"deepseek"})
        detected = models.detect_provider_for_model("deepseek/deepseek-chat", "anthropic")
        assert detected == ("openrouter", "deepseek/deepseek-chat")

    def test_bare_model_detection_unchanged(self, monkeypatch):
        monkeypatch.setattr(models, "_find_openrouter_slug", lambda _name: None)
        detected = models.detect_provider_for_model("deepseek-chat", "anthropic")
        assert detected == ("deepseek", "deepseek-chat")


class TestDictModelAliases:
    """``model.aliases`` accepts dict entries with an explicit provider."""

    def _load_with(self, monkeypatch, cfg):
        monkeypatch.setattr("hermes_cli.config.load_config", lambda: cfg)
        return model_switch._load_direct_aliases()

    def test_dict_entry_with_explicit_provider(self, monkeypatch):
        cfg = {
            "model": {
                "aliases": {
                    "localqwen": {"model": "qwen3.5:4b", "provider": "custom"},
                },
            },
        }
        aliases = self._load_with(monkeypatch, cfg)
        da = aliases["localqwen"]
        assert (da.model, da.provider) == ("qwen3.5:4b", "custom")

    def test_dict_entry_with_base_url(self, monkeypatch):
        cfg = {
            "model": {
                "aliases": {
                    "qwen": {
                        "model": "qwen3.5:4b",
                        "provider": "ollama",
                        "base_url": "http://localhost:11434/v1",
                    },
                },
            },
        }
        aliases = self._load_with(monkeypatch, cfg)
        da = aliases["qwen"]
        assert (da.model, da.provider, da.base_url) == (
            "qwen3.5:4b", "ollama", "http://localhost:11434/v1",
        )

    def test_dict_entry_without_provider_uses_model_provider(self, monkeypatch):
        cfg = {
            "model": {
                "provider": "openrouter",
                "aliases": {"bare": {"model": "some-model"}},
            },
        }
        aliases = self._load_with(monkeypatch, cfg)
        da = aliases["bare"]
        assert (da.model, da.provider) == ("some-model", "openrouter")

    def test_string_entries_still_parse(self, monkeypatch):
        cfg = {
            "model": {
                "aliases": {"ds-flash": "deepseek/deepseek-v4-flash"},
            },
        }
        aliases = self._load_with(monkeypatch, cfg)
        da = aliases["ds-flash"]
        assert (da.model, da.provider) == ("deepseek-v4-flash", "deepseek")

    def test_model_aliases_block_keeps_priority_over_model_aliases(self, monkeypatch):
        cfg = {
            "model_aliases": {
                "shared": {"model": "from-top-block", "provider": "custom"},
            },
            "model": {
                "aliases": {"shared": {"model": "from-nested", "provider": "ollama"}},
            },
        }
        aliases = self._load_with(monkeypatch, cfg)
        assert aliases["shared"].model == "from-top-block"
