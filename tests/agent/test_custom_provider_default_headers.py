"""Tests for per-custom-provider ``default_headers`` (both wire formats).

Extends the ``model.default_headers`` support from #40033 so a specific
custom provider (e.g. Requesty via ``api_mode: chat_completions`` OR
``anthropic_messages``) can carry its own cache-control / attribution headers
on every request, without forcing them onto every other endpoint.

Covers:
- config normalizer preserves ``custom_providers[].default_headers``
- ``_resolve_user_default_headers`` merges provider-level + model-level
- the Anthropic-wire client (``build_anthropic_client``) applies them
"""

from unittest.mock import patch, MagicMock

import pytest


@pytest.fixture(autouse=True)
def _isolate(tmp_path, monkeypatch):
    hermes_home = tmp_path / ".hermes"
    hermes_home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    (hermes_home / "config.yaml").write_text("model:\n  default: test-model\n")


def _write_config(tmp_path, config_dict):
    import yaml
    (tmp_path / ".hermes" / "config.yaml").write_text(yaml.dump(config_dict))


class TestConfigNormalizerPreservesHeaders:
    def test_default_headers_survive_normalization(self, tmp_path):
        from hermes_cli.config import get_compatible_custom_providers, load_config
        _write_config(tmp_path, {
            "custom_providers": [{
                "name": "requesty",
                "base_url": "https://router.requesty.ai/v1",
                "api_key": "k",
                "default_headers": {"X-Requesty-Cache": "true", "X-Title": "Hermes"},
            }],
        })
        providers = get_compatible_custom_providers(load_config())
        entry = next(p for p in providers if p["name"] == "requesty")
        assert entry["default_headers"] == {"X-Requesty-Cache": "true", "X-Title": "Hermes"}

    def test_non_string_values_stringified(self, tmp_path):
        """YAML bare `true`/numbers must reach the SDK as str, not bool/int."""
        from hermes_cli.config import get_compatible_custom_providers, load_config
        _write_config(tmp_path, {
            "custom_providers": [{
                "name": "requesty",
                "base_url": "https://router.requesty.ai/v1",
                "api_key": "k",
                "default_headers": {"X-Requesty-Cache": True, "X-N": 5},
            }],
        })
        entry = next(
            p for p in get_compatible_custom_providers(load_config())
            if p["name"] == "requesty"
        )
        assert entry["default_headers"] == {"X-Requesty-Cache": "True", "X-N": "5"}


class TestResolveUserDefaultHeaders:
    def test_provider_level_headers_resolved(self, tmp_path):
        _write_config(tmp_path, {
            "custom_providers": [{
                "name": "requesty",
                "base_url": "https://router.requesty.ai/v1",
                "api_key": "k",
                "default_headers": {"X-Requesty-Cache": "true"},
            }],
        })
        from agent.auxiliary_client import _resolve_user_default_headers
        assert _resolve_user_default_headers("requesty") == {"X-Requesty-Cache": "true"}

    def test_model_level_wins_over_provider_level(self, tmp_path):
        _write_config(tmp_path, {
            "model": {"default": "m", "default_headers": {"X-Title": "FromModel"}},
            "custom_providers": [{
                "name": "requesty",
                "base_url": "https://router.requesty.ai/v1",
                "api_key": "k",
                "default_headers": {"X-Title": "FromProvider", "X-Requesty-Cache": "true"},
            }],
        })
        from agent.auxiliary_client import _resolve_user_default_headers
        resolved = _resolve_user_default_headers("requesty")
        assert resolved["X-Title"] == "FromModel"          # model wins
        assert resolved["X-Requesty-Cache"] == "true"       # provider-only survives

    def test_unknown_provider_only_gets_model_headers(self, tmp_path):
        _write_config(tmp_path, {
            "model": {"default": "m", "default_headers": {"X-Title": "Hermes"}},
            "custom_providers": [{
                "name": "requesty", "base_url": "https://x/v1", "api_key": "k",
                "default_headers": {"X-Requesty-Cache": "true"},
            }],
        })
        from agent.auxiliary_client import _resolve_user_default_headers
        assert _resolve_user_default_headers("someone-else") == {"X-Title": "Hermes"}

    def test_no_config_returns_empty(self, tmp_path):
        _write_config(tmp_path, {"model": {"default": "m"}})
        from agent.auxiliary_client import _resolve_user_default_headers
        assert _resolve_user_default_headers("requesty") == {}


class TestAnthropicWireAppliesHeaders:
    def test_build_anthropic_client_merges_provider_headers(self, tmp_path):
        _write_config(tmp_path, {
            "custom_providers": [{
                "name": "requesty",
                "base_url": "https://router.requesty.ai/anthropic",
                "api_key": "k",
                "default_headers": {"X-Requesty-Cache": "true"},
            }],
        })
        with patch("agent.anthropic_adapter._get_anthropic_sdk") as mock_sdk:
            sdk = MagicMock()
            mock_sdk.return_value = sdk
            from agent.anthropic_adapter import build_anthropic_client
            build_anthropic_client(
                "sk-test-key",
                "https://router.requesty.ai/anthropic",
                provider="requesty",
            )
        headers = sdk.Anthropic.call_args.kwargs.get("default_headers", {})
        assert headers.get("X-Requesty-Cache") == "true"

    def test_user_header_does_not_clobber_anthropic_beta(self, tmp_path):
        """Hermes' own anthropic-beta header is preserved when user adds others."""
        _write_config(tmp_path, {
            "custom_providers": [{
                "name": "requesty", "base_url": "https://x/anthropic", "api_key": "k",
                "default_headers": {"X-Requesty-Cache": "true"},
            }],
        })
        with patch("agent.anthropic_adapter._get_anthropic_sdk") as mock_sdk, \
             patch("agent.anthropic_adapter._common_betas_for_base_url", return_value=["beta-x"]):
            sdk = MagicMock()
            mock_sdk.return_value = sdk
            from agent.anthropic_adapter import build_anthropic_client
            build_anthropic_client("sk-test-key", "https://x/anthropic", provider="requesty")
        headers = sdk.Anthropic.call_args.kwargs.get("default_headers", {})
        assert headers.get("X-Requesty-Cache") == "true"
        assert "beta-x" in headers.get("anthropic-beta", "")
