"""Tests for ${ENV_VAR} expansion in gateway config loading paths.

Regression tests for #14457: custom_providers.base_url placeholders were
validated before expansion because _load_gateway_config() returned raw
YAML without calling _expand_env_vars().
"""

import os
import textwrap
from pathlib import Path
from unittest.mock import patch

import pytest
import yaml


class TestLoadGatewayConfigExpandsEnvVars:
    """_load_gateway_config() must expand ${VAR} placeholders."""

    def _write_config(self, tmp_path, content):
        config_file = tmp_path / "config.yaml"
        config_file.write_text(textwrap.dedent(content))
        return tmp_path

    def test_base_url_placeholder_expanded(self, tmp_path, monkeypatch):
        """${VAR} in custom_providers base_url must be expanded."""
        self._write_config(tmp_path, """\
            custom_providers:
              - name: my-provider
                base_url: ${TEST_PROVIDER_URL_14457}
                key_env: TEST_KEY
        """)
        monkeypatch.setenv("TEST_PROVIDER_URL_14457", "https://api.example.com/v1")

        import gateway.run as gw
        monkeypatch.setattr(gw, "_hermes_home", tmp_path)

        cfg = gw._load_gateway_config()
        entry = cfg["custom_providers"][0]
        assert entry["base_url"] == "https://api.example.com/v1"

    def test_model_base_url_placeholder_expanded(self, tmp_path, monkeypatch):
        """${VAR} in model.base_url must be expanded."""
        self._write_config(tmp_path, """\
            model:
              provider: custom
              default: gpt-4
              base_url: ${TEST_MODEL_URL_14457}
              api_key: ${TEST_MODEL_KEY_14457}
        """)
        monkeypatch.setenv("TEST_MODEL_URL_14457", "https://custom-api.example.com/v1")
        monkeypatch.setenv("TEST_MODEL_KEY_14457", "sk-test-key-123")

        import gateway.run as gw
        monkeypatch.setattr(gw, "_hermes_home", tmp_path)

        cfg = gw._load_gateway_config()
        assert cfg["model"]["base_url"] == "https://custom-api.example.com/v1"
        assert cfg["model"]["api_key"] == "sk-test-key-123"

    def test_unresolved_var_kept_verbatim(self, tmp_path, monkeypatch):
        """Unresolved ${VAR} should be kept as-is, not cause errors."""
        self._write_config(tmp_path, """\
            custom_providers:
              - name: offline
                base_url: ${UNSET_PROVIDER_URL_14457}
        """)
        monkeypatch.delenv("UNSET_PROVIDER_URL_14457", raising=False)

        import gateway.run as gw
        monkeypatch.setattr(gw, "_hermes_home", tmp_path)

        cfg = gw._load_gateway_config()
        assert cfg["custom_providers"][0]["base_url"] == "${UNSET_PROVIDER_URL_14457}"

    def test_literal_values_unchanged(self, tmp_path, monkeypatch):
        """Non-placeholder values must not be altered."""
        self._write_config(tmp_path, """\
            custom_providers:
              - name: literal-provider
                base_url: https://api.real.com/v1
                key_env: MY_KEY
        """)
        import gateway.run as gw
        monkeypatch.setattr(gw, "_hermes_home", tmp_path)

        cfg = gw._load_gateway_config()
        assert cfg["custom_providers"][0]["base_url"] == "https://api.real.com/v1"

    def test_multiple_providers_all_expanded(self, tmp_path, monkeypatch):
        """All providers in a multi-provider config must be expanded."""
        self._write_config(tmp_path, """\
            custom_providers:
              - name: provider-a
                base_url: ${TEST_URL_A_14457}
                key_env: KEY_A
              - name: provider-b
                base_url: ${TEST_URL_B_14457}
                key_env: KEY_B
        """)
        monkeypatch.setenv("TEST_URL_A_14457", "https://a.example.com/v1")
        monkeypatch.setenv("TEST_URL_B_14457", "https://b.example.com/v1")

        import gateway.run as gw
        monkeypatch.setattr(gw, "_hermes_home", tmp_path)

        cfg = gw._load_gateway_config()
        assert cfg["custom_providers"][0]["base_url"] == "https://a.example.com/v1"
        assert cfg["custom_providers"][1]["base_url"] == "https://b.example.com/v1"


class TestGetCompatibleCustomProvidersWithEnvVars:
    """get_compatible_custom_providers() must work with gateway-loaded config."""

    def test_env_var_providers_are_found(self, tmp_path, monkeypatch):
        """Providers with ${VAR} base_url must be returned after expansion."""
        config = {
            "custom_providers": [
                {
                    "name": "my-api",
                    "base_url": "${TEST_CP_URL_14457}",
                    "key_env": "MY_API_KEY",
                },
            ],
        }
        monkeypatch.setenv("TEST_CP_URL_14457", "https://my-api.example.com/v1")

        from hermes_cli.config import _expand_env_vars, get_compatible_custom_providers

        expanded = _expand_env_vars(config)
        providers = get_compatible_custom_providers(expanded)
        assert len(providers) == 1
        assert providers[0]["base_url"] == "https://my-api.example.com/v1"
        assert providers[0]["name"] == "my-api"

    def test_unexpanded_base_url_not_rejected_as_invalid(self, monkeypatch):
        """Unresolved ${VAR} in base_url must not be rejected by URL validation.

        _normalize_custom_provider_entry() validates URLs with urlparse().
        A literal '${VAR}' has no scheme/netloc and was rejected as invalid.
        The fix: recognise env-ref patterns and pass them through unvalidated.
        """
        monkeypatch.delenv("NONEXISTENT_URL_VAR_14457", raising=False)
        config = {
            "custom_providers": [
                {
                    "name": "pending",
                    "base_url": "${NONEXISTENT_URL_VAR_14457}",
                },
            ],
        }
        from hermes_cli.config import get_compatible_custom_providers

        providers = get_compatible_custom_providers(config)
        assert len(providers) == 1
        assert providers[0]["base_url"] == "${NONEXISTENT_URL_VAR_14457}"
