"""Tests for _resolve_api_key_provider_secret credential_pool fallback.

Covers: https://github.com/NousResearch/hermes-agent/issues/15914
When os.environ has no API key but ~/.hermes/auth.json has valid entries
in the credential_pool, the function should fall back to the pool.
"""

import json
import os
import pytest


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    """Strip all API-key env vars so tests don't leak secrets."""
    for key in (
        "OPENAI_API_KEY", "ANTHROPIC_API_KEY", "DEEPSEEK_API_KEY",
        "GLM_API_KEY", "ZAI_API_KEY", "Z_AI_API_KEY",
        "MINIMAX_API_KEY", "KIMI_API_KEY", "XAI_API_KEY",
    ):
        monkeypatch.delenv(key, raising=False)


@pytest.fixture
def _auth_store_with_pool(tmp_path, monkeypatch):
    """Write a credential pool into auth.json for a provider."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes"))
    hermes_home = tmp_path / "hermes"
    hermes_home.mkdir(parents=True, exist_ok=True)

    auth_store = {
        "version": 1,
        "providers": {},
        "credential_pool": {
            "deepseek": [
                {"id": "deepseek-0", "access_token": "sk-deepseek-test-key-12345", "label": "deepseek"},
            ],
            "openai": [
                {"id": "openai-0", "access_token": "sk-openai-test-key-67890", "label": "openai"},
            ],
        },
    }
    (hermes_home / "auth.json").write_text(json.dumps(auth_store))
    return hermes_home


class TestResolveApiKeyProviderSecret:
    """Test _resolve_api_key_provider_secret credential pool fallback."""

    def test_returns_empty_when_no_env_and_no_pool(self, tmp_path, monkeypatch):
        """No env var and no credential_pool → returns empty string."""
        monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes"))
        (tmp_path / "hermes").mkdir(parents=True, exist_ok=True)

        from hermes_cli.auth import _resolve_api_key_provider_secret, ProviderConfig

        pconfig = ProviderConfig(
            provider_id="deepseek",
            name="DeepSeek",
            auth_type="api_key",
            inference_base_url="https://api.deepseek.com/v1",
            api_key_env_vars=("DEEPSEEK_API_KEY",),
            base_url_env_var=None,
        )
        key, source = _resolve_api_key_provider_secret("deepseek", pconfig)
        assert key == ""
        assert source == ""

    def test_returns_env_var_when_present(self, tmp_path, monkeypatch, _auth_store_with_pool):
        """os.environ has the key → returns it (pool is not consulted)."""
        monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-env-key-from-os-environ")

        from hermes_cli.auth import _resolve_api_key_provider_secret, ProviderConfig

        pconfig = ProviderConfig(
            provider_id="deepseek",
            name="DeepSeek",
            auth_type="api_key",
            inference_base_url="https://api.deepseek.com/v1",
            api_key_env_vars=("DEEPSEEK_API_KEY",),
            base_url_env_var=None,
        )
        key, source = _resolve_api_key_provider_secret("deepseek", pconfig)
        assert key == "sk-env-key-from-os-environ"
        assert source == "DEEPSEEK_API_KEY"

    def test_falls_back_to_credential_pool_when_env_missing(self, tmp_path, monkeypatch, _auth_store_with_pool):
        """No env var but pool has valid entry → returns pool key."""
        # Ensure env is clear
        assert os.getenv("DEEPSEEK_API_KEY") is None

        from hermes_cli.auth import _resolve_api_key_provider_secret, ProviderConfig

        pconfig = ProviderConfig(
            provider_id="deepseek",
            name="DeepSeek",
            auth_type="api_key",
            inference_base_url="https://api.deepseek.com/v1",
            api_key_env_vars=("DEEPSEEK_API_KEY",),
            base_url_env_var=None,
        )
        key, source = _resolve_api_key_provider_secret("deepseek", pconfig)
        assert key == "sk-deepseek-test-key-12345"
        assert source == "credential_pool:deepseek-0"

    def test_credential_pool_fallback_is_provider_specific(self, tmp_path, monkeypatch, _auth_store_with_pool):
        """Pool entry is for deepseek, not openai → deepseek works, openai returns empty."""
        # openai has no env and no pool entries
        assert os.getenv("OPENAI_API_KEY") is None

        from hermes_cli.auth import _resolve_api_key_provider_secret, ProviderConfig

        pconfig_openai = ProviderConfig(
            provider_id="openai",
            name="OpenAI",
            auth_type="api_key",
            inference_base_url="https://api.openai.com/v1",
            api_key_env_vars=("OPENAI_API_KEY",),
            base_url_env_var=None,
        )
        key, source = _resolve_api_key_provider_secret("openai", pconfig_openai)
        assert key == "sk-openai-test-key-67890"
        assert source == "credential_pool:openai-0"

    def test_empty_pool_entry_skipped(self, tmp_path, monkeypatch):
        """Pool entry with empty access_token is skipped."""
        monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes"))
        hermes_home = tmp_path / "hermes"
        hermes_home.mkdir(parents=True, exist_ok=True)

        auth_store = {
            "version": 1,
            "credential_pool": {
                "deepseek": [
                    {"id": "deepseek-0", "access_token": "", "label": "empty"},
                ],
            },
        }
        (hermes_home / "auth.json").write_text(json.dumps(auth_store))

        from hermes_cli.auth import _resolve_api_key_provider_secret, ProviderConfig

        pconfig = ProviderConfig(
            provider_id="deepseek",
            name="DeepSeek",
            auth_type="api_key",
            inference_base_url="https://api.deepseek.com/v1",
            api_key_env_vars=("DEEPSEEK_API_KEY",),
            base_url_env_var=None,
        )
        key, source = _resolve_api_key_provider_secret("deepseek", pconfig)
        # Empty token filtered → falls back to empty
        assert key == ""