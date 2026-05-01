"""Tests for _get_api_key_for_base_url and custom endpoint API key resolution.

This tests the fix for GitHub issue where auxiliary tasks with custom base_url
but no explicit api_key would fail with 401 errors because the resolution chain
did not look up credentials from known providers based on the base_url pattern.

The fix adds:
1. _BASE_URL_TO_PROVIDER mapping for known Chinese/international providers
2. _get_api_key_for_base_url() to match base_url to provider credentials
3. Integration in resolve_provider_client() custom branch
"""

import os
import logging
from unittest.mock import patch, MagicMock
import pytest

from agent.auxiliary_client import (
    _get_api_key_for_base_url,
    _BASE_URL_TO_PROVIDER,
    _read_env_var_from_file,
    resolve_provider_client,
)


class TestBaseUrlToProviderMapping:
    """Tests for _BASE_URL_TO_PROVIDER static mapping."""

    def test_mapping_contains_dashscope(self):
        """Alibaba DashScope endpoints are mapped."""
        assert "dashscope.aliyuncs.com" in _BASE_URL_TO_PROVIDER
        assert "coding.dashscope.aliyuncs.com" in _BASE_URL_TO_PROVIDER
        assert _BASE_URL_TO_PROVIDER["dashscope.aliyuncs.com"] == ("alibaba", "DASHSCOPE_API_KEY")
        assert _BASE_URL_TO_PROVIDER["coding.dashscope.aliyuncs.com"] == ("alibaba", "DASHSCOPE_API_KEY")

    def test_mapping_contains_deepseek(self):
        """DeepSeek endpoint is mapped."""
        assert "api.deepseek.com" in _BASE_URL_TO_PROVIDER
        assert _BASE_URL_TO_PROVIDER["api.deepseek.com"] == ("deepseek", "DEEPSEEK_API_KEY")

    def test_mapping_contains_moonshot(self):
        """Moonshot (Kimi) endpoint is mapped."""
        assert "api.moonshot.cn" in _BASE_URL_TO_PROVIDER
        assert _BASE_URL_TO_PROVIDER["api.moonshot.cn"] == ("moonshot", "MOONSHOT_API_KEY")

    def test_mapping_contains_minimax(self):
        """MiniMax endpoint is mapped."""
        assert "api.minimax.chat" in _BASE_URL_TO_PROVIDER
        assert _BASE_URL_TO_PROVIDER["api.minimax.chat"] == ("minimax", "MINIMAX_API_KEY")

    def test_mapping_contains_zhipu(self):
        """Zhipu (GLM) endpoint is mapped."""
        assert "api.zhipuai.cn" in _BASE_URL_TO_PROVIDER
        assert _BASE_URL_TO_PROVIDER["api.zhipuai.cn"] == ("zhipu", "ZHIPU_API_KEY")

    def test_mapping_contains_siliconflow(self):
        """SiliconFlow endpoint is mapped."""
        assert "api.siliconflow.cn" in _BASE_URL_TO_PROVIDER
        assert _BASE_URL_TO_PROVIDER["api.siliconflow.cn"] == ("siliconflow", "SILICONFLOW_API_KEY")

    def test_mapping_contains_modelscope(self):
        """ModelScope endpoint is mapped."""
        assert "api-inference.modelscope.cn" in _BASE_URL_TO_PROVIDER
        assert _BASE_URL_TO_PROVIDER["api-inference.modelscope.cn"] == ("modelscope", "MODELSCOPE_API_KEY")


class TestReadEnvVarFromFile:
    """Tests for _read_env_var_from_file helper."""

    def test_reads_valid_key_from_env_file(self, tmp_path, monkeypatch):
        """Can read a valid API key from .env file."""
        env_file = tmp_path / ".env"
        env_file.write_text("DASHSCOPE_API_KEY=sk-test-12345\nOTHER_KEY=value\n")
        monkeypatch.setattr("hermes_cli.config.get_hermes_home", lambda: tmp_path)
        result = _read_env_var_from_file("DASHSCOPE_API_KEY")
        assert result == "sk-test-12345"

    def test_skips_masked_placeholder_values(self, tmp_path, monkeypatch):
        """Skips '***' masked placeholder values."""
        env_file = tmp_path / ".env"
        env_file.write_text("DASHSCOPE_API_KEY=***\n")
        monkeypatch.setattr("hermes_cli.config.get_hermes_home", lambda: tmp_path)
        result = _read_env_var_from_file("DASHSCOPE_API_KEY")
        assert result is None

    def test_skips_masked_values_with_prefix(self, tmp_path, monkeypatch):
        """Skips values starting with '***'."""
        env_file = tmp_path / ".env"
        env_file.write_text("DASHSCOPE_API_KEY=***masked\n")
        monkeypatch.setattr("hermes_cli.config.get_hermes_home", lambda: tmp_path)
        result = _read_env_var_from_file("DASHSCOPE_API_KEY")
        assert result is None

    def test_returns_none_for_missing_var(self, tmp_path, monkeypatch):
        """Returns None when var is not in .env."""
        env_file = tmp_path / ".env"
        env_file.write_text("OTHER_KEY=value\n")
        monkeypatch.setattr("hermes_cli.config.get_hermes_home", lambda: tmp_path)
        result = _read_env_var_from_file("DASHSCOPE_API_KEY")
        assert result is None

    def test_returns_none_when_env_file_missing(self, tmp_path, monkeypatch):
        """Returns None when .env file does not exist."""
        monkeypatch.setattr("hermes_cli.config.get_hermes_home", lambda: tmp_path)
        result = _read_env_var_from_file("DASHSCOPE_API_KEY")
        assert result is None

    def test_handles_quoted_values(self, tmp_path, monkeypatch):
        """Strips quotes from values."""
        env_file = tmp_path / ".env"
        env_file.write_text('DASHSCOPE_API_KEY="sk-test-quoted"\n')
        monkeypatch.setattr("hermes_cli.config.get_hermes_home", lambda: tmp_path)
        result = _read_env_var_from_file("DASHSCOPE_API_KEY")
        assert result == "sk-test-quoted"


class TestGetApiKeyForBaseUrl:
    """Tests for _get_api_key_for_base_url resolution chain."""

    def test_returns_none_for_empty_url(self):
        """Empty base_url returns None."""
        result = _get_api_key_for_base_url("")
        assert result is None
        result = _get_api_key_for_base_url(None)
        assert result is None

    def test_returns_none_for_unknown_provider(self, monkeypatch):
        """Unknown base_url pattern returns None."""
        monkeypatch.setenv("OPENAI_API_KEY", "sk-openai-key")
        result = _get_api_key_for_base_url("https://unknown.example.com/v1")
        assert result is None

    def test_finds_key_from_env_var(self, monkeypatch):
        """Finds key from matching env var."""
        monkeypatch.setenv("DASHSCOPE_API_KEY", "sk-dashscope-test")
        result = _get_api_key_for_base_url("https://dashscope.aliyuncs.com/v1")
        assert result == "sk-dashscope-test"

    def test_finds_key_from_coding_dashscope_endpoint(self, monkeypatch):
        """Matches coding.dashscope.aliyuncs.com pattern."""
        monkeypatch.setenv("DASHSCOPE_API_KEY", "sk-codeplan-test")
        result = _get_api_key_for_base_url("https://coding.dashscope.aliyuncs.com/v1")
        assert result == "sk-codeplan-test"

    def test_matches_subdomain_pattern(self, monkeypatch):
        """Matches pattern embedded in larger URL."""
        monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-deepseek-test")
        # Pattern 'api.deepseek.com' should match this URL
        result = _get_api_key_for_base_url("https://api.deepseek.com/chat/completions")
        assert result == "sk-deepseek-test"

    def test_skips_masked_env_var_value(self, monkeypatch):
        """Skips '***' values in env vars."""
        monkeypatch.setenv("DASHSCOPE_API_KEY", "***")
        result = _get_api_key_for_base_url("https://dashscope.aliyuncs.com/v1")
        assert result is None

    def test_fallback_to_env_file_reading(self, tmp_path, monkeypatch):
        """Falls back to reading .env file when env var is masked."""
        # Set env to masked value
        monkeypatch.setenv("DASHSCOPE_API_KEY", "***")
        # Write real value to .env file
        env_file = tmp_path / ".env"
        env_file.write_text("DASHSCOPE_API_KEY=sk-real-from-file\n")
        monkeypatch.setattr("hermes_cli.config.get_hermes_home", lambda: tmp_path)
        result = _get_api_key_for_base_url("https://dashscope.aliyuncs.com/v1")
        assert result == "sk-real-from-file"

    def test_case_insensitive_matching(self, monkeypatch):
        """Pattern matching is case-insensitive."""
        monkeypatch.setenv("DASHSCOPE_API_KEY", "sk-test-case")
        # Uppercase URL should still match
        result = _get_api_key_for_base_url("HTTPS://DASHSCOPE.ALIYUNCS.COM/V1")
        assert result == "sk-test-case"

    @patch("agent.auxiliary_client._select_pool_entry")
    def test_prefers_credential_pool_over_env(self, mock_pool, monkeypatch):
        """Credential pool takes priority over env var."""
        # Create a mock entry with runtime_api_key attribute
        mock_entry = MagicMock()
        mock_entry.runtime_api_key = "sk-from-pool"
        mock_pool.return_value = (True, mock_entry)
        monkeypatch.setenv("DASHSCOPE_API_KEY", "sk-from-env")
        result = _get_api_key_for_base_url("https://dashscope.aliyuncs.com/v1")
        assert result == "sk-from-pool"

    @patch("agent.auxiliary_client._select_pool_entry")
    def test_falls_back_to_env_when_pool_empty(self, mock_pool, monkeypatch):
        """Falls back to env var when pool has no entry."""
        mock_pool.return_value = (False, None)
        monkeypatch.setenv("DASHSCOPE_API_KEY", "sk-from-env")
        result = _get_api_key_for_base_url("https://dashscope.aliyuncs.com/v1")
        assert result == "sk-from-env"

    @patch("agent.auxiliary_client._select_pool_entry")
    def test_skips_masked_pool_value(self, mock_pool, monkeypatch):
        """Skips masked values from credential pool."""
        # Create a mock entry with masked runtime_api_key
        mock_entry = MagicMock()
        mock_entry.runtime_api_key = "***"
        mock_pool.return_value = (True, mock_entry)
        monkeypatch.setenv("DASHSCOPE_API_KEY", "sk-from-env")
        result = _get_api_key_for_base_url("https://dashscope.aliyuncs.com/v1")
        assert result == "sk-from-env"


class TestResolveProviderClientCustomBranch:
    """Tests for resolve_provider_client custom branch with base_url."""

    def test_resolves_key_from_base_url_pattern(self, monkeypatch):
        """Custom endpoint with known base_url gets API key from provider env."""
        monkeypatch.setenv("DASHSCOPE_API_KEY", "sk-dashscope-key")
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        
        client, model = resolve_provider_client(
            provider="custom",
            model="kimi-k2.5",
            explicit_base_url="https://coding.dashscope.aliyuncs.com/v1",
            explicit_api_key=None,  # Should be resolved from base_url
        )
        
        assert client is not None
        assert client.api_key == "sk-dashscope-key"

    def test_falls_back_to_openai_api_key(self, monkeypatch):
        """Unknown base_url falls back to OPENAI_API_KEY."""
        monkeypatch.setenv("OPENAI_API_KEY", "sk-openai-key")
        monkeypatch.delenv("DASHSCOPE_API_KEY", raising=False)
        
        client, model = resolve_provider_client(
            provider="custom",
            model="local-model",
            explicit_base_url="https://local-server.example.com/v1",
            explicit_api_key=None,
        )
        
        assert client is not None
        assert client.api_key == "sk-openai-key"

    def test_uses_no_key_required_for_local_servers(self, monkeypatch):
        """Uses 'no-key-required' when no keys available."""
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        monkeypatch.delenv("DASHSCOPE_API_KEY", raising=False)
        
        client, model = resolve_provider_client(
            provider="custom",
            model="local-model",
            explicit_base_url="http://localhost:8080/v1",
            explicit_api_key=None,
        )
        
        assert client is not None
        assert client.api_key == "no-key-required"

    def test_explicit_api_key_takes_priority(self, monkeypatch):
        """Explicit api_key takes priority over base_url resolution."""
        monkeypatch.setenv("DASHSCOPE_API_KEY", "sk-dashscope-env")
        
        client, model = resolve_provider_client(
            provider="custom",
            model="kimi-k2.5",
            explicit_base_url="https://coding.dashscope.aliyuncs.com/v1",
            explicit_api_key="sk-explicit-key",  # Should win
        )
        
        assert client is not None
        assert client.api_key == "sk-explicit-key"


class TestIntegrationWithAuxiliaryConfig:
    """Integration tests simulating auxiliary.vision config scenario."""

    def test_vision_config_with_base_url_no_api_key(self, monkeypatch):
        """Simulates auxiliary.vision with base_url but api_key: null."""
        # This is the bug scenario: config.yaml has:
        # auxiliary:
        #   vision:
        #     provider: alibaba
        #     base_url: https://coding.dashscope.aliyuncs.com/v1
        #     api_key: null
        
        monkeypatch.setenv("DASHSCOPE_API_KEY", "sk-vision-key")
        
        # _resolve_task_provider_model would return provider="custom", base_url=...
        client, model = resolve_provider_client(
            provider="custom",
            model="kimi-k2.5",
            explicit_base_url="https://coding.dashscope.aliyuncs.com/v1",
            explicit_api_key=None,
        )
        
        assert client is not None
        assert client.api_key == "sk-vision-key"
        assert model == "kimi-k2.5"