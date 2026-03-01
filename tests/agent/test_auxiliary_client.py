"""Tests for agent.auxiliary_client resolution chain, especially the Codex fallback."""

import json
import os
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

from agent.auxiliary_client import (
    get_text_auxiliary_client,
    get_vision_auxiliary_client,
    auxiliary_max_tokens_param,
    _read_codex_access_token,
)


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    """Strip provider env vars so each test starts clean."""
    for key in (
        "OPENROUTER_API_KEY", "OPENAI_BASE_URL", "OPENAI_API_KEY",
        "OPENAI_MODEL", "LLM_MODEL", "NOUS_INFERENCE_BASE_URL",
    ):
        monkeypatch.delenv(key, raising=False)


@pytest.fixture
def codex_auth_dir(tmp_path, monkeypatch):
    """Provide a writable ~/.codex/ directory with a valid auth.json."""
    codex_dir = tmp_path / ".codex"
    codex_dir.mkdir()
    auth_file = codex_dir / "auth.json"
    auth_file.write_text(json.dumps({
        "tokens": {
            "access_token": "codex-test-token-abc123",
            "refresh_token": "codex-refresh-xyz",
        }
    }))
    monkeypatch.setattr(
        "agent.auxiliary_client._read_codex_access_token",
        lambda: "codex-test-token-abc123",
    )
    return codex_dir


class TestReadCodexAccessToken:
    def test_valid_auth_file(self, tmp_path):
        codex_dir = tmp_path / ".codex"
        codex_dir.mkdir()
        auth = codex_dir / "auth.json"
        auth.write_text(json.dumps({
            "tokens": {"access_token": "tok-123", "refresh_token": "r-456"}
        }))
        with patch("agent.auxiliary_client.Path.home", return_value=tmp_path):
            result = _read_codex_access_token()
        assert result == "tok-123"

    def test_missing_file_returns_none(self, tmp_path):
        with patch("agent.auxiliary_client.Path.home", return_value=tmp_path):
            result = _read_codex_access_token()
        assert result is None

    def test_empty_token_returns_none(self, tmp_path):
        codex_dir = tmp_path / ".codex"
        codex_dir.mkdir()
        auth = codex_dir / "auth.json"
        auth.write_text(json.dumps({"tokens": {"access_token": "  "}}))
        with patch("agent.auxiliary_client.Path.home", return_value=tmp_path):
            result = _read_codex_access_token()
        assert result is None

    def test_malformed_json_returns_none(self, tmp_path):
        codex_dir = tmp_path / ".codex"
        codex_dir.mkdir()
        (codex_dir / "auth.json").write_text("{bad json")
        with patch("agent.auxiliary_client.Path.home", return_value=tmp_path):
            result = _read_codex_access_token()
        assert result is None

    def test_missing_tokens_key_returns_none(self, tmp_path):
        codex_dir = tmp_path / ".codex"
        codex_dir.mkdir()
        (codex_dir / "auth.json").write_text(json.dumps({"other": "data"}))
        with patch("agent.auxiliary_client.Path.home", return_value=tmp_path):
            result = _read_codex_access_token()
        assert result is None


class TestGetTextAuxiliaryClient:
    """Test the full resolution chain for get_text_auxiliary_client."""

    def test_openrouter_takes_priority(self, monkeypatch, codex_auth_dir):
        monkeypatch.setenv("OPENROUTER_API_KEY", "or-key")
        with patch("agent.auxiliary_client.OpenAI") as mock_openai:
            client, model = get_text_auxiliary_client()
        assert model == "google/gemini-3-flash-preview"
        mock_openai.assert_called_once()
        call_kwargs = mock_openai.call_args
        assert call_kwargs.kwargs["api_key"] == "or-key"

    def test_nous_takes_priority_over_codex(self, monkeypatch, codex_auth_dir):
        with patch("agent.auxiliary_client._read_nous_auth") as mock_nous, \
             patch("agent.auxiliary_client.OpenAI") as mock_openai:
            mock_nous.return_value = {"access_token": "nous-tok"}
            client, model = get_text_auxiliary_client()
        assert model == "gemini-3-flash"

    def test_custom_endpoint_over_codex(self, monkeypatch, codex_auth_dir):
        monkeypatch.setenv("OPENAI_BASE_URL", "http://localhost:1234/v1")
        monkeypatch.setenv("OPENAI_API_KEY", "lm-studio-key")
        # Override the autouse monkeypatch for codex
        monkeypatch.setattr(
            "agent.auxiliary_client._read_codex_access_token",
            lambda: "codex-test-token-abc123",
        )
        with patch("agent.auxiliary_client._read_nous_auth", return_value=None), \
             patch("agent.auxiliary_client.OpenAI") as mock_openai:
            client, model = get_text_auxiliary_client()
        assert model == "gpt-4o-mini"
        call_kwargs = mock_openai.call_args
        assert call_kwargs.kwargs["base_url"] == "http://localhost:1234/v1"

    def test_codex_fallback_when_nothing_else(self, codex_auth_dir):
        with patch("agent.auxiliary_client._read_nous_auth", return_value=None), \
             patch("agent.auxiliary_client.OpenAI") as mock_openai:
            client, model = get_text_auxiliary_client()
        assert model == "gpt-5.3-codex"
        # Returns a CodexAuxiliaryClient wrapper, not a raw OpenAI client
        from agent.auxiliary_client import CodexAuxiliaryClient
        assert isinstance(client, CodexAuxiliaryClient)

    def test_returns_none_when_nothing_available(self):
        with patch("agent.auxiliary_client._read_nous_auth", return_value=None), \
             patch("agent.auxiliary_client._read_codex_access_token", return_value=None):
            client, model = get_text_auxiliary_client()
        assert client is None
        assert model is None


class TestCodexNotInVisionClient:
    """Codex fallback should NOT apply to vision tasks."""

    def test_vision_returns_none_without_openrouter_nous(self):
        with patch("agent.auxiliary_client._read_nous_auth", return_value=None):
            client, model = get_vision_auxiliary_client()
        assert client is None
        assert model is None


class TestAuxiliaryMaxTokensParam:
    def test_codex_fallback_uses_max_tokens(self, monkeypatch):
        """Codex adapter translates max_tokens internally, so we return max_tokens."""
        with patch("agent.auxiliary_client._read_nous_auth", return_value=None), \
             patch("agent.auxiliary_client._read_codex_access_token", return_value="tok"):
            result = auxiliary_max_tokens_param(1024)
        assert result == {"max_tokens": 1024}

    def test_openrouter_uses_max_tokens(self, monkeypatch):
        monkeypatch.setenv("OPENROUTER_API_KEY", "or-key")
        result = auxiliary_max_tokens_param(1024)
        assert result == {"max_tokens": 1024}

    def test_no_provider_uses_max_tokens(self):
        with patch("agent.auxiliary_client._read_nous_auth", return_value=None), \
             patch("agent.auxiliary_client._read_codex_access_token", return_value=None):
            result = auxiliary_max_tokens_param(1024)
        assert result == {"max_tokens": 1024}
