"""Tests for agent/auxiliary_client.py — API client resolution chain."""

import json
import os
from pathlib import Path
from unittest.mock import patch

from agent.auxiliary_client import (
    _read_nous_auth,
    _nous_api_key,
    _nous_base_url,
    auxiliary_max_tokens_param,
    get_auxiliary_extra_body,
    _AUTH_JSON_PATH,
    _NOUS_DEFAULT_BASE_URL,
    NOUS_EXTRA_BODY,
)


# ---------------------------------------------------------------------------
# _read_nous_auth
# ---------------------------------------------------------------------------


class TestReadNousAuth:
    def test_missing_file(self, tmp_path):
        with patch("agent.auxiliary_client._AUTH_JSON_PATH", tmp_path / "nope.json"):
            assert _read_nous_auth() is None

    def test_wrong_active_provider(self, tmp_path):
        auth_file = tmp_path / "auth.json"
        auth_file.write_text(json.dumps({
            "active_provider": "openrouter",
            "providers": {"nous": {"access_token": "tok"}}
        }))
        with patch("agent.auxiliary_client._AUTH_JSON_PATH", auth_file):
            assert _read_nous_auth() is None

    def test_missing_tokens(self, tmp_path):
        auth_file = tmp_path / "auth.json"
        auth_file.write_text(json.dumps({
            "active_provider": "nous",
            "providers": {"nous": {}}
        }))
        with patch("agent.auxiliary_client._AUTH_JSON_PATH", auth_file):
            assert _read_nous_auth() is None

    def test_valid_access_token(self, tmp_path):
        auth_file = tmp_path / "auth.json"
        auth_file.write_text(json.dumps({
            "active_provider": "nous",
            "providers": {"nous": {"access_token": "my-token"}}
        }))
        with patch("agent.auxiliary_client._AUTH_JSON_PATH", auth_file):
            result = _read_nous_auth()
        assert result is not None
        assert result["access_token"] == "my-token"

    def test_valid_agent_key(self, tmp_path):
        auth_file = tmp_path / "auth.json"
        auth_file.write_text(json.dumps({
            "active_provider": "nous",
            "providers": {"nous": {"agent_key": "agent-key-123"}}
        }))
        with patch("agent.auxiliary_client._AUTH_JSON_PATH", auth_file):
            result = _read_nous_auth()
        assert result is not None

    def test_corrupt_json(self, tmp_path):
        auth_file = tmp_path / "auth.json"
        auth_file.write_text("not json{{{")
        with patch("agent.auxiliary_client._AUTH_JSON_PATH", auth_file):
            assert _read_nous_auth() is None


# ---------------------------------------------------------------------------
# _nous_api_key
# ---------------------------------------------------------------------------


class TestNousApiKey:
    def test_prefers_agent_key(self):
        provider = {"agent_key": "agent-key", "access_token": "access-tok"}
        assert _nous_api_key(provider) == "agent-key"

    def test_falls_back_to_access_token(self):
        provider = {"access_token": "access-tok"}
        assert _nous_api_key(provider) == "access-tok"

    def test_empty_provider(self):
        assert _nous_api_key({}) == ""


# ---------------------------------------------------------------------------
# _nous_base_url
# ---------------------------------------------------------------------------


class TestNousBaseUrl:
    def test_default(self):
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("NOUS_INFERENCE_BASE_URL", None)
            assert _nous_base_url() == _NOUS_DEFAULT_BASE_URL

    def test_env_override(self):
        with patch.dict(os.environ, {"NOUS_INFERENCE_BASE_URL": "https://custom.api/v1"}):
            assert _nous_base_url() == "https://custom.api/v1"


# ---------------------------------------------------------------------------
# auxiliary_max_tokens_param
# ---------------------------------------------------------------------------


class TestAuxiliaryMaxTokensParam:
    def test_openrouter_uses_max_tokens(self):
        with patch.dict(os.environ, {"OPENROUTER_API_KEY": "key"}):
            result = auxiliary_max_tokens_param(1000)
        assert result == {"max_tokens": 1000}

    def test_direct_openai_uses_max_completion_tokens(self, tmp_path):
        """Direct api.openai.com endpoint uses max_completion_tokens."""
        auth_file = tmp_path / "auth.json"
        auth_file.write_text(json.dumps({"active_provider": "other"}))

        env = {
            "OPENAI_BASE_URL": "https://api.openai.com/v1",
            "OPENAI_API_KEY": "sk-test",
        }
        with patch.dict(os.environ, env, clear=False), \
             patch("agent.auxiliary_client._AUTH_JSON_PATH", auth_file):
            os.environ.pop("OPENROUTER_API_KEY", None)
            result = auxiliary_max_tokens_param(500)
        assert result == {"max_completion_tokens": 500}

    def test_custom_non_openai_uses_max_tokens(self, tmp_path):
        auth_file = tmp_path / "auth.json"
        auth_file.write_text(json.dumps({"active_provider": "other"}))

        env = {
            "OPENAI_BASE_URL": "https://my-custom-api.com/v1",
            "OPENAI_API_KEY": "key",
        }
        with patch.dict(os.environ, env, clear=False), \
             patch("agent.auxiliary_client._AUTH_JSON_PATH", auth_file):
            os.environ.pop("OPENROUTER_API_KEY", None)
            result = auxiliary_max_tokens_param(500)
        assert result == {"max_tokens": 500}


# ---------------------------------------------------------------------------
# get_auxiliary_extra_body
# ---------------------------------------------------------------------------


class TestGetAuxiliaryExtraBody:
    def test_returns_nous_tags_when_nous(self):
        with patch("agent.auxiliary_client.auxiliary_is_nous", True):
            result = get_auxiliary_extra_body()
        assert result == NOUS_EXTRA_BODY
        # Should be a copy, not the original
        assert result is not NOUS_EXTRA_BODY

    def test_returns_empty_when_not_nous(self):
        with patch("agent.auxiliary_client.auxiliary_is_nous", False):
            result = get_auxiliary_extra_body()
        assert result == {}
