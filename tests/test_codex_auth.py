"""Tests for agent/codex_auth.py.

Run with: python -m pytest tests/test_codex_auth.py -v
"""

import base64
import json
import time
from pathlib import Path
from unittest.mock import patch

import pytest


class TestJwtClaims:
    """Tests for _jwt_claims JWT parsing."""

    def test_valid_jwt(self):
        from agent.codex_auth import _jwt_claims
        payload = {"sub": "user-123", "exp": 9999999999}
        encoded = base64.urlsafe_b64encode(json.dumps(payload).encode()).rstrip(b"=").decode()
        token = f"header.{encoded}.signature"
        claims = _jwt_claims(token)
        assert claims["sub"] == "user-123"
        assert claims["exp"] == 9999999999

    def test_empty_string(self):
        from agent.codex_auth import _jwt_claims
        assert _jwt_claims("") == {}

    def test_no_dots(self):
        from agent.codex_auth import _jwt_claims
        assert _jwt_claims("nodots") == {}

    def test_invalid_base64(self):
        from agent.codex_auth import _jwt_claims
        assert _jwt_claims("a.!!!.b") == {}

    def test_none_input(self):
        from agent.codex_auth import _jwt_claims
        assert _jwt_claims(None) == {}


class TestTokenExpired:
    """Tests for _token_expired."""

    def _make_token(self, exp: int) -> str:
        from agent.codex_auth import _jwt_claims
        payload = {"exp": exp}
        encoded = base64.urlsafe_b64encode(json.dumps(payload).encode()).rstrip(b"=").decode()
        return f"h.{encoded}.s"

    def test_not_expired(self):
        from agent.codex_auth import _token_expired
        future = int(time.time()) + 3600
        assert not _token_expired(self._make_token(future))

    def test_expired(self):
        from agent.codex_auth import _token_expired
        past = int(time.time()) - 100
        assert _token_expired(self._make_token(past))

    def test_within_skew(self):
        from agent.codex_auth import _token_expired
        # Expires 30s from now, but skew is 90s → should be considered expired
        almost = int(time.time()) + 30
        assert _token_expired(self._make_token(almost), skew_seconds=90)

    def test_no_exp_claim(self):
        from agent.codex_auth import _token_expired
        payload = {"sub": "test"}
        encoded = base64.urlsafe_b64encode(json.dumps(payload).encode()).rstrip(b"=").decode()
        token = f"h.{encoded}.s"
        assert not _token_expired(token)


class TestHasCodexCredentials:
    """Tests for has_codex_credentials with mock auth.json."""

    def test_no_auth_file(self, tmp_path):
        auth_file = tmp_path / "nonexistent.json"
        with patch("agent.codex_auth.CODEX_AUTH_FILE", auth_file):
            from agent.codex_auth import has_codex_credentials
            assert not has_codex_credentials()

    def test_unknown_auth_mode(self, tmp_path):
        auth_file = tmp_path / "auth.json"
        auth_file.write_text(json.dumps({"auth_mode": "unknown"}))
        with patch("agent.codex_auth.CODEX_AUTH_FILE", auth_file):
            from agent.codex_auth import has_codex_credentials
            assert not has_codex_credentials()

    def test_api_key_mode_ignored(self, tmp_path):
        """api_key mode is not supported — should return False."""
        auth_file = tmp_path / "auth.json"
        auth_file.write_text(json.dumps({
            "auth_mode": "api_key",
            "OPENAI_API_KEY": "sk-test-key-12345",
        }))
        with patch("agent.codex_auth.CODEX_AUTH_FILE", auth_file):
            from agent.codex_auth import has_codex_credentials
            assert not has_codex_credentials()


class TestGetCodexModelIds:
    """Tests for get_codex_model_ids with mock cache files."""

    def test_from_cache(self, tmp_path):
        cache_file = tmp_path / "models_cache.json"
        cache_file.write_text(json.dumps({
            "models": [
                {"slug": "gpt-5.1-codex", "priority": 1, "supported_in_api": True},
                {"slug": "gpt-5-codex", "priority": 2, "supported_in_api": True},
                {"slug": "gpt-4o", "priority": 0},  # not a codex model
            ]
        }))
        config_file = tmp_path / "config.toml"
        with (
            patch("agent.codex_models.CODEX_MODELS_CACHE_FILE", cache_file),
            patch("agent.codex_models.CODEX_CONFIG_FILE", config_file),
        ):
            from agent.codex_models import get_codex_model_ids
            models = get_codex_model_ids()
            assert "gpt-5.1-codex" in models
            assert "gpt-5-codex" in models
            assert "gpt-4o" not in models

    def test_default_model_first(self, tmp_path):
        cache_file = tmp_path / "models_cache.json"
        cache_file.write_text(json.dumps({
            "models": [
                {"slug": "gpt-5.1-codex", "priority": 1},
                {"slug": "gpt-5-codex", "priority": 2},
            ]
        }))
        config_file = tmp_path / "config.toml"
        config_file.write_text('model = "gpt-5-codex"\n')
        with (
            patch("agent.codex_models.CODEX_MODELS_CACHE_FILE", cache_file),
            patch("agent.codex_models.CODEX_CONFIG_FILE", config_file),
        ):
            from agent.codex_models import get_codex_model_ids
            models = get_codex_model_ids()
            assert models[0] == "gpt-5-codex"

    def test_hidden_models_excluded(self, tmp_path):
        cache_file = tmp_path / "models_cache.json"
        cache_file.write_text(json.dumps({
            "models": [
                {"slug": "gpt-5.1-codex", "priority": 1, "visibility": "hidden"},
                {"slug": "gpt-5-codex", "priority": 2},
            ]
        }))
        config_file = tmp_path / "config.toml"
        with (
            patch("agent.codex_models.CODEX_MODELS_CACHE_FILE", cache_file),
            patch("agent.codex_models.CODEX_CONFIG_FILE", config_file),
        ):
            from agent.codex_models import get_codex_model_ids
            models = get_codex_model_ids()
            assert "gpt-5.1-codex" not in models
            assert "gpt-5-codex" in models

    def test_empty_cache(self, tmp_path):
        cache_file = tmp_path / "nonexistent.json"
        config_file = tmp_path / "config.toml"
        with (
            patch("agent.codex_models.CODEX_MODELS_CACHE_FILE", cache_file),
            patch("agent.codex_models.CODEX_CONFIG_FILE", config_file),
        ):
            from agent.codex_models import get_codex_model_ids
            assert get_codex_model_ids() == []


