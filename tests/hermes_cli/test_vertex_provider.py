"""Tests for Vertex AI provider registration and resolution in auth.py.

Mirrors tests/hermes_cli/test_gmi_provider.py pattern.
"""

from __future__ import annotations

import sys
import types
from unittest.mock import patch

import pytest

if "dotenv" not in sys.modules:
    fake_dotenv = types.ModuleType("dotenv")
    fake_dotenv.load_dotenv = lambda *args, **kwargs: None
    sys.modules["dotenv"] = fake_dotenv

from hermes_cli.auth import PROVIDER_REGISTRY, resolve_provider, get_auth_status


@pytest.fixture(autouse=True)
def _clear_provider_env(monkeypatch):
    """Clear provider env vars to prevent leakage."""
    for key in (
        "OPENROUTER_API_KEY", "OPENAI_API_KEY", "GOOGLE_API_KEY",
        "VERTEX_PROJECT_ID", "GOOGLE_CLOUD_PROJECT",
        "GOOGLE_APPLICATION_CREDENTIALS", "VERTEX_CREDENTIALS_PATH",
        "VERTEX_BASE_URL", "VERTEX_REGION",
    ):
        monkeypatch.delenv(key, raising=False)


class TestVertexProviderRegistration:
    """Test that vertex is registered in PROVIDER_REGISTRY."""

    def test_vertex_in_registry(self):
        assert "vertex" in PROVIDER_REGISTRY

    def test_vertex_config_auth_type(self):
        config = PROVIDER_REGISTRY["vertex"]
        assert config.auth_type == "gcp_sdk"

    def test_vertex_config_name(self):
        config = PROVIDER_REGISTRY["vertex"]
        assert config.name == "Google Vertex AI"

    def test_vertex_config_no_api_key_env_vars(self):
        config = PROVIDER_REGISTRY["vertex"]
        assert config.api_key_env_vars == ()

    def test_vertex_config_base_url_env_var(self):
        config = PROVIDER_REGISTRY["vertex"]
        assert config.base_url_env_var == "VERTEX_BASE_URL"


class TestVertexAliases:
    """Test provider alias resolution."""

    @pytest.mark.parametrize("alias", [
        "vertex-ai", "google-vertex", "gcp-vertex", "google-vertex-ai",
    ])
    def test_alias_resolves_to_vertex(self, alias, monkeypatch):
        monkeypatch.setenv("VERTEX_PROJECT_ID", "test-project")
        assert resolve_provider(alias) == "vertex"

    def test_canonical_name_resolves(self, monkeypatch):
        monkeypatch.setenv("VERTEX_PROJECT_ID", "test-project")
        assert resolve_provider("vertex") == "vertex"


class TestVertexAuthStatus:
    """Test get_auth_status for vertex provider."""

    def test_vertex_has_credentials(self, monkeypatch):
        monkeypatch.setenv("VERTEX_PROJECT_ID", "test-project")
        status = get_auth_status("vertex")
        assert status["logged_in"] is True
        assert status["provider"] == "vertex"

    def test_vertex_no_credentials(self, monkeypatch):
        monkeypatch.delenv("VERTEX_PROJECT_ID", raising=False)
        monkeypatch.delenv("GOOGLE_CLOUD_PROJECT", raising=False)
        monkeypatch.delenv("GOOGLE_APPLICATION_CREDENTIALS", raising=False)
        monkeypatch.delenv("VERTEX_CREDENTIALS_PATH", raising=False)

        status = get_auth_status("vertex")
        assert status["logged_in"] is False
        assert status["provider"] == "vertex"

    def test_vertex_auth_status_import_error_fallback(self, monkeypatch):
        """When vertex_adapter can't be imported, auth_status returns
        logged_in=False with an error message."""
        monkeypatch.setenv("VERTEX_PROJECT_ID", "test-project")

        with patch.dict("sys.modules", {
            "agent.vertex_adapter": None,
        }):
            status = get_auth_status("vertex")
            assert status["logged_in"] is False
            assert "error" in status
