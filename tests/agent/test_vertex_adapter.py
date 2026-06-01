"""Tests for the Google Vertex AI adapter.

Covers:
  - Credential path detection from environment variables
  - Project ID resolution (env vars + SA JSON)
  - Auth source labeling
  - Region resolution
  - Token caching and refresh
  - Base URL construction
  - Edge cases: missing files, corrupt JSON, missing keys

Mirrors tests/agent/test_bedrock_adapter.py pattern.
"""

import json
import os
import tempfile
from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Credential path detection
# ---------------------------------------------------------------------------

class TestResolveCredentialsPath:
    """Test service account JSON path detection."""

    def test_prefers_vertex_credentials_path(self, monkeypatch):
        from agent.vertex_adapter import _resolve_credentials_path

        sa_path = "/tmp/fake-sa.json"
        # Create a real file so os.path.exists returns True
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            f.write('{"project_id": "test"}')
            sa_path = f.name

        monkeypatch.setenv("VERTEX_CREDENTIALS_PATH", sa_path)
        monkeypatch.setenv("GOOGLE_APPLICATION_CREDENTIALS", "/other/path.json")

        result = _resolve_credentials_path()
        assert result == sa_path
        os.unlink(sa_path)

    def test_falls_back_to_application_default_credentials(self, monkeypatch):
        from agent.vertex_adapter import _resolve_credentials_path

        sa_path = "/tmp/fake-gac.json"
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            f.write('{"project_id": "test"}')
            sa_path = f.name

        monkeypatch.delenv("VERTEX_CREDENTIALS_PATH", raising=False)
        monkeypatch.setenv("GOOGLE_APPLICATION_CREDENTIALS", sa_path)

        result = _resolve_credentials_path()
        assert result == sa_path
        os.unlink(sa_path)

    def test_returns_none_when_file_does_not_exist(self, monkeypatch):
        from agent.vertex_adapter import _resolve_credentials_path

        monkeypatch.setenv("GOOGLE_APPLICATION_CREDENTIALS", "/nonexistent/sa.json")
        monkeypatch.delenv("VERTEX_CREDENTIALS_PATH", raising=False)

        assert _resolve_credentials_path() is None

    def test_returns_none_when_no_env_vars_set(self, monkeypatch):
        from agent.vertex_adapter import _resolve_credentials_path

        monkeypatch.delenv("VERTEX_CREDENTIALS_PATH", raising=False)
        monkeypatch.delenv("GOOGLE_APPLICATION_CREDENTIALS", raising=False)

        assert _resolve_credentials_path() is None

    def test_ignores_whitespace_only_values(self, monkeypatch):
        from agent.vertex_adapter import _resolve_credentials_path

        monkeypatch.setenv("GOOGLE_APPLICATION_CREDENTIALS", "   ")
        monkeypatch.delenv("VERTEX_CREDENTIALS_PATH", raising=False)

        assert _resolve_credentials_path() is None


# ---------------------------------------------------------------------------
# Project ID resolution from SA JSON
# ---------------------------------------------------------------------------

class TestResolveProjectIdFromSA:
    """Test reading project_id from a service account JSON file."""

    def test_reads_project_id(self):
        from agent.vertex_adapter import _resolve_project_id_from_sa

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump({"project_id": "my-gcp-project-123"}, f)
            path = f.name

        result = _resolve_project_id_from_sa(path)
        assert result == "my-gcp-project-123"
        os.unlink(path)

    def test_returns_none_when_key_missing(self):
        from agent.vertex_adapter import _resolve_project_id_from_sa

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump({"client_email": "sa@project.iam.gserviceaccount.com"}, f)
            path = f.name

        result = _resolve_project_id_from_sa(path)
        assert result is None
        os.unlink(path)

    def test_returns_none_on_corrupt_json(self):
        from agent.vertex_adapter import _resolve_project_id_from_sa

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            f.write("not valid json {{{")
            path = f.name

        result = _resolve_project_id_from_sa(path)
        assert result is None
        os.unlink(path)

    def test_returns_none_on_missing_file(self):
        from agent.vertex_adapter import _resolve_project_id_from_sa

        assert _resolve_project_id_from_sa("/nonexistent/sa.json") is None


# ---------------------------------------------------------------------------
# Project ID resolution (priority chain)
# ---------------------------------------------------------------------------

class TestResolveProjectId:
    """Test project ID priority: VERTEX_PROJECT_ID > GOOGLE_CLOUD_PROJECT > SA JSON."""

    def test_prefers_vertex_project_id(self, monkeypatch):
        from agent.vertex_adapter import _resolve_project_id

        monkeypatch.setenv("VERTEX_PROJECT_ID", "vertex-pid")
        monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", "cloud-pid")
        monkeypatch.delenv("GOOGLE_APPLICATION_CREDENTIALS", raising=False)
        monkeypatch.delenv("VERTEX_CREDENTIALS_PATH", raising=False)

        assert _resolve_project_id() == "vertex-pid"

    def test_falls_back_to_google_cloud_project(self, monkeypatch):
        from agent.vertex_adapter import _resolve_project_id

        monkeypatch.delenv("VERTEX_PROJECT_ID", raising=False)
        monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", "cloud-pid")
        monkeypatch.delenv("GOOGLE_APPLICATION_CREDENTIALS", raising=False)
        monkeypatch.delenv("VERTEX_CREDENTIALS_PATH", raising=False)

        assert _resolve_project_id() == "cloud-pid"

    def test_falls_back_to_sa_json(self, monkeypatch):
        from agent.vertex_adapter import _resolve_project_id

        monkeypatch.delenv("VERTEX_PROJECT_ID", raising=False)
        monkeypatch.delenv("GOOGLE_CLOUD_PROJECT", raising=False)

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump({"project_id": "sa-pid"}, f)
            sa_path = f.name

        monkeypatch.setenv("GOOGLE_APPLICATION_CREDENTIALS", sa_path)

        result = _resolve_project_id()
        assert result == "sa-pid"
        os.unlink(sa_path)

    def test_returns_none_when_nothing_configured(self, monkeypatch):
        from agent.vertex_adapter import _resolve_project_id

        monkeypatch.delenv("VERTEX_PROJECT_ID", raising=False)
        monkeypatch.delenv("GOOGLE_CLOUD_PROJECT", raising=False)
        monkeypatch.delenv("GOOGLE_APPLICATION_CREDENTIALS", raising=False)
        monkeypatch.delenv("VERTEX_CREDENTIALS_PATH", raising=False)

        assert _resolve_project_id() is None


# ---------------------------------------------------------------------------
# Region resolution
# ---------------------------------------------------------------------------

class TestResolveVertexRegion:
    """Test Vertex AI region resolution."""

    def test_returns_global_by_default(self, monkeypatch):
        from agent.vertex_adapter import resolve_vertex_region

        monkeypatch.delenv("VERTEX_REGION", raising=False)
        assert resolve_vertex_region() == "global"

    def test_respects_vertex_region_env(self, monkeypatch):
        from agent.vertex_adapter import resolve_vertex_region

        monkeypatch.setenv("VERTEX_REGION", "us-central1")
        assert resolve_vertex_region() == "us-central1"

    def test_falls_back_to_global_for_whitespace(self, monkeypatch):
        from agent.vertex_adapter import resolve_vertex_region

        monkeypatch.setenv("VERTEX_REGION", "   ")
        assert resolve_vertex_region() == "global"


# ---------------------------------------------------------------------------
# Credential detection
# ---------------------------------------------------------------------------

class TestHasVertexCredentials:
    """Test credential detection for auto-detection flow."""

    def test_true_when_sa_json_exists(self, monkeypatch):
        from agent.vertex_adapter import has_vertex_credentials

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            f.write('{"project_id": "test"}')
            sa_path = f.name

        monkeypatch.setenv("GOOGLE_APPLICATION_CREDENTIALS", sa_path)
        monkeypatch.delenv("VERTEX_PROJECT_ID", raising=False)

        assert has_vertex_credentials() is True
        os.unlink(sa_path)

    def test_true_when_project_id_set(self, monkeypatch):
        from agent.vertex_adapter import has_vertex_credentials

        monkeypatch.setenv("VERTEX_PROJECT_ID", "my-project")
        monkeypatch.delenv("GOOGLE_APPLICATION_CREDENTIALS", raising=False)
        monkeypatch.delenv("VERTEX_CREDENTIALS_PATH", raising=False)

        assert has_vertex_credentials() is True

    def test_false_when_nothing_configured(self, monkeypatch):
        from agent.vertex_adapter import has_vertex_credentials

        monkeypatch.delenv("VERTEX_PROJECT_ID", raising=False)
        monkeypatch.delenv("GOOGLE_CLOUD_PROJECT", raising=False)
        monkeypatch.delenv("GOOGLE_APPLICATION_CREDENTIALS", raising=False)
        monkeypatch.delenv("VERTEX_CREDENTIALS_PATH", raising=False)

        assert has_vertex_credentials() is False


# ---------------------------------------------------------------------------
# Auth source labeling
# ---------------------------------------------------------------------------

class TestResolveVertexAuthSource:
    """Test human-readable auth source labels."""

    def test_labels_vertex_credentials_path(self, monkeypatch):
        from agent.vertex_adapter import resolve_vertex_auth_source

        monkeypatch.setenv("VERTEX_CREDENTIALS_PATH", "/path/to/sa.json")
        monkeypatch.delenv("GOOGLE_APPLICATION_CREDENTIALS", raising=False)

        assert resolve_vertex_auth_source() == "VERTEX_CREDENTIALS_PATH"

    def test_labels_application_default_credentials(self, monkeypatch):
        from agent.vertex_adapter import resolve_vertex_auth_source

        monkeypatch.delenv("VERTEX_CREDENTIALS_PATH", raising=False)
        monkeypatch.setenv("GOOGLE_APPLICATION_CREDENTIALS", "/path/to/sa.json")

        assert resolve_vertex_auth_source() == "GOOGLE_APPLICATION_CREDENTIALS"

    def test_labels_vertex_project_id(self, monkeypatch):
        from agent.vertex_adapter import resolve_vertex_auth_source

        monkeypatch.delenv("VERTEX_CREDENTIALS_PATH", raising=False)
        monkeypatch.delenv("GOOGLE_APPLICATION_CREDENTIALS", raising=False)
        monkeypatch.setenv("VERTEX_PROJECT_ID", "my-project")

        assert resolve_vertex_auth_source() == "VERTEX_PROJECT_ID"

    def test_labels_cloud_project(self, monkeypatch):
        from agent.vertex_adapter import resolve_vertex_auth_source

        monkeypatch.delenv("VERTEX_CREDENTIALS_PATH", raising=False)
        monkeypatch.delenv("GOOGLE_APPLICATION_CREDENTIALS", raising=False)
        monkeypatch.delenv("VERTEX_PROJECT_ID", raising=False)
        monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", "my-project")

        assert resolve_vertex_auth_source() == "GOOGLE_CLOUD_PROJECT"

    def test_returns_none_when_nothing_set(self, monkeypatch):
        from agent.vertex_adapter import resolve_vertex_auth_source

        monkeypatch.delenv("VERTEX_CREDENTIALS_PATH", raising=False)
        monkeypatch.delenv("GOOGLE_APPLICATION_CREDENTIALS", raising=False)
        monkeypatch.delenv("VERTEX_PROJECT_ID", raising=False)
        monkeypatch.delenv("GOOGLE_CLOUD_PROJECT", raising=False)

        assert resolve_vertex_auth_source() is None


# ---------------------------------------------------------------------------
# Token caching
# ---------------------------------------------------------------------------

class TestResolveVertexToken:
    """Test OAuth2 token resolution with caching behavior."""

    def test_returns_cached_token_when_fresh(self):
        import time
        from agent.vertex_adapter import _token_cache, _resolve_vertex_token

        # Prime the cache with a fake token
        _token_cache.clear()
        _token_cache["token"] = "cached-token"
        _token_cache["expires_at"] = time.time() + 3600

        # Mock the actual credential fetching to ensure it's NOT called
        with patch("agent.vertex_adapter._resolve_credentials_path",
                   return_value=None):
            result = _resolve_vertex_token()
            assert result == "cached-token"

    def test_refreshes_when_cache_stale(self):
        import time
        from agent.vertex_adapter import (
            _token_cache, _resolve_vertex_token,
        )

        # Set an expired cache
        _token_cache.clear()
        _token_cache["token"] = "old-token"
        _token_cache["expires_at"] = time.time() - 3600

        # Prime the module-scoped google-auth refs so _import_google_auth()
        # succeeds without actually importing google.auth.
        # Route through service_account path by providing a SA JSON path.
        mock_creds = MagicMock()
        mock_creds.token = "fresh-token"

        mock_sa = MagicMock()
        mock_sa.Credentials.from_service_account_file = MagicMock(
            return_value=mock_creds,
        )
        mock_transport = MagicMock()

        with patch(
            "agent.vertex_adapter._ga_service_account", mock_sa,
        ), patch(
            "agent.vertex_adapter._ga_transport", mock_transport,
        ), patch(
            "agent.vertex_adapter._ga_auth", MagicMock(),
        ), patch(
            "agent.vertex_adapter._resolve_credentials_path",
            return_value="/fake/sa.json",
        ):
            result = _resolve_vertex_token()
            assert result == "fresh-token"

    def test_returns_none_when_google_auth_not_installed(self, monkeypatch):
        from agent.vertex_adapter import _token_cache, _resolve_vertex_token

        _token_cache.clear()
        monkeypatch.delenv("GOOGLE_APPLICATION_CREDENTIALS", raising=False)
        monkeypatch.delenv("VERTEX_CREDENTIALS_PATH", raising=False)

        # Remove google from sys.modules so the import fails
        with patch.dict("sys.modules", {
            "google": None, "google.auth": None,
            "google.oauth2": None,
        }):
            result = _resolve_vertex_token()
            assert result is None


# ---------------------------------------------------------------------------
# Base URL construction
# ---------------------------------------------------------------------------

class TestGetVertexBaseUrl:
    """Test Vertex AI OpenAI-compatible base URL construction."""

    def test_builds_url_with_explicit_params(self, monkeypatch):
        from agent.vertex_adapter import get_vertex_base_url

        monkeypatch.delenv("VERTEX_REGION", raising=False)

        url = get_vertex_base_url(
            project_id="my-project",
            region="us-central1",
        )
        assert url == (
            "https://aiplatform.googleapis.com/v1/projects/my-project"
            "/locations/us-central1/endpoints/openapi"
        )

    def test_defaults_to_global_region(self, monkeypatch):
        from agent.vertex_adapter import get_vertex_base_url

        monkeypatch.setenv("VERTEX_PROJECT_ID", "my-project")
        monkeypatch.delenv("VERTEX_REGION", raising=False)

        url = get_vertex_base_url()
        assert "locations/global/endpoints/openapi" in url

    def test_returns_none_when_no_project_id(self, monkeypatch):
        from agent.vertex_adapter import get_vertex_base_url

        monkeypatch.delenv("VERTEX_PROJECT_ID", raising=False)
        monkeypatch.delenv("GOOGLE_CLOUD_PROJECT", raising=False)
        monkeypatch.delenv("GOOGLE_APPLICATION_CREDENTIALS", raising=False)
        monkeypatch.delenv("VERTEX_CREDENTIALS_PATH", raising=False)

        assert get_vertex_base_url() is None

    def test_respects_vertex_region_env(self, monkeypatch):
        from agent.vertex_adapter import get_vertex_base_url

        monkeypatch.setenv("VERTEX_PROJECT_ID", "my-project")
        monkeypatch.setenv("VERTEX_REGION", "europe-west4")

        url = get_vertex_base_url()
        assert "locations/europe-west4/endpoints/openapi" in url


# ---------------------------------------------------------------------------
# Integration: full resolution chain
# ---------------------------------------------------------------------------

class TestVertexAdapterIntegration:
    """End-to-end resolution without live API calls."""

    def test_full_chain_with_sa_json(self, monkeypatch):
        """When GOOGLE_APPLICATION_CREDENTIALS points to a valid SA JSON,
        the full resolution chain should produce a project_id, region,
        auth source label, and credential detection without errors."""
        from agent.vertex_adapter import (
            has_vertex_credentials,
            resolve_vertex_auth_source,
            _resolve_project_id,
            resolve_vertex_region,
            get_vertex_base_url,
        )

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump({
                "project_id": "test-project-123",
                "client_email": "sa@test-project-123.iam.gserviceaccount.com",
            }, f)
            sa_path = f.name

        monkeypatch.setenv("GOOGLE_APPLICATION_CREDENTIALS", sa_path)
        monkeypatch.delenv("VERTEX_PROJECT_ID", raising=False)
        monkeypatch.delenv("VERTEX_REGION", raising=False)

        assert has_vertex_credentials() is True
        assert resolve_vertex_auth_source() == "GOOGLE_APPLICATION_CREDENTIALS"
        assert _resolve_project_id() == "test-project-123"
        assert resolve_vertex_region() == "global"

        url = get_vertex_base_url()
        assert "test-project-123" in url
        assert "global" in url

        os.unlink(sa_path)
