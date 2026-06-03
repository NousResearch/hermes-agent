"""Tests for v2.10 Multi-Entry Session Binding API endpoints."""

import pytest
from fastapi.testclient import TestClient

from hermes_cli import web_server


@pytest.fixture
def client():
    """Create a test client for the web server with auth headers."""
    return TestClient(web_server.app)


def _auth_headers() -> dict[str, str]:
    return {web_server._SESSION_HEADER_NAME: web_server._SESSION_TOKEN}


class TestV210WorkspacesAPI:
    """Tests for /api/v2.10/workspaces endpoints."""

    def test_list_workspaces_returns_ok(self, client):
        resp = client.get("/api/v2.10/workspaces", headers=_auth_headers())
        assert resp.status_code == 200
        data = resp.json()
        assert "workspaces" in data
        assert "total" in data
        assert isinstance(data["workspaces"], list)

    def test_get_workspace_not_found(self, client):
        resp = client.get("/api/v2.10/workspaces/nonexistent-id", headers=_auth_headers())
        assert resp.status_code in (404, 500)

    def test_list_workspaces_requires_auth(self, client):
        resp = client.get("/api/v2.10/workspaces")
        assert resp.status_code == 401

    def test_list_workspaces_schema(self, client):
        resp = client.get("/api/v2.10/workspaces", headers=_auth_headers())
        assert resp.status_code == 200
        data = resp.json()
        for ws in data["workspaces"]:
            assert "workspace_id" in ws
            assert "name" in ws
            assert "entrypoint" in ws
            assert "created_at" in ws


class TestV210SessionsAPI:
    """Tests for /api/v2.10/sessions endpoints."""

    def test_list_sessions_returns_ok(self, client):
        resp = client.get("/api/v2.10/sessions", headers=_auth_headers())
        assert resp.status_code == 200
        data = resp.json()
        assert "sessions" in data
        assert "total" in data
        assert isinstance(data["sessions"], list)

    def test_list_sessions_with_workspace_filter(self, client):
        resp = client.get("/api/v2.10/sessions?workspace_id=hermes-local", headers=_auth_headers())
        assert resp.status_code == 200
        data = resp.json()
        assert "sessions" in data

    def test_get_session_not_found(self, client):
        resp = client.get("/api/v2.10/sessions/nonexistent-id", headers=_auth_headers())
        assert resp.status_code in (404, 500)

    def test_list_sessions_requires_auth(self, client):
        resp = client.get("/api/v2.10/sessions")
        assert resp.status_code == 401

    def test_list_sessions_schema(self, client):
        resp = client.get("/api/v2.10/sessions", headers=_auth_headers())
        assert resp.status_code == 200
        data = resp.json()
        for s in data["sessions"]:
            assert "session_id" in s
            assert "workspace_id" in s
            assert "name" in s
            assert "entrypoint" in s
            assert "created_at" in s
            assert "updated_at" in s


class TestV210AdaptersHealthAPI:
    """Tests for /api/v2.10/adapters/health endpoint."""

    def test_adapters_health_returns_ok(self, client):
        resp = client.get("/api/v2.10/adapters/health", headers=_auth_headers())
        assert resp.status_code == 200
        data = resp.json()
        assert "adapters" in data
        assert "registered_entrypoints" in data
        assert "mode" in data
        assert isinstance(data["adapters"], dict)
        assert isinstance(data["registered_entrypoints"], list)

    def test_adapters_health_schema(self, client):
        resp = client.get("/api/v2.10/adapters/health", headers=_auth_headers())
        data = resp.json()
        assert data["mode"] in ("cli_legacy", "multi_entry")
        if data["mode"] == "cli_legacy":
            assert "note" in data

    def test_adapters_health_requires_auth(self, client):
        resp = client.get("/api/v2.10/adapters/health")
        assert resp.status_code == 401
