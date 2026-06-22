"""Tests for dashboard configured-model shortcut library endpoints."""

import json

import pytest
import yaml
from starlette.testclient import TestClient

from hermes_constants import get_hermes_home
from hermes_cli import web_server
from hermes_cli.dashboard_auth.public_paths import PUBLIC_API_PATHS


@pytest.fixture
def client(_isolate_hermes_home):
    previous_auth = getattr(web_server.app.state, "auth_required", None)
    previous_host = getattr(web_server.app.state, "bound_host", None)
    web_server.app.state.auth_required = False
    web_server.app.state.bound_host = None
    test_client = TestClient(web_server.app)
    test_client.headers[web_server._SESSION_HEADER_NAME] = web_server._SESSION_TOKEN
    try:
        yield test_client
    finally:
        test_client.close()
        if previous_auth is None:
            if hasattr(web_server.app.state, "auth_required"):
                delattr(web_server.app.state, "auth_required")
        else:
            web_server.app.state.auth_required = previous_auth
        if previous_host is None:
            if hasattr(web_server.app.state, "bound_host"):
                delattr(web_server.app.state, "bound_host")
        else:
            web_server.app.state.bound_host = previous_host


def test_model_library_route_is_not_public():
    assert "/api/model/library" not in PUBLIC_API_PATHS


def test_model_library_requires_dashboard_auth(_isolate_hermes_home):
    previous_auth = getattr(web_server.app.state, "auth_required", None)
    previous_host = getattr(web_server.app.state, "bound_host", None)
    web_server.app.state.auth_required = False
    web_server.app.state.bound_host = None
    unauthenticated = TestClient(web_server.app)
    try:
        response = unauthenticated.get("/api/model/library")
    finally:
        unauthenticated.close()
        if previous_auth is None:
            if hasattr(web_server.app.state, "auth_required"):
                delattr(web_server.app.state, "auth_required")
        else:
            web_server.app.state.auth_required = previous_auth
        if previous_host is None:
            if hasattr(web_server.app.state, "bound_host"):
                delattr(web_server.app.state, "bound_host")
        else:
            web_server.app.state.bound_host = previous_host

    assert response.status_code == 401


def test_model_library_crud_persists_profile_scoped_metadata_only(client, _isolate_hermes_home):
    response = client.post(
        "/api/model/library",
        json={
            "provider": "openrouter",
            "model": "anthropic/claude-sonnet-4",
            "baseUrl": "https://openrouter.ai/api/v1",
            "name": "Sonnet",
            "api_key": "sk-should-not-persist",
            "token": "tok-should-not-persist",
        },
    )

    assert response.status_code == 200
    row = response.json()
    assert row["provider"] == "openrouter"
    assert row["model"] == "anthropic/claude-sonnet-4"
    assert row["baseUrl"] == "https://openrouter.ai/api/v1"
    assert "api_key" not in row
    assert "token" not in row

    path = get_hermes_home() / "models.json"
    persisted = json.loads(path.read_text(encoding="utf-8"))
    assert persisted == [row]
    assert "api_key" not in persisted[0]
    assert "token" not in persisted[0]

    duplicate = client.post(
        "/api/model/library",
        json={
            "provider": "openrouter",
            "model": "anthropic/claude-sonnet-4",
            "baseUrl": "https://openrouter.ai/api/v1/",
            "name": "Duplicate Name",
        },
    )
    assert duplicate.status_code == 200
    assert duplicate.json()["id"] == row["id"]
    assert len(json.loads(path.read_text(encoding="utf-8"))) == 1

    listed = client.get("/api/model/library")
    assert listed.status_code == 200
    assert listed.json()["models"] == [row]

    patched = client.patch(
        f"/api/model/library/{row['id']}",
        json={"name": "Claude Sonnet", "api_key": "still-not-persisted"},
    )
    assert patched.status_code == 200
    updated = patched.json()["model"]
    assert updated["name"] == "Claude Sonnet"
    assert "api_key" not in updated
    assert "api_key" not in json.loads(path.read_text(encoding="utf-8"))[0]

    deleted = client.delete(f"/api/model/library/{row['id']}")
    assert deleted.status_code == 200
    assert json.loads(path.read_text(encoding="utf-8")) == []


def test_model_library_honors_profile_query_scope(client, tmp_path, monkeypatch):
    profile_home = tmp_path / "profiles" / "coder"
    profile_home.mkdir(parents=True)
    monkeypatch.setattr(web_server, "_resolve_profile_dir", lambda name: profile_home)

    response = client.post(
        "/api/model/library?profile=coder",
        json={"provider": "custom", "model": "local-model", "baseUrl": "http://127.0.0.1:8000/v1"},
    )

    assert response.status_code == 200
    assert not (tmp_path / "models.json").exists()
    profile_rows = json.loads((profile_home / "models.json").read_text(encoding="utf-8"))
    assert profile_rows[0]["provider"] == "custom"
    assert profile_rows[0]["model"] == "local-model"


def test_model_library_rejects_missing_provider_or_model(client):
    response = client.post("/api/model/library", json={"provider": "openrouter"})
    assert response.status_code == 400
    assert response.json()["detail"] == "provider and model required"


@pytest.mark.parametrize(
    "base_url",
    [
        "https://user:password@example.com/v1",
        "https://example.com/v1?api_key=secret",
        "https://example.com/v1?token=secret",
    ],
)
def test_model_library_rejects_credential_bearing_base_url(client, base_url):
    response = client.post(
        "/api/model/library",
        json={"provider": "custom", "model": "local-model", "baseUrl": base_url},
    )

    assert response.status_code == 400
    assert response.json()["detail"] == "baseUrl must not contain credentials"


def test_model_library_sanitizes_credential_bearing_active_base_url(client):
    config_path = get_hermes_home() / "config.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "model": {
                    "provider": "custom",
                    "default": "local-model",
                    "base_url": "https://user:password@example.com/v1",
                }
            }
        ),
        encoding="utf-8",
    )

    response = client.get("/api/model/library")

    assert response.status_code == 200
    active = response.json()["models"][0]
    assert active["provider"] == "custom"
    assert active["model"] == "local-model"
    assert active["baseUrl"] == ""
