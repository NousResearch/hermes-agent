"""The Models card negotiates runtime support, not a user-editable config key."""

from contextlib import contextmanager

from fastapi import FastAPI
from fastapi.testclient import TestClient


def test_schema_advertises_delegation_runtime_support_in_requested_profile(monkeypatch):
    from hermes_cli.web_routers import config_env

    scopes = []

    @contextmanager
    def scoped(profile):
        scopes.append(profile)
        yield

    fields = {"delegation.model": {"type": "string"}}
    monkeypatch.setattr(config_env, "_config_profile_scope", scoped)
    monkeypatch.setattr(config_env, "_schema_with_dynamic_provider_options", lambda: fields)
    app = FastAPI()
    app.include_router(config_env.config_router)
    with TestClient(app) as client:
        response = client.get("/api/config/schema", params={"profile": "worker-profile"})
    assert response.status_code == 200
    payload = response.json()
    assert payload["capabilities"]["delegation_fallbacks"] is True
    assert payload["fields"] == fields
    assert scopes == ["worker-profile"]
