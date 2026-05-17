"""Tests for dashboard model configuration REST endpoints.

TDD: write failing tests, then implement in web_server.py.
"""
from __future__ import annotations

from unittest.mock import patch

import pytest

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient


@pytest.fixture
def client():
    """FastAPI test client with mocked load_config and valid session token."""
    from hermes_cli.web_server import app, _SESSION_TOKEN

    with patch("hermes_cli.web_server.load_config") as mock_load:
        mock_load.return_value = {}
        with TestClient(app, headers={
            "X-Hermes-Session-Token": _SESSION_TOKEN,
        }) as c:
            yield c, mock_load


# ── GET /api/model/configured ─────────────────────────────────────────

class TestGetConfigured:
    def test_returns_main_and_empty_fallbacks(self, client):
        c, mock = client
        mock.return_value = {
            "model": {"provider": "openrouter", "default": "anthropic/claude-sonnet-4"},
        }
        resp = c.get("/api/model/configured")
        assert resp.status_code == 200
        data = resp.json()
        assert data["main"]["provider"] == "openrouter"
        assert data["main"]["model"] == "anthropic/claude-sonnet-4"
        assert data["main"]["id"] == "openrouter/anthropic/claude-sonnet-4"
        assert data["fallbacks"] == []

    def test_returns_fallback_chain(self, client):
        c, mock = client
        mock.return_value = {
            "model": {"provider": "openrouter", "default": "anthropic/claude-sonnet-4"},
            "fallback_providers": [
                {"provider": "gemini", "model": "gemini-2.5-flash"},
                {"provider": "openrouter", "model": "openai/gpt-4o-mini"},
            ],
        }
        resp = c.get("/api/model/configured")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["fallbacks"]) == 2
        assert data["fallbacks"][0]["id"] == "gemini/gemini-2.5-flash"
        assert data["fallbacks"][1]["id"] == "openrouter/openai/gpt-4o-mini"

    def test_returns_auxiliary_assignments(self, client):
        c, mock = client
        mock.return_value = {
            "model": {"provider": "openrouter", "default": "anthropic/claude-sonnet-4"},
            "auxiliary": {
                "vision": {"provider": "openrouter", "model": "openai/gpt-4o-mini"},
                "compression": {"provider": "auto"},
            },
        }
        resp = c.get("/api/model/configured")
        assert resp.status_code == 200
        data = resp.json()
        tasks = {t["task"]: t for t in data["auxiliary"]}
        assert tasks["vision"]["provider"] == "openrouter"
        assert tasks["vision"]["model"] == "openai/gpt-4o-mini"
        # auto → resolved to main model
        assert tasks["compression"]["provider"] == "openrouter"
        assert tasks["compression"]["model"] == "anthropic/claude-sonnet-4"

    def test_empty_config(self, client):
        c, mock = client
        mock.return_value = {}
        resp = c.get("/api/model/configured")
        assert resp.status_code == 200
        data = resp.json()
        assert data["main"] is None
        assert data["fallbacks"] == []


# ── PUT /api/model/fallbacks ──────────────────────────────────────────

class TestSetFallbacks:
    def test_reorders_fallback_providers(self, client):
        c, mock = client
        mock.return_value = {
            "model": {"provider": "openrouter", "default": "anthropic/claude-sonnet-4"},
            "fallback_providers": [
                {"provider": "openrouter", "model": "openai/gpt-4o-mini"},
                {"provider": "gemini", "model": "gemini-2.5-flash"},
            ],
        }
        new_order = [
            {"provider": "gemini", "model": "gemini-2.5-flash"},
            {"provider": "openrouter", "model": "openai/gpt-4o-mini"},
        ]
        resp = c.put("/api/model/fallbacks", json={"fallbacks": new_order})
        assert resp.status_code == 200
        data = resp.json()
        assert data["ok"] is True
        assert len(data["fallbacks"]) == 2
        assert data["fallbacks"][0]["id"] == "gemini/gemini-2.5-flash"
        assert data["fallbacks"][1]["id"] == "openrouter/openai/gpt-4o-mini"

    def test_clears_fallback_providers(self, client):
        c, mock = client
        mock.return_value = {
            "model": {"provider": "openrouter", "default": "anthropic/claude-sonnet-4"},
            "fallback_providers": [
                {"provider": "gemini", "model": "gemini-2.5-flash"},
            ],
        }
        resp = c.put("/api/model/fallbacks", json={"fallbacks": []})
        assert resp.status_code == 200
        data = resp.json()
        assert data["ok"] is True
        assert data["fallbacks"] == []

    def test_rejects_empty_provider(self, client):
        c, mock = client
        mock.return_value = {"model": {"provider": "openrouter", "default": "anthropic/claude-sonnet-4"}}
        bad_order = [{"provider": "", "model": "some-model"}]
        resp = c.put("/api/model/fallbacks", json={"fallbacks": bad_order})
        assert resp.status_code == 400

    def test_clears_legacy_fallback_model_key(self, client):
        c, mock = client
        mock.return_value = {
            "model": {"provider": "openrouter", "default": "anthropic/claude-sonnet-4"},
            "fallback_model": {"provider": "gemini", "model": "gemini-2.5-flash"},
        }
        new_order = [
            {"provider": "openrouter", "model": "openai/gpt-4o-mini"},
        ]
        resp = c.put("/api/model/fallbacks", json={"fallbacks": new_order})
        assert resp.status_code == 200


# ── POST /api/model/register ──────────────────────────────────────────

class TestRegisterModel:
    def test_adds_model_to_registry(self, client):
        c, mock = client
        mock.return_value = {"model_registry": {"providers": {}}}
        resp = c.post("/api/model/register", json={
            "provider": "openrouter",
            "model": "custom-model-v1",
            "capabilities": {"vision": True},
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["ok"] is True
        assert data["id"] == "openrouter/custom-model-v1"

    def test_returns_conflict_when_model_already_registered(self, client):
        c, mock = client
        mock.return_value = {
            "model_registry": {
                "providers": {
                    "openrouter": {
                        "models": [
                            {"id": "openrouter/existing-model", "vision": False},
                        ]
                    }
                }
            }
        }
        resp = c.post("/api/model/register", json={
            "provider": "openrouter",
            "model": "existing-model",
        })
        assert resp.status_code == 409


# ── Existing endpoints still work ─────────────────────────────────────

class TestExistingEndpointsStillWork:
    def test_post_model_set_main(self, client):
        c, mock = client
        mock.return_value = {
            "model": {"provider": "openrouter", "default": "anthropic/claude-sonnet-4"},
        }
        resp = c.post("/api/model/set", json={
            "scope": "main",
            "provider": "gemini",
            "model": "gemini-2.5-flash",
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["ok"] is True
