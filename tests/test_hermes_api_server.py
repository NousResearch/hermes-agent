from unittest.mock import MagicMock

import pytest


fastapi = pytest.importorskip("fastapi")
pytest.importorskip("starlette")

from fastapi.testclient import TestClient

import hermes_api_server as api_server


@pytest.fixture
def client():
    return TestClient(api_server.app)


def test_status_rejects_missing_api_key(client, monkeypatch):
    monkeypatch.delenv("API_SERVER_KEY", raising=False)

    assert client.get("/status").status_code == 503


def test_status_requires_matching_bearer_token(client, monkeypatch):
    monkeypatch.setenv("API_SERVER_KEY", "a" * 32)

    assert client.get("/status").status_code == 401
    assert client.get("/status", headers={"Authorization": "Bearer wrong"}).status_code == 401
    assert client.get("/status", headers={"Authorization": "Bearer " + "a" * 32}).status_code == 200


def test_send_reaches_agent_only_with_valid_token(client, monkeypatch):
    monkeypatch.setenv("API_SERVER_KEY", "a" * 32)
    agent = MagicMock()
    agent.chat.return_value = "ok"
    monkeypatch.setattr(api_server, "AIAgent", lambda **_kwargs: agent)

    assert client.post("/send", json={"message": "hello"}).status_code == 401

    response = client.post(
        "/send",
        json={"message": "hello"},
        headers={"Authorization": "Bearer " + "a" * 32},
    )

    assert response.status_code == 200
    agent.chat.assert_called_once_with("hello")


def test_healthz_stays_available_without_api_key(client, monkeypatch):
    monkeypatch.delenv("API_SERVER_KEY", raising=False)

    assert client.get("/healthz").status_code == 200
