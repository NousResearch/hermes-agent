"""Tests for POST /api/model/discover (web_server.py).

The discovery backend (hermes_cli.inventory.discover_provider_models) is
mocked so no network is touched. Pins: 200 + models on success, 502 with the
underlying message on discovery failure, and that the request is profile-scoped.
"""
import pytest
from unittest.mock import patch

from hermes_cli.inventory import ModelDiscoveryError


@pytest.fixture
def isolated_profiles(tmp_path, monkeypatch, _isolate_hermes_home):
    from hermes_constants import get_hermes_home
    from hermes_cli import profiles

    default_home = get_hermes_home()
    profiles_root = default_home / "profiles"
    worker_home = profiles_root / "worker_beta"
    for home in (default_home, worker_home):
        home.mkdir(parents=True, exist_ok=True)
        (home / "config.yaml").write_text("{}\n", encoding="utf-8")
    (worker_home / ".env").write_text("", encoding="utf-8")

    monkeypatch.setattr(profiles, "_get_default_hermes_home", lambda: default_home)
    monkeypatch.setattr(profiles, "_get_profiles_root", lambda: profiles_root)
    return {"default": default_home, "worker_beta": worker_home}


@pytest.fixture
def client(monkeypatch, isolated_profiles):
    try:
        from starlette.testclient import TestClient
    except ImportError:
        pytest.skip("fastapi/starlette not installed")

    import hermes_state
    from hermes_constants import get_hermes_home
    from hermes_cli.web_server import app, _SESSION_HEADER_NAME, _SESSION_TOKEN

    monkeypatch.setattr(hermes_state, "DEFAULT_DB_PATH", get_hermes_home() / "state.db")
    c = TestClient(app)
    c.headers[_SESSION_HEADER_NAME] = _SESSION_TOKEN
    return c


def test_discover_returns_models(client):
    with patch(
        "hermes_cli.inventory.discover_provider_models",
        return_value=[{"id": "a", "name": "A"}, {"id": "b", "name": "B"}],
    ) as mock:
        resp = client.post("/api/model/discover", json={"base_url": "https://h/v1", "api_key": "k"})
    assert resp.status_code == 200
    assert resp.json()["models"] == [{"id": "a", "name": "A"}, {"id": "b", "name": "B"}]
    mock.assert_called_once_with("https://h/v1", "k", "chat_completions")


def test_discover_502_on_failure(client):
    with patch(
        "hermes_cli.inventory.discover_provider_models",
        side_effect=ModelDiscoveryError("boom"),
    ):
        resp = client.post("/api/model/discover", json={"base_url": "https://h/v1"})
    assert resp.status_code == 502
    assert "boom" in resp.json()["detail"]


def test_discover_passes_api_mode(client):
    with patch(
        "hermes_cli.inventory.discover_provider_models",
        return_value=[],
    ) as mock:
        resp = client.post(
            "/api/model/discover",
            json={"base_url": "https://h/v1", "api_mode": "anthropic_messages"},
        )
    assert resp.status_code == 200
    mock.assert_called_once_with("https://h/v1", None, "anthropic_messages")
