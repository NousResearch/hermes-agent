"""E2E tests for POST /api/providers/custom-endpoints/{id}/enable.

The enablement toggle writes the canonical ``providers.<id>.enabled`` flag that
the runtime resolver and model picker honour (``is_provider_enabled``). These
tests exercise the real endpoint against a temp HERMES_HOME config (no network):
disable sets ``enabled: false``; re-enable removes the key; unknown id → 404.
"""
import pytest


@pytest.fixture
def client(tmp_path, monkeypatch, _isolate_hermes_home):
    try:
        from starlette.testclient import TestClient
    except ImportError:
        pytest.skip("fastapi/starlette not installed")

    import hermes_state
    from hermes_constants import get_hermes_home
    from hermes_cli.web_server import app, _SESSION_HEADER_NAME, _SESSION_TOKEN

    home = get_hermes_home()
    home.mkdir(parents=True, exist_ok=True)
    (home / "config.yaml").write_text(
        "providers:\n"
        "  demo:\n"
        "    name: Demo\n"
        "    base_url: https://demo.example/v1\n"
        "    model: demo-model\n",
        encoding="utf-8",
    )
    (home / ".env").write_text("", encoding="utf-8")

    monkeypatch.setattr(hermes_state, "DEFAULT_DB_PATH", home / "state.db")
    c = TestClient(app)
    c.headers[_SESSION_HEADER_NAME] = _SESSION_TOKEN
    return c


def _read_providers():
    from hermes_cli.config import read_raw_config

    return read_raw_config().get("providers", {})


def test_disable_sets_enabled_false(client):
    resp = client.post("/api/providers/custom-endpoints/demo/enable", json={"enabled": False})

    assert resp.status_code == 200
    body = resp.json()
    assert body["ok"] is True
    assert body["enabled"] is False
    assert _read_providers()["demo"]["enabled"] is False


def test_reenable_removes_enabled_key(client):
    # First disable, then re-enable.
    client.post("/api/providers/custom-endpoints/demo/enable", json={"enabled": False})
    assert _read_providers()["demo"]["enabled"] is False

    resp = client.post("/api/providers/custom-endpoints/demo/enable", json={"enabled": True})

    assert resp.status_code == 200
    assert resp.json()["enabled"] is True
    # Absence == enabled (config stays clean).
    assert "enabled" not in _read_providers()["demo"]


def test_enable_default_true_when_body_omitted(client):
    client.post("/api/providers/custom-endpoints/demo/enable", json={"enabled": False})

    resp = client.post("/api/providers/custom-endpoints/demo/enable", json={})

    assert resp.status_code == 200
    assert resp.json()["enabled"] is True
    assert "enabled" not in _read_providers()["demo"]


def test_unknown_endpoint_404(client):
    resp = client.post("/api/providers/custom-endpoints/nope/enable", json={"enabled": False})

    assert resp.status_code == 404


def test_disable_survives_other_entry_fields(client):
    """Toggling enabled must not clobber the endpoint's other config fields."""
    client.post("/api/providers/custom-endpoints/demo/enable", json={"enabled": False})

    demo = _read_providers()["demo"]
    assert demo["base_url"] == "https://demo.example/v1"
    assert demo["model"] == "demo-model"
    assert demo["enabled"] is False
