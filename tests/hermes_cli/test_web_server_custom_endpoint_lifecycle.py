"""E2E lifecycle tests for the canonical custom-endpoint REST API.

Covers the PR #74297 review requirements end-to-end against a temp HERMES_HOME
(no network): v12 provider edit, credential storage in ``.env`` (never
config.yaml), ``api_mode`` → canonical ``transport``, disable, and re-enable.
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
    (home / "config.yaml").write_text("{}\n", encoding="utf-8")
    (home / ".env").write_text("", encoding="utf-8")

    monkeypatch.setattr(hermes_state, "DEFAULT_DB_PATH", home / "state.db")
    c = TestClient(app)
    c.headers[_SESSION_HEADER_NAME] = _SESSION_TOKEN
    return c


def _raw_config():
    from hermes_cli.config import read_raw_config

    return read_raw_config()


def _env_text():
    from hermes_constants import get_hermes_home

    return (get_hermes_home() / ".env").read_text(encoding="utf-8")


def test_create_endpoint_stores_key_in_env_not_config(client):
    """Issue 1: the API key must land in .env behind key_env, never in config.yaml."""
    resp = client.post(
        "/api/providers/custom-endpoints",
        json={
            "name": "Demo",
            "base_url": "https://demo.example/v1",
            "model": "demo-model",
            "api_key": "sk-secret-value",
        },
    )

    assert resp.status_code == 200
    body = resp.json()
    assert body["ok"] is True

    providers = _raw_config().get("providers", {})
    assert "demo" in providers
    entry = providers["demo"]
    # Key is referenced via key_env, NOT stored in plaintext.
    assert "api_key" not in entry
    assert entry.get("key_env")
    # The secret is in .env under the key_env var.
    assert "sk-secret-value" in _env_text()
    assert entry["key_env"] in _env_text()


def test_create_endpoint_without_model_is_allowed(client):
    """A custom endpoint can be created before its models are discovered."""
    resp = client.post(
        "/api/providers/custom-endpoints",
        json={"name": "Bare", "base_url": "https://bare.example/v1"},
    )

    assert resp.status_code == 200
    providers = _raw_config().get("providers", {})
    assert "bare" in providers
    assert providers["bare"]["base_url"] == "https://bare.example/v1"


def test_create_endpoint_carries_api_mode_as_transport(client):
    """api_mode maps to the canonical `transport` key the runtime reads."""
    resp = client.post(
        "/api/providers/custom-endpoints",
        json={
            "name": "Anth",
            "base_url": "https://anth.example/v1",
            "api_mode": "anthropic_messages",
        },
    )

    assert resp.status_code == 200
    entry = _raw_config()["providers"]["anth"]
    assert entry["transport"] == "anthropic_messages"


def test_blank_api_key_clears_env(client):
    """Re-saving with a blank key removes the .env secret + key_env reference."""
    client.post(
        "/api/providers/custom-endpoints",
        json={"name": "Demo", "base_url": "https://demo.example/v1", "api_key": "sk-first"},
    )
    assert "sk-first" in _env_text()

    client.post(
        "/api/providers/custom-endpoints",
        json={"name": "Demo", "base_url": "https://demo.example/v1", "api_key": ""},
    )

    entry = _raw_config()["providers"]["demo"]
    assert "key_env" not in entry
    assert "api_key" not in entry
    assert "sk-first" not in _env_text()


def test_edit_preserves_unrelated_fields(client):
    """Editing base_url must not drop a previously-stored transport/key_env."""
    client.post(
        "/api/providers/custom-endpoints",
        json={
            "name": "Demo",
            "base_url": "https://demo.example/v1",
            "api_key": "sk-keep",
            "api_mode": "anthropic_messages",
        },
    )

    # Edit only the base URL (no api_key, no api_mode supplied).
    client.post(
        "/api/providers/custom-endpoints",
        json={"name": "Demo", "base_url": "https://new.example/v1"},
    )

    entry = _raw_config()["providers"]["demo"]
    assert entry["base_url"] == "https://new.example/v1"
    assert entry["transport"] == "anthropic_messages"  # preserved
    assert entry.get("key_env")  # preserved


def test_disable_then_reenable_round_trip(client):
    """Issue 3: enablement is the canonical providers.<id>.enabled flag."""
    client.post(
        "/api/providers/custom-endpoints",
        json={"name": "Demo", "base_url": "https://demo.example/v1"},
    )

    # Disable.
    resp = client.post("/api/providers/custom-endpoints/demo/enable", json={"enabled": False})
    assert resp.status_code == 200
    assert _raw_config()["providers"]["demo"]["enabled"] is False

    # The custom-endpoints list reports it as disabled (authoritative source).
    listing = client.get("/api/providers/custom-endpoints").json()
    demo = next(e for e in listing["endpoints"] if e["id"] == "demo")
    assert demo["enabled"] is False

    # Re-enable removes the key.
    resp = client.post("/api/providers/custom-endpoints/demo/enable", json={"enabled": True})
    assert resp.status_code == 200
    assert "enabled" not in _raw_config()["providers"]["demo"]

    listing = client.get("/api/providers/custom-endpoints").json()
    demo = next(e for e in listing["endpoints"] if e["id"] == "demo")
    assert demo["enabled"] is True


def test_delete_removes_provider_and_env(client):
    client.post(
        "/api/providers/custom-endpoints",
        json={"name": "Demo", "base_url": "https://demo.example/v1", "api_key": "sk-gone"},
    )
    assert "sk-gone" in _env_text()

    resp = client.delete("/api/providers/custom-endpoints/demo")

    assert resp.status_code == 200
    assert "demo" not in _raw_config().get("providers", {})
    assert "sk-gone" not in _env_text()
