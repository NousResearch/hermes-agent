"""PUT /api/env refuses a blank value instead of clobbering a working credential.

A blank save used to write ``KEY=`` over the live key AND, through the credential
lifecycle's mirror scrub, copy the blank into any config.yaml ``api_key`` that held the
old value. Removal has its own route (DELETE /api/env).
"""

import pytest


@pytest.fixture
def client(monkeypatch, _isolate_hermes_home):
    from starlette.testclient import TestClient

    import hermes_state
    from hermes_cli.web_server import _SESSION_HEADER_NAME, _SESSION_TOKEN, app
    from hermes_constants import get_hermes_home

    monkeypatch.setattr(hermes_state, "DEFAULT_DB_PATH", get_hermes_home() / "state.db")
    test_client = TestClient(app)
    test_client.headers[_SESSION_HEADER_NAME] = _SESSION_TOKEN
    return test_client


def test_env_rejects_blank_value_without_touching_stored_key(client):
    from hermes_cli.config import load_config, load_env, save_config, save_env_value

    key = "OPENROUTER_API_KEY"
    real = "sk-or-live-secret-abcdef1234567890"
    save_env_value(key, real)
    config = load_config()
    config["model"] = {"provider": "openrouter", "default": "x/y", "api_key": real}
    save_config(config)

    for blank in ("", "   ", "\t"):
        response = client.put("/api/env", json={"key": key, "value": blank})
        assert response.status_code == 400
        assert load_env()[key] == real
        assert load_config()["model"]["api_key"] == real

    rotated = "sk-or-rotated-0987654321"
    assert client.put("/api/env", json={"key": key, "value": rotated}).status_code == 200
    assert load_env()[key] == rotated
