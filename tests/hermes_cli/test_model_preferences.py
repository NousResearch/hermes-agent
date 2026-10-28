from concurrent.futures import ThreadPoolExecutor

import pytest

from hermes_cli.model_preferences import ModelPreferences, PreferenceConflict


def test_instance_scope_cancellation_and_independent_fields(tmp_path):
    a = ModelPreferences(tmp_path / "a")
    b = ModelPreferences(tmp_path / "b")
    first = a.snapshot()
    assert first["instanceId"] != b.snapshot()["instanceId"]
    a.set(first["instanceId"], "p/a", "m::b", "favorite", 0, True)
    a.set(first["instanceId"], "p/a", "m::b", "hidden", 0, True)
    a.set(first["instanceId"], "p/a", "m::b", "favorite", 1, False)
    with pytest.raises(PreferenceConflict):
        a.set(first["instanceId"], "p/a", "m::b", "favorite", 0, True)
    restored = ModelPreferences(tmp_path / "a").snapshot()
    assert restored["instanceId"] == first["instanceId"]
    assert restored["rows"][0]["favorite"] is False
    assert restored["rows"][0]["hidden"] is True
    assert b.snapshot()["rows"] == []
    with pytest.raises(PreferenceConflict):
        b.set(first["instanceId"], "p/a", "m::b", "favorite", 0, True)


def test_concurrent_clients_have_one_winner_and_no_partial_write(tmp_path):
    store = ModelPreferences(tmp_path)
    instance = store.snapshot()["instanceId"]

    def attempt(value):
        try:
            return store.set(instance, "p", "m", "favorite", 0, value)
        except PreferenceConflict:
            return None

    with ThreadPoolExecutor(2) as pool:
        outcomes = list(pool.map(attempt, [True, False]))
    assert sum(value is not None for value in outcomes) == 1
    assert store.snapshot()["revision"] == "1"


def test_authenticated_api_uses_launch_instance_and_rejects_invalid_writes(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient
    from hermes_cli import web_server
    from hermes_cli.web_routers import models

    monkeypatch.setattr(models, "get_routing_process_hermes_home", lambda: tmp_path)
    client = TestClient(web_server.app)
    assert client.get("/api/model/preferences").status_code == 401
    headers = {"X-Hermes-Session-Token": web_server._SESSION_TOKEN}
    first = client.get("/api/model/preferences", headers=headers)
    assert first.status_code == 200
    assert first.headers["cache-control"] == "private, no-store"
    snapshot = first.json()
    body = {"instanceId": snapshot["instanceId"], "provider": "p", "model": "m", "field": "favorite", "baseRevision": "0", "value": True}
    assert client.post("/api/model/preferences", headers=headers, json=body).status_code == 200
    assert client.post("/api/model/preferences", headers=headers, json=body).status_code == 409
    assert client.post("/api/model/preferences", headers=headers, json={**body, "profile": "other"}).status_code == 422
    assert client.post("/api/model/preferences", headers=headers, json={**body, "value": "true"}).status_code == 422
    assert client.get("/api/model/preferences", headers=headers).json()["rows"][0]["favorite"] is True
