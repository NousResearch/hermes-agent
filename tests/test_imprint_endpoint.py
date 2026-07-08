"""HTTP-surface tests for the imprint endpoints in hermes_cli.web_server."""

import pytest
from fastapi.testclient import TestClient

from hermes_cli import web_server


@pytest.fixture
def client(tmp_path, monkeypatch):
    # Isolate the profile home and neutralize the loopback token gate so the
    # endpoints are reachable in-process.
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setattr(web_server, "_has_valid_session_token", lambda request: True)
    return TestClient(web_server.app)


def test_record_flip_and_clear_roundtrip(client):
    assert client.get("/api/memory/imprints").json() == {"enabled": True, "imprints": []}

    up = client.post("/api/memory/imprint", json={"message_id": "m1", "valence": "up", "excerpt": "tight answer"})
    assert up.json() == {"ok": True, "recorded": True, "valence": "up"}
    assert client.get("/api/memory/imprints").json()["imprints"] == [{"message_id": "m1", "valence": "up"}]

    # Flipping to 👎 replaces, does not stack.
    client.post("/api/memory/imprint", json={"message_id": "m1", "valence": "down"})
    assert client.get("/api/memory/imprints").json()["imprints"] == [{"message_id": "m1", "valence": "down"}]

    cleared = client.post("/api/memory/imprint", json={"message_id": "m1", "valence": "none"})
    assert cleared.json() == {"ok": True, "recorded": False, "cleared": True}
    assert client.get("/api/memory/imprints").json()["imprints"] == []


def test_bad_input_rejected(client):
    assert client.post("/api/memory/imprint", json={"message_id": "m1", "valence": "sideways"}).status_code == 400
    assert client.post("/api/memory/imprint", json={"message_id": "", "valence": "up"}).status_code == 400


def test_disabled_records_nothing(client, monkeypatch):
    monkeypatch.setattr(web_server, "_imprints_enabled", lambda: False)
    r = client.post("/api/memory/imprint", json={"message_id": "m1", "valence": "up"})
    assert r.json() == {"ok": True, "recorded": False, "reason": "disabled"}
    # And the store stays empty.
    monkeypatch.setattr(web_server, "_imprints_enabled", lambda: True)
    assert client.get("/api/memory/imprints").json()["imprints"] == []


def test_requires_auth_without_token(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setattr(web_server, "_has_valid_session_token", lambda request: False)
    unauthed = TestClient(web_server.app)
    assert unauthed.get("/api/memory/imprints").status_code == 401
