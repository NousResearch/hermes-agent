"""Trust levels over HTTP: the level written through the API is the one the approval gate reads."""
import pytest


@pytest.fixture
def client():
    try:
        from starlette.testclient import TestClient
    except ImportError:
        pytest.skip("fastapi/starlette not installed")
    from hermes_cli.web_server import app, _SESSION_HEADER_NAME, _SESSION_TOKEN

    c = TestClient(app)
    c.headers[_SESSION_HEADER_NAME] = _SESSION_TOKEN
    return c


def test_level_round_trips_through_config_and_is_what_the_gate_reads(client):
    from agent import integration_trust as trust

    assert client.get("/api/trust").json()["levels"] == {"google": "ask"}

    assert client.post("/api/trust/level", json={"integration": "google", "level": "propose"}).json()["levels"] == {"google": "propose"}
    assert trust.trust_level("google", None) == "propose"


def test_unknown_integration_or_level_is_refused_and_changes_nothing(client):
    assert client.post("/api/trust/level", json={"integration": "slack", "level": "auto"}).status_code == 400
    assert client.post("/api/trust/level", json={"integration": "google", "level": "yolo"}).status_code == 400
    assert client.get("/api/trust").json()["levels"] == {"google": "ask"}
