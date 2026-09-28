"""The dashboard session-import endpoint honors sessions.import_max_* from config.yaml."""

import pytest


@pytest.fixture()
def client(monkeypatch, _isolate_hermes_home):
    """A TestClient with the state DB isolated under the test HERMES_HOME."""
    try:
        from starlette.testclient import TestClient
    except ImportError:
        pytest.skip("fastapi/starlette not installed")

    import hermes_state
    from hermes_constants import get_hermes_home
    from hermes_cli.web_server import app, _SESSION_HEADER_NAME, _SESSION_TOKEN

    monkeypatch.setattr(hermes_state, "DEFAULT_DB_PATH", get_hermes_home() / "state.db")
    test_client = TestClient(app)
    test_client.headers[_SESSION_HEADER_NAME] = _SESSION_TOKEN
    return test_client


def test_import_sessions_endpoint_honors_configured_import_limit(client):
    """sessions.import_max_* in the served profile's config.yaml reaches the endpoint."""
    from hermes_constants import get_hermes_home

    payload = {"id": "limited-web-session", "source": "cli",
               "messages": [{"role": "user", "content": "x"}] * 3}
    config = get_hermes_home() / "config.yaml"
    original = config.read_text() if config.exists() else None
    try:
        config.write_text("sessions:\n  import_max_messages_per_session: 2\n")
        limited = client.post("/api/sessions/import", json={"sessions": [payload]})
        assert limited.status_code == 400
        assert limited.json()["detail"]["errors"][0]["error"] == (
            "messages exceeds the per-session import limit")

        config.write_text("sessions:\n  import_max_messages_per_session: 0\n")
        unbounded = client.post("/api/sessions/import", json={"sessions": [payload]})
        assert unbounded.status_code == 200
        assert unbounded.json()["imported"] == 1
    finally:
        if original is None:
            config.unlink(missing_ok=True)
        else:
            config.write_text(original)
