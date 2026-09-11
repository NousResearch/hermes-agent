"""Regression coverage for dashboard usage reads against a damaged state.db."""

import pytest


@pytest.fixture
def dashboard_client(monkeypatch, _isolate_hermes_home):
    try:
        from starlette.testclient import TestClient
    except ImportError:
        pytest.skip("fastapi/starlette not installed")

    import hermes_state
    from hermes_constants import get_hermes_home
    from hermes_cli.web_server import app, _SESSION_HEADER_NAME, _SESSION_TOKEN

    monkeypatch.setattr(
        hermes_state, "DEFAULT_DB_PATH", get_hermes_home() / "state.db"
    )
    client = TestClient(app)
    client.headers[_SESSION_HEADER_NAME] = _SESSION_TOKEN
    return client


def test_corrupt_usage_db_fails_closed_and_opens_circuit(
    monkeypatch, dashboard_client
):
    from hermes_constants import get_hermes_home
    from hermes_cli.web_routers import analytics

    db_path = get_hermes_home() / "state.db"
    damaged_bytes = b"this is not a sqlite database"
    db_path.write_bytes(damaged_bytes)

    monkeypatch.setattr(analytics, "_usage_db_circuit", {})
    monkeypatch.setattr(analytics, "_usage_db_last_error_log", {})
    log_calls = []
    monkeypatch.setattr(
        analytics._log,
        "error",
        lambda *args, **kwargs: log_calls.append((args, kwargs)),
    )
    real_get_usage = analytics._get_usage_analytics
    db_attempts = []

    def counted_get_usage(*args, **kwargs):
        db_attempts.append((args, kwargs))
        return real_get_usage(*args, **kwargs)

    monkeypatch.setattr(analytics, "_get_usage_analytics", counted_get_usage)

    first = dashboard_client.get("/api/analytics/usage?days=7")
    second = dashboard_client.get("/api/analytics/usage?days=7")

    assert first.status_code == 503
    assert second.status_code == 503
    assert first.headers["retry-after"] == "300"
    assert "state.db is corrupt" in first.json()["detail"]
    assert "hermes doctor" in first.json()["detail"]
    assert len(db_attempts) == 1
    assert len(log_calls) == 1
    assert "Traceback" not in str(log_calls[0])
    assert db_path.read_bytes() == damaged_bytes
