"""Dashboard API tests for Memory Wiki endpoints."""

import pytest


@pytest.fixture
def memory_client(monkeypatch, _isolate_hermes_home):
    try:
        from starlette.testclient import TestClient
    except ImportError:
        pytest.skip("fastapi/starlette not installed")

    import hermes_state
    from hermes_constants import get_hermes_home
    from hermes_cli.web_server import app, _SESSION_HEADER_NAME, _SESSION_TOKEN

    db_path = get_hermes_home() / "state.db"
    monkeypatch.setattr(hermes_state, "DEFAULT_DB_PATH", db_path)

    db = hermes_state.SessionDB(db_path=db_path)
    try:
        db.create_session("s-memory", "discord")
        db.append_message("s-memory", "user", "Build Memory Wiki dashboard in web/src/App.tsx")
        db.append_message("s-memory", "assistant", "Implemented Memory Wiki subject cards.")
        db.append_message("s-memory", "tool", "", tool_name="search_files")

        db.create_session("s-fastapi", "cli")
        db.append_message("s-fastapi", "user", "Debug FastAPI pytest failure")
        db.append_message("s-fastapi", "assistant", "Planned and fixed the FastAPI route bug.")
    finally:
        db.close()

    client = TestClient(app)
    client.headers[_SESSION_HEADER_NAME] = _SESSION_TOKEN
    return client


def test_memory_overview_endpoint(memory_client):
    resp = memory_client.get("/api/memory/overview")

    assert resp.status_code == 200
    data = resp.json()
    assert "subjects" in data
    assert "daily_logs" in data
    assert "recent_sessions" in data
    assert any(subject["slug"] == "memory-wiki" for subject in data["subjects"])


def test_memory_subjects_endpoint_supports_query(memory_client):
    resp = memory_client.get("/api/memory/subjects?q=fastapi&limit=10")

    assert resp.status_code == 200
    data = resp.json()
    assert data["query"] == "fastapi"
    assert data["limit"] == 10
    assert any(subject["slug"] == "fastapi" for subject in data["subjects"])


def test_memory_subject_detail_endpoint(memory_client):
    resp = memory_client.get("/api/memory/subjects/memory-wiki")

    assert resp.status_code == 200
    data = resp.json()
    assert data["subject"]["slug"] == "memory-wiki"
    assert data["subject"]["session_count"] >= 1


def test_memory_subject_detail_returns_404(memory_client):
    resp = memory_client.get("/api/memory/subjects/not-a-real-subject")

    assert resp.status_code == 404


def test_memory_days_and_day_detail_endpoints(memory_client):
    days_resp = memory_client.get("/api/memory/days?limit=10")

    assert days_resp.status_code == 200
    days_data = days_resp.json()
    assert days_data["limit"] == 10
    assert days_data["daily_logs"]

    date = days_data["daily_logs"][0]["date"]
    day_resp = memory_client.get(f"/api/memory/days/{date}")
    assert day_resp.status_code == 200
    day_data = day_resp.json()
    assert day_data["daily_log"]["date"] == date
    assert day_data["daily_log"]["work_items"]


def test_memory_day_detail_returns_404(memory_client):
    resp = memory_client.get("/api/memory/days/1999-01-01")

    assert resp.status_code == 404
