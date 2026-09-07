"""Dashboard ownership contracts with the real runtime and a disposable home."""
import runpy
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[2]
PLUGIN = ROOT / "plugins/hermes-realms"
pytestmark = pytest.mark.linux_only


@pytest.fixture
def dashboard(tmp_path, monkeypatch):
    home = tmp_path / "hermes"
    home.mkdir()
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("HERMES_HOME", str(home))
    (home / "config.yaml").write_text(
        "plugins:\n  realms:\n    default_mode: ask\n", encoding="utf-8"
    )
    api = runpy.run_path(str(PLUGIN / "dashboard/plugin_api.py"))
    service = api["get_integration"]()
    assert service.home == home.resolve()
    assert Path(api["_integration"].__file__).resolve() == PLUGIN / "realms/integration.py"
    import hermes_constants

    assert Path(hermes_constants.__file__).resolve() == ROOT / "hermes_constants.py"
    app = FastAPI()
    app.include_router(api["router"], prefix="/api/plugins/hermes-realms")
    try:
        with TestClient(app, base_url="http://localhost") as client:
            yield client, service
    finally:
        service.unload()


def ownership_rows(service):
    with service.owners.connection() as db:
        return (
            db.execute("SELECT id, mode FROM owners ORDER BY id").fetchall(),
            db.execute("SELECT kind, value, owner FROM aliases ORDER BY kind, value").fetchall(),
        )


def test_historical_listing_is_empty_without_registering_ownership(dashboard):
    client, service = dashboard
    owner = service.bind(session_id="current", runtime_session_id="current-runtime")
    service.owners.set_mode(owner, "host")
    before = ownership_rows(service)
    for identity in (
        {},
        {"stored_session_id": "historical-stored"},
        {"runtime_session_id": "historical-runtime"},
        {"runtime_session_id": "historical-runtime", "stored_session_id": "historical-stored"},
    ):
        response = client.get("/api/plugins/hermes-realms/realms", params=identity)
        assert response.status_code == 200, response.text
        assert response.json() == {"mode": service.manager.config.default_mode, "realms": []}
        assert ownership_rows(service) == before
    response = client.get(
        "/api/plugins/hermes-realms/realms", params={"runtime_session_id": "current-runtime"}
    )
    assert response.status_code == 200, response.text
    assert response.json() == {"mode": "host", "realms": []}
    assert ownership_rows(service) == before


def test_listing_and_watch_reject_invalid_ownership_without_binding(dashboard):
    client, service = dashboard
    service.bind(session_id="a", runtime_session_id="runtime-a", stored_session_id="stored-a")
    service.bind(session_id="b", runtime_session_id="runtime-b", stored_session_id="stored-b")
    before = ownership_rows(service)
    rejected = (
        {"runtime_session_id": "runtime-a", "stored_session_id": "unknown"},
        {"runtime_session_id": "unknown", "stored_session_id": "stored-a"},
        {"runtime_session_id": "runtime-a", "stored_session_id": "stored-b"},
        {"stored_session_id": " invalid "},
        {"runtime_session_id": ""},
        {"stored_session_id": ""},
        {"runtime_session_id": "", "stored_session_id": "unknown"},
        {"runtime_session_id": "unknown", "stored_session_id": ""},
        {"runtime_session_id": "", "stored_session_id": "stored-a"},
        {"runtime_session_id": "runtime-a", "stored_session_id": ""},
    )
    for identity in rejected:
        response = client.get("/api/plugins/hermes-realms/realms", params=identity)
        assert response.status_code == 403, (identity, response.text)
    for identity in (*rejected, {}, {"stored_session_id": "unknown"},
                     {"runtime_session_id": "unknown-runtime", "stored_session_id": "unknown"}):
        response = client.post("/api/plugins/hermes-realms/realms/missing/watch", json=identity)
        assert response.status_code == 403, (identity, response.text)
        assert response.json() == {"detail": "Session ownership mismatch"}
    assert ownership_rows(service) == before
