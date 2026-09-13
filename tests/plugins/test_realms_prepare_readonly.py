"""A cold setup preview must not create or migrate profile state."""
from pathlib import Path
import runpy

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

PLUGIN = Path(__file__).resolve().parents[2] / "plugins/hermes-realms"
load = runpy.run_path(str(PLUGIN / "realms/_binding.py"))["load_runtime"]


@pytest.mark.linux_only
@pytest.mark.parametrize("registered", [False, True])
def test_cold_prepare_is_read_only_even_for_unknown_owner(tmp_path, monkeypatch, registered):
    home = tmp_path / "profile"
    monkeypatch.setenv("HERMES_HOME", str(home))
    integration = load("integration")
    if registered:
        integration.RealmIntegration(home).bind(session_id="owner", runtime_session_id="runtime")
    integration._services.pop(home, None)
    api = runpy.run_path(str(PLUGIN / "dashboard/plugin_api.py"))
    app = FastAPI()
    app.include_router(api["router"])
    with TestClient(app) as client, monkeypatch.context() as guarded:
        def mutation(*args, **kwargs):
            raise AssertionError("cold prepare attempted a filesystem mutation")
        guarded.setattr(Path, "mkdir", mutation)
        guarded.setattr(Path, "chmod", mutation)
        response = client.post("/realms/setup/prepare", json={"runtime_session_id": "runtime", "kind": "realm"})
    assert response.status_code == (200 if registered else 403)
    assert home.exists() == registered
    assert home not in integration._services
