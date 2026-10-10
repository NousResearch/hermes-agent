"""POST /api/cron/blueprints/render: a blueprint's filled job, without creating it.

Desktop's "Customize prompt" turns a recipe into an ordinary editable job, so the render must be
exactly what instantiate would create (prompt, schedule, delivery, skills) and must never create.
"""

import pytest
from starlette.testclient import TestClient

from hermes_cli import web_server
import hermes_cli.web_server_cron as _web_server_cron
from cron.blueprint_catalog import fill_blueprint, get_blueprint


@pytest.fixture()
def client(monkeypatch):
    monkeypatch.setattr(web_server, "_has_valid_session_token", lambda req: True)

    def no_create(*args, **kwargs):
        raise AssertionError("render must not create a cron job")

    monkeypatch.setattr(_web_server_cron, "_call_cron_for_profile", no_create)
    return TestClient(web_server.app)


def test_render_returns_the_job_instantiate_would_create(client):
    values = {"time": "07:30", "deliver": "local"}

    resp = client.post("/api/cron/blueprints/render", json={"blueprint": "morning-brief", "values": values})

    assert resp.status_code == 200
    expected = fill_blueprint(get_blueprint("morning-brief"), values)
    expected.pop("origin", None)
    assert resp.json() == expected
    assert resp.json()["schedule"] == "30 7 * * *"
    assert resp.json()["skills"] == ["google-workspace"]


def test_render_rejects_an_unknown_slot_as_a_field_error(client):
    resp = client.post(
        "/api/cron/blueprints/render",
        json={"blueprint": "morning-brief", "values": {"tiem": "07:30"}},
    )

    assert resp.status_code == 422
    assert "tiem" in resp.json()["detail"]
