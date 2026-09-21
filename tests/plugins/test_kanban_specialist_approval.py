"""Dashboard-authenticated operator approvals for specialist promotion."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest
from fastapi import FastAPI, Request
from fastapi.testclient import TestClient

from gateway.operator_approval_authority import canonical_dashboard_subject
from hermes_cli import kanban_db as kb
from hermes_cli.dashboard_auth.base import Session


def _router():
    source = Path(__file__).resolve().parents[2] / "plugins/kanban/dashboard/plugin_api.py"
    spec = importlib.util.spec_from_file_location("specialist_approval_plugin", source)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module.router


@pytest.fixture
def client(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setattr(
        "hermes_cli.config.load_config",
        lambda: {
            "kanban": {
                "specialist_operator_approvals": {
                    "allowed_subjects": [
                        canonical_dashboard_subject(provider="portal", user_id="operator-1")
                    ]
                }
            }
        },
    )
    kb.init_db()
    app = FastAPI()
    app.state.auth_required = True

    @app.middleware("http")
    async def authenticated_session(request: Request, call_next):
        request.state.session = Session(
            user_id="operator-1",
            email="operator@example.test",
            display_name="Operator",
            org_id="org",
            provider="portal",
            expires_at=4_102_444_800,
            access_token="opaque-access-token",
            refresh_token="opaque-refresh-token",
        )
        return await call_next(request)

    app.include_router(_router(), prefix="/api/plugins/kanban")
    return TestClient(app)


def test_authenticated_allowlisted_dashboard_session_records_operator_approval(client):
    response = client.post(
        "/api/plugins/kanban/specialist-approvals",
        json={
            "candidate_id": "candidate-1",
            "verification_result_hash": "a" * 64,
            "target_state": "staged",
        },
    )

    assert response.status_code == 200, response.text
    assert response.json()["operator_identity"] == canonical_dashboard_subject(
        provider="portal", user_id="operator-1"
    )


def test_authenticated_dashboard_can_inspect_its_approval_subject(client):
    response = client.get("/api/plugins/kanban/specialist-approval-authority")

    assert response.status_code == 200, response.text
    assert response.json() == {
        "authenticated_subject": canonical_dashboard_subject(
            provider="portal", user_id="operator-1"
        ),
        "approval_authorized": True,
    }


def test_loopback_token_mode_cannot_record_operator_approval(client):
    client.app.state.auth_required = False

    response = client.post(
        "/api/plugins/kanban/specialist-approvals",
        json={
            "candidate_id": "candidate-1",
            "verification_result_hash": "a" * 64,
            "target_state": "staged",
        },
    )

    assert response.status_code == 403
