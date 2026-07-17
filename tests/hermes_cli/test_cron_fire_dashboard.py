"""Tests for the Chronos cron-fire webhook ON THE DASHBOARD APP (web_server).

Regression guard for the relocation bug: the fire webhook MUST live on the
dashboard FastAPI app (`hermes_cli.web_server.app`) — the agent's public HTTP
surface on hosted deployments — not only on the aiohttp APIServerAdapter (which
hosted agents don't expose). It must:
  - be a registered route on the dashboard app,
  - be in PUBLIC_API_PATHS so the dashboard cookie gate doesn't 401 it before
    the JWT verifier runs,
  - reject a bad/missing NAS-JWT with 401 (the JWT is the real gate),
  - 400 on missing job_id,
  - on a valid token, resolve the job's profile and run fire_due in the
    background, returning 202.
"""

import asyncio
from types import SimpleNamespace

import pytest
from starlette.testclient import TestClient

from hermes_cli import web_server
from hermes_cli.dashboard_auth.public_paths import PUBLIC_API_PATHS


def _client(auth_required: bool):
    prev_auth = getattr(web_server.app.state, "auth_required", None)
    prev_host = getattr(web_server.app.state, "bound_host", None)
    web_server.app.state.auth_required = auth_required
    web_server.app.state.bound_host = None
    client = TestClient(web_server.app)
    return client, prev_auth, prev_host


def _restore(prev_auth, prev_host):
    if prev_auth is None:
        if hasattr(web_server.app.state, "auth_required"):
            delattr(web_server.app.state, "auth_required")
    else:
        web_server.app.state.auth_required = prev_auth
    if prev_host is None:
        if hasattr(web_server.app.state, "bound_host"):
            delattr(web_server.app.state, "bound_host")
    else:
        web_server.app.state.bound_host = prev_host


def test_route_registered_on_dashboard_app():
    """The fire webhook is served by the dashboard app (the hosted-agent public
    surface), not only the aiohttp adapter."""
    paths = {r.path for r in web_server.app.routes if hasattr(r, "path")}
    assert "/api/cron/fire" in paths


def test_fire_path_is_public():
    """Must bypass the dashboard cookie gate so the NAS bearer-JWT callback
    reaches the verifier (the JWT is the real auth)."""
    assert "/api/cron/fire" in PUBLIC_API_PATHS


def test_bad_token_401(monkeypatch):
    """Invalid NAS-JWT -> 401, even with the dashboard auth gate ENGAGED
    (proves the route is reachable past the cookie gate and the verifier is the
    gate). fire_due must NOT run."""
    fired = []
    monkeypatch.setattr(
        "plugins.cron_providers.chronos.verify.get_fire_verifier",
        lambda: (lambda **kw: None),  # verification fails
    )
    monkeypatch.setattr(web_server, "_find_cron_job_profile", lambda jid: "default")
    monkeypatch.setattr(web_server, "_fire_cron_job_for_profile",
                        lambda p, j: fired.append((p, j)))

    client, pa, ph = _client(auth_required=True)
    try:
        resp = client.post("/api/cron/fire",
                           headers={"Authorization": "Bearer forged"},
                           json={"job_id": "abc"})
        assert resp.status_code == 401
        assert fired == []
    finally:
        _restore(pa, ph)
        client.close()


def test_missing_job_id_400(monkeypatch):
    monkeypatch.setattr(
        "plugins.cron_providers.chronos.verify.get_fire_verifier",
        lambda: (lambda **kw: {"purpose": "cron_fire"}),
    )
    client, pa, ph = _client(auth_required=False)
    try:
        resp = client.post("/api/cron/fire",
                           headers={"Authorization": "Bearer good"},
                           json={})
        assert resp.status_code == 400
    finally:
        _restore(pa, ph)
        client.close()


def test_unknown_job_200_gone(monkeypatch):
    """Valid token but the job isn't found in any profile -> 200 'gone'
    (NAS shouldn't retry a fire for a cancelled/completed job)."""
    monkeypatch.setattr(
        "plugins.cron_providers.chronos.verify.get_fire_verifier",
        lambda: (lambda **kw: {"purpose": "cron_fire"}),
    )
    monkeypatch.setattr(web_server, "_find_cron_job_profile", lambda jid: None)
    client, pa, ph = _client(auth_required=False)
    try:
        resp = client.post("/api/cron/fire",
                           headers={"Authorization": "Bearer good"},
                           json={"job_id": "ghost"})
        assert resp.status_code == 200
        assert resp.json().get("status") == "gone"
    finally:
        _restore(pa, ph)
        client.close()


def test_valid_token_accepts_and_fires(monkeypatch):
    """Valid token + known job -> 202 and fire_due invoked for the resolved
    profile."""
    fired = []
    monkeypatch.setattr(
        "plugins.cron_providers.chronos.verify.get_fire_verifier",
        lambda: (lambda **kw: {"purpose": "cron_fire", "aud": "agent:x"}),
    )
    monkeypatch.setattr(web_server, "_find_cron_job_profile", lambda jid: "default")
    monkeypatch.setattr(web_server, "_fire_cron_job_for_profile",
                        lambda p, j: fired.append((p, j)) or True)

    client, pa, ph = _client(auth_required=False)
    try:
        resp = client.post("/api/cron/fire",
                           headers={"Authorization": "Bearer good"},
                           json={"job_id": "j1"})
        assert resp.status_code == 202
        assert resp.json()["job_id"] == "j1"
    finally:
        _restore(pa, ph)
        client.close()
    # background task ran the fire for the resolved profile
    assert fired == [("default", "j1")]


@pytest.mark.asyncio
async def test_worker_exception_is_consumed_and_redacted(monkeypatch, caplog):
    raw_secret = "RAW_SECRET_SENTINEL"
    monkeypatch.setattr(
        "plugins.cron_providers.chronos.verify.get_fire_verifier",
        lambda: (lambda **kw: {"purpose": "cron_fire"}),
    )

    async def dashboard_io(_callback, *_args):
        return "default"

    monkeypatch.setattr(web_server, "_run_cron_dashboard_io", dashboard_io)
    monkeypatch.setattr(
        web_server,
        "_fire_cron_job_for_profile",
        lambda *_args: (_ for _ in ()).throw(RuntimeError(raw_secret)),
    )
    request = SimpleNamespace(
        headers={"Authorization": "Bearer good"},
        json=lambda: asyncio.sleep(0, result={"job_id": "failing"}),
    )
    created = []
    original_create_task = asyncio.create_task

    def capture_task(coro):
        task = original_create_task(coro)
        created.append(task)
        return task

    monkeypatch.setattr(web_server.asyncio, "create_task", capture_task)
    response = await web_server.cron_fire_webhook(request)
    assert response.status_code == 202
    await created[0]
    assert created[0].exception() is None
    assert raw_secret not in caplog.text
    assert "Cron dashboard fire worker failed" in caplog.text


@pytest.mark.asyncio
async def test_task_start_failure_is_sanitized(monkeypatch, caplog):
    raw_secret = "TASK_START_SECRET_SENTINEL"
    monkeypatch.setattr(
        "plugins.cron_providers.chronos.verify.get_fire_verifier",
        lambda: (lambda **kw: {"purpose": "cron_fire"}),
    )

    async def dashboard_io(_callback, *_args):
        return "default"

    monkeypatch.setattr(web_server, "_run_cron_dashboard_io", dashboard_io)
    monkeypatch.setattr(
        web_server.asyncio,
        "create_task",
        lambda _coro: (_ for _ in ()).throw(RuntimeError(raw_secret)),
    )
    request = SimpleNamespace(
        headers={"Authorization": "Bearer good"},
        json=lambda: asyncio.sleep(0, result={"job_id": "startup-failure"}),
    )
    response = await web_server.cron_fire_webhook(request)
    assert response.status_code == 503
    assert raw_secret not in caplog.text
    assert raw_secret not in bytes(response.body).decode()
