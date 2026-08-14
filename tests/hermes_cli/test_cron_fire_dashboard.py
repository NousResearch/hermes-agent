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


def test_valid_fire_forwards_to_gateway(monkeypatch):
    """The dashboard is the public door only: a verified fire is FORWARDED to
    the gateway api_server (which owns the live adapters — relay/E2EE
    delivery), with the NAS bearer preserved, and the gateway's response is
    passed through. The deprecated in-dashboard execution must NOT run."""
    forwarded = []
    executed = []

    async def fake_forward(profile, job_id, authorization):
        forwarded.append((profile, job_id, authorization))
        return 202, {"status": "accepted", "job_id": job_id}

    monkeypatch.setattr(
        "plugins.cron_providers.chronos.verify.get_fire_verifier",
        lambda: (lambda **kw: {"purpose": "cron_fire"}),
    )
    monkeypatch.setattr(web_server, "_find_cron_job_profile", lambda jid: "default")
    monkeypatch.setattr(web_server, "_forward_cron_fire_to_gateway", fake_forward)
    monkeypatch.setattr(web_server, "_fire_cron_job_for_profile",
                        lambda p, j: executed.append((p, j)))

    client, pa, ph = _client(auth_required=False)
    try:
        resp = client.post("/api/cron/fire",
                           headers={"Authorization": "Bearer nas-jwt"},
                           json={"job_id": "j1"})
        assert resp.status_code == 202
        assert resp.json().get("status") == "accepted"
        assert forwarded == [("default", "j1", "Bearer nas-jwt")]
        assert executed == []  # no local execution — gateway owns cron
    finally:
        _restore(pa, ph)
        client.close()


def test_gateway_unreachable_503_for_nas_retry(monkeypatch):
    """Gateway down (scale-to-zero wake window / restart) -> 503 so NAS
    retries per the Chronos contract. Deliberately NO local-execution
    fallback — delivering from the dashboard process cannot serve
    relay-fronted or E2EE targets."""
    executed = []

    async def fake_forward(profile, job_id, authorization):
        return None  # unreachable

    monkeypatch.setattr(
        "plugins.cron_providers.chronos.verify.get_fire_verifier",
        lambda: (lambda **kw: {"purpose": "cron_fire"}),
    )
    monkeypatch.setattr(web_server, "_find_cron_job_profile", lambda jid: "default")
    monkeypatch.setattr(web_server, "_forward_cron_fire_to_gateway", fake_forward)
    monkeypatch.setattr(web_server, "_fire_cron_job_for_profile",
                        lambda p, j: executed.append((p, j)))

    client, pa, ph = _client(auth_required=False)
    try:
        resp = client.post("/api/cron/fire",
                           headers={"Authorization": "Bearer nas-jwt"},
                           json={"job_id": "j2"})
        assert resp.status_code == 503
        assert executed == []
    finally:
        _restore(pa, ph)
        client.close()


def test_gateway_error_status_passes_through(monkeypatch):
    """A non-2xx gateway response (e.g. its own 401 on a replayed JWT) passes
    through unchanged — the dashboard adds no interpretation of its own."""

    async def fake_forward(profile, job_id, authorization):
        return 401, {"error": "invalid fire token"}

    monkeypatch.setattr(
        "plugins.cron_providers.chronos.verify.get_fire_verifier",
        lambda: (lambda **kw: {"purpose": "cron_fire"}),
    )
    monkeypatch.setattr(web_server, "_find_cron_job_profile", lambda jid: "default")
    monkeypatch.setattr(web_server, "_forward_cron_fire_to_gateway", fake_forward)

    client, pa, ph = _client(auth_required=False)
    try:
        resp = client.post("/api/cron/fire",
                           headers={"Authorization": "Bearer nas-jwt"},
                           json={"job_id": "j3"})
        assert resp.status_code == 401
    finally:
        _restore(pa, ph)
        client.close()


# ── _gateway_fire_endpoint URL resolution ────────────────────────────────


def test_fire_endpoint_default_port(tmp_path, monkeypatch):
    monkeypatch.delenv("API_SERVER_PORT", raising=False)
    monkeypatch.delenv("GATEWAY_MULTIPLEX_PROFILES", raising=False)
    monkeypatch.setattr(web_server, "load_config", lambda: {})
    url = web_server._gateway_fire_endpoint("default", tmp_path)
    assert url == "http://127.0.0.1:8642/api/cron/fire"


def test_fire_endpoint_config_yaml_port_wins(tmp_path, monkeypatch):
    """The profile config.yaml port (read via the CANONICAL load_config, per
    the config-read guard) wins over the process env API_SERVER_PORT."""
    monkeypatch.setenv("API_SERVER_PORT", "9999")
    monkeypatch.delenv("GATEWAY_MULTIPLEX_PROFILES", raising=False)
    monkeypatch.setattr(
        web_server,
        "load_config",
        lambda: {"platforms": {"api_server": {"extra": {"port": 8700}}}},
    )
    url = web_server._gateway_fire_endpoint("default", tmp_path)
    assert url == "http://127.0.0.1:8700/api/cron/fire"


def test_fire_endpoint_profile_env_port(tmp_path, monkeypatch):
    """A non-default profile reads API_SERVER_PORT from its own .env, not the
    dashboard process env (per-profile-gateway topology)."""
    monkeypatch.setenv("API_SERVER_PORT", "9999")  # dashboard process env
    monkeypatch.delenv("GATEWAY_MULTIPLEX_PROFILES", raising=False)
    monkeypatch.setattr(web_server, "load_config", lambda: {})
    monkeypatch.setattr(web_server, "_cron_default_profile", lambda: "default")
    (tmp_path / ".env").write_text("API_SERVER_PORT=8701\n", encoding="utf-8")
    url = web_server._gateway_fire_endpoint("worker_alpha", tmp_path)
    assert url == "http://127.0.0.1:8701/api/cron/fire"


def test_fire_endpoint_multiplex_profile_prefix(tmp_path, monkeypatch):
    """Multiplex mode: a non-default profile routes through the default
    gateway's port with the /p/<profile>/ prefix mirror."""
    monkeypatch.delenv("API_SERVER_PORT", raising=False)
    monkeypatch.setenv("GATEWAY_MULTIPLEX_PROFILES", "1")
    monkeypatch.setattr(web_server, "load_config", lambda: {})
    url = web_server._gateway_fire_endpoint("worker_alpha", tmp_path)
    assert url == "http://127.0.0.1:8642/p/worker_alpha/api/cron/fire"


