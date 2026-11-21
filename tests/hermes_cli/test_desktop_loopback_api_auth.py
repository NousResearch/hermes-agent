"""Trust-boundary regression for the Desktop-owned loopback backend (#126391).

A Desktop-spawned backend (``HERMES_DESKTOP=1`` + per-spawn
``HERMES_DASHBOARD_SESSION_TOKEN``, loopback bind) must answer the dashboard's
``/api/*`` reads for the token it serves, and refuse everything else. The
#126391 workaround (appending the polled paths to ``PUBLIC_API_PATHS``) makes
those reads pass by dropping auth entirely: the allowlist is exact-match,
method-agnostic and bind-agnostic, so it also opens unauthenticated
``POST /api/profiles``. These tests are green on main, red on that widening,
and red if the session-token path regresses.

The auth state comes from the real ``_configure_auth_gate`` (the only
production writer of ``app.state.auth_required``), never hand-set flags, so
the suite cannot pin a configuration startup never emits.
"""

import hermes_state
import pytest
from fastapi.testclient import TestClient

from hermes_cli import web_server
from hermes_cli.dashboard_auth import clear_providers, register_provider
from hermes_cli.dashboard_auth.public_paths import PUBLIC_API_PATHS
from tests.hermes_cli.conftest_dashboard_auth import StubAuthProvider

_SPAWN_TOKEN = "desktop-minted-per-spawn-token"

# Reads the Desktop polls: the issue's list endpoints, the subpath the view
# needs, and the two requests seen 401ing in the field (desktop.log).
_DESKTOP_READS = (
    "/api/profiles",
    "/api/sessions",
    "/api/profiles/sessions",
    "/api/profiles/projects/tree?preview_limit=3",
    "/api/cron/jobs?profile=all",
)

# Paths #126391 proposed allowlisting. /api/kanban and /api/artifacts are not
# routes; they must still never become public.
_REPORTED_PATHS = ("/api/profiles", "/api/sessions", "/api/kanban", "/api/artifacts")


@pytest.fixture
def client(tmp_path, monkeypatch):
    """A Desktop-spawned loopback backend, configured the way startup does it.

    A non-loopback ``dashboard.public_url`` is declared so the #96490
    exemption, not the absence of a public URL, is what keeps the backend in
    token mode. A provider is registered so a regressed exemption engages a
    working gate (401) instead of failing closed at startup.
    """
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    # The gateway (owner) initializes the session store; GET /api/sessions only browses it.
    monkeypatch.setattr(hermes_state, "DEFAULT_DB_PATH", tmp_path / "state.db")
    hermes_state.SessionDB().close()
    monkeypatch.setenv("HERMES_DESKTOP", "1")
    monkeypatch.setenv("HERMES_DASHBOARD_SESSION_TOKEN", _SPAWN_TOKEN)
    monkeypatch.setenv("HERMES_DASHBOARD_PUBLIC_URL", "https://dashboard.example.test")
    # The token the backend serves is resolved at import; pin it to the spawn token.
    monkeypatch.setattr(web_server, "_SESSION_TOKEN", _SPAWN_TOKEN)
    for name in ("auth_required", "bound_host", "trusted_public_hosts"):
        monkeypatch.setattr(
            web_server.app.state,
            name,
            getattr(web_server.app.state, name, None),
            raising=False,
        )
    clear_providers()
    register_provider(StubAuthProvider())
    try:
        web_server._configure_auth_gate("127.0.0.1", False, None, None)
        web_server.app.state.bound_host = "127.0.0.1"
        assert web_server.app.state.auth_required is False, (
            "not a Desktop-exempt loopback backend"
        )
        yield TestClient(web_server.app, base_url="http://127.0.0.1")
    finally:
        clear_providers()


@pytest.mark.parametrize("path", _DESKTOP_READS)
def test_desktop_loopback_token_authenticates_reads(client, path):
    resp = client.get(path, headers={"X-Hermes-Session-Token": _SPAWN_TOKEN})
    assert resp.status_code == 200, resp.text


@pytest.mark.parametrize("path", _DESKTOP_READS)
def test_desktop_loopback_rejects_absent_and_wrong_token(client, path):
    absent = client.get(path)
    wrong = client.get(path, headers={"X-Hermes-Session-Token": "not-the-spawn-token"})
    for resp in (absent, wrong):
        assert resp.status_code == 401, resp.text
        # Loopback token mode's refusal, never the cookie gate's envelope.
        assert resp.json() == {"detail": "Unauthorized"}


def test_reported_endpoints_are_not_publicly_allowlisted():
    assert PUBLIC_API_PATHS.isdisjoint(_REPORTED_PATHS)
    assert not any(p.split("?")[0] in PUBLIC_API_PATHS for p in _DESKTOP_READS)


def test_unauthenticated_profile_creation_stays_gated(client):
    resp = client.post("/api/profiles", json={"name": "boundary-probe"})
    assert resp.status_code == 401, resp.text
