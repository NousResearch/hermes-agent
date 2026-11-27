"""``GET /api/health`` ``commit`` contract (review G4).

The public health probe reports the backend's BOOT ``commit`` so an attaching client can refuse a
backend whose boot commit differs from its checkout: a ``serve`` that outlived ``hermes update``
is never re-adopted. The field must always be present
(``null`` off git) and must come from ``get_version_info().commit``.
"""

from __future__ import annotations

import pytest

from hermes_cli.version_info import VersionInfo


def _info(commit: str | None) -> VersionInfo:
    return VersionInfo(base_version="2026.10.1", derived_version="2026.10.1", distance=0,
                       commit=commit, branch="main", source="git")


@pytest.mark.parametrize("commit", ["0fbe50a9d88aa7e1c0b5f3d2e4a6b8c9d0e1f2a3", None])
def test_health_reports_the_boot_commit_from_version_info(monkeypatch, commit):
    from starlette.testclient import TestClient

    from hermes_cli.web_routers import status
    import hermes_cli.web_server as ws

    monkeypatch.setattr(status, "get_version_info", lambda: _info(commit))

    response = TestClient(ws.app).get("/api/health")  # public: no session token

    assert response.status_code == 200
    body = response.json()
    assert body["ok"] is True
    assert "commit" in body, "attaching clients key the boot-identity decision on this field"
    assert body["commit"] == commit


def test_bare_health_is_the_api_server_probe_and_stays_public(monkeypatch):
    """``GET /health`` answers the api_server's documented health contract, and stays public.

    Clients and uptime monitors written against ``docs/features/api-server.md`` probe the bare
    path. Both gates answer a path they do not mount with the sign-in redirect, so a JSON probe
    there parses the login page and reports a protocol error instead of a health result. The body
    is the ``/api/health`` payload plus the api_server's ``status``/``platform`` keys; asserted as
    a relationship (it carries every field of both contracts), never as a value snapshot.
    """
    from starlette.testclient import TestClient

    from hermes_cli.dashboard_auth.public_paths import PUBLIC_API_PATHS
    from hermes_cli.web_routers import status
    import hermes_cli.web_server as ws

    monkeypatch.setattr(status, "get_version_info", lambda: _info("0fbe50a9d88aa7e1c0b5f3d2e4a6b8c9d0e1f2a3"))

    client = TestClient(ws.app)
    bare = client.get("/health")  # public: no session token, no cookie
    api = client.get("/api/health")

    assert bare.status_code == 200, f"/health is not a mounted route any more: {bare.status_code}"
    body, dashboard = bare.json(), api.json()
    # The api_server contract (what a bare-path probe reads).
    assert body["status"] == "ok"
    assert body["platform"] == "hermes-agent"
    assert body["version"] == dashboard["version"]
    # And every field the dashboard probe relies on, unchanged.
    for key, value in dashboard.items():
        assert body.get(key) == value, f"/health dropped the /api/health field {key!r}"
    assert "/health" in PUBLIC_API_PATHS, "the gate would redirect a bare /health probe to /login"
