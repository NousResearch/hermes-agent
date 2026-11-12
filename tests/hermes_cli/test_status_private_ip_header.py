"""/api/status names this machine's Fly private (6PN) address in a response header.

The agent router wakes a suspended agent by requesting /api/status through Fly's edge, then
retries the caller's request over 6PN. A resumed machine accepts on its 6PN address within a
second, but ``<app>.internal`` keeps failing to resolve for ~9 s, so the router dials the address
this header names. The router's request is unauthenticated, so the header must be on the public
(signed-out) answer too.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from hermes_cli import web_server
from hermes_cli.dashboard_auth import clear_providers, register_provider
from hermes_cli.web_routers.status import PRIVATE_IP_HEADER
from tests.hermes_cli.conftest_dashboard_auth import StubAuthProvider

_FLY_IP = "fdaa:0:1:a7b:1::2"


@pytest.fixture
def gated_client():
    clear_providers()
    register_provider(StubAuthProvider())
    prev = {k: getattr(web_server.app.state, k, None) for k in ("bound_host", "bound_port", "auth_required")}
    web_server.app.state.bound_host = "fly-app.fly.dev"
    web_server.app.state.bound_port = 443
    web_server.app.state.auth_required = True
    yield TestClient(web_server.app, base_url="https://fly-app.fly.dev")
    clear_providers()
    for k, v in prev.items():
        setattr(web_server.app.state, k, v)


def test_status_names_fly_private_ip_to_an_unauthenticated_caller(gated_client, monkeypatch):
    monkeypatch.setenv("FLY_PRIVATE_IP", _FLY_IP)
    r = gated_client.get("/api/status")
    assert r.status_code == 200
    assert r.headers[PRIVATE_IP_HEADER] == _FLY_IP


@pytest.mark.parametrize("value", [None, "", "10.0.0.5", "fd00::1"])
def test_status_omits_header_without_a_fly_private_ip(gated_client, monkeypatch, value):
    if value is None:
        monkeypatch.delenv("FLY_PRIVATE_IP", raising=False)
    else:
        monkeypatch.setenv("FLY_PRIVATE_IP", value)
    r = gated_client.get("/api/status")
    assert r.status_code == 200
    assert PRIVATE_IP_HEADER not in r.headers
