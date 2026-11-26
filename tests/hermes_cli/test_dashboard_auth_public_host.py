"""Sign-in started on another name for the listener than its public URL's host.

The PKCE cookie belongs to the host that set it and the IDP returns to ``{public_url}/auth/callback``,
so the start moves to the public URL's host first (once: a proxy that hides the browser's host
must not loop it).
"""
from __future__ import annotations

from urllib.parse import parse_qs, urlsplit

import pytest
from fastapi.testclient import TestClient

from hermes_cli import web_server
from hermes_cli.dashboard_auth import clear_providers, register_provider
from tests.hermes_cli.conftest_dashboard_auth import StubAuthProvider

PUBLIC = "https://agent--dash.example.test"


@pytest.fixture
def agent(monkeypatch):
    """The gated dashboard as a hosted agent runs it: bound to all interfaces, reached through a
    proxy under several names, with ``HERMES_DASHBOARD_PUBLIC_URL`` naming one of them."""
    clear_providers()
    register_provider(StubAuthProvider())
    monkeypatch.setattr("hermes_cli.config.load_config", dict)
    monkeypatch.setenv("HERMES_DASHBOARD_PUBLIC_URL", PUBLIC)
    for name, value in {"bound_host": "0.0.0.0", "bound_port": 9119, "auth_required": True}.items():
        monkeypatch.setattr(web_server.app.state, name, value, raising=False)
    yield TestClient(web_server.app, base_url="https://agent.example.test")
    clear_providers()


def test_a_sign_in_started_on_another_host_completes_on_the_public_one(agent):
    started = agent.get("/auth/login", params={"provider": "stub", "next": "/sessions"},
                        follow_redirects=False)
    moved = urlsplit(started.headers["location"])
    assert (started.status_code, f"{moved.scheme}://{moved.netloc}", moved.path) == (
        302, PUBLIC, "/auth/login")
    assert parse_qs(moved.query)["next"] == ["/sessions"]
    # The loop guard: a proxy that hides the browser's host must not send it round again.
    assert parse_qs(moved.query)["public_host_hop"] == ["1"]

    to_idp = agent.get(started.headers["location"], follow_redirects=False)
    landed = agent.get(to_idp.headers["location"], follow_redirects=False)

    assert landed.status_code == 302, landed.text
    assert landed.headers["location"] == "/sessions"
    assert agent.get(f"{PUBLIC}/api/auth/me").status_code == 200


def test_the_desktop_sign_in_moves_to_the_public_host_too(agent):
    """The desktop's RFC 8252 start sets the same PKCE cookie, so it must not stay on the old host."""
    native = {"provider": "stub", "code_challenge": "challenge", "code_challenge_method": "S256",
              "redirect_uri": "http://127.0.0.1:53682/callback", "state": "desktop-state"}
    started = agent.get("/auth/native/authorize", params=native, follow_redirects=False)

    moved = urlsplit(started.headers["location"])
    assert (started.status_code, f"{moved.scheme}://{moved.netloc}", moved.path) == (
        302, PUBLIC, "/auth/native/authorize")
    assert {key: parse_qs(moved.query)[key] for key in native} == {key: [value] for key, value in native.items()}


@pytest.mark.parametrize("own, lands_on", [
    ("https://hermes.example.test", "https://hermes.example.test/auth/login?"),
    ("", "https://agent.example.test/auth/callback?code="),
], ids=["webapp-public-url", "webapp-without-one"])
def test_a_webapp_listener_signs_in_on_its_own_public_url_never_the_dashboards(agent, monkeypatch, own, lands_on):
    """The Webapp's IDP callback is its own ``webapp.public_url``; the dashboard's would lose the cookie."""
    monkeypatch.setattr(web_server.app.state, "listener_public_url", own, raising=False)

    started = agent.get("/auth/login", params={"provider": "stub"}, follow_redirects=False)

    assert started.status_code == 302
    assert started.headers["location"].startswith(lands_on), started.headers["location"]


@pytest.mark.parametrize("headers, query", [
    ({"host": "sandbox.azure.example", "x-forwarded-host": "agent--dash.example.test"}, ""),
    ({"host": "sandbox.azure.example", "x-forwarded-host": "agent--dash.example.test, sandbox.azure.example"}, ""),
    ({"host": "sandbox.azure.example"}, "&public_host_hop=1"),
], ids=["proxy-forwards-the-browser-host", "proxy-chain-puts-the-browser-host-first",
        "proxy-hides-it-after-one-hop"])
def test_a_proxy_that_rewrites_host_goes_to_the_idp_without_looping(agent, headers, query):
    started = agent.get(f"/auth/login?provider=stub{query}", headers=headers, follow_redirects=False)

    assert started.status_code == 302
    assert started.headers["location"].startswith(f"{PUBLIC}/auth/callback?code=")
