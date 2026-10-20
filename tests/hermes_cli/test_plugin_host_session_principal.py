"""A hosted dashboard plugin learns WHO an admitted account request is, never how it authenticated.

``plugins.isolation: host`` serves ``/api/plugins/<name>/`` in another process, so the plugin never
sees the gateway's ``request.state.session``. The forward carries the guard's decision over the private
plugin-host channel; for a request the gate verified as a dashboard account Session it now also carries
that account's four stable identity fields as ``request.state.verified_session_principal``: provider,
issuer, org_id and user_id, never tokens, email or expiry. Every other admission (native ticket,
loopback token, service token), every malformed value and anything a client sends leaves it absent.
"""
from __future__ import annotations

import json
import time
from types import SimpleNamespace

import pytest

from hermes_cli import web_server
from hermes_cli.dashboard_auth import clear_providers, register_provider
from hermes_cli.dashboard_auth.base import Session, TokenPrincipal
from hermes_cli.plugin_host import PluginHost
from tests.hermes_cli import test_plugin_secondary_profile_host as secondary
from tests.hermes_cli.conftest_dashboard_auth import StubAuthProvider

daemon = secondary.daemon

BEARER = "account-bearer-never-forwarded"
ACCOUNT = {"provider": "stub", "issuer": "https://issuer.example", "org_id": "", "user_id": "acct-1"}
KEY = "verified_session_principal"

_API = '''
from fastapi import APIRouter, Request

router = APIRouter()

@router.api_route("/whoami", methods=["GET", "POST"])
async def whoami(request: Request):
    return {"principal": getattr(request.state, "verified_session_principal", None),
            "admitted": getattr(request.state, "hosted_http_authenticated", None)}
'''


class _AccountProvider(StubAuthProvider):
    """Verifies exactly one bearer, as the account Session a real provider would mint for it."""

    def verify_session(self, *, access_token: str):
        if access_token != BEARER:
            return None
        return Session(user_id=ACCOUNT["user_id"], email="me@example.test", display_name="Me",
                       org_id=ACCOUNT["org_id"], provider=ACCOUNT["provider"], expires_at=int(time.time()) + 600,
                       access_token=BEARER, refresh_token="refresh-never-forwarded", issuer=ACCOUNT["issuer"])


@pytest.fixture
def gated(daemon, monkeypatch):
    clear_providers()
    register_provider(_AccountProvider())
    monkeypatch.setattr(web_server.app.state, "auth_required", True, raising=False)
    monkeypatch.setattr(PluginHost, "asgi_request", secondary._REAL_ASGI_REQUEST)
    try:
        yield daemon
    finally:
        clear_providers()
        secondary._shutdown_hosts()


def _hosted(home, name):
    secondary._install(home, name)
    (home / "plugins" / name / "dashboard" / "plugin_api.py").write_text(_API, encoding="utf-8")


def _launch_hosted(daemon):
    _hosted(daemon.launch, "launch-who")
    secondary._plugins(daemon.launch, enabled=["launch-who"], isolation="host")
    return "/api/plugins/launch-who/whoami", {}


def _secondary_hosted(daemon):
    _hosted(daemon.beta, secondary.NAME)
    secondary._plugins(daemon.beta, enabled=[secondary.NAME], isolation="in_process")
    secondary._plugins(daemon.launch)
    return f"/api/plugins/{secondary.NAME}/whoami", {"profile": "beta"}


ROUTES = pytest.mark.parametrize("route", [_launch_hosted, _secondary_hosted], ids=["launch", "secondary"])


@pytest.mark.platforms("any")
@ROUTES
def test_account_session_reaches_the_host_as_its_four_identity_fields_only(gated, route):
    url, params = route(gated)
    response = gated.client.get(url, params=params, headers={"Authorization": f"Bearer {BEARER}"})

    assert response.status_code == 200, response.text
    assert response.json() == {"principal": ACCOUNT, "admitted": True}
    for secret in (BEARER, "refresh-never-forwarded", "me@example.test"):
        assert secret not in response.text


@pytest.mark.platforms("any")
@ROUTES
def test_client_supplied_identity_cannot_replace_or_invent_the_account(gated, route):
    url, params = route(gated)
    forged = json.dumps({**ACCOUNT, "user_id": "someone-else"})
    spoof = {"X-Verified-Session-Principal": forged, "X-Hosted-Http-Authenticated": "true"}

    verified = gated.client.post(url, params={**params, KEY: forged}, content=forged,
                                 headers={**spoof, "Authorization": f"Bearer {BEARER}"})
    assert verified.json()["principal"] == ACCOUNT
    assert gated.client.post(url, params={**params, KEY: forged}, content=forged,
                             headers={**spoof, "Authorization": "Bearer forged"}).status_code == 401
    assert gated.client.post(url, params={**params, KEY: forged}, content=forged, headers=spoof).status_code == 401


@pytest.mark.platforms("any")
def test_native_ticket_admits_without_an_account(gated):
    url, _ = _secondary_hosted(gated)
    response = gated.client.get(url, headers=gated.beta_ticket())

    assert response.status_code == 200, response.text
    assert response.json() == {"principal": None, "admitted": True}


@pytest.mark.platforms("any")
@ROUTES
def test_loopback_token_admits_without_an_account(daemon, monkeypatch, route):
    monkeypatch.setattr(PluginHost, "asgi_request", secondary._REAL_ASGI_REQUEST)
    url, params = route(daemon)
    try:
        response = daemon.client.get(url, params=params, headers={
            **daemon.owner, "X-Verified-Session-Principal": json.dumps(ACCOUNT)})
    finally:
        secondary._shutdown_hosts()

    assert response.status_code == 200, response.text
    assert response.json() == {"principal": None, "admitted": True}


def _request(state, *, auth_required=True):
    from fastapi import FastAPI
    from starlette.requests import Request

    app = FastAPI()
    app.state.auth_required = auth_required
    return Request({"type": "http", "method": "GET", "path": "/x", "headers": [], "app": app, "state": state})


def _session(**overrides):
    fields = dict(user_id="acct-1", email="me@example.test", display_name="Me", org_id="", provider="stub",
                  expires_at=1, access_token="at", refresh_token="rt", issuer="https://issuer.example")
    return Session(**{**fields, **overrides})


def test_only_an_admitted_account_session_yields_a_principal():
    from hermes_cli.plugin_host_http_auth import hosted_admission

    assert hosted_admission(_request({"session": _session()})) == (True, ACCOUNT)
    assert hosted_admission(_request({"session": _session(org_id="org-9", issuer="")})) == (
        True, {**ACCOUNT, "org_id": "org-9", "issuer": ""})
    assert hosted_admission(_request({})) == (False, None)
    assert hosted_admission(_request({"native_http_principal": {"profile_id": "/p"}})) == (True, None)
    service = TokenPrincipal(principal="svc", provider="tokens")
    assert hosted_admission(_request({"session": service, "token_principal": service})) == (True, None)
    assert hosted_admission(_request({"session": SimpleNamespace(**ACCOUNT)})) == (True, None)
    assert hosted_admission(_request({"session": dict(ACCOUNT)})) == (True, None)
    for broken in ({"user_id": ""}, {"user_id": "  "}, {"provider": ""}, {"issuer": None}, {"org_id": 7}):
        assert hosted_admission(_request({"session": _session(**broken)})) == (True, None)


_MALFORMED = [
    None, "", json.dumps(ACCOUNT), list(ACCOUNT.items()), tuple(ACCOUNT),
    {k: v for k, v in ACCOUNT.items() if k != "issuer"},
    {**ACCOUNT, "email": "me@example.test"}, {**ACCOUNT, "access_token": "at"},
    {**ACCOUNT, "user_id": ""}, {**ACCOUNT, "user_id": "\t"}, {**ACCOUNT, "provider": " "},
    {**ACCOUNT, "user_id": 1}, {**ACCOUNT, "issuer": None}, {**ACCOUNT, "org_id": ["o"]},
]


@pytest.mark.platforms("any")
def test_host_stamps_only_an_exact_principal_on_an_admitted_request_and_never_caches_it(tmp_path):
    api = tmp_path / "api"
    api.mkdir()
    (api / "plugin_api.py").write_text(_API, encoding="utf-8")
    home = tmp_path / "home"
    home.mkdir()
    host = PluginHost(SimpleNamespace(home_path=home, _plugins={}))
    other = {**ACCOUNT, "user_id": "acct-2", "org_id": "org-2"}

    def who(*, query="", headers=(), **kwargs):
        reply = host.asgi_request("who-probe", str(api), "plugin_api.py", "GET", "/whoami", query,
                                  list(headers), b"", **kwargs)
        assert reply["status"] == 200, reply
        return json.loads(reply["body"])

    try:
        assert who(authenticated=True, principal=ACCOUNT) == {"principal": ACCOUNT, "admitted": True}
        # Request-local: the cached plugin app never carries the previous request's account.
        assert who(authenticated=True) == {"principal": None, "admitted": True}
        assert who(authenticated=True, principal=other)["principal"] == other
        assert who()["principal"] is None
        for marker in (False, None, "true", 1):
            assert who(authenticated=marker, principal=ACCOUNT) == {"principal": None, "admitted": False}
        for value in _MALFORMED:
            assert who(authenticated=True, principal=value)["principal"] is None, value
        assert who(authenticated=True, query=f"{KEY}={json.dumps(ACCOUNT)}",
                   headers=[("X-Verified-Session-Principal", json.dumps(ACCOUNT))])["principal"] is None
    finally:
        host.shutdown()


@pytest.mark.platforms("any")
def test_child_validates_what_arrives_and_tolerates_a_parent_that_sends_no_principal(tmp_path):
    """Mixed versions: an older gateway sends no ``principal`` key; a gateway that bypassed its own
    check still cannot get a malformed or unadmitted principal stamped by the child."""
    api = tmp_path / "api"
    api.mkdir()
    (api / "plugin_api.py").write_text(_API, encoding="utf-8")
    home = tmp_path / "home"
    home.mkdir()
    host = PluginHost(SimpleNamespace(home_path=home, _plugins={}))
    legacy = {"plugin": "who-probe", "dashboard_dir": str(api), "api_file": "plugin_api.py", "method": "GET",
              "path": "/whoami", "query": "", "headers": [], "body": None}

    def raw(**params):
        host.ensure_started()
        reply = host._call("asgi", {**legacy, **params})
        return json.loads(reply["body"])

    try:
        assert raw(authenticated=True) == {"principal": None, "admitted": True}
        assert raw(authenticated=True, principal=ACCOUNT)["principal"] == ACCOUNT
        assert raw(authenticated=False, principal=ACCOUNT) == {"principal": None, "admitted": False}
        for value in _MALFORMED:
            assert raw(authenticated=True, principal=value)["principal"] is None, value
    finally:
        host.shutdown()


@pytest.mark.platforms("any")
def test_gateway_credential_headers_never_reach_the_launch_profiles_host(daemon):
    """The launch profile's hosted forward drops the same credential headers the secondary one does."""
    secondary._install(daemon.launch, "launch-creds")
    secondary._plugins(daemon.launch, enabled=["launch-creds"], isolation="host")
    response = daemon.client.get("/api/plugins/launch-creds/ping", headers={
        **daemon.owner, "Authorization": f"Bearer {BEARER}", "Cookie": "hermes_session_at=x", "X-Keep": "1"})

    assert response.json() == {"hosted": True}
    (call,) = daemon.calls
    forbidden = {"x-hermes-gateway-ticket", web_server._SESSION_HEADER_NAME.lower(), "authorization", "cookie", "host"}
    assert not call.headers & forbidden
    assert "x-keep" in call.headers
