"""Verified dashboard accounts survive only allowlisted hosted HTTP requests."""
import asyncio
import json
import os
from dataclasses import asdict, replace
from types import SimpleNamespace

import httpx
import pytest
from fastapi import FastAPI

from hermes_cli.dashboard_auth.base import Session
from hermes_cli.dashboard_auth.routes import router as auth_router
from hermes_cli.plugin_host import PluginHost
from hermes_cli.plugin_host_principal import (
    VerifiedSessionPrincipal, asgi_with_principal, encode_principal, principal_for_request,
)
from starlette.requests import Request


API = '''
import asyncio
import os
from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

router = APIRouter()
waiting = 0
ready = asyncio.Event()

@router.api_route("/history", methods=["GET"])
@router.api_route("/history/sync", methods=["POST"])
@router.api_route("/history/import", methods=["POST"])
@router.api_route("/history/context", methods=["POST"])
@router.api_route("/locations", methods=["GET"])
async def context(request: Request):
    global waiting
    before = getattr(request.state, "verified_session_principal", None)
    if request.query_params.get("barrier"):
        waiting += 1
        if waiting == 2:
            ready.set()
        await asyncio.wait_for(ready.wait(), 5)
    after = getattr(request.state, "verified_session_principal", None)
    if before is None:
        return JSONResponse({"error": "owner_unresolved"}, status_code=401)
    credentials = [key for key in request.headers
                   if key in {"authorization", "cookie", "x-hermes-gateway-ticket", "x-hermes-session-token"}]
    credentials += [key for key in request.query_params
                    if key in {"token", "ticket", "internal", "display_ticket", "attach", "profile"}]
    return {"principal": after, "unchanged": before == after,
            "credentials": credentials, "pid": os.getpid()}
'''


def _session(user_id="alice"):
    return Session(
        provider="test", issuer=f"https://issuer.example/{user_id}",
        org_id=f"org-{user_id}", user_id=user_id,
        email="private@example.test", display_name="Private Name", expires_at=2**40,
        access_token=f"secret-access-{user_id}", refresh_token="secret-refresh",
    )


def _request(session=None, *, path="/history", method="GET"):
    return Request({
        "type": "http", "method": method, "path": path,
        "state": {"session": session}, "headers": [],
    })


@pytest.fixture(params=["launch", "secondary"])
def hosted(tmp_path, monkeypatch, request):
    from hermes_cli import web_server_dashboard as dashboard
    from hermes_cli import web_server_plugin_profiles as profiles
    from hermes_cli.dashboard_auth import middleware

    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("HOME", str(tmp_path))
    api = tmp_path / "api"
    api.mkdir()
    (api / "plugin_api.py").write_text(API)
    host = PluginHost(SimpleNamespace(home_path=home, _plugins={}))
    monkeypatch.setattr(dashboard, "_hosted_plugin_for_request",
                        lambda name: (host, str(api), "plugin_api.py"))
    monkeypatch.setattr(dashboard, "_plugin_route_secret_scope", lambda: None)
    monkeypatch.setattr(middleware, "_verify_access_token",
                        lambda request, *, access_token, **kw:
                        _session(access_token) if access_token in {"alice", "bob"} else None)
    app = FastAPI()
    app.state.auth_required = True
    app.middleware("http")(middleware.gated_auth_middleware)
    app.include_router(auth_router)
    if request.param == "launch":
        dashboard._mount_hosted_plugin_api(app, {"name": "hermes-locations"}, "plugin_api.py")
        dashboard._mount_hosted_plugin_api(app, {"name": "other-plugin"}, "plugin_api.py")
    else:
        from hermes_cli import plugins
        monkeypatch.setattr(profiles, "_secondary_hosted_plugin",
                            lambda name: {"_dir": str(api), "_api_file": "plugin_api.py"})
        monkeypatch.setattr(profiles, "secondary_profile_home", lambda: home)
        monkeypatch.setattr(plugins, "get_plugin_manager",
                            lambda: SimpleNamespace(_plugin_host=lambda: host))

        @app.api_route("/api/plugins/{name}/{path:path}", methods=["GET", "POST"])
        async def secondary(request: Request, name: str, path: str):
            return await profiles.forward_secondary_plugin_request(request, name)
    try:
        yield app, host, api
    finally:
        host.shutdown()


def test_only_real_sessions_produce_a_credential_free_principal():
    session = _session()
    principal = principal_for_request(_request(session), "hermes-locations", "/history")
    assert asdict(principal) == {
        "provider": session.provider, "issuer": session.issuer,
        "org_id": session.org_id, "user_id": session.user_id,
    }
    for fake in (None, asdict(session), SimpleNamespace(**asdict(session))):
        assert principal_for_request(_request(fake), "hermes-locations", "/history") is None
    assert principal_for_request(_request(session), "other-plugin", "/history") is None
    assert principal_for_request(_request(session), "hermes-locations", "/locations") is None


@pytest.mark.parametrize("provider,issuer,org_id,user_id", [
    ("test", "https://issuer.example/a", "org-a", "alice"),
    ("test", "https://issuer.example/b", "org-a", "alice"),
    ("test", "https://issuer.example/a", "org-b", "alice"),
    ("other", "https://issuer.example/a", "org-a", "alice"),
    ("test", "https://issuer.example/a", "org-a", "bob"),
    ("basic", "", "", "alice"),
])
def test_auth_me_exposes_exact_verified_account_scope(
    hosted, monkeypatch, provider, issuer, org_id, user_id,
):
    from hermes_cli.dashboard_auth import middleware
    app, _, _ = hosted
    session = replace(_session(), provider=provider, issuer=issuer, org_id=org_id, user_id=user_id)
    monkeypatch.setattr(middleware, "_verify_access_token", lambda *a, **kw: session)

    async def scenario():
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app),
                                     base_url="https://testserver") as client:
            return await client.get("/api/auth/me", headers={"Authorization": "Bearer test"})

    response = asyncio.run(scenario())
    assert response.status_code == 200, response.text
    identity = response.json()
    assert {key: identity[key] for key in ("provider", "issuer", "org_id", "user_id")} == {
        "provider": session.provider, "issuer": issuer, "org_id": org_id, "user_id": session.user_id,
    }
    assert "access_token" not in identity and "refresh_token" not in identity


@pytest.mark.parametrize("raw", [
    {}, {"provider": "test", "issuer": "", "org_id": "", "user_id": ""},
    {"provider": "test", "issuer": "", "org_id": None, "user_id": "alice"},
    {**asdict(VerifiedSessionPrincipal("test", "", "", "alice")), "access_token": "secret"},
])
def test_malformed_internal_principals_are_rejected(raw):
    with pytest.raises(ValueError, match="invalid verified"):
        asgi_with_principal(FastAPI(), {
            "plugin": "hermes-locations", "method": "GET", "path": "/history",
            "verified_session_principal": raw,
        })


@pytest.mark.platforms("any")
@pytest.mark.parametrize("method,path", [
    ("GET", "/history"), ("POST", "/history/sync"),
    ("POST", "/history/import"), ("POST", "/history/context"),
])
def test_verified_account_crosses_real_host_without_session_secrets(hosted, method, path):
    app, host, _ = hosted

    async def scenario():
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app),
                                     base_url="https://testserver") as client:
            return await client.request(
                method, f"/api/plugins/hermes-locations{path}?token=secret-query&profile=other",
                headers={"Authorization": "Bearer alice",
                         "Cookie": "private-cookie=secret",
                         "X-Hermes-Gateway-Ticket": "secret-ticket",
                         "X-Verified-Session-Principal": '{"user_id":"bob"}'},
                json={"verified_session_principal": {"user_id": "bob"}},
            )

    response = asyncio.run(scenario())
    assert response.status_code == 200, response.text
    assert response.json() == {
        "principal": {"provider": "test", "issuer": "https://issuer.example/alice",
                      "org_id": "org-alice", "user_id": "alice"},
        "unchanged": True, "credentials": [], "pid": host.pid,
    }
    assert host.pid != os.getpid()


@pytest.mark.platforms("any")
def test_absent_spoofed_and_nonallowlisted_accounts_remain_unresolved(hosted):
    app, host, api = hosted

    async def scenario():
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app),
                                     base_url="https://testserver") as client:
            denied = await client.get("/api/plugins/hermes-locations/history", headers={
                "X-Verified-Session-Principal": json.dumps(asdict(_session("bob"))),
            })
            assert denied.status_code == 401
            for url in ("/api/plugins/other-plugin/history",
                        "/api/plugins/hermes-locations/locations"):
                response = await client.get(url, headers={"Authorization": "Bearer alice"})
                assert response.status_code == 401
                assert response.json() == {"error": "owner_unresolved"}
            app.state.auth_required = False
            response = await client.get("/api/plugins/hermes-locations/history", headers={
                "Authorization": "Bearer alice", "X-User-Id": "alice",
                "X-Verified-Session-Principal": json.dumps(asdict(_session("alice"))),
            })
            assert response.status_code == 401

    asyncio.run(scenario())
    principal = VerifiedSessionPrincipal("test", "", "", "alice")
    # Both the sending and receiving sides enforce the allowlist.
    assert encode_principal(principal, "other-plugin", "GET", "/history") is None
    result = host._call("asgi", {
        "plugin": "other-plugin", "dashboard_dir": str(api), "api_file": "plugin_api.py",
        "method": "GET", "path": "/history", "query": "", "headers": [], "body": None,
        "verified_session_principal": asdict(principal),
    })
    assert result["status"] == 401


@pytest.mark.platforms("any")
def test_concurrent_accounts_and_later_missing_identity_do_not_share_state(hosted):
    app, _, _ = hosted

    async def scenario():
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app),
                                     base_url="https://testserver") as client:
            # Load the cached plugin app before holding two requests in its shared barrier.
            warm = await client.get("/api/plugins/hermes-locations/history",
                                    headers={"Authorization": "Bearer alice"})
            assert warm.status_code == 200, warm.text
            responses = await asyncio.gather(*(
                client.get("/api/plugins/hermes-locations/history?barrier=1",
                           headers={"Authorization": f"Bearer {user}"})
                for user in ("alice", "bob")
            ))
            for user, response in zip(("alice", "bob"), responses):
                assert response.status_code == 200, response.text
                assert response.json()["principal"] == {
                    "provider": "test", "issuer": f"https://issuer.example/{user}",
                    "org_id": f"org-{user}", "user_id": user,
                }
                assert response.json()["unchanged"]
            app.state.auth_required = False
            missing = await client.get("/api/plugins/hermes-locations/history")
            assert missing.status_code == 401
            assert missing.json() == {"error": "owner_unresolved"}

    asyncio.run(scenario())
