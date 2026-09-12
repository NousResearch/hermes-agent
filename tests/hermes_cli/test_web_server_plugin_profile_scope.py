"""Plugin API routers (``/api/plugins/<name>/*``) honour the Desktop's ``?profile=`` scope.

The Desktop appends ``?profile=<name>`` to every REST call; core routers read it per handler, plugin
routers did not, so a plugin's state landed in the serve process's own profile. The middleware sets
the context-local HERMES_HOME override once for the whole namespace.
"""

from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, FastAPI
from fastapi.testclient import TestClient


def _app(router: APIRouter) -> TestClient:
    from hermes_cli.web_server_dashboard import _PluginProfileScopeMiddleware
    app = FastAPI()
    app.include_router(router, prefix="/api/plugins/probe")
    app.add_middleware(_PluginProfileScopeMiddleware)
    return TestClient(app)


def test_plugin_routes_resolve_hermes_home_for_the_requested_profile(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    (home / "profiles" / "research").mkdir(parents=True)
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(home))
    router = APIRouter()

    @router.get("/home")
    def where():
        from hermes_constants import get_hermes_home
        return {"home": str(get_hermes_home())}

    http = _app(router)
    assert http.get("/api/plugins/probe/home").json()["home"] == str(home)
    assert http.get("/api/plugins/probe/home?profile=research").json()["home"] == str(home / "profiles" / "research")
    assert http.get("/api/plugins/probe/home?profile=missing").status_code == 404
    # Scope is per request: the override never leaks into the next unscoped call.
    assert http.get("/api/plugins/probe/home").json()["home"] == str(home)
