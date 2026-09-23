"""Plugin API routes must run inside the launch/profile secret scope.

Regression for #120310: under multi-profile hosting the ``/api/plugins/<id>/…``
dispatch bound no secret scope, so a plugin handler's ``get_secret()`` raised
``UnscopedSecretError`` and the plugin's "no data" failure contract swallowed it.
The fix binds the scope via ``_plugin_route_secret_scope`` on every plugin router,
mirroring the sibling install/update (#116816) and MCP connect (#113746) fixes.

These tests mount a router through the *same* seam the production code uses
(``include_router(..., dependencies=[Depends(_plugin_route_secret_scope)])``) so
they cover the real wiring, not just the helper in isolation.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, FastAPI
from fastapi.testclient import TestClient

from agent.secret_scope import get_secret, is_multiplex_active, set_multiplex_active
from hermes_cli.web_server_dashboard import _plugin_route_secret_scope


def _whoami_router() -> APIRouter:
    """A stand-in for a third-party plugin backend that reads a credential and
    honors the plugin contract (never raises out of the handler)."""
    router = APIRouter()

    @router.get("/whoami")
    def whoami() -> dict:
        try:
            return {"ok": True, "key": get_secret("DEEPSEEK_API_KEY")}
        except Exception as exc:  # plugin contract: fold failures into "no data"
            return {"ok": False, "error": type(exc).__name__}

    return router


def _client(app: FastAPI) -> TestClient:
    return TestClient(app)


def _write_home_env(tmp_path, **secrets) -> str:
    home = tmp_path / "launch"
    home.mkdir()
    body = "".join(f"{k}={v}\n" for k, v in secrets.items())
    (home / ".env").write_text(body, encoding="utf-8")
    return str(home)


def _multiplex(active: bool):
    """Set the process-global multiplex flag, restoring it afterwards."""
    import contextlib

    @contextlib.contextmanager
    def _cm():
        previous = is_multiplex_active()
        set_multiplex_active(active)
        try:
            yield
        finally:
            set_multiplex_active(previous)

    return _cm()


def test_plugin_route_resolves_secret_under_multiplexing(tmp_path, monkeypatch):
    """The fix: with the scope dependency, a plugin route reading a credential
    resolves the launch profile's value under multi-profile hosting."""
    monkeypatch.setenv("HERMES_HOME", _write_home_env(tmp_path, DEEPSEEK_API_KEY="sk-live"))
    app = FastAPI()
    app.include_router(
        _whoami_router(),
        prefix="/api/plugins/example",
        dependencies=[Depends(_plugin_route_secret_scope)],
    )
    with _multiplex(True):
        resp = _client(app).get("/api/plugins/example/whoami")
    assert resp.status_code == 200
    assert resp.json() == {"ok": True, "key": "sk-live"}


def test_plugin_route_without_scope_fails_closed_under_multiplexing(tmp_path, monkeypatch):
    """Control: the same router mounted WITHOUT the dependency fails closed —
    proving the dependency is what fixes the bug, not the fixtures."""
    monkeypatch.setenv("HERMES_HOME", _write_home_env(tmp_path, DEEPSEEK_API_KEY="sk-live"))
    app = FastAPI()
    app.include_router(_whoami_router(), prefix="/api/plugins/example")
    with _multiplex(True):
        resp = _client(app).get("/api/plugins/example/whoami")
    assert resp.status_code == 200
    assert resp.json() == {"ok": False, "error": "UnscopedSecretError"}


def test_plugin_route_reads_environ_without_multiplexing(tmp_path, monkeypatch):
    """No regression: with multiplexing off the dependency is a no-op and the
    handler reads ``os.environ`` exactly as a single-profile host does today."""
    monkeypatch.setenv("HERMES_HOME", _write_home_env(tmp_path))  # empty .env
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-env")
    app = FastAPI()
    app.include_router(
        _whoami_router(),
        prefix="/api/plugins/example",
        dependencies=[Depends(_plugin_route_secret_scope)],
    )
    with _multiplex(False):
        resp = _client(app).get("/api/plugins/example/whoami")
    assert resp.status_code == 200
    assert resp.json() == {"ok": True, "key": "sk-env"}
