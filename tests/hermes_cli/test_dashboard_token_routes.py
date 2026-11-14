"""Provider-bound plugin token routes, including the unified dashboard launch topology."""
from __future__ import annotations

import json
import secrets
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi import FastAPI, Request, WebSocket
from fastapi.responses import JSONResponse
from fastapi.testclient import TestClient

from hermes_cli.dashboard_auth import DashboardAuthProvider, ProviderError, TokenPrincipal
from hermes_cli.dashboard_auth import registry, token_auth
from hermes_cli.plugins import PluginContext, PluginManager, PluginManifest, PluginRegistration
from hermes_constants import (
    get_hermes_home, hermes_home_key, reset_hermes_home_override, set_hermes_home_override,
)

PATH = "/api/plugins/machine/jobs"


class _Provider(DashboardAuthProvider):
    name = "machine-token"
    display_name = "Machine Token"
    supports_token = True
    supports_session = False

    def __init__(self, result, secret="owned"):
        self.result, self.secret, self.tokens = result, secret, []

    def verify_token(self, *, token):
        self.tokens.append(token)
        if isinstance(self.result, Exception):
            raise self.result
        return self.result if token == self.secret else None

    def unused(self, **kwargs):
        return None

    start_login = complete_login = verify_session = refresh_session = revoke_session = unused


@pytest.fixture(autouse=True)
def _isolate_auth(monkeypatch):
    if web_server := sys.modules.get("hermes_cli.web_server"):
        monkeypatch.setattr(web_server, "_dashboard_plugins_cache", None)
    registry.clear_providers()
    token_auth.clear_token_routes()
    yield
    registry.clear_providers()
    token_auth.clear_token_routes()


@pytest.fixture
def context():
    directory = get_hermes_home() / "plugins" / "machine"
    (directory / "dashboard").mkdir(parents=True)
    (directory / "dashboard" / "manifest.json").write_text("{}", encoding="utf-8")
    manager = PluginManager(scope_key=hermes_home_key())
    ctx = PluginContext(PluginManifest(name="machine", path=str(directory)), manager)
    yield manager, ctx
    manager.unload("machine")


def _principal(provider="machine-token", scopes=("r", "w", "extra")):
    return TokenPrincipal(principal="client", provider=provider, scopes=scopes)


def _client(path=PATH):
    app = FastAPI()

    @app.get(path)
    def endpoint(request: Request):
        if not getattr(request.state, "token_authenticated", False):
            return JSONResponse({"error": "session required"}, status_code=401)
        return {"provider": request.state.token_principal.provider}

    @app.websocket(path)
    async def socket(websocket: WebSocket):
        await websocket.accept()
        await websocket.send_json({"token_authenticated": getattr(websocket.state, "token_authenticated", False)})
        await websocket.close()

    app.middleware("http")(token_auth.token_auth_middleware)
    return TestClient(app)


@pytest.mark.parametrize(
    "result,secret,status,reason",
    [
        (_principal(), "owned", 200, None),
        (_principal(), "foreign", 401, "token_rejected"),
        (_principal(scopes=("r",)), "owned", 403, "insufficient_scope"),
        ({"provider": "machine-token"}, "owned", 401, "invalid_principal"),
        (object(), "owned", 401, "invalid_principal"),
        (_principal(scopes=None), "owned", 401, "invalid_principal"),
        (_principal(scopes=([],)), "owned", 401, "invalid_principal"),
        (_principal(scopes="rw"), "owned", 401, "invalid_principal"),
        (_principal(scopes=b"rw"), "owned", 401, "invalid_principal"),
        (_principal(provider="other"), "owned", 401, "provider_mismatch"),
        (ProviderError("store down"), "owned", 503, "provider_unreachable"),
        (RuntimeError("broken verifier"), "owned", 401, "provider_error"),
        (None, "owned", 401, "token_rejected"),
        (_principal(), "", 401, "missing_token"),
        ("removed-provider", "owned", 401, "token_provider_unavailable"),
        ("session-only", "owned", 401, "token_provider_unavailable"),
        ("replaced-provider", "owned", 401, "token_provider_replaced"),
    ],
    ids=["valid", "foreign-token", "missing-scope", "dict", "object", "noniterable-scopes",
         "unhashable-scopes", "str-scopes", "bytes-scopes", "foreign-principal",
         "unreachable", "buggy", "unrecognised", "no-token", "removed-provider", "session-only",
         "replaced-provider"],
)
def test_bound_route_enforces_provider_principal_and_scopes(
    context, monkeypatch, result, secret, status, reason,
):
    manager, ctx = context
    other = _Provider(_principal(provider="other"), secret="foreign")
    other.name = "other"
    registry.register_provider(other)
    provider = _Provider(result)
    provider_handle = ctx.register_dashboard_auth_provider(provider)
    handle = ctx.register_dashboard_token_route(PATH, provider=provider.name, required_scopes=("r", "w"))
    assert isinstance(handle, PluginRegistration)
    if result == "removed-provider":
        provider_handle.dispose()
    if result == "session-only":
        provider.supports_token = False
    if result == "replaced-provider":
        impostor = _Provider(_principal())
        PluginContext(PluginManifest(name="other"), manager).register_dashboard_auth_provider(impostor)
    events = []
    monkeypatch.setattr(token_auth, "audit_log", lambda event, **fields: events.append(fields))

    with _client() as client:
        response = client.get(PATH, headers={"Authorization": f"Bearer {secret}"} if secret else {})

    assert response.status_code == status
    expected_tokens = [secret] if secret and result not in (
        "removed-provider", "session-only", "replaced-provider",
    ) else []
    assert provider.tokens == expected_tokens
    assert other.tokens == []
    if result == "replaced-provider":
        assert impostor.tokens == []
        manager.unload("other")
    if reason:
        assert events == [{
            "provider": provider.name, "reason": reason, "path": PATH, "ip": "testclient",
        }]
        if status == 503:
            assert response.json() == {"detail": f"Auth provider {provider.name!r} unreachable"}
        else:
            assert response.json()["error"] == ("insufficient_scope" if status == 403 else "unauthenticated")
    else:
        assert response.json() == {"provider": provider.name}
        assert events == []


@pytest.mark.parametrize("case", [
    "/api/gateway/drain", "/api/plugins/other/jobs", "/api/plugins/machine-extra/jobs",
    "/api/plugins/machine", "/api/plugins/machine/../other", PATH + "?page=1",
    PATH + "/{id}", PATH + "/*",
    ("", ()), (None, ()), ("machine-token", "read"), ("machine-token", ("",)),
    ("machine-token", ([],)),
    "entrypoint", "file", "missing", "unset", "non-dashboard", "shadowed", "secondary",
    "no-provider", "disposed-provider", "other-plugin", "other-manager",
])
def test_registration_requires_owned_mount_provider_and_valid_policy(context, tmp_path, case):
    manager, ctx = context
    registrar = ctx
    if case == "other-plugin":
        registrar = PluginContext(PluginManifest(name="other"), manager)
    elif case == "other-manager":
        registrar = PluginContext(ctx.manifest, PluginManager(scope_key=manager.scope_key))
    handle = None if case == "no-provider" else registrar.register_dashboard_auth_provider(_Provider(_principal()))
    if case == "disposed-provider":
        handle.dispose()
    path, provider, scopes = PATH, "machine-token", ()
    if isinstance(case, tuple):
        provider, scopes = case
    elif case.startswith("/"):
        path = case
    elif case == "entrypoint":
        ctx.manifest.source = "entrypoint"
    elif case == "unset":
        ctx.manifest.path = None
    elif case == "shadowed":
        winner = Path(ctx.manifest.path).parent / "00-winner"
        (winner / "dashboard").mkdir(parents=True)
        (winner / "dashboard" / "manifest.json").write_text('{"name":"machine"}', encoding="utf-8")
    elif case == "secondary":
        manager = PluginManager(scope_key=hermes_home_key(tmp_path / "worker"))
        ctx = PluginContext(ctx.manifest, manager)
    elif case in ("file", "missing", "non-dashboard"):
        directory = tmp_path / case / "machine"
        if case == "file":
            directory.parent.mkdir(parents=True)
            directory.write_text("not a directory", encoding="utf-8")
        elif case == "non-dashboard":
            directory.mkdir(parents=True)
        ctx.manifest.path = str(directory)

    assert ctx.register_dashboard_token_route(path, provider=provider, required_scopes=scopes) is None
    assert not token_auth.is_token_route(path)
    assert not any(r.kind == "dashboard_token_route" for r in manager._ownership_ledger.get("machine", ()))
    if handle is not None:
        handle.dispose()


@pytest.mark.parametrize("dashboard_manifest", [{"name": "mounted"}, {}], ids=["dashboard-name", "directory-name"])
def test_mount_guard_uses_cached_router_owner(context, dashboard_manifest):
    manager, ctx = context
    ctx.manifest.name = "different-native-name"
    directory = Path(ctx.manifest.path)
    (directory / "dashboard" / "manifest.json").write_text(json.dumps(dashboard_manifest), encoding="utf-8")
    ctx.register_dashboard_auth_provider(_Provider(_principal()))
    name = dashboard_manifest.get("name", directory.name)
    prefix = f"/api/plugins/{name}/"
    assert isinstance(ctx.register_dashboard_token_route(prefix + "jobs", provider="machine-token"), PluginRegistration)
    # A load sweep keeps its discovery winner even if another plugin is installed midway.
    winner = directory.parent / "00-winner"
    (winner / "dashboard").mkdir(parents=True)
    (winner / "dashboard" / "manifest.json").write_text(
        json.dumps({"name": name}), encoding="utf-8")
    shadow = PluginContext(PluginManifest(name="shadow", path=str(winner)), manager)
    shadow.register_dashboard_auth_provider(_Provider(_principal(provider="shadow")))
    assert isinstance(ctx.register_dashboard_token_route(prefix + "child", provider="machine-token"), PluginRegistration)
    assert shadow.register_dashboard_token_route(prefix + "jobs", provider="machine-token") is None
    manager.unload("shadow")
    assert ctx.register_dashboard_token_route(
        "/api/plugins/different-native-name/jobs", provider="machine-token") is None


def test_binding_lifetime_preserves_identity_and_legacy_exclusion(context):
    manager, ctx = context
    owner = _Provider(_principal())
    ctx.register_dashboard_auth_provider(owner)
    assert token_auth.register_token_route("/legacy") is None
    assert not token_auth.is_token_route("/legacy/child")
    with _client("/legacy") as legacy, _client() as bound:
        headers = {"Authorization": "Bearer owned"}
        assert legacy.get("/legacy", headers=headers).status_code == 200
        old = ctx.register_dashboard_token_route(PATH, provider=owner.name, required_scopes=("r", "w"))
        current = ctx.register_dashboard_token_route(PATH, provider=owner.name, required_scopes=("w", "r", "r"))
        second = ctx.register_dashboard_token_route(PATH + "/child", provider=owner.name)
        assert isinstance(current, PluginRegistration)
        assert ctx.register_dashboard_token_route(PATH, provider=owner.name) is None
        old.dispose()
        assert bound.get(PATH, headers=headers).status_code == 200
        with bound.websocket_connect(PATH) as socket:
            assert socket.receive_json() == {"token_authenticated": False}
        owner.tokens.clear()
        for handle in (current, second):
            assert legacy.get("/legacy", headers=headers).status_code == 401
            assert owner.tokens == []
            handle.dispose()
        assert legacy.get("/legacy", headers=headers).status_code == 200
        current = ctx.register_dashboard_token_route(PATH, provider=owner.name)
        assert manager.unload("machine")
        assert not current.active
        assert not token_auth.is_token_route(PATH)
        registry.register_provider(_Provider(ProviderError("store down")))
        assert legacy.get("/legacy", headers=headers).status_code == 503
        assert legacy.get("/legacy").status_code == 401


def _write_native(directory, *, provider="machine-token", scopes=("read",), secret="owned", bind=True, fail=False):
    (directory / "__init__.py").write_text(
        f"""from hermes_cli.dashboard_auth import DashboardAuthProvider, TokenPrincipal
class Provider(DashboardAuthProvider):
    name = {provider!r}
    display_name = "Machine Token"
    supports_token = True
    supports_session = False
    def __init__(self):
        self.tokens = []
    def verify_token(self, *, token):
        self.tokens.append(token)
        return TokenPrincipal("client", self.name, {scopes!r}) if token == {secret!r} else None
    def unused(self, **kwargs):
        return None
    start_login = complete_login = verify_session = refresh_session = revoke_session = unused
def register(ctx):
    global route
    ctx.register_dashboard_auth_provider(Provider())
    route = ctx.register_dashboard_token_route({PATH!r}, provider={provider!r}, required_scopes={scopes!r}) if {bind!r} else None
    if {fail!r}:
        raise RuntimeError("registration failed")
""", encoding="utf-8",
    )


@pytest.fixture
def real_dashboard(tmp_path, monkeypatch):
    """Real discovery, native registration, API mount, auth gates and secondary-profile host."""
    from agent.secret_scope import is_multiplex_active, set_multiplex_active
    from hermes_cli import plugins as plugins_mod, web_server
    from plugins.dashboard_auth import drain
    from plugins.dashboard_auth.session_token import SessionTokenProvider
    import tui_gateway.launch_profile_policy as lpp

    launch = tmp_path / ".hermes"
    profile = launch / "profiles" / "worker"
    profile.mkdir(parents=True)
    config = "plugins:\n  enabled: [machine]\n"
    (profile / "config.yaml").write_text(config, encoding="utf-8")
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(launch))
    original_routes = list(web_server.app.router.routes)
    monkeypatch.setattr(web_server.app.router, "routes", list(original_routes))
    monkeypatch.setattr(plugins_mod, "_plugin_managers_by_home", {})
    monkeypatch.setattr(plugins_mod, "_plugin_manager", None)
    monkeypatch.setattr(lpp, "_snapshot", None)
    previous_multiplex = is_multiplex_active()
    (launch / "config.yaml").write_text(config, encoding="utf-8")
    api = (
        "from fastapi import APIRouter\n"
        "router = APIRouter()\n"
        "@router.get('/jobs')\n"
        "def jobs(): return {'copy': %r}\n"
        "@router.get('/session')\n"
        "def session(): return {'ok': True}\n"
    )
    for home, copy in ((launch, "launch"), (profile, "worker")):
        directory = home / "plugins" / "machine"
        (directory / "dashboard").mkdir(parents=True)
        (directory / "dashboard" / "manifest.json").write_text(
            '{"name": "machine", "api": "api.py"}', encoding="utf-8")
        (directory / "dashboard" / "api.py").write_text(api % copy, encoding="utf-8")
        (directory / "plugin.yaml").write_text("name: machine\n", encoding="utf-8")
        (directory / "__init__.py").write_text("def register(ctx): pass\n", encoding="utf-8")
    _write_native(launch / "plugins" / "machine")
    manager = plugins_mod.get_plugin_manager()
    monkeypatch.setattr(plugins_mod, "get_bundled_plugins_dir", lambda: tmp_path / "bundled")
    monkeypatch.setattr(web_server.app.state, "initial_profile", "worker", raising=False)
    events = []
    monkeypatch.setattr(token_auth, "audit_log", lambda event, **fields: events.append(fields))
    drain_secret = secrets.token_urlsafe(48)
    monkeypatch.setenv("HERMES_DASHBOARD_DRAIN_SECRET", drain_secret)
    try:
        impostor = _Provider(_principal(), secret="foreign")
        registry.register_provider(impostor, scope=hermes_home_key(profile))
        manager.discover_and_load()
        drain.register(PluginContext(PluginManifest(name="drain"), manager))
        assert registry.snapshot_registration("drain-secret") is not None
        registry.register_global_provider(SessionTokenProvider(token="dashboard-session"))
        web_server._mount_plugin_api_routes()
        # Mounting after import must put the new router before the SPA and hot-mount fallbacks.
        fresh = [r for r in web_server.app.router.routes if r not in original_routes]
        web_server.app.router.routes[:] = fresh + original_routes
        client = TestClient(web_server.app)
        try:
            yield SimpleNamespace(
                client=client, profile=profile, home=launch, events=events, impostor=impostor, manager=manager,
                provider=registry.snapshot_registration("machine-token"), drain_secret=drain_secret,
            )
        finally:
            client.close()
    finally:
        manager.unload("machine")
        for scoped in plugins_mod._plugin_managers_by_home.values():
            host = getattr(scoped, "_plugin_host_instance", None)
            if host is not None:
                host.shutdown()
        set_multiplex_active(previous_multiplex)


@pytest.mark.parametrize("auth_required", [False, True], ids=["loopback", "gated"])
def test_real_stack_keeps_bound_and_drain_credentials_separate(real_dashboard, monkeypatch, auth_required):
    from hermes_cli import web_server
    from hermes_cli.dashboard_auth.cookies import SESSION_AT_COOKIE

    monkeypatch.setattr(web_server.app.state, "auth_required", auth_required, raising=False)
    dashboard = real_dashboard
    assert dashboard.client.get(PATH, headers={"Authorization": "Bearer foreign"}).status_code == 401
    client = dashboard.client
    assert client.get(PATH, headers={"Authorization": "Bearer owned"}).json() == {"copy": "launch"}
    client.cookies.set(SESSION_AT_COOKIE, "dashboard-session")
    session_headers = {web_server._SESSION_HEADER_NAME: web_server._SESSION_TOKEN}
    assert client.get("/api/plugins/machine/session", headers=session_headers).status_code == 200
    assert client.get(PATH, headers=session_headers).status_code == 401
    assert dashboard.events[-1]["reason"] == "missing_token"
    client.cookies.clear()
    assert client.get(PATH, headers={"Authorization": f"Bearer {dashboard.drain_secret}"}).status_code == 401
    ctx = PluginContext(dashboard.manager._plugins["machine"].manifest, dashboard.manager)
    assert ctx.register_dashboard_token_route(PATH + "/foreign", provider="drain-secret") is None
    assert client.post("/api/gateway/drain", json={"action": "drain"}, headers={
        "Authorization": f"Bearer {dashboard.drain_secret}",
    }).status_code == 200
    marker = dashboard.home / ".drain_request.json"
    assert json.loads(marker.read_text(encoding="utf-8"))["principal"] == "drain-control"
    calls = list(dashboard.provider.tokens)
    assert client.post("/api/gateway/drain", json={"action": "cancel"}, headers={
        "Authorization": "Bearer owned",
    }).status_code == 401
    assert dashboard.events[-1]["reason"] == "no_provider_recognises_token"
    assert dashboard.provider.tokens == calls
    assert marker.exists()


@pytest.mark.parametrize("selection", [
    "profile=worker", "profile=worker&profile=", "profile=worker&profile", "ambient", "unknown", "resolution-error",
])
@pytest.mark.parametrize("auth_required", [False, True], ids=["loopback", "gated"])
def test_unified_dashboard_launch_registration_refuses_secondary_profile(
    real_dashboard, monkeypatch, selection, auth_required,
):
    from agent.secret_scope import is_multiplex_active
    from hermes_cli import env_loader, profiles, web_server

    monkeypatch.setattr(web_server.app.state, "auth_required", auth_required, raising=False)
    dashboard = real_dashboard
    if selection == "resolution-error":
        def broken_resolution(name):
            raise OSError("unreadable profile directory")
        monkeypatch.setattr(profiles, "get_profile_dir", broken_resolution)
        selection = "profile=worker"
    elif selection == "unknown":
        selection = "profile=unknown"
    override = set_hermes_home_override(dashboard.profile) if selection == "ambient" else None
    previous_multiplex = is_multiplex_active()
    try:
        query = "" if selection == "ambient" else "?" + selection
        response = dashboard.client.get(PATH + query, headers={"Authorization": "Bearer owned"})
        assert response.status_code == 401, response.text
        assert dashboard.events[-1]["reason"] == "profile_scope"
        assert dashboard.events[-1]["provider"] == "machine-token"
        assert dashboard.impostor.tokens == []
        assert is_multiplex_active() == previous_multiplex
        def forbidden_hydration(*args, **kwargs):
            pytest.fail("token refusal entered profile secret hydration")
        monkeypatch.setattr(env_loader, "hydrate_profile_secret_sources", forbidden_hydration)
        assert dashboard.client.get(PATH + query, headers={"Authorization": "Bearer owned"}).status_code == 401
    finally:
        if override is not None:
            reset_hermes_home_override(override)


@pytest.mark.parametrize("mode", ["reregistered", "changed-scopes", "changed-provider", "omitted", "failed"])
def test_force_rediscovery_preserves_binding_until_reconciliation(real_dashboard, monkeypatch, mode):
    dashboard, manager = real_dashboard, real_dashboard.manager
    old = next(r for r in manager._ownership_ledger["machine"] if r.kind == "dashboard_token_route")
    original = manager._discover_and_load_inner
    manifest = manager._plugins["machine"].manifest
    provider = "replacement-token" if mode == "changed-provider" else "machine-token"
    bearer = "updated" if mode == "changed-provider" else "owned"
    scopes = ("read", "write") if mode == "changed-scopes" else ("read",)
    _write_native(Path(manifest.path), provider=provider, scopes=scopes, secret=bearer,
                  bind=mode != "omitted", fail=mode == "failed")
    assert dashboard.client.post("/api/gateway/drain", json={"action": "drain"}, headers={
        "Authorization": f"Bearer {dashboard.drain_secret}",
    }).status_code == 200
    marker = dashboard.home / ".drain_request.json"

    def assert_guard():
        assert dashboard.client.post("/api/gateway/drain", json={"action": "cancel"}, headers={
            "Authorization": f"Bearer {bearer}",
        }).status_code == 401
        assert marker.exists()
        assert token_auth.is_token_route(PATH)
        assert dashboard.client.get(PATH).status_code == 401
        assert dashboard.events[-1]["reason"] == "missing_token"

    def sweep():
        assert_guard()
        if mode == "changed-scopes":
            foreign = PluginContext(PluginManifest(name="other", path=manifest.path), manager)
            handle = foreign.register_dashboard_auth_provider(_Provider(_principal()))
            assert foreign.register_dashboard_token_route(PATH, provider="machine-token", required_scopes=scopes) is None
            handle.dispose()
        original()
        if mode != "failed":
            assert_guard()

    monkeypatch.setattr(manager, "_discover_and_load_inner", sweep)
    manager.discover_and_load(force=True)
    if mode not in ("omitted", "failed"):
        loaded = manager._plugins["machine"]
        assert loaded.module.route is not None
        assert token_auth.is_token_route(PATH)
        assert dashboard.client.get(PATH, headers={"Authorization": f"Bearer {bearer}"}).status_code == 200
        if mode == "changed-scopes":
            registry.snapshot_registration(provider).verify_token = lambda **kwargs: _principal(scopes=("read",))
            assert dashboard.client.get(PATH, headers={"Authorization": "Bearer owned"}).status_code == 403
        assert manager.unload("machine")
        assert not token_auth.is_token_route(PATH)
        # Discarded parked generations must never be restored by the current release.
        old.dispose()
    assert not token_auth.is_token_route(PATH)
    if mode == "failed":
        assert registry.snapshot_registration(provider) is None
