"""Plugin-contributed dashboard pages at ``/p/<slug>``.

The acceptance list this file encodes:

* a plugin router answers under ``/p/<slug>`` even with the SPA catch-all mounted,
* an unauthenticated caller gets 401 (never the page),
* WebSockets below ``/p/<slug>/`` work, several at once,
* under the auth gate a socket needs a single-use ``?ticket=``, not the session token,
* slug validation refuses path escapes and core-route collisions,
* two plugins do not interfere,
* unloading the plugin removes the route,
* ``hermes serve --headless`` still serves no plugin page.
"""

from __future__ import annotations

import pytest
from fastapi import APIRouter, FastAPI, WebSocket
from starlette.testclient import TestClient

from hermes_cli import dashboard_pages as dp


@pytest.fixture(autouse=True)
def _clean_registry():
    dp.clear_pages()
    yield
    dp.clear_pages()


def _page_router(marker: str) -> APIRouter:
    router = APIRouter()

    @router.get("/")
    async def index():
        return {"page": marker}

    @router.get("/deep/thing")
    async def deep():
        return {"deep": marker}

    @router.websocket("/ws")
    async def socket(ws: WebSocket):
        await ws.accept()
        await ws.send_text(f"hello-{marker}")
        got = await ws.receive_text()
        await ws.send_text(f"echo:{got}")
        await ws.close()

    return router


def _dashboard_app(monkeypatch, tmp_path, *, gated: bool = False):
    """A FastAPI app wired in the SAME order as ``web_server``: pages first, SPA last."""
    import hermes_cli.web_server as ws
    import hermes_cli.web_server_dashboard as wsd

    dist = tmp_path / "web_dist"
    (dist / "assets").mkdir(parents=True)
    (dist / "index.html").write_text("<html><head></head><body>SPA</body></html>", encoding="utf-8")
    monkeypatch.setattr(ws, "WEB_DIST", dist)
    monkeypatch.setattr(wsd, "WEB_DIST", dist, raising=False)
    monkeypatch.delenv("HERMES_SERVE_HEADLESS", raising=False)

    app = FastAPI()
    app.state.auth_required = gated
    # The WS gate reads the gate flag and the bound host off the module-level dashboard app
    # (HTTP middleware does not run for WS scopes), so pin them there for the test. bound_host
    # stays unset = "not listening yet", which is what start_server also leaves before bind.
    monkeypatch.setattr(ws.app.state, "auth_required", gated, raising=False)
    monkeypatch.setattr(ws.app.state, "bound_host", "", raising=False)
    dp.mount_pages(app)
    wsd.mount_spa(app)
    return app, ws


class TestSlugValidation:
    @pytest.mark.parametrize("slug", ["i3", "my-page", "a", "x9", "a-b-c"])
    def test_accepts_plain_slugs(self, slug):
        assert dp.validate_slug(slug) == slug

    @pytest.mark.parametrize(
        "slug",
        ["../etc", "a/b", "", "-lead", "trail-", "UPPER", "with space", "dot.dot",
         "%2e%2e", "a" * 40, None, 7],
    )
    def test_rejects_traversal_and_malformed(self, slug):
        with pytest.raises(dp.PageSlugError):
            dp.validate_slug(slug)

    @pytest.mark.parametrize("slug", ["api", "auth", "chat", "login", "assets"])
    def test_rejects_core_route_collisions(self, slug):
        with pytest.raises(dp.PageSlugError):
            dp.validate_slug(slug)

    def test_register_rejects_non_asgi_router(self):
        with pytest.raises(TypeError):
            dp.register_page(dp.DashboardPage(slug="x", title="X", router=object(), plugin="p"))

    def test_second_plugin_cannot_steal_a_live_slug(self):
        dp.register_page(dp.DashboardPage("i3", "A", _page_router("a"), plugin="alpha"))
        with pytest.raises(ValueError):
            dp.register_page(dp.DashboardPage("i3", "B", _page_router("b"), plugin="beta"))

    def test_same_plugin_rotates_in_place(self):
        dp.register_page(dp.DashboardPage("i3", "A", _page_router("a"), plugin="alpha"))
        dp.register_page(dp.DashboardPage("i3", "A2", _page_router("a2"), plugin="alpha"))
        assert dp.get_page("i3").title == "A2"


class TestRoutingAgainstTheSpa:
    def test_page_reachable_with_spa_mounted(self, monkeypatch, tmp_path):
        app, ws = _dashboard_app(monkeypatch, tmp_path)
        dp.register_page(dp.DashboardPage("i3", "Terminals", _page_router("i3"), plugin="hermes-i3"))
        client = TestClient(app)
        headers = {ws._SESSION_HEADER_NAME: ws._SESSION_TOKEN}

        # The SPA catch-all is live (it answers an unrelated client-side route)…
        assert "SPA" in client.get("/settings").text
        # …and the plugin page still wins under its own prefix.
        assert client.get("/p/i3", headers=headers).json() == {"page": "i3"}
        assert client.get("/p/i3/deep/thing", headers=headers).json() == {"deep": "i3"}

    def test_unknown_slug_is_404_not_the_spa(self, monkeypatch, tmp_path):
        app, ws = _dashboard_app(monkeypatch, tmp_path)
        resp = TestClient(app).get("/p/nope")
        assert resp.status_code == 404
        assert resp.json()["detail"] == "Plugin page not found"

    def test_two_plugins_do_not_interfere(self, monkeypatch, tmp_path):
        app, ws = _dashboard_app(monkeypatch, tmp_path)
        dp.register_page(dp.DashboardPage("i3", "T", _page_router("one"), plugin="alpha"))
        dp.register_page(dp.DashboardPage("board", "B", _page_router("two"), plugin="beta"))
        client = TestClient(app)
        headers = {ws._SESSION_HEADER_NAME: ws._SESSION_TOKEN}
        assert client.get("/p/i3", headers=headers).json() == {"page": "one"}
        assert client.get("/p/board", headers=headers).json() == {"page": "two"}
        assert client.get("/p/i3/deep/thing", headers=headers).json() == {"deep": "one"}

    def test_query_token_authenticates_a_navigation(self, monkeypatch, tmp_path):
        app, ws = _dashboard_app(monkeypatch, tmp_path)
        dp.register_page(dp.DashboardPage("i3", "T", _page_router("i3"), plugin="alpha"))
        resp = TestClient(app).get(f"/p/i3?token={ws._SESSION_TOKEN}")
        assert resp.status_code == 200

    def test_navigation_issues_a_cookie_so_subresources_load(self, monkeypatch, tmp_path):
        """A ``<script src>`` carries neither header nor query token. Without the cookie the page
        would render and every asset under it would 401 — the hole a plugin would otherwise
        plug with a token of its own."""
        app, ws = _dashboard_app(monkeypatch, tmp_path)
        dp.register_page(dp.DashboardPage("i3", "T", _page_router("i3"), plugin="alpha"))
        client = TestClient(app)
        nav = client.get(f"/p/i3/?token={ws._SESSION_TOKEN}")
        assert nav.status_code == 200
        cookie = nav.headers["set-cookie"]
        assert "HttpOnly" in cookie and "Path=/p" in cookie and "SameSite=Strict" in cookie
        # TestClient keeps the cookie jar: the follow-up asset request carries no token.
        assert client.get("/p/i3/deep/thing").status_code == 200

    def test_cookie_does_not_reach_outside_the_page_prefix(self, monkeypatch, tmp_path):
        app, ws = _dashboard_app(monkeypatch, tmp_path)
        dp.register_page(dp.DashboardPage("i3", "T", _page_router("i3"), plugin="alpha"))
        client = TestClient(app)
        client.get(f"/p/i3/?token={ws._SESSION_TOKEN}")
        jar = client.cookies.jar
        paths = {c.path for c in jar if c.name == dp._PAGE_COOKIE}
        assert paths == {"/p"}

    def test_header_auth_does_not_hand_out_a_cookie(self, monkeypatch, tmp_path):
        app, ws = _dashboard_app(monkeypatch, tmp_path)
        dp.register_page(dp.DashboardPage("i3", "T", _page_router("i3"), plugin="alpha"))
        resp = TestClient(app).get("/p/i3", headers={ws._SESSION_HEADER_NAME: ws._SESSION_TOKEN})
        assert resp.status_code == 200
        assert "set-cookie" not in resp.headers


class TestAuth:
    def test_unauthenticated_http_gets_401_not_the_page(self, monkeypatch, tmp_path):
        app, _ws = _dashboard_app(monkeypatch, tmp_path)
        dp.register_page(dp.DashboardPage("i3", "T", _page_router("i3"), plugin="alpha"))
        resp = TestClient(app).get("/p/i3")
        assert resp.status_code == 401
        assert "i3" not in resp.text

    def test_wrong_token_gets_401(self, monkeypatch, tmp_path):
        app, ws = _dashboard_app(monkeypatch, tmp_path)
        dp.register_page(dp.DashboardPage("i3", "T", _page_router("i3"), plugin="alpha"))
        resp = TestClient(app).get("/p/i3", headers={ws._SESSION_HEADER_NAME: "nope"})
        assert resp.status_code == 401

    def test_non_ascii_token_is_401_not_500(self, monkeypatch, tmp_path):
        """A pasted ``?token=…`` (literal ellipsis) must be a wrong token, not a crash —
        the exact bug the i3 server had to learn about ``secrets.compare_digest``."""
        app, _ws = _dashboard_app(monkeypatch, tmp_path)
        dp.register_page(dp.DashboardPage("i3", "T", _page_router("i3"), plugin="alpha"))
        resp = TestClient(app).get("/p/i3?token=abc…")
        assert resp.status_code == 401

    def test_gated_mode_requires_a_verified_session(self, monkeypatch, tmp_path):
        app, ws = _dashboard_app(monkeypatch, tmp_path, gated=True)
        dp.register_page(dp.DashboardPage("i3", "T", _page_router("i3"), plugin="alpha"))
        client = TestClient(app)
        # The session token must NOT open a gated page.
        resp = client.get("/p/i3", headers={ws._SESSION_HEADER_NAME: ws._SESSION_TOKEN})
        assert resp.status_code == 401

    def test_gated_mode_serves_a_verified_session(self, monkeypatch, tmp_path):
        app, _ws = _dashboard_app(monkeypatch, tmp_path, gated=True)
        dp.register_page(dp.DashboardPage("i3", "T", _page_router("i3"), plugin="alpha"))

        @app.middleware("http")
        async def _fake_gate(request, call_next):
            request.state.session = object()
            return await call_next(request)

        assert TestClient(app).get("/p/i3").status_code == 200

    def test_public_api_paths_are_not_consulted(self):
        """The plugin prefix must never be reachable via the auth bypass allow-list."""
        from hermes_cli.dashboard_auth.public_paths import PUBLIC_API_PATHS
        assert not any(p.startswith("/p/") or p == "/p" for p in PUBLIC_API_PATHS)


class TestWebSockets:
    def test_websocket_below_the_prefix(self, monkeypatch, tmp_path):
        app, ws = _dashboard_app(monkeypatch, tmp_path)
        dp.register_page(dp.DashboardPage("i3", "T", _page_router("i3"), plugin="alpha"))
        client = TestClient(app)
        with client.websocket_connect(f"/p/i3/ws?token={ws._SESSION_TOKEN}") as sock:
            assert sock.receive_text() == "hello-i3"
            sock.send_text("ping")
            assert sock.receive_text() == "echo:ping"

    def test_several_websockets_at_once(self, monkeypatch, tmp_path):
        """i3 needs one socket per pane, so concurrency is the requirement, not a nicety."""
        app, ws = _dashboard_app(monkeypatch, tmp_path)
        dp.register_page(dp.DashboardPage("i3", "T", _page_router("i3"), plugin="alpha"))
        client = TestClient(app)
        url = f"/p/i3/ws?token={ws._SESSION_TOKEN}"
        with client.websocket_connect(url) as a, client.websocket_connect(url) as b:
            assert a.receive_text() == "hello-i3"
            assert b.receive_text() == "hello-i3"
            a.send_text("one")
            b.send_text("two")
            assert a.receive_text() == "echo:one"
            assert b.receive_text() == "echo:two"

    def test_unauthenticated_websocket_is_refused(self, monkeypatch, tmp_path):
        from starlette.websockets import WebSocketDisconnect

        app, _ws = _dashboard_app(monkeypatch, tmp_path)
        dp.register_page(dp.DashboardPage("i3", "T", _page_router("i3"), plugin="alpha"))
        client = TestClient(app)
        with pytest.raises(WebSocketDisconnect) as exc:
            with client.websocket_connect("/p/i3/ws") as sock:
                sock.receive_text()
        assert exc.value.code == 4401

    def test_websocket_on_unknown_slug_is_refused(self, monkeypatch, tmp_path):
        from starlette.websockets import WebSocketDisconnect

        app, ws = _dashboard_app(monkeypatch, tmp_path)
        client = TestClient(app)
        with pytest.raises(WebSocketDisconnect) as exc:
            with client.websocket_connect(f"/p/gone/ws?token={ws._SESSION_TOKEN}") as sock:
                sock.receive_text()
        assert exc.value.code == 4404

    def test_websocket_inherits_the_dns_rebinding_guard(self, monkeypatch, tmp_path):
        """A plugin WS gets the dashboard's Host check for free — the reason the surface exists
        rather than each plugin re-deriving it."""
        from starlette.websockets import WebSocketDisconnect

        import hermes_cli.web_server as ws_mod

        app, ws = _dashboard_app(monkeypatch, tmp_path)
        dp.register_page(dp.DashboardPage("i3", "T", _page_router("i3"), plugin="alpha"))
        # Bound to loopback, but the upgrade claims a foreign Host: rebinding shape.
        monkeypatch.setattr(ws_mod.app.state, "bound_host", "127.0.0.1", raising=False)
        client = TestClient(app, base_url="http://evil.example")
        with pytest.raises(WebSocketDisconnect) as exc:
            with client.websocket_connect(f"/p/i3/ws?token={ws._SESSION_TOKEN}") as sock:
                sock.receive_text()
        assert exc.value.code == 4401


class TestGatedWebSockets:
    """Under the auth gate a page renders from its session but its sockets need a ``?ticket=``.

    A browser cannot set a header on a WS upgrade and the session cookie is not accepted there,
    so core mints a single-use ticket via ``POST /api/auth/ws-ticket``. A page that connects
    without one shows an empty shell: HTML and assets load, every socket is rejected. These pin
    both directions so a plugin author can rely on the contract instead of rediscovering it.
    """

    def test_gated_socket_without_a_ticket_is_refused(self, monkeypatch, tmp_path):
        from starlette.websockets import WebSocketDisconnect

        app, _ws = _dashboard_app(monkeypatch, tmp_path, gated=True)
        dp.register_page(dp.DashboardPage("i3", "T", _page_router("i3"), plugin="alpha"))
        with pytest.raises(WebSocketDisconnect) as exc:
            with TestClient(app).websocket_connect("/p/i3/ws") as sock:
                sock.receive_text()
        assert exc.value.code == 4401

    def test_gated_socket_rejects_the_session_token(self, monkeypatch, tmp_path):
        """The loopback credential must not open a socket on a gated deployment."""
        from starlette.websockets import WebSocketDisconnect

        app, ws = _dashboard_app(monkeypatch, tmp_path, gated=True)
        dp.register_page(dp.DashboardPage("i3", "T", _page_router("i3"), plugin="alpha"))
        with pytest.raises(WebSocketDisconnect) as exc:
            with TestClient(app).websocket_connect(f"/p/i3/ws?token={ws._SESSION_TOKEN}") as sock:
                sock.receive_text()
        assert exc.value.code == 4401

    def test_gated_socket_opens_with_a_minted_ticket(self, monkeypatch, tmp_path):
        from hermes_cli.dashboard_auth.ws_tickets import mint_ticket

        app, _ws = _dashboard_app(monkeypatch, tmp_path, gated=True)
        dp.register_page(dp.DashboardPage("i3", "T", _page_router("i3"), plugin="alpha"))
        ticket = mint_ticket(user_id="probe", provider="local")
        with TestClient(app).websocket_connect(f"/p/i3/ws?ticket={ticket}") as sock:
            assert sock.receive_text() == "hello-i3"
            sock.send_text("ping")
            assert sock.receive_text() == "echo:ping"

    def test_gated_ticket_is_single_use(self, monkeypatch, tmp_path):
        from starlette.websockets import WebSocketDisconnect

        from hermes_cli.dashboard_auth.ws_tickets import mint_ticket

        app, _ws = _dashboard_app(monkeypatch, tmp_path, gated=True)
        dp.register_page(dp.DashboardPage("i3", "T", _page_router("i3"), plugin="alpha"))
        ticket = mint_ticket(user_id="probe", provider="local")
        client = TestClient(app)
        with client.websocket_connect(f"/p/i3/ws?ticket={ticket}") as sock:
            assert sock.receive_text() == "hello-i3"
            sock.send_text("ping")
            assert sock.receive_text() == "echo:ping"
        with pytest.raises(WebSocketDisconnect) as exc:
            with client.websocket_connect(f"/p/i3/ws?ticket={ticket}") as sock:
                sock.receive_text()
        assert exc.value.code == 4401


class TestHeadless:
    def test_headless_serves_no_plugin_page(self, monkeypatch, tmp_path):
        import hermes_cli.web_server_dashboard as wsd

        monkeypatch.setenv("HERMES_SERVE_HEADLESS", "1")
        app = FastAPI()
        app.state.auth_required = False
        dp.mount_pages(app)          # no-op under headless
        wsd.mount_spa(app)
        dp.register_page(dp.DashboardPage("i3", "T", _page_router("i3"), plugin="alpha"))
        resp = TestClient(app).get("/p/i3")
        assert resp.status_code == 404
        assert "Headless backend" in resp.text

    def test_headless_flip_after_mount_still_refuses(self, monkeypatch, tmp_path):
        """The dispatcher re-checks per request, so a late flag flip cannot leak a page."""
        app, ws = _dashboard_app(monkeypatch, tmp_path)
        dp.register_page(dp.DashboardPage("i3", "T", _page_router("i3"), plugin="alpha"))
        monkeypatch.setenv("HERMES_SERVE_HEADLESS", "1")
        resp = TestClient(app).get("/p/i3", headers={ws._SESSION_HEADER_NAME: ws._SESSION_TOKEN})
        assert resp.status_code == 404


class TestMountOrder:
    def test_pages_mount_before_the_spa_catch_all(self):
        """The SPA's ``/{full_path:path}`` matches every otherwise-unclaimed path, so a ``/p``
        mount registered after it is unreachable. Assert the real dashboard app's route order,
        not the order of two statements in a source file."""
        import hermes_cli.web_server as ws

        paths = [getattr(route, "path", "") or "" for route in ws.app.routes]
        assert "/p" in paths, "the dashboard app does not mount the plugin page dispatcher"
        catch_all = [i for i, path in enumerate(paths) if "{full_path" in path]
        assert catch_all, "the SPA catch-all route is missing"
        assert paths.index("/p") < catch_all[0]


class TestPluginLifecycle:
    def test_registration_survives_unload_all_but_dies_on_targeted_unload(self, monkeypatch, tmp_path):
        """#91701 answer: the page registry is process-global, so a routine unload-all must not
        blank it (``persistent=True``), while a targeted unload of the owning plugin must."""
        from hermes_cli.plugins import PluginManager, PluginContext
        from hermes_cli.plugins_manifest import PluginManifest

        manager = PluginManager()
        manifest = PluginManifest(name="alpha", version="1.0.0", description="")
        ctx = PluginContext(manifest, manager)
        handle = ctx.register_dashboard_page(slug="i3", title="Terminals", router=_page_router("i3"))

        assert handle is not None and handle.persistent
        assert dp.get_page("i3") is not None
        # Not in the reverse-order teardown list: an unload-all cannot dispose it.
        assert handle not in manager._registration_order
        handle.dispose()
        assert dp.get_page("i3") is None

    def test_bad_slug_warns_and_returns_none(self):
        from hermes_cli.plugins import PluginManager, PluginContext
        from hermes_cli.plugins_manifest import PluginManifest

        ctx = PluginContext(PluginManifest(name="alpha", version="1", description=""),
                            PluginManager())
        assert ctx.register_dashboard_page(slug="../escape", title="x", router=_page_router("x")) is None
        assert ctx.register_dashboard_page(slug="api", title="x", router=_page_router("x")) is None
        assert ctx.register_dashboard_page(slug="ok", title="x", router=object()) is None
        assert dp.list_pages() == []

    def test_unregister_is_identity_conditional(self):
        first = dp.DashboardPage("i3", "A", _page_router("a"), plugin="alpha")
        second = dp.DashboardPage("i3", "B", _page_router("b"), plugin="alpha")
        dp.register_page(first)
        dp.register_page(second)
        assert dp.unregister_page("i3", first) is False   # stale handle, newer page survives
        assert dp.get_page("i3") is second
        assert dp.unregister_page("i3", second) is True
        assert dp.get_page("i3") is None
