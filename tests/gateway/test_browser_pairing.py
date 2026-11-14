"""Tests for loopback browser-extension pairing.

Covers:
- BrowserPairingStore lifecycle: create, get, grant, deny, expiry, prune,
  persistence across instances, token validation.
- The five /api/browser-extension/pair/* HTTP handlers: start, approve page
  (button IDs + form actions contract), grant, deny, status polling.
- Loopback enforcement on every pairing endpoint.
- The scoped-token auth hook in _check_auth (loopback profile exception,
  remote fail-closed behavior).
"""
from unittest.mock import MagicMock

import pytest
import pytest_asyncio
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from gateway.browser_pairing import (
    BrowserPairingStore,
    PAIRING_TTL_SECONDS,
    TOKEN_MAX_AGE_SECONDS,
)
from gateway.config import PlatformConfig
from gateway.platforms.api_server import APIServerAdapter


# ---------------------------------------------------------------------------
# BrowserPairingStore unit tests
# ---------------------------------------------------------------------------


class TestBrowserPairingStore:

    def test_create_pairing_is_pending_with_ttl(self, tmp_path):
        store = BrowserPairingStore(tmp_path / "state.json")
        record = store.create_pairing(name="Hermes Browser Extension", extension_id="ext-123")
        assert record["status"] == "pending"
        assert record["name"] == "Hermes Browser Extension"
        assert record["extension_id"] == "ext-123"
        assert record["token"] is None
        assert float(record["expires_at"]) - float(record["created_at"]) == PAIRING_TTL_SECONDS
        # No token is minted at start.
        assert store.snapshot()["tokens"] == {}

    def test_get_pairing_returns_copy(self, tmp_path):
        store = BrowserPairingStore(tmp_path / "state.json")
        record = store.create_pairing()
        fetched = store.get_pairing(record["pairing_id"])
        # create_pairing returns the pairing_id as an extra top-level key;
        # get_pairing returns the bare record.
        assert fetched == {k: v for k, v in record.items() if k != "pairing_id"}
        # Mutation of the returned dict must not leak into the store.
        fetched["status"] = "approved"
        assert store.get_pairing(record["pairing_id"])["status"] == "pending"
        # And the original record object is unaffected too (it is a copy).
        assert record["status"] == "pending"

    def test_get_unknown_pairing_returns_none(self, tmp_path):
        store = BrowserPairingStore(tmp_path / "state.json")
        assert store.get_pairing("does-not-exist") is None

    def test_pairing_expires(self, tmp_path):
        clock = {"now": 1000.0}
        store = BrowserPairingStore(tmp_path / "state.json", now=lambda: clock["now"])
        record = store.create_pairing()
        assert store.get_pairing(record["pairing_id"])["status"] == "pending"
        clock["now"] += PAIRING_TTL_SECONDS + 1
        expired = store.get_pairing(record["pairing_id"])
        assert expired["status"] == "expired"
        # Granting an expired pairing is refused.
        assert store.grant_pairing(record["pairing_id"]) is None

    def test_grant_mints_token_and_persists(self, tmp_path):
        store = BrowserPairingStore(tmp_path / "state.json")
        record = store.create_pairing(extension_id="ext-abc")
        granted = store.grant_pairing(record["pairing_id"])
        assert granted["status"] == "approved"
        token = granted["token"]
        assert token and len(token) >= 32
        assert store.is_valid_token(token) is True
        # Token record carries the extension id and is not revoked.
        snapshot = store.snapshot()
        assert snapshot["tokens"][token]["extension_id"] == "ext-abc"
        assert snapshot["tokens"][token]["revoked"] is False

    def test_grant_twice_is_refused(self, tmp_path):
        store = BrowserPairingStore(tmp_path / "state.json")
        record = store.create_pairing()
        assert store.grant_pairing(record["pairing_id"]) is not None
        assert store.grant_pairing(record["pairing_id"]) is None

    def test_deny_marks_pairing_denied(self, tmp_path):
        store = BrowserPairingStore(tmp_path / "state.json")
        record = store.create_pairing()
        denied = store.deny_pairing(record["pairing_id"])
        assert denied["status"] == "denied"
        # No token is minted for a denied pairing.
        assert store.snapshot()["tokens"] == {}

    def test_is_valid_token_rejects_short_garbage(self, tmp_path):
        store = BrowserPairingStore(tmp_path / "state.json")
        assert store.is_valid_token("") is False
        assert store.is_valid_token("short") is False

    def test_is_valid_token_rejects_unknown(self, tmp_path):
        store = BrowserPairingStore(tmp_path / "state.json")
        assert store.is_valid_token("a" * 64) is False

    def test_state_persists_across_instances(self, tmp_path):
        path = tmp_path / "state.json"
        store = BrowserPairingStore(path)
        record = store.create_pairing(extension_id="ext-persist")
        granted = store.grant_pairing(record["pairing_id"])
        token = granted["token"]

        reloaded = BrowserPairingStore(path)
        assert reloaded.is_valid_token(token) is True
        assert reloaded.get_pairing(record["pairing_id"])["status"] == "approved"

    def test_prune_drops_expired_pending_pairings(self, tmp_path):
        clock = {"now": 1000.0}
        store = BrowserPairingStore(tmp_path / "state.json", now=lambda: clock["now"])
        record = store.create_pairing()
        clock["now"] += PAIRING_TTL_SECONDS + 5
        store._prune()
        assert record["pairing_id"] not in store.snapshot()["pairings"]

    def test_prune_keeps_approved_pairings(self, tmp_path):
        clock = {"now": 1000.0}
        store = BrowserPairingStore(tmp_path / "state.json", now=lambda: clock["now"])
        record = store.create_pairing()
        store.grant_pairing(record["pairing_id"])
        clock["now"] += PAIRING_TTL_SECONDS * 10
        store._prune()
        assert record["pairing_id"] in store.snapshot()["pairings"]

    def test_token_expires_after_max_age(self, tmp_path):
        clock = {"now": 1000.0}
        store = BrowserPairingStore(tmp_path / "state.json", now=lambda: clock["now"])
        record = store.create_pairing()
        token = store.grant_pairing(record["pairing_id"])["token"]
        clock["now"] += TOKEN_MAX_AGE_SECONDS + 1
        assert store.is_valid_token(token) is False

    def test_corrupt_state_file_falls_back_to_empty(self, tmp_path):
        path = tmp_path / "state.json"
        path.write_text("{not valid json", encoding="utf-8")
        store = BrowserPairingStore(path)
        assert store.snapshot() == {"pairings": {}, "tokens": {}}


# ---------------------------------------------------------------------------
# HTTP handler tests
# ---------------------------------------------------------------------------


def _make_adapter() -> APIServerAdapter:
    return APIServerAdapter(PlatformConfig(enabled=True))


def _create_pairing_app(adapter: APIServerAdapter) -> web.Application:
    app = web.Application()
    app.router.add_post("/api/browser-extension/pair/start", adapter._handle_browser_pair_start)
    app.router.add_get(
        "/api/browser-extension/pair/approve/{pairing_id}",
        adapter._handle_browser_pair_approve,
    )
    app.router.add_post(
        "/api/browser-extension/pair/grant/{pairing_id}",
        adapter._handle_browser_pair_grant,
    )
    app.router.add_post(
        "/api/browser-extension/pair/deny/{pairing_id}",
        adapter._handle_browser_pair_deny,
    )
    app.router.add_get(
        "/api/browser-extension/pair/status/{pairing_id}",
        adapter._handle_browser_pair_status,
    )
    return app


@pytest_asyncio.fixture
async def pair_client(tmp_path, monkeypatch):
    # Point the store's default state path at a temp dir so tests never touch
    # the real HERMES_HOME/state/browser_pairing.json.
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    adapter = _make_adapter()
    app = _create_pairing_app(adapter)
    app["pairing_adapter"] = adapter
    client = TestClient(TestServer(app))
    return client


def _from_page(client) -> dict:
    """Headers a browser sends when the gateway's own approval page submits a form."""
    return {"Origin": f"http://127.0.0.1:{client.port}", "Sec-Fetch-Site": "same-origin"}


class TestPairStart:

    @pytest.mark.asyncio
    async def test_start_mints_short_lived_pairing(self, pair_client):
        async with pair_client:
            resp = await pair_client.post(
                "/api/browser-extension/pair/start",
                json={"name": "Hermes Browser Extension", "extensionId": "ext-abc"},
            )
            assert resp.status == 200
            body = await resp.json()
            assert body["pairing_id"]
            assert body["approval_url"].endswith(f"/api/browser-extension/pair/approve/{body['pairing_id']}")
            assert body["ttl_seconds"] == PAIRING_TTL_SECONDS

    @pytest.mark.asyncio
    async def test_start_accepts_missing_body(self, pair_client):
        async with pair_client:
            resp = await pair_client.post("/api/browser-extension/pair/start")
            assert resp.status == 200
            body = await resp.json()
            assert body["pairing_id"]

    @pytest.mark.asyncio
    async def test_start_blocks_non_loopback(self, pair_client):
        async with pair_client:
            adapter = pair_client.server.app["pairing_adapter"]
            adapter._request_is_loopback = lambda request: False
            app2 = _create_pairing_app(adapter)
            app2["pairing_adapter"] = adapter
            async with TestClient(TestServer(app2)) as cli2:
                resp = await cli2.post(
                    "/api/browser-extension/pair/start",
                    json={"name": "Hermes Browser Extension"},
                )
                assert resp.status == 403


class TestPairApprovePage:

    @pytest.mark.asyncio
    async def test_approve_page_has_contract_buttons(self, pair_client):
        """The e2e pairing contract: #approveButton / #denyButton + form actions."""
        async with pair_client:
            start = await pair_client.post("/api/browser-extension/pair/start", json={"name": "Hermes Browser Extension"})
            pairing_id = (await start.json())["pairing_id"]
            resp = await pair_client.get(f"/api/browser-extension/pair/approve/{pairing_id}")
            assert resp.status == 200
            html = await resp.text()
            assert "Approve connection" in html
            assert 'id="approveButton"' in html
            assert 'id="denyButton"' in html
            assert f'action="/api/browser-extension/pair/grant/{pairing_id}"' in html
            assert f'action="/api/browser-extension/pair/deny/{pairing_id}"' in html
            # Branded split layout markers.
            assert 'class="frame"' in html
            assert 'class="rail"' in html
            assert 'class="pane"' in html

    def test_pairing_page_assets_all_ship(self, tmp_path, monkeypatch):
        """Every asset the page asks for ships in gateway/assets/browser_pairing/.

        A missing asset only degrades the page (fallback fonts, no artwork),
        so a renamed or dropped file would otherwise go unnoticed.
        """
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        adapter = _make_adapter()
        load = adapter._browser_pairing_asset
        requested = []

        def record(name):
            requested.append(name)
            return load(name)

        adapter._browser_pairing_asset = record
        adapter._pairing_page("Approve connection", "", rail_rows=[("STATUS", "PENDING APPROVAL")])
        assert requested
        assert [name for name in requested if not load(name)] == []

    @pytest.mark.asyncio
    async def test_approve_unknown_pairing_404(self, pair_client):
        async with pair_client:
            resp = await pair_client.get("/api/browser-extension/pair/approve/nope")
            assert resp.status == 404

    @pytest.mark.asyncio
    async def test_approve_already_approved_shows_state(self, pair_client):
        async with pair_client:
            start = await pair_client.post("/api/browser-extension/pair/start")
            pairing_id = (await start.json())["pairing_id"]
            await pair_client.post(f"/api/browser-extension/pair/grant/{pairing_id}", headers=_from_page(pair_client))
            resp = await pair_client.get(f"/api/browser-extension/pair/approve/{pairing_id}")
            assert resp.status == 200
            assert "Already approved" in await resp.text()


class TestPairGrantDenyStatus:

    @pytest.mark.asyncio
    async def test_grant_then_status_returns_token(self, pair_client):
        async with pair_client:
            start = await pair_client.post("/api/browser-extension/pair/start")
            pairing_id = (await start.json())["pairing_id"]

            status_before = await pair_client.get(f"/api/browser-extension/pair/status/{pairing_id}")
            assert (await status_before.json())["status"] == "pending"

            grant = await pair_client.post(f"/api/browser-extension/pair/grant/{pairing_id}", headers=_from_page(pair_client))
            assert grant.status == 200
            assert "Approved" in await grant.text()

            status_after = await pair_client.get(f"/api/browser-extension/pair/status/{pairing_id}")
            body = await status_after.json()
            assert body["status"] == "approved"
            assert body.get("token")

    @pytest.mark.asyncio
    async def test_deny_then_status_410(self, pair_client):
        async with pair_client:
            start = await pair_client.post("/api/browser-extension/pair/start")
            pairing_id = (await start.json())["pairing_id"]

            deny = await pair_client.post(f"/api/browser-extension/pair/deny/{pairing_id}", headers=_from_page(pair_client))
            assert deny.status == 200
            assert "Denied" in await deny.text()

            status = await pair_client.get(f"/api/browser-extension/pair/status/{pairing_id}")
            assert status.status == 410
            assert (await status.json())["error"]["code"] == "pairing_denied"

    @pytest.mark.asyncio
    async def test_status_unknown_404(self, pair_client):
        async with pair_client:
            resp = await pair_client.get("/api/browser-extension/pair/status/nope")
            assert resp.status == 404

    @pytest.mark.asyncio
    async def test_grant_unknown_410(self, pair_client):
        async with pair_client:
            resp = await pair_client.post("/api/browser-extension/pair/grant/nope", headers=_from_page(pair_client))
            assert resp.status == 410


class TestScopedTokenAuth:

    @pytest.mark.asyncio
    async def test_paired_token_authenticates_default_profile(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        adapter = _make_adapter()
        store = adapter._browser_pairing
        record = store.create_pairing()
        token = store.grant_pairing(record["pairing_id"])["token"]

        mock_request = MagicMock()
        mock_request.headers = {"Authorization": f"Bearer {token}"}
        # _check_auth consults the profile ContextVar, default "" == default profile.
        assert adapter._check_auth(mock_request) is None

    @pytest.mark.asyncio
    async def test_paired_token_authenticates_named_profile_only_on_loopback(self, tmp_path):
        from gateway.platforms.api_server import _api_request_profile

        adapter = APIServerAdapter(PlatformConfig(enabled=True))
        adapter._browser_pairing = BrowserPairingStore(tmp_path / "state.json")
        record = adapter._browser_pairing.create_pairing()
        token = adapter._browser_pairing.grant_pairing(record["pairing_id"])["token"]
        profile_token = _api_request_profile.set("worker")
        class Transport:
            def __init__(self, host):
                self.host = host

            def get_extra_info(self, name):
                return (self.host, 0) if name == "peername" else None

        try:
            local_request = MagicMock()
            local_request.headers = {"Authorization": f"Bearer {token}"}
            local_request.transport = Transport("127.0.0.1")
            assert adapter._check_auth(local_request) is None

            remote_request = MagicMock()
            remote_request.headers = {"Authorization": f"Bearer {token}"}
            remote_request.transport = Transport("10.0.0.8")
            rejected = adapter._check_auth(remote_request)
            assert rejected is not None
            assert rejected.status == 401
        finally:
            _api_request_profile.reset(profile_token)

    @pytest.mark.asyncio
    async def test_full_flow_http_token_works_on_models(self, pair_client):
        """Mint a token over HTTP, then use it to authenticate /v1/models."""
        async with pair_client:
            start = await pair_client.post("/api/browser-extension/pair/start")
            pairing_id = (await start.json())["pairing_id"]
            await pair_client.post(f"/api/browser-extension/pair/grant/{pairing_id}", headers=_from_page(pair_client))
            status = await pair_client.get(f"/api/browser-extension/pair/status/{pairing_id}")
            token = (await status.json())["token"]

            # The pairing app does not register /v1/models; instead assert the
            # auth hook accepts the token on a request with that header.
            adapter = pair_client.server.app["pairing_adapter"]
            mock_request = MagicMock()
            mock_request.headers = {"Authorization": f"Bearer {token}"}
            assert adapter._check_auth(mock_request) is None


# ---------------------------------------------------------------------------
# In-panel approval: frameable pages and the grant/deny origin check
# ---------------------------------------------------------------------------

EXT_CHROME = "chrome-extension://abcdefghijklmnopabcdefghijklmnop"
EXT_FIREFOX = "moz-extension://6f1c2a52-9d1c-4e2a-8f7e-0a4a5b3c9d10"


class TestExtensionOriginStore:

    @pytest.mark.parametrize(
        "origin",
        [EXT_CHROME, EXT_FIREFOX, "safari-web-extension://6F1C2A52-9D1C-4E2A-8F7E-0A4A5B3C9D10"],
    )
    def test_store_keeps_an_exact_extension_origin(self, tmp_path, origin):
        store = BrowserPairingStore(tmp_path / "state.json")
        record = store.create_pairing(extension_origin=origin)
        assert record["extension_origin"] == origin
        assert store.get_pairing(record["pairing_id"])["extension_origin"] == origin

    @pytest.mark.parametrize(
        "origin",
        [
            "",
            "*",
            "http://127.0.0.1:8642",
            "https://example.com",
            "chrome-extension://*",
            "chrome-extension://abcdefghijklmnopabcdefghijklmnop/",
            "chrome-extension://abcdefghijklmnopabcdefghijklmnop/page.html",
            "chrome-extension://abcdefghijklmnopabcdefghijklmnop:80",
            "chrome-extension://abcdefghijklmnopabcdefghijklmnop https://evil.example",
            "chrome-extension://abcdefghijklmnopabcdefghijklmnop; script-src *",
            "chrome-extension://abcdefghijklmnopabcdefghijklmnop\r\nX-Evil: 1",
            "chrome-extension://short",
            "null",
        ],
    )
    def test_store_refuses_anything_but_one_extension_origin(self, tmp_path, origin):
        store = BrowserPairingStore(tmp_path / "state.json")
        assert store.create_pairing(extension_origin=origin)["extension_origin"] == ""

    def test_extension_origin_survives_a_restart(self, tmp_path):
        path = tmp_path / "state.json"
        record = BrowserPairingStore(path).create_pairing(extension_origin=EXT_FIREFOX)
        again = BrowserPairingStore(path).get_pairing(record["pairing_id"])
        assert again["extension_origin"] == EXT_FIREFOX


def _framing_app(adapter: APIServerAdapter) -> web.Application:
    """The pairing routes behind the gateway's real CORS and security-header middleware."""
    from gateway.platforms.api_server import cors_middleware, security_headers_middleware

    app = web.Application(middlewares=[cors_middleware, security_headers_middleware])
    app["api_server_adapter"] = adapter
    for method, path, handler in (
        ("POST", "start", adapter._handle_browser_pair_start),
        ("GET", "approve", adapter._handle_browser_pair_approve),
        ("POST", "grant", adapter._handle_browser_pair_grant),
        ("POST", "deny", adapter._handle_browser_pair_deny),
        ("GET", "status", adapter._handle_browser_pair_status),
    ):
        suffix = "" if path == "start" else "/{pairing_id}"
        app.router.add_route(method, f"/api/browser-extension/pair/{path}{suffix}", handler)
    return app


@pytest_asyncio.fixture
async def framing_client(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    adapter = _make_adapter()
    client = TestClient(TestServer(_framing_app(adapter)))
    client.adapter = adapter
    return client


async def _start(client, **body) -> dict:
    resp = await client.post("/api/browser-extension/pair/start", json=body)
    assert resp.status == 200
    return await resp.json()


def _frame_ancestors(resp) -> str:
    csp = resp.headers["Content-Security-Policy"]
    return [d for d in csp.split(";") if "frame-ancestors" in d][0].strip()


class TestApproveDenyOriginCheck:

    @staticmethod
    async def _pending(client, **body) -> str:
        return (await _start(client, **body))["pairing_id"]

    @staticmethod
    async def _status(client, pairing_id) -> str:
        resp = await client.get(f"/api/browser-extension/pair/status/{pairing_id}")
        return (await resp.json()).get("status") or "denied"

    @pytest.mark.asyncio
    @pytest.mark.parametrize("action", ["grant", "deny"])
    async def test_a_post_with_no_origin_is_refused(self, framing_client, action):
        async with framing_client:
            pairing_id = await self._pending(framing_client)
            resp = await framing_client.post(f"/api/browser-extension/pair/{action}/{pairing_id}")
            assert resp.status == 403
            assert await self._status(framing_client, pairing_id) == "pending"

    @pytest.mark.asyncio
    @pytest.mark.parametrize("action", ["grant", "deny"])
    @pytest.mark.parametrize(
        "origin", [EXT_CHROME, EXT_FIREFOX, "null", "https://example.com", "http://localhost:9"]
    )
    async def test_a_post_from_another_origin_is_refused(self, framing_client, action, origin):
        """Any local extension or page, with or without a CORS allow-list entry."""
        async with framing_client:
            framing_client.adapter._cors_origins = (EXT_CHROME, EXT_FIREFOX, "https://example.com")
            pairing_id = await self._pending(framing_client, extensionOrigin=EXT_CHROME)
            resp = await framing_client.post(
                f"/api/browser-extension/pair/{action}/{pairing_id}", headers={"Origin": origin}
            )
            assert resp.status == 403
            assert await self._status(framing_client, pairing_id) == "pending"

    @pytest.mark.asyncio
    async def test_a_cross_site_post_with_the_gateway_origin_is_refused(self, framing_client):
        async with framing_client:
            pairing_id = await self._pending(framing_client)
            for site in ("cross-site", "same-site", "none"):
                headers = {**_from_page(framing_client), "Sec-Fetch-Site": site}
                resp = await framing_client.post(
                    f"/api/browser-extension/pair/grant/{pairing_id}", headers=headers
                )
                assert resp.status == 403, site
            assert await self._status(framing_client, pairing_id) == "pending"

    @pytest.mark.asyncio
    async def test_a_rebound_hostname_is_not_the_gateways_own_page(self, framing_client):
        async with framing_client:
            pairing_id = await self._pending(framing_client)
            evil = f"evil.example:{framing_client.port}"
            resp = await framing_client.post(
                f"/api/browser-extension/pair/grant/{pairing_id}",
                headers={"Host": evil, "Origin": f"http://{evil}", "Sec-Fetch-Site": "same-origin"},
            )
            assert resp.status == 403
            assert await self._status(framing_client, pairing_id) == "pending"

    @pytest.mark.asyncio
    @pytest.mark.parametrize("host", ["127.0.0.1", "localhost"])
    async def test_the_gateways_own_page_can_approve_and_deny(self, framing_client, host):
        async with framing_client:
            authority = f"{host}:{framing_client.port}"
            headers = {"Origin": f"http://{authority}", "Host": authority}
            granted = await self._pending(framing_client)
            denied = await self._pending(framing_client)
            grant = await framing_client.post(f"/api/browser-extension/pair/grant/{granted}", headers=headers)
            deny = await framing_client.post(f"/api/browser-extension/pair/deny/{denied}", headers=headers)
            assert (grant.status, deny.status) == (200, 200)
            assert await self._status(framing_client, granted) == "approved"

    @pytest.mark.asyncio
    async def test_the_own_origin_exemption_covers_only_grant_and_deny(self, framing_client):
        from aiohttp.test_utils import make_mocked_request

        adapter = framing_client.adapter
        own = {"Origin": "http://127.0.0.1:8642", "Host": "127.0.0.1:8642"}

        def post(path):
            return make_mocked_request("POST", path, headers=own)

        assert adapter._is_pairing_form_post(post("/api/browser-extension/pair/grant/abc")) is True
        assert adapter._is_pairing_form_post(post("/p/work/api/browser-extension/pair/deny/abc")) is True
        assert adapter._is_pairing_form_post(post("/api/browser-extension/pair/start")) is False
        assert adapter._is_pairing_form_post(post("/v1/chat/completions")) is False
        get = make_mocked_request("GET", "/api/browser-extension/pair/grant/abc", headers=own)
        assert adapter._is_pairing_form_post(get) is False

    @pytest.mark.asyncio
    async def test_start_is_still_loopback_gated(self, framing_client):
        async with framing_client:
            framing_client.adapter._request_is_loopback = lambda request: False
            resp = await framing_client.post(
                "/api/browser-extension/pair/start", json={"extensionOrigin": EXT_CHROME}
            )
            assert resp.status == 403


class TestExtensionOriginStart:

    @pytest.mark.asyncio
    async def test_start_says_whether_the_page_can_be_embedded(self, framing_client):
        async with framing_client:
            plain = await _start(framing_client, name="Hermes Browser Extension")
            named = await _start(framing_client, extensionOrigin=EXT_CHROME)
            junk = await _start(framing_client, extensionOrigin="chrome-extension://*")
            assert plain["embeddable"] is False
            assert named["embeddable"] is True
            assert junk["embeddable"] is False

    @pytest.mark.asyncio
    async def test_start_refuses_an_origin_other_than_the_caller(self, framing_client):
        async with framing_client:
            framing_client.adapter._cors_origins = (EXT_CHROME, EXT_FIREFOX)
            for sent, declared, embeddable in (
                (EXT_CHROME, EXT_CHROME, True),
                (EXT_CHROME, EXT_FIREFOX, False),
                (EXT_CHROME, "", False),
            ):
                resp = await framing_client.post(
                    "/api/browser-extension/pair/start",
                    json={"extensionOrigin": declared},
                    headers={"Origin": sent},
                )
                assert (await resp.json())["embeddable"] is embeddable, (sent, declared)


class TestFramingHeaders:

    @pytest.mark.asyncio
    async def test_approval_page_is_framed_by_the_asking_extension_only(self, framing_client):
        async with framing_client:
            started = await _start(framing_client, extensionOrigin=EXT_FIREFOX)
            resp = await framing_client.get(f"/api/browser-extension/pair/approve/{started['pairing_id']}")
            assert resp.status == 200
            assert _frame_ancestors(resp) == f"frame-ancestors {EXT_FIREFOX}"
            assert "X-Frame-Options" not in resp.headers
            # Nothing else about the page's CSP is relaxed.
            csp = resp.headers["Content-Security-Policy"]
            assert csp.startswith("default-src 'none'")
            assert "script-src" not in csp
            assert "*" not in csp

    @pytest.mark.asyncio
    async def test_approval_page_without_an_origin_cannot_be_framed(self, framing_client):
        async with framing_client:
            started = await _start(framing_client, name="Hermes Browser Extension")
            resp = await framing_client.get(f"/api/browser-extension/pair/approve/{started['pairing_id']}")
            assert _frame_ancestors(resp) == "frame-ancestors 'none'"
            assert resp.headers["X-Frame-Options"] == "DENY"

    @pytest.mark.asyncio
    async def test_the_approval_page_posts_with_a_real_origin(self, framing_client):
        async with framing_client:
            for body in ({}, {"extensionOrigin": EXT_CHROME}):
                started = await _start(framing_client, **body)
                resp = await framing_client.get(f"/api/browser-extension/pair/approve/{started['pairing_id']}")
                assert resp.headers["Referrer-Policy"] == "same-origin"

    @pytest.mark.asyncio
    async def test_approved_and_denied_pages_are_framed_by_the_same_origin(self, framing_client):
        async with framing_client:
            headers = _from_page(framing_client)
            for action in ("grant", "deny"):
                started = await _start(framing_client, extensionOrigin=EXT_CHROME)
                resp = await framing_client.post(
                    f"/api/browser-extension/pair/{action}/{started['pairing_id']}", headers=headers
                )
                assert resp.status == 200
                assert _frame_ancestors(resp) == f"frame-ancestors {EXT_CHROME}", action
                assert "X-Frame-Options" not in resp.headers, action

    @pytest.mark.asyncio
    async def test_approved_and_denied_pages_stay_unframed_without_an_origin(self, framing_client):
        async with framing_client:
            headers = _from_page(framing_client)
            for action in ("grant", "deny"):
                started = await _start(framing_client)
                resp = await framing_client.post(
                    f"/api/browser-extension/pair/{action}/{started['pairing_id']}", headers=headers
                )
                assert _frame_ancestors(resp) == "frame-ancestors 'none'", action
                assert resp.headers["X-Frame-Options"] == "DENY", action

    @pytest.mark.asyncio
    async def test_already_approved_page_is_framed_too(self, framing_client):
        async with framing_client:
            started = await _start(framing_client, extensionOrigin=EXT_CHROME)
            pairing_id = started["pairing_id"]
            await framing_client.post(
                f"/api/browser-extension/pair/grant/{pairing_id}", headers=_from_page(framing_client)
            )
            resp = await framing_client.get(f"/api/browser-extension/pair/approve/{pairing_id}")
            assert "Already approved" in await resp.text()
            assert _frame_ancestors(resp) == f"frame-ancestors {EXT_CHROME}"
            assert "X-Frame-Options" not in resp.headers

    @pytest.mark.asyncio
    async def test_everything_else_keeps_x_frame_options_deny(self, framing_client):
        async with framing_client:
            started = await _start(framing_client, extensionOrigin=EXT_CHROME)
            for resp in (
                await framing_client.get("/api/browser-extension/pair/approve/nope"),
                await framing_client.get(f"/api/browser-extension/pair/status/{started['pairing_id']}"),
            ):
                assert resp.headers["X-Frame-Options"] == "DENY"
                assert "frame-ancestors 'none'" in resp.headers["Content-Security-Policy"]


class TestEmbeddedLayout:
    """The compact page the extension frames in its panel (``?embed=1``)."""

    @staticmethod
    async def _approve_html(client, pairing_id, query=""):
        resp = await client.get(f"/api/browser-extension/pair/approve/{pairing_id}{query}")
        assert resp.status == 200
        return await resp.text()

    @pytest.mark.asyncio
    async def test_a_framed_approval_page_asks_for_the_compact_layout(self, framing_client):
        async with framing_client:
            pairing_id = (await _start(framing_client, extensionOrigin=EXT_CHROME))["pairing_id"]
            html = await self._approve_html(framing_client, pairing_id, "?embed=1")
            assert '<body class="embed">' in html
            assert f'action="/api/browser-extension/pair/grant/{pairing_id}?embed=1"' in html
            assert f'action="/api/browser-extension/pair/deny/{pairing_id}?embed=1"' in html
            assert 'id="approveButton"' in html and 'id="denyButton"' in html

    @pytest.mark.asyncio
    async def test_the_full_page_stays_the_default(self, framing_client):
        async with framing_client:
            pairing_id = (await _start(framing_client, extensionOrigin=EXT_CHROME))["pairing_id"]
            html = await self._approve_html(framing_client, pairing_id)
            assert "<body>" in html and 'class="embed"' not in html
            assert "?embed=1" not in html

    @pytest.mark.asyncio
    async def test_a_page_nobody_may_frame_ignores_the_flag(self, framing_client):
        async with framing_client:
            pairing_id = (await _start(framing_client))["pairing_id"]
            html = await self._approve_html(framing_client, pairing_id, "?embed=1")
            assert 'class="embed"' not in html
            assert "?embed=1" not in html

    @pytest.mark.asyncio
    @pytest.mark.parametrize("action", ["grant", "deny"])
    async def test_the_result_pages_keep_the_compact_layout(self, framing_client, action):
        async with framing_client:
            pairing_id = (await _start(framing_client, extensionOrigin=EXT_CHROME))["pairing_id"]
            resp = await framing_client.post(
                f"/api/browser-extension/pair/{action}/{pairing_id}?embed=1",
                headers=_from_page(framing_client),
            )
            assert resp.status == 200
            assert '<body class="embed">' in await resp.text()

    @pytest.mark.asyncio
    async def test_every_page_scrolls_with_the_portals_own_scrollbar(self, framing_client):
        async with framing_client:
            for body in ({}, {"extensionOrigin": EXT_CHROME}):
                pairing_id = (await _start(framing_client, **body))["pairing_id"]
                for query in ("", "?embed=1"):
                    html = await self._approve_html(framing_client, pairing_id, query)
                    assert "::-webkit-scrollbar-thumb{background:rgba(5,5,232,.55)" in html
                    assert "scrollbar-width:thin" in html
