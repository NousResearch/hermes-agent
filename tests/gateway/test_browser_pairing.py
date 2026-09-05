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
            await pair_client.post(f"/api/browser-extension/pair/grant/{pairing_id}")
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

            grant = await pair_client.post(f"/api/browser-extension/pair/grant/{pairing_id}")
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

            deny = await pair_client.post(f"/api/browser-extension/pair/deny/{pairing_id}")
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
            resp = await pair_client.post("/api/browser-extension/pair/grant/nope")
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
            await pair_client.post(f"/api/browser-extension/pair/grant/{pairing_id}")
            status = await pair_client.get(f"/api/browser-extension/pair/status/{pairing_id}")
            token = (await status.json())["token"]

            # The pairing app does not register /v1/models; instead assert the
            # auth hook accepts the token on a request with that header.
            adapter = pair_client.server.app["pairing_adapter"]
            mock_request = MagicMock()
            mock_request.headers = {"Authorization": f"Bearer {token}"}
            assert adapter._check_auth(mock_request) is None
