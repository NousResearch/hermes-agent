"""Authenticated Mattermost session-binding API contracts."""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest
import pytest_asyncio
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from plugins.platforms.mattermost.session_binding_api import MattermostSessionBindingAPI
from plugins.platforms.mattermost.session_bindings import MattermostSessionBindingStore


class _FakeAPIServerAdapter:
    def __init__(self) -> None:
        self.sessions = {"session-1": {"id": "session-1"}}

    def _check_auth(self, request):
        if request.headers.get("Authorization") == "Bearer test-key":
            return None
        return web.json_response({"error": {"code": "gateway_auth_failed"}}, status=401)

    async def _get_existing_session_or_404(self, session_id):
        session = self.sessions.get(session_id)
        if session is None:
            return None, web.json_response({"error": {"code": "session_not_found"}}, status=404)
        return session, None

    async def _read_json_body(self, request):
        try:
            body = await request.json()
        except Exception:
            return {}, web.json_response({"error": {"code": "invalid_json"}}, status=400)
        return body, None


@pytest_asyncio.fixture
async def client(tmp_path):
    app = web.Application()
    api = MattermostSessionBindingAPI(
        _FakeAPIServerAdapter(),
        store_factory=lambda: MattermostSessionBindingStore(tmp_path / "bindings.db"),
    )
    api.register_routes(app)
    test_client = TestClient(TestServer(app))
    await test_client.start_server()
    try:
        yield test_client
    finally:
        await test_client.close()


def _auth():
    return {"Authorization": "Bearer test-key"}


@pytest.mark.asyncio
async def test_routes_reject_missing_api_server_bearer(client):
    response = await client.get("/api/plugins/mattermost/v1/capabilities")
    assert response.status == 401


@pytest.mark.asyncio
async def test_binding_crud_and_reverse_resolution(client):
    response = await client.put(
        "/api/plugins/mattermost/v1/session-bindings/session-1",
        headers=_auth(),
        json={"channel_id": "channel1", "root_post_id": "root1"},
    )
    assert response.status == 200
    assert (await response.json())["binding"]["session_id"] == "session-1"

    response = await client.get(
        "/api/plugins/mattermost/v1/session-bindings/resolve",
        headers=_auth(),
        params={"channel_id": "channel1", "root_post_id": "root1"},
    )
    assert response.status == 200
    assert (await response.json())["binding"]["session_id"] == "session-1"

    response = await client.get("/api/plugins/mattermost/v1/session-bindings", headers=_auth())
    assert response.status == 200
    assert [item["session_id"] for item in (await response.json())["data"]] == ["session-1"]

    response = await client.delete(
        "/api/plugins/mattermost/v1/session-bindings/session-1", headers=_auth()
    )
    assert response.status == 200
    assert (await response.json())["deleted"] is True


@pytest.mark.asyncio
async def test_binding_requires_existing_hermes_session(client):
    response = await client.put(
        "/api/plugins/mattermost/v1/session-bindings/missing",
        headers=_auth(),
        json={"channel_id": "channel1", "root_post_id": "root1"},
    )
    assert response.status == 404
    assert (await response.json())["error"]["code"] == "session_not_found"


@pytest.mark.asyncio
async def test_connected_target_normalizer_is_used(tmp_path):
    normalizer = AsyncMock(return_value=("canonical-channel", "canonical-root"))
    app = web.Application()
    api = MattermostSessionBindingAPI(
        _FakeAPIServerAdapter(),
        target_normalizer=normalizer,
        store_factory=lambda: MattermostSessionBindingStore(tmp_path / "bindings.db"),
    )
    api.register_routes(app)
    async with TestClient(TestServer(app)) as test_client:
        response = await test_client.put(
            "/api/plugins/mattermost/v1/session-bindings/session-1",
            headers=_auth(),
            json={"channel_id": "requested-channel", "root_post_id": "reply-id"},
        )
        assert response.status == 200
        payload = await response.json()
    normalizer.assert_awaited_once_with("requested-channel", "reply-id")
    assert payload["binding"]["root_post_id"] == "canonical-root"
