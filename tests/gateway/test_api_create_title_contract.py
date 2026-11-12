"""Create title validation preserves its HTTP contract before any state write."""
import pytest
from aiohttp.test_utils import TestClient, TestServer

from tests.gateway import test_session_api_ownership as ownership

stores = ownership.stores
snapshot = ownership.snapshot


@pytest.mark.asyncio
@pytest.mark.parametrize("length,expected_status", [(100, 201), (101, 400)])
async def test_create_title_length_boundary(stores, length, expected_status):
    local, sibling, app = stores
    before = snapshot(local), snapshot(sibling)
    async with TestClient(TestServer(app)) as client:
        response = await client.post("/api/sessions", json={"id": "created", "title": "x" * length})
        assert response.status == expected_status, await response.text()
        if expected_status == 400:
            assert (await response.json())["error"]["code"] == "invalid_title"
            assert (snapshot(local), snapshot(sibling)) == before
        else:
            assert local.get_session("created")["title"] == "x" * length
            assert snapshot(sibling) == before[1]
