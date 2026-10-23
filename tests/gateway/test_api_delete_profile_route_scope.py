"""Deleting a profile-local row cannot remove a foreign colliding routing ID."""

import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from tests.gateway.test_secondary_route_fence_scope import _served, _entry, _durable, _AUTH


@pytest.mark.asyncio
async def test_api_delete_preserves_a_foreign_route_with_the_same_session_id(tmp_path, monkeypatch):
    from gateway.run import _profile_runtime_scope

    async with _served(tmp_path, monkeypatch) as served:
        sid = "colliding-local-id"
        own = "agent:alpha:telegram:own"
        foreign = "agent:main:telegram:foreign"
        for db in (served.alpha.db, served.launch.db):
            db.create_session(sid, source="telegram")
        served.store._entries[own] = _entry(own, sid)
        served.store._entries[foreign] = _entry(foreign, sid)
        served.store._save()
        assert {own, foreign} <= set(_durable(served.store))
        app = web.Application()
        app.router.add_delete('/api/sessions/{session_id}', served.adapter._handle_delete_session)
        with _profile_runtime_scope(served.home, {"API_SERVER_KEY": "route-fence-key"}):
            async with TestClient(TestServer(app)) as client:
                response = await client.delete(f'/api/sessions/{sid}', headers=_AUTH)
                assert response.status == 200, await response.text()
                assert (await response.json())["deleted"] is True
        assert served.alpha.db.get_session(sid) is None
        assert served.launch.db.get_session(sid) is not None
        assert own not in served.store._entries
        assert foreign in served.store._entries
        assert own not in _durable(served.store)
        assert foreign in _durable(served.store)
