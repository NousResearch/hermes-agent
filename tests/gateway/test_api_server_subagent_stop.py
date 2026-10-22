"""HTTP cooperative cancellation is scoped to an API session and its profile DB."""
from types import SimpleNamespace

import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from gateway.config import PlatformConfig
from gateway.platforms.api_server import APIServerAdapter
from hermes_state import SessionDB
from tools import delegate_tool_registry as registry


class Child:
    def __init__(self):
        self.interrupts = 0

    def hard_interrupt(self, message=None, **kwargs):
        self.interrupts += 1


def test_real_child_registration_pins_profile_db_and_retains_finished_ownership(tmp_path):
    from tools.delegate_tool_child_run import _register_child

    db = SessionDB(tmp_path / 'state.db')
    parent = SimpleNamespace(session_id='owned', _session_db=db)
    child = Child()
    child._subagent_id = 'http-test-registered'
    try:
        _register_child(child, parent, 'fixture', owner_session_id=None, owner_transport=None, owner_session_record=None)
        assert registry.request_session_subagent_stop(child._subagent_id, 'owned', db) == 'stopping'
        registry._unregister_subagent(child._subagent_id)
        assert registry.request_session_subagent_stop(child._subagent_id, 'owned', db) == 'finished'
        assert child.interrupts == 1
    finally:
        registry._unregister_subagent(child._subagent_id)
        registry._recent_subagents.pop(child._subagent_id, None)
        db.close()


@pytest.mark.asyncio
async def test_owned_http_stop_interrupts_only_one_child_and_duplicate_is_idempotent(tmp_path):
    db = SessionDB(tmp_path / "state.db")
    db.create_session("owned", "api_server")
    adapter = APIServerAdapter(PlatformConfig(enabled=True, extra={"key": "fixture-api-key"}))
    adapter._session_db = db
    child, sibling = Child(), Child()
    records = {
        "http-test-child": {"subagent_id": "http-test-child", "agent": child, "owner_agent_session_id": "owned", "owner_db_path": str(db.db_path)},
        "http-test-sibling": {"subagent_id": "http-test-sibling", "agent": sibling, "owner_agent_session_id": "owned", "owner_db_path": str(db.db_path)},
    }
    for record in records.values():
        registry._register_subagent(record)
    app = web.Application()
    for method, path, handler in adapter._http_route_table():
        if "subagents" in path:
            app.router.add_route(method, path, handler)
    try:
        async with TestClient(TestServer(app)) as client:
            url = "/api/sessions/owned/subagents/http-test-child/stop"
            headers = {"Authorization": "Bearer fixture-api-key"}
            assert (await client.post(url)).status == 401
            first = await client.post(url, headers=headers)
            assert first.status == 202
            assert (await first.json())["status"] == "stopping"
            assert (await client.post(url, headers=headers)).status == 202
            assert child.interrupts == 1
            assert sibling.interrupts == 0
    finally:
        for sid in records:
            registry._unregister_subagent(sid)
            registry._recent_subagents.pop(sid, None)
        db.close()


@pytest.mark.asyncio
@pytest.mark.parametrize('case, expected', [('session', 404), ('profile', 404), ('unknown', 404), ('finished', 409), ('keyless', 403)])
async def test_http_stop_rejects_foreign_missing_finished_or_unauthenticated_scope(tmp_path, case, expected):
    db = SessionDB(tmp_path / 'state.db')
    db.create_session('owned', 'api_server')
    db.create_session('foreign', 'api_server')
    adapter = APIServerAdapter(PlatformConfig(enabled=True, extra={'key': '' if case == 'keyless' else 'fixture-api-key'}))
    adapter._session_db = db
    child = Child()
    sid = 'http-denied-child'
    registry._register_subagent({'subagent_id': sid, 'agent': child, 'owner_agent_session_id': 'owned', 'owner_db_path': str(tmp_path / 'other/state.db') if case == 'profile' else str(db.db_path)})
    if case == 'finished':
        registry._unregister_subagent(sid)
    app = web.Application()
    for method, path, handler in adapter._http_route_table():
        if 'subagents' in path:
            app.router.add_route(method, path, handler)
    try:
        async with TestClient(TestServer(app)) as client:
            owner = 'foreign' if case == 'session' else 'owned'
            target = 'unknown' if case == 'unknown' else sid
            response = await client.post(f'/api/sessions/{owner}/subagents/{target}/stop', headers={'Authorization': 'Bearer fixture-api-key'})
            assert response.status == expected
            assert child.interrupts == 0
    finally:
        registry._unregister_subagent(sid)
        registry._recent_subagents.pop(sid, None)
        db.close()
