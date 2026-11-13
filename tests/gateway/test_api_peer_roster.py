"""Authenticated capabilities advertise the selected profile's RoomLink identity."""
from pathlib import Path

import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from gateway.config import PlatformConfig
from gateway.platforms.api_server import APIServerAdapter, _api_request_profile
from gateway.run import _profile_runtime_scope


@pytest.mark.asyncio
@pytest.mark.parametrize('selected_profile', ['secondary', None])
async def test_capability_bot_uses_authenticated_mirror_policy(tmp_path, monkeypatch, selected_profile):
    from agent import secret_scope
    root = tmp_path / 'state'
    home = root / 'profiles' / 'secondary'
    home.mkdir(parents=True)
    monkeypatch.setattr(Path, 'home', lambda: tmp_path)
    monkeypatch.setenv('HERMES_HOME', str(root))
    (root / 'config.yaml').write_text('agent:\n  max_turns: 3\n')
    (home / 'config.yaml').write_text('agent:\n  max_turns: 9\n')
    (home / '.env').write_text('API_SERVER_KEY=' + 's' * 32 + '\n')
    previous = secret_scope.is_multiplex_active()
    secret_scope.set_multiplex_active(True)
    adapter_key = 'launch-key' if selected_profile else 's' * 32
    adapter = APIServerAdapter(PlatformConfig(enabled=True, extra={'key': adapter_key}))

    @web.middleware
    async def mirror(request, handler):
        token = _api_request_profile.set(selected_profile)
        try:
            with _profile_runtime_scope(home):
                return await handler(request)
        finally:
            _api_request_profile.reset(token)

    app = web.Application(middlewares=[mirror])
    app.router.add_get('/v1/capabilities', adapter._handle_capabilities)
    try:
        async with TestClient(TestServer(app)) as client:
            refused = await client.get('/v1/capabilities', headers={'Authorization': 'Bearer launch-key'})
            assert refused.status == 401
            response = await client.get('/v1/capabilities', headers={'Authorization': 'Bearer ' + 's' * 32})
            assert response.status == 200
            bot = (await response.json())['bot']
            assert bot['profile'] == 'secondary'
            assert bot['room_link']['execution_policy']['target_profile'] == 'secondary'
            assert bot['room_link']['execution_policy']['max_iterations'] == 9
            assert bot['installation_id'] == bot['room_link']['installation_id']
            assert 's' * 32 not in str(bot)
    finally:
        secret_scope.set_multiplex_active(previous)
        adapter._run_idempotency_store.close()
