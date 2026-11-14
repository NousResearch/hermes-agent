"""A connector replay of one admitted Discord interaction is one durable input."""
import asyncio
import base64
import json

import pytest

from gateway.config import GatewayConfig, Platform
from gateway.relay.admitted_authors import note_delivered
from gateway.run import GatewayRunner
from gateway.session import SessionSource
from gateway.session_authority import initialize_session_authority
from tests.gateway.relay.test_relay_native_recovery_and_passthrough import (
    _Connector, _rows, _stub_turns, _wired_relay, websockets,
)
from tests.waiting import async_wait_until


@pytest.mark.asyncio
async def test_replayed_component_commits_and_runs_once(tmp_path, monkeypatch, caplog):
    connector, turns = _Connector('discord'), []
    runner = GatewayRunner(GatewayConfig(sessions_dir=tmp_path / 'sessions'))
    _stub_turns(runner, monkeypatch, turns)
    authority = await initialize_session_authority(runner, profile_id='default', instance_id='component-replay')
    note_delivered(SessionSource(platform=Platform.DISCORD, chat_id='replay-chat', user_id='replay-user',
                                 delivered_via_upstream_relay=True))
    body = {'type': 3, 'id': 'same-interaction', 'channel_id': 'replay-chat', 'guild_id': 'replay-guild',
            'member': {'user': {'id': 'replay-user'}}, 'data': {'custom_id': 'choose-once'}}
    forward = {'platform': 'discord', 'botId': 'fixture-bot', 'method': 'POST', 'path': '/interactions',
               'headers': [], 'bodyB64': base64.b64encode(json.dumps(body).encode()).decode()}
    try:
        async with websockets.serve(connector, '127.0.0.1', 0) as server:
            url = f'ws://127.0.0.1:{server.sockets[0].getsockname()[1]}/relay'
            async with _wired_relay(runner, connector, url) as (adapter, ws):
                handled = asyncio.Queue()
                original = adapter.handle_message
                async def handle(event):
                    await original(event)
                    await handled.put(None)
                monkeypatch.setattr(adapter, 'handle_message', handle)
                for _ in range(2):
                    await ws.send(json.dumps({'type': 'passthrough_forward', 'forward': forward}) + '\n')
                    await asyncio.wait_for(handled.get(), 10)
                    await async_wait_until(lambda: _rows(authority) and all(
                        row['status'] == 'terminal' for row in _rows(authority)), timeout=20,
                        what='interaction admission settled')
                assert len(_rows(authority)) == 1, 'a replay created another durable input'
                assert turns == ['choose-once']
                assert 'admission_conflict' not in caplog.text
    finally:
        authority.db.close()
