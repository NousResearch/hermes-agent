"""Relay admissions queued before a restart run after it, answer with full egress routing, and a
forwarded Discord component is admitted like a relayed text message.

After a restart the queued native admissions are bound to adapters from the persisted session
origin, which never carries the wire-invisible relay marker; the Relay (registered under
``Platform.RELAY``) must still be found for the logical platform it fronts, and its per-chat egress
caches (filled only on live inbound) must be warm before the recovered turn replies. A Discord
interaction arrives over the same authenticated socket as ``passthrough_forward``; native admission
requires the relay callback provenance on that leg too.
"""
import asyncio
import base64
import json
from contextlib import asynccontextmanager
from pathlib import Path

import pytest

from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.run import GatewayRunner
from gateway.session_results import execution_result
from tests.waiting import async_wait_until

websockets = pytest.importorskip('websockets')


class _Connector:
    """A fake connector: hands out descriptors, records frames, acks every outbound."""

    def __init__(self, platform):
        from gateway.relay.descriptor import CapabilityDescriptor
        self.descriptor = CapabilityDescriptor(1, platform, 'Fixture', 4096, False, True, False, 'plain', 'chars')
        self.outgoing, self.ready = [], asyncio.Queue()

    async def __call__(self, ws):
        await ws.recv()  # hello
        await ws.send(json.dumps({'type': 'descriptor', 'descriptor': json.loads(self.descriptor.to_json())}) + '\n')
        await self.ready.put(ws)
        async for line in ws:
            frame = json.loads(line)
            self.outgoing.append(frame)
            if frame['type'] == 'outbound':
                await ws.send(json.dumps({'type': 'outbound_result', 'requestId': frame['requestId'],
                                          'result': {'success': True, 'message_id': 'sent'}}) + '\n')
            elif frame['type'] == 'going_idle':
                await ws.send(json.dumps({'type': 'going_idle_ack'}) + '\n')

    def replies(self, chat_id):
        return [f['action'] for f in self.outgoing if f.get('type') == 'outbound'
                and f['action'].get('op') in ('send', 'edit') and f['action'].get('chat_id') == chat_id]


@asynccontextmanager
async def _wired_relay(runner, connector, url):
    from gateway.relay.adapter import RelayAdapter
    from gateway.relay.ws_transport import WebSocketRelayTransport
    transport = WebSocketRelayTransport(url, connector.descriptor.platform, 'fixture-bot',
                                        gateway_id='fixture-gateway', upgrade_secret='fixture-secret',
                                        reconnect=False)
    adapter = RelayAdapter(PlatformConfig(enabled=True), connector.descriptor, transport)
    adapter.gateway_runner = runner
    runner.adapters[Platform.RELAY] = adapter
    runner._wire_adapter_handlers(adapter)
    try:
        assert await adapter.connect()
        yield adapter, await asyncio.wait_for(connector.ready.get(), 5)
    finally:
        await adapter.disconnect()


def _stub_turns(runner, monkeypatch, turns):
    async def turn(event, *_args, **_kwargs):
        turns.append(event.text)
        execution_result.get()['result'] = {'final_response': 'reply ' + event.text, 'messages': []}
        return 'reply ' + event.text

    monkeypatch.setattr(runner, '_handle_message_with_agent', turn)


def _rows(authority):
    from hermes_state_runtime import list_session_admissions
    return [r for sid in list(authority.sessions)
            for r in list_session_admissions(authority.db, session_id=sid, pending_only=False)]


@pytest.mark.asyncio
async def test_relay_admission_queued_across_a_restart_runs_and_replies_with_egress_routing(tmp_path, monkeypatch):
    from gateway.run_runtime import recover_gateway_native_sessions
    from gateway.session_authority import SessionAuthority, initialize_session_authority
    from hermes_state_runtime import list_session_admissions

    connector = _Connector('discord')
    raw = {'text': 'queued before restart', 'message_id': 'm-1', 'source': {
        'platform': 'discord', 'chat_id': 'chan-1', 'chat_type': 'group', 'user_id': 'owner-1',
        'scope_id': 'guild-1'}}
    async with websockets.serve(connector, '127.0.0.1', 0) as server:
        url = f'ws://127.0.0.1:{server.sockets[0].getsockname()[1]}/relay'

        # Boot 1: admit and ack the frame, but never start it (the process stops first).
        runner = GatewayRunner(GatewayConfig(sessions_dir=tmp_path / 'sessions'))
        turns = []
        _stub_turns(runner, monkeypatch, turns)
        with monkeypatch.context() as held:
            held.setattr(SessionAuthority, '_schedule', lambda self, ref: None)
            authority = await initialize_session_authority(runner, profile_id='default', instance_id='boot-1')
            async with _wired_relay(runner, connector, url) as (_adapter, ws):
                await ws.send(json.dumps({'type': 'inbound', 'event': raw, 'bufferId': 'b-1'}) + '\n')
                await async_wait_until(lambda: any(f.get('type') == 'inbound_ack' for f in connector.outgoing),
                                       timeout=10, what='the frame committed and acked')
            sid = next(iter(authority.sessions))
            assert [r['status'] for r in list_session_admissions(authority.db, session_id=sid)] == ['queued']
            authority.db.close()
        assert not turns

        # Boot 2: a fresh runner, authority and relay socket over the same home. The hosted-room
        # service (local surfaces) is not part of this path.
        async def _no_hosted_service(_runner):
            return None
        monkeypatch.setattr('gateway.session_hosted_service.ensure_hosted_service', _no_hosted_service)
        runner = GatewayRunner(GatewayConfig(sessions_dir=tmp_path / 'sessions'))
        _stub_turns(runner, monkeypatch, turns)
        authority = await initialize_session_authority(runner, profile_id='default', instance_id='boot-2')
        try:
            async with _wired_relay(runner, connector, url):
                outcome = await recover_gateway_native_sessions(runner)
                assert list(outcome.values()) == ['ready'], outcome
                await async_wait_until(lambda: turns and connector.replies('chan-1'), timeout=20,
                                       what='the recovered turn answered through the relay')
                assert turns == ['queued before restart']
                reply = connector.replies('chan-1')[-1]
                # The connector resolves the tenant from these; a cold cache sent neither.
                assert reply['metadata'].get('scope_id') == 'guild-1'
                assert reply['metadata'].get('user_id') == 'owner-1'
                assert [r['status'] for r in _rows(authority)] == ['terminal']
        finally:
            authority.db.close()


def _reserve_homes(tmp_path, monkeypatch):
    root = tmp_path / '.hermes'
    root.mkdir(mode=0o700)
    monkeypatch.setattr(Path, 'home', lambda: tmp_path)
    monkeypatch.setenv('HERMES_HOME', str(root))
    import hermes_state
    monkeypatch.setattr(hermes_state, 'DEFAULT_DB_PATH', hermes_state._IMPORT_DEFAULT_DB_PATH)
    work = root / 'profiles' / 'work'
    work.mkdir(parents=True, mode=0o700)
    (work / 'config.yaml').write_text('{}')
    return root, work


def _multiplexed_runner(tmp_path, homes):
    config = GatewayConfig(sessions_dir=tmp_path / 'sessions')
    config.multiplex_profiles = True
    config._runtime_profile_homes = tuple(homes)
    return GatewayRunner(config)


@pytest.mark.asyncio
async def test_multiplexed_stamped_admission_queued_across_a_restart_runs_in_its_profile(tmp_path, monkeypatch):
    from gateway.run_runtime import initialize_gateway_runtime, recover_gateway_native_sessions
    from gateway.runtime_ownership import process_ownership
    from gateway.session_authority import SessionAuthority

    root, work = _reserve_homes(tmp_path, monkeypatch)
    homes = (('default', root), ('work', work))
    connector = _Connector('discord')
    raw = {'text': 'work queued before restart', 'message_id': 'm-w', 'source': {
        'platform': 'discord', 'chat_id': 'chan-w', 'chat_type': 'group', 'user_id': 'owner-w',
        'scope_id': 'guild-w', 'profile': 'work'}}

    async def _no_hosted_service(_runner):
        return None
    monkeypatch.setattr('gateway.session_hosted_service.ensure_hosted_service', _no_hosted_service)

    def _close(runner):
        for authority in list(getattr(runner, 'session_authorities', None) or []):
            authority.db.close()
        for _, home in homes:
            process_ownership.release(home)

    turns = []
    async with websockets.serve(connector, '127.0.0.1', 0) as server:
        url = f'ws://127.0.0.1:{server.sockets[0].getsockname()[1]}/relay'
        process_ownership.reserve([home for _, home in homes])
        runner = _multiplexed_runner(tmp_path, homes)
        _stub_turns(runner, monkeypatch, turns)
        try:
            with monkeypatch.context() as held:
                held.setattr(SessionAuthority, '_schedule', lambda self, ref: None)
                await initialize_gateway_runtime(runner)
                async with _wired_relay(runner, connector, url) as (_adapter, ws):
                    await ws.send(json.dumps({'type': 'inbound', 'event': raw, 'bufferId': 'b-w'}) + '\n')
                    await async_wait_until(lambda: any(f.get('type') == 'inbound_ack' for f in connector.outgoing),
                                           timeout=10, what='the frame committed and acked')
                assert [r['status'] for r in _rows(runner.session_authorities.for_home(work))] == ['queued']
        finally:
            _close(runner)
        assert not turns

        process_ownership.reserve([home for _, home in homes])
        runner = _multiplexed_runner(tmp_path, homes)
        _stub_turns(runner, monkeypatch, turns)
        try:
            await initialize_gateway_runtime(runner)
            async with _wired_relay(runner, connector, url):
                outcome = await recover_gateway_native_sessions(runner)
                assert list(outcome.values()) == ['ready'], outcome
                await async_wait_until(lambda: turns and connector.replies('chan-w'), timeout=20,
                                       what='the recovered work turn answered through the relay')
                assert turns == ['work queued before restart']
                reply = connector.replies('chan-w')[-1]['metadata']
                assert (reply.get('scope_id'), reply.get('user_id'), reply.get('profile')) == ('guild-w', 'owner-w', 'work')
                assert [r['status'] for r in _rows(runner.session_authorities.for_home(work))] == ['terminal']
                assert not _rows(runner.session_authorities.for_home(root))
        finally:
            _close(runner)


@pytest.mark.asyncio
async def test_forwarded_discord_component_is_admitted_like_relayed_text(tmp_path, monkeypatch, caplog):
    from gateway.session_authority import initialize_session_authority

    connector = _Connector('discord')
    runner = GatewayRunner(GatewayConfig(sessions_dir=tmp_path / 'sessions'))
    turns = []
    _stub_turns(runner, monkeypatch, turns)
    authority = await initialize_session_authority(runner, profile_id='default', instance_id='passthrough')
    try:
        async with websockets.serve(connector, '127.0.0.1', 0) as server:
            url = f'ws://127.0.0.1:{server.sockets[0].getsockname()[1]}/relay'
            async with _wired_relay(runner, connector, url) as (_adapter, ws):
                # A MESSAGE_COMPONENT press with a foreign custom_id: ordinary text, not a command.
                # The connector stamped a profile; this single-profile gateway serves it anyway.
                body = {'type': 3, 'id': 'ix-1', 'channel_id': 'chan-2', 'guild_id': 'guild-2',
                        'member': {'user': {'id': 'owner-2', 'username': 'owner'}},
                        'data': {'custom_id': 'pick-blue'}}
                forward = {'platform': 'discord', 'botId': 'fixture-bot', 'method': 'POST',
                           'path': '/interactions', 'headers': [], 'profile': 'work',
                           'bodyB64': base64.b64encode(json.dumps(body).encode()).decode()}
                await ws.send(json.dumps({'type': 'passthrough_forward', 'forward': forward}) + '\n')
                await async_wait_until(lambda: turns and connector.replies('chan-2'), timeout=20,
                                       what='the component turn answered')
                assert turns == ['pick-blue']
                assert len(_rows(authority)) == 1
                assert 'relay inbound refused' not in caplog.text
                assert 'permission_denied' not in caplog.text
    finally:
        authority.db.close()
