"""A connector-stamped ``source.profile`` is a route, on single-profile and multiplexed gateways.

The connector (Team Gateway per-scope profile routing) stamps the Hermes profile that serves an
inbound event. A single-profile gateway serves it from its one profile (the stamp is not a route
there, as before shared session authority, and never reaches durable state); a multiplexed gateway admits it into that profile's
own ledger. Neither may acknowledge the frame and drop it. An unserved stamped profile is refused
the way an unserved profile route is: acknowledged, never run.
"""
import asyncio
import base64
import json
from contextlib import asynccontextmanager
from pathlib import Path
from urllib.parse import urlsplit

import pytest

from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.run import GatewayRunner
from gateway.session_results import execution_result
from tests.waiting import async_wait_until

websockets = pytest.importorskip('websockets')

SOURCE = {'platform': 'telegram', 'chat_type': 'dm'}


@asynccontextmanager
async def _relay(runner, platforms=('telegram',), port=0):
    """A real WebSocketRelayTransport + RelayAdapter wired as ``runner``'s primary relay, against a
    fake connector that acks every outbound. Yields ``(send, outgoing, adapter)``."""
    from gateway.relay.adapter import RelayAdapter
    from gateway.relay.descriptor import CapabilityDescriptor
    from gateway.relay.ws_transport import WebSocketRelayTransport

    outgoing, ready = [], asyncio.Queue()
    descriptor = CapabilityDescriptor(1, platforms[0], 'Fixture', 4096, False, True, False, 'plain', 'chars')

    async def connector(ws):
        await ws.recv()  # hello
        await ws.send(json.dumps({'type': 'descriptor', 'descriptor': json.loads(descriptor.to_json())}) + '\n')
        await ready.put(ws)
        async for line in ws:
            frame = json.loads(line)
            outgoing.append(frame)
            if frame['type'] == 'outbound':
                await ws.send(json.dumps({'type': 'outbound_result', 'requestId': frame['requestId'],
                                          'result': {'success': True, 'message_id': 'sent'}}) + '\n')
            elif frame['type'] == 'going_idle':
                await ws.send(json.dumps({'type': 'going_idle_ack'}) + '\n')

    async with websockets.serve(connector, '127.0.0.1', port) as server:
        transport = WebSocketRelayTransport(
            f'ws://127.0.0.1:{server.sockets[0].getsockname()[1]}/relay', platforms[0], 'fixture-bot',
            gateway_id='fixture-gateway', upgrade_secret='fixture-secret', reconnect=False)
        adapter = RelayAdapter(PlatformConfig(enabled=True), descriptor, transport)
        adapter.gateway_runner = runner
        runner.adapters[Platform.RELAY] = adapter
        runner._wire_adapter_handlers(adapter)
        try:
            assert await adapter.connect()
            ws = await asyncio.wait_for(ready.get(), 5)

            async def send(message_id, chat_id, profile=None, text=None):
                source = {**SOURCE, 'chat_id': chat_id, 'user_id': 'u-' + chat_id}
                if profile is not None:
                    source['profile'] = profile
                event = {'text': text or f'hello {message_id}', 'message_id': message_id, 'source': source}
                await ws.send(json.dumps({'type': 'inbound', 'event': event, 'bufferId': 'buf-' + message_id}) + '\n')

            yield send, outgoing, adapter
        finally:
            await adapter.disconnect()


def _stub_turns(runner, monkeypatch):
    """Record each executed turn's (text, runtime home, session key) instead of calling a model."""
    from hermes_constants import get_hermes_home
    turns = []

    async def turn(event, *_args, **_kwargs):
        turns.append((event.text, Path(get_hermes_home()).resolve(),
                      runner._session_key_for_source(event.source)))
        execution_result.get()['result'] = {'final_response': 'reply ' + event.text, 'messages': []}
        return 'reply ' + event.text

    monkeypatch.setattr(runner, '_handle_message_with_agent', turn)
    return turns


def _acked(outgoing):
    return {f.get('bufferId') for f in outgoing if f.get('type') == 'inbound_ack'}


def _replies(outgoing, chat_id):
    return [f['action'] for f in outgoing if f.get('type') == 'outbound'
            and f['action'].get('op') in ('send', 'edit') and f['action'].get('chat_id') == chat_id]


def _requests(authority):
    from hermes_state_runtime import list_session_admissions
    rows = []
    for sid in list(authority.sessions):
        rows += list_session_admissions(authority.db, session_id=sid, pending_only=False)
    return rows


@pytest.mark.asyncio
async def test_single_profile_gateway_serves_a_connector_stamped_profile(tmp_path, monkeypatch):
    from gateway.session_authority import initialize_session_authority
    from hermes_constants import get_hermes_home
    launch = Path(get_hermes_home()).resolve()
    # A profile of the stamped name exists on disk: the single-profile gateway must still not use it.
    (launch / 'profiles' / 'work').mkdir(parents=True, exist_ok=True)
    runner = GatewayRunner(GatewayConfig(sessions_dir=tmp_path / 'sessions'))
    turns = _stub_turns(runner, monkeypatch)
    authority = await initialize_session_authority(runner, profile_id='default', instance_id='relay-profile-single')
    try:
        async with _relay(runner) as (send, outgoing, _adapter):
            await send('plain', 'chat-plain')
            await send('stamped', 'chat-stamped', profile='work')
            await async_wait_until(lambda: len(turns) == 2 and _replies(outgoing, 'chat-stamped')
                                   and _replies(outgoing, 'chat-plain'), timeout=20,
                                   what='both relay frames answered')
            assert {'buf-plain', 'buf-stamped'} <= _acked(outgoing)
            by_text = {text: (home, key) for text, home, key in turns}
            # The stamp is not a route on a single-profile gateway: the one profile, historical namespace.
            assert by_text['hello stamped'] == (launch, 'agent:main:telegram:dm:chat-stamped')
            assert by_text['hello plain'] == (launch, 'agent:main:telegram:dm:chat-plain')
            rows = {r['request_id']: r for r in _requests(authority)}
            assert sorted(rows) == ['plain', 'stamped']
            # No profile reaches durable state: the admission and the session origin (which later
            # completion/notification scopes resolve a home from) both name the launch profile only.
            assert 'profile' not in rows['stamped']['payload']['native_text_v1']['source']
            entry = runner.session_store.lookup_by_session_key('agent:main:telegram:dm:chat-stamped')
            assert entry is not None and not entry.origin.profile
            assert Path(runner._resolve_profile_home_for_source(entry.origin)).resolve() == launch
            # The connector still gets its profile echoed on egress (persona attribution).
            assert _replies(outgoing, 'chat-stamped')[-1]['metadata'].get('profile') == 'work'
            # A replay of the stamped frame is the same admission, not a second turn.
            await send('stamped', 'chat-stamped', profile='work')
            await async_wait_until(lambda: len(_acked(outgoing)) >= 2 and sum(
                1 for f in outgoing if f.get('type') == 'inbound_ack' and f['bufferId'] == 'buf-stamped') == 2,
                timeout=10, what='the replay acknowledged')
            assert len(turns) == 2
    finally:
        authority.db.close()


def _reserve_homes(tmp_path, monkeypatch, names):
    root = tmp_path / '.hermes'
    root.mkdir(mode=0o700)
    monkeypatch.setattr(Path, 'home', lambda: tmp_path)
    monkeypatch.setenv('HERMES_HOME', str(root))
    import hermes_state
    monkeypatch.setattr(hermes_state, 'DEFAULT_DB_PATH', hermes_state._IMPORT_DEFAULT_DB_PATH)
    homes = [('default', root)]
    for name in names:
        home = root / 'profiles' / name
        home.mkdir(parents=True, mode=0o700)
        (home / 'config.yaml').write_text('{}')
        homes.append((name, home))
    return root, homes


@pytest.mark.asyncio
async def test_multiplexed_gateway_admits_a_connector_stamped_profile_into_its_own_ledger(tmp_path, monkeypatch):
    from gateway.run_runtime import initialize_gateway_runtime
    from gateway.runtime_ownership import process_ownership
    from gateway.session_envelope import restore_native
    from hermes_state_runtime import RuntimeStoreError

    root, homes = _reserve_homes(tmp_path, monkeypatch, names=('work',))
    work = homes[1][1]
    process_ownership.reserve([home for _, home in homes])
    config = GatewayConfig(sessions_dir=tmp_path / 'sessions')
    config.multiplex_profiles = True
    config._runtime_profile_homes = tuple(homes)
    runner = GatewayRunner(config)
    turns = _stub_turns(runner, monkeypatch)
    try:
        await initialize_gateway_runtime(runner)
        default_authority = runner.session_authorities.for_home(root)
        work_authority = runner.session_authorities.for_home(work)
        async with _relay(runner) as (send, outgoing, _adapter):
            await send('plain', 'chat-plain')
            await send('to-default', 'chat-default', profile='default')
            await send('to-work', 'chat-work', profile='work')
            await send('to-ghost', 'chat-ghost', profile='ghost')
            await async_wait_until(lambda: len(turns) == 3 and all(
                _replies(outgoing, chat) for chat in ('chat-plain', 'chat-default', 'chat-work'))
                and 'buf-to-ghost' in _acked(outgoing), timeout=30, what='served frames answered, unserved acked')
            by_text = {text: (home, key) for text, home, key in turns}
            assert by_text['hello plain'] == (root.resolve(), 'agent:main:telegram:dm:chat-plain')
            assert by_text['hello to-default'] == (root.resolve(), 'agent:main:telegram:dm:chat-default')
            # The stamped profile's turn runs under its own home, in its own namespace and ledger.
            assert by_text['hello to-work'] == (work.resolve(), 'agent:work:telegram:dm:chat-work')
            assert sorted(r['request_id'] for r in _requests(default_authority)) == ['plain', 'to-default']
            work_rows = _requests(work_authority)
            assert [r['request_id'] for r in work_rows] == ['to-work']
            # An unserved stamped profile is refused like an unserved profile route: acked, never run.
            assert not _replies(outgoing, 'chat-ghost')
            assert 'hello to-ghost' not in by_text
            # The stamp is sealed into the admission: rewriting it in storage does not reroute a replay.
            forged = json.loads(json.dumps(work_rows[0]['payload']))
            forged['native_text_v1']['source']['profile'] = 'default'
            with pytest.raises(RuntimeStoreError):
                restore_native(forged, runner)
            restore_native(work_rows[0]['payload'], runner)  # the genuine row still restores
    finally:
        for authority in list(getattr(runner, 'session_authorities', None) or []):
            authority.db.close()
        for _, home in homes:
            process_ownership.release(home)


@pytest.mark.asyncio
async def test_stamped_profile_completion_finds_its_relay_after_a_restart(tmp_path, monkeypatch):
    """A persisted origin omits relay trust, and a stamped profile with no local route cannot borrow
    the primary relay by route: after a restart its background completion must still reach the relay,
    proven by the route's committed relay admission, instead of being deferred or refused."""
    from gateway.run_runtime import initialize_gateway_runtime
    from gateway.runtime_ownership import process_ownership

    _root, homes = _reserve_homes(tmp_path, monkeypatch, names=('work',))
    process_ownership.reserve([home for _, home in homes])
    config = GatewayConfig(sessions_dir=tmp_path / 'sessions')
    config.multiplex_profiles = True
    config._runtime_profile_homes = tuple(homes)
    key = 'agent:work:telegram:dm:chat-work'
    runners = []
    try:
        first = GatewayRunner(config)
        runners.append(first)
        turns = _stub_turns(first, monkeypatch)
        await initialize_gateway_runtime(first)
        async with _relay(first) as (send, outgoing, adapter):
            # The relay seal binds the connector URL: the restarted gateway dials the same one.
            port = urlsplit(adapter._transport._url).port
            await send('to-work', 'chat-work', profile='work')
            await async_wait_until(lambda: turns and _replies(outgoing, 'chat-work'), timeout=30,
                                   what='the stamped turn answered')
        parent = first.session_store.lookup_by_session_key(key).session_id
        for authority in list(first.session_authorities):
            authority.db.close()

        second = GatewayRunner(config)  # restart: fresh runner, authorities and relay over persisted state
        runners.append(second)
        turns = _stub_turns(second, monkeypatch)
        await initialize_gateway_runtime(second)
        async with _relay(second, port=port) as (_send, outgoing, _adapter):
            evt = {'type': 'completion', 'session_key': key, 'parent_session_id': parent,
                   'session_id': 'proc_restart', 'started_at': 1.0}
            async with second._completion_event_scope(evt):
                assert await second._inject_watch_notification('[process finished]', evt) is True
            await async_wait_until(lambda: turns and _replies(outgoing, 'chat-work'), timeout=30,
                                   what='the completion answered through the relay')
            assert turns[0][1:] == (homes[1][1].resolve(), key)
    finally:
        for runner in runners:
            for authority in list(getattr(runner, 'session_authorities', None) or []):
                authority.db.close()
        for _, home in homes:
            process_ownership.release(home)


def _discord_modal(profile):
    body = {'type': 5, 'id': 'modal-1', 'channel_id': 'chat-modal', 'user': {'id': 'u-modal'},
            'data': {'custom_id': 'feedback', 'components': [{'type': 1, 'components': [
                {'type': 4, 'custom_id': 'answer', 'value': 'Please summarize'}]}]}}
    return {'platform': 'discord', 'botId': 'fixture-bot', 'method': 'POST', 'path': '/interactions',
            'headers': [], 'bodyB64': base64.b64encode(json.dumps(body).encode()).decode(), 'profile': profile}


@pytest.mark.asyncio
async def test_single_profile_gateway_serves_stamped_commands_and_discord_interactions(tmp_path, monkeypatch):
    from gateway.session_authority import initialize_session_authority
    from hermes_constants import get_hermes_home
    launch = Path(get_hermes_home()).resolve()
    (launch / 'profiles' / 'work').mkdir(parents=True, exist_ok=True)
    runner = GatewayRunner(GatewayConfig(sessions_dir=tmp_path / 'sessions'))
    turns = _stub_turns(runner, monkeypatch)
    authority = await initialize_session_authority(runner, profile_id='default', instance_id='relay-stamp-paths')
    try:
        async with _relay(runner, platforms=('discord',)) as (_send, outgoing, adapter):
            # A stamped slash command that becomes a turn (``_on_inbound`` skips native admission).
            await adapter._transport._on_inbound({'event': {'text': '/queue summarize', 'message_id': 'cmd-1', 'source': {
                'platform': 'discord', 'chat_type': 'dm', 'chat_id': 'chat-cmd', 'user_id': 'u-cmd', 'profile': 'work'}}})
            # A stamped Discord modal submit (``_on_passthrough`` calls handle_message directly).
            await adapter._transport._on_passthrough_forward({'forward': _discord_modal('work')})
            await async_wait_until(lambda: _replies(outgoing, 'chat-cmd') and _replies(outgoing, 'chat-modal'),
                                   timeout=20, what='both stamped inputs answered')
            assert len(turns) == 2, turns
            assert {home for _, home, _ in turns} == {launch}
            for chat in ('chat-cmd', 'chat-modal'):
                sent = [a.get('content') or '' for a in _replies(outgoing, chat)]
                assert sent and not any('profile_mismatch' in text for text in sent), sent
                # The stamp is still echoed on egress.
                assert _replies(outgoing, chat)[-1]['metadata'].get('profile') == 'work'
    finally:
        authority.db.close()
