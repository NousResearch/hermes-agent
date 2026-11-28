"""``session.info`` carries what main's classic CLI read off itself for /status, the status bar and the
exit summary (wave A2): title, started/last-active times, yolo, api_mode, tool_count and a masked
credential hint (never the key); re-published when the auto-title lands."""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from gateway.config import Platform


@pytest.mark.asyncio
async def test_session_info_carries_the_classic_cli_fields(tmp_path, monkeypatch):
    from gateway.session_contract import SessionRef
    from gateway.session_local import local_session_info
    from tests.gateway.test_surface_admission import _authority

    async def answer(event):
        return 'ok'
    authority = await _authority(tmp_path, monkeypatch, answer)
    agent = SimpleNamespace(model='m', provider='custom', api_mode='chat_completions',
                            api_key='sk-abcdefghijklmnop1234', tools=[{'function': {'name': 'a'}}] * 3)
    authority.runner._cached_agent_for = lambda route: agent
    authority.persistence.sessions.set_title('s', 'Some Title')
    info = local_session_info(authority, SessionRef('p', 's'))
    assert info['title'] == 'Some Title'
    assert info['provider'] == 'custom'
    assert isinstance(info['started_at'], float) and isinstance(info['last_active_at'], float)
    assert info['yolo'] is False and info['api_mode'] == 'chat_completions' and info['tool_count'] == 3
    assert info['credential_hint'] == 'sk-a...1234' and 'abcdefgh' not in str(info)
    from tui_gateway.contracts.registry import CANONICAL_METHODS
    CANONICAL_METHODS['session.info'].result.model_validate(
        {**info, 'profile_name': 'default', 'desktop_protocol': 'hermes-gateway-v1', 'lazy': False})


def test_lazy_session_info_uses_the_policy_route_provider(tmp_path):
    """Regression for #139015: resumed Desktop sessions need the picker route identity."""
    from gateway.session_local import _session_provider
    from gateway.session_policy import build_policy

    policy = build_policy({'cwd': str(tmp_path)}, {'model': {'default': 'model', 'provider': 'custom'}})
    auto_policy = build_policy({'cwd': str(tmp_path)}, {'model': {'default': 'model', 'provider': 'auto'}})

    assert _session_provider(None, None, policy) == 'custom'
    assert _session_provider(None, None, auto_policy) == ''



@pytest.mark.asyncio
async def test_an_unresolved_route_provider_is_left_out_not_sent_empty(tmp_path, monkeypatch):
    """An ``auto`` route's provider is unknown until its agent builds. Desktop stores any string it
    gets, so an empty one cleared the session's provider on every idle frame of a lazy session."""
    from gateway.session_contract import SessionRef
    from gateway.session_local import local_session_info
    from tests.gateway.test_session_create_tier_cwd import _owner

    monkeypatch.chdir(tmp_path)
    store, _runner, authority, owner = _owner(
        tmp_path / 'owner', monkeypatch, {'model': {'default': 'base-model', 'provider': 'auto'}},
    )
    try:
        created = await owner.dispatch({'id': 'c1', 'method': 'session.create', 'params': {
            'request_id': 'c1', 'source': 'gui', 'cwd': str(tmp_path), 'toolsets': [],
        }})
        ref = SessionRef(authority.profile_id, created['result']['session_id'])
        info = local_session_info(authority, ref)
        assert info['model'] == 'base-model' and info['lazy'] is True
        assert 'provider' not in info
    finally:
        await owner.close()
        store.close_all_db_handles()

@pytest.mark.asyncio
async def test_local_viewer_title_republishes_session_info(monkeypatch):
    import asyncio

    from gateway.run_turn_runner import TurnRunner
    from tests.waiting import async_wait_until
    published = []
    live = SimpleNamespace(event_stream=SimpleNamespace(
        publish=lambda sid, payload, event_type: published.append((event_type, payload))))
    authority = SimpleNamespace(sessions={'s': live})
    turn = object.__new__(TurnRunner)
    turn._ctx = SimpleNamespace(source=SimpleNamespace(platform=Platform.LOCAL), session_key='k',
                                _loop_for_step=asyncio.get_running_loop())
    turn._approval_owner = (authority, 's', 1)
    turn._runner = SimpleNamespace(_recover_discord_auto_thread_source=lambda source, key: source,
                                   _is_telegram_topic_lane=lambda s: False,
                                   _is_discord_auto_thread_lane=lambda s: False,
                                   _is_relay_discord_channel_lane=lambda s: False)
    from gateway import session_local
    monkeypatch.setattr(session_local, 'local_session_info', lambda authority, ref: {'title': 'Auto'})
    agent = SimpleNamespace(session_id='s')
    turn._attach_session_title_callback(agent, turn._ctx)
    agent._on_session_title('Auto', 'llm')
    await async_wait_until(lambda: published, what='the title republish')
    assert published == [('session.info', {'title': 'Auto'})]


@pytest.mark.asyncio
async def test_local_session_info_preserves_named_custom_provider_identity(tmp_path, monkeypatch):
    from gateway.session_contract import SessionRef
    from gateway.session_local import local_session_info
    from gateway.session_local_inventory import snapshot_info
    from tests.gateway.test_session_create_tier_cwd import _owner

    monkeypatch.chdir(tmp_path)
    cfg = {
        'model': {'default': 'lab-model', 'provider': 'custom:lab'},
        'providers': {'lab': {'base_url': 'http://127.0.0.1:18999/v1', 'api_key': 'sk-lab'}},
    }
    store, runner, authority, owner = _owner(tmp_path / 'owner', monkeypatch, cfg)
    try:
        created = await owner.dispatch({'id': 'c1', 'method': 'session.create', 'params': {
            'request_id': 'c1', 'source': 'gui', 'cwd': str(tmp_path), 'toolsets': [],
        }})
        sid = created['result']['session_id']
        ref = SessionRef(authority.profile_id, sid)
        agent = SimpleNamespace(
            model='lab-model',
            provider='custom',
            requested_provider='custom:lab',
            base_url='http://127.0.0.1:18999/v1',
            api_mode='chat_completions',
            tools=[],
        )
        runner._cached_agent_for = lambda route: agent
        assert local_session_info(authority, ref)['provider'] == 'custom:lab'
        assert (await snapshot_info(authority, ref))['provider'] == 'custom:lab'
    finally:
        await owner.close()
        store.close_all_db_handles()


@pytest.mark.asyncio
async def test_idle_policy_mutation_publishes_stamped_session_info_with_provider(tmp_path, monkeypatch):
    from dataclasses import asdict, replace
    import json

    from gateway import session_mutations
    from gateway.session_contract import SessionRef
    from gateway.session_policy import restore_policy
    from tests.gateway.test_session_create_tier_cwd import _owner

    monkeypatch.chdir(tmp_path)
    store, _runner, authority, owner = _owner(
        tmp_path / 'owner', monkeypatch, {'model': {'default': 'base-model', 'provider': 'openrouter'}},
    )

    async def prepare(auth, live, payload, prepared):
        old = restore_policy(prepared['snapshot']['receipt']['policy'])
        config = old.config()
        config['model'] = {'default': 'switched-model', 'provider': 'claude-code'}
        return {
            **prepared,
            'policy': asdict(replace(old, model='switched-model', config_json=json.dumps(config))),
        }

    monkeypatch.setattr('gateway.session_mutation_model.prepare_model', prepare)
    try:
        created = await owner.dispatch({'id': 'c1', 'method': 'session.create', 'params': {
            'request_id': 'c1', 'source': 'gui', 'cwd': str(tmp_path), 'toolsets': [],
        }})
        ref = SessionRef(authority.profile_id, created['result']['session_id'])
        handle = authority._handle(ref)
        receipt = await session_mutations.mutate_session(
            authority,
            owner.actor,
            ref,
            {
                'session_id': ref.session_id,
                'request_id': 'pick-claude',
                'operation': 'model',
                'payload': {'model': 'switched-model', 'provider': 'claude-code'},
                'expected_revision': handle.revision,
                'expected_generation': handle.execution_generation,
            },
        )
        stream = authority.sessions[ref.session_id].event_stream
        events = [
            e for e in stream.since(stream.watermark()[0], 0)['events']
            if e.get('type') == 'session.info' and e.get('payload', {}).get('model') == 'switched-model'
        ]
        assert events
        payload = events[-1]['payload']
        assert payload['provider'] == 'claude-code'
        assert payload['execution_generation'] == receipt['execution_generation']
        assert payload['revision'] == receipt['revision']
        assert 'authority_epoch' not in payload
    finally:
        await owner.close()
        store.close_all_db_handles()


@pytest.mark.asyncio
async def test_sibling_idle_publishers_stamp_generation_and_revision(tmp_path, monkeypatch):
    import asyncio

    from gateway.run_turn_runner import TurnRunner
    from gateway.session_contract import SessionRef
    from gateway.session_policy_refresh import _publish as publish_refreshed_policy
    from tests.gateway.test_session_create_tier_cwd import _owner
    from tests.waiting import async_wait_until

    monkeypatch.chdir(tmp_path)
    store, runner, authority, owner = _owner(
        tmp_path / 'owner', monkeypatch, {'model': {'default': 'base-model', 'provider': 'openrouter'}},
    )
    try:
        created = await owner.dispatch({'id': 'c1', 'method': 'session.create', 'params': {
            'request_id': 'c1', 'source': 'gui', 'cwd': str(tmp_path), 'toolsets': [],
        }})
        ref = SessionRef(authority.profile_id, created['result']['session_id'])
        await publish_refreshed_policy(authority, ref)
        handle = authority._handle(ref)
        stream = authority.sessions[ref.session_id].event_stream
        refresh_events = [
            e['payload'] for e in stream.since(stream.watermark()[0], 0)['events']
            if e.get('type') == 'session.info' and e.get('payload', {}).get('model') == 'base-model'
        ]
        assert refresh_events
        assert refresh_events[-1]['provider'] == 'openrouter'
        assert refresh_events[-1]['execution_generation'] == handle.execution_generation
        assert refresh_events[-1]['revision'] == handle.revision

        turn = object.__new__(TurnRunner)
        turn._ctx = SimpleNamespace(source=SimpleNamespace(platform=Platform.LOCAL), session_key='k',
                                    _loop_for_step=asyncio.get_running_loop())
        turn._approval_owner = (authority, ref.session_id, handle.execution_generation)
        turn._runner = runner
        agent = SimpleNamespace(session_id=ref.session_id)
        turn._attach_session_title_callback(agent, turn._ctx)
        authority.persistence.sessions.set_title(ref.session_id, 'Renamed')
        agent._on_session_title('Renamed', 'llm')
        title_events = await async_wait_until(lambda: [
            e['payload'] for e in stream.since(stream.watermark()[0], 0)['events']
            if e.get('type') == 'session.info' and e.get('payload', {}).get('title') == 'Renamed'
        ], what='the title republish')
        assert title_events[-1]['execution_generation'] == handle.execution_generation
        assert title_events[-1]['revision'] == handle.revision
    finally:
        await owner.close()
        store.close_all_db_handles()


@pytest.mark.asyncio
async def test_messaging_chat_pick_publishes_stamped_session_info_with_provider(tmp_path, monkeypatch):
    from gateway import run
    from gateway.config import GatewayConfig
    from gateway.run import GatewayRunner
    from gateway.session import SessionSource
    from gateway.session_authority import initialize_session_authority
    from gateway.session_contract import SessionRef
    from gateway.session_controls import AuthorityConnection
    from gateway.session_messaging_picks import apply_chat_pick

    msg_home = tmp_path / 'msg'
    msg_home.mkdir()
    monkeypatch.setenv('HERMES_HOME', str(msg_home))
    monkeypatch.setattr(run, '_hermes_home', msg_home)
    (msg_home / 'config.yaml').write_text(
        'model:\n'
        '  default: base-model\n'
        '  provider: openrouter\n'
    )
    runner = GatewayRunner(GatewayConfig())
    authority = await initialize_session_authority(runner, profile_id=str(msg_home), instance_id='owner')
    source = SessionSource(platform=Platform.TELEGRAM, chat_id='chat-1', user_id='u-1', chat_type='dm')
    msg_ref = authority.register(source)
    live = authority.sessions[msg_ref.session_id]
    viewer = AuthorityConnection(authority, object(), {
        'provider': 'local', 'user_id': 'viewer', 'profile_id': str(msg_home), 'instance_id': 'owner',
        'capabilities': ['session:read', 'session:control'],
    })
    try:
        handle = authority._handle(msg_ref)
        receipt = await apply_chat_pick(
            authority,
            msg_ref,
            SessionRef(authority.profile_id, msg_ref.session_id),
            live,
            {
                'session_id': msg_ref.session_id,
                'request_id': 'pick-effort',
                'operation': 'reasoning',
                'payload': {'effort': 'medium'},
                'expected_revision': handle.revision,
                'expected_generation': handle.execution_generation,
            },
            principal_id=viewer.actor.subject,
        )
        msg_events = [
            e['payload'] for e in live.event_stream.since(live.event_stream.watermark()[0], 0)['events']
            if e.get('type') == 'session.info' and 'model' in e.get('payload', {})
        ]
        assert msg_events
        assert msg_events[-1]['provider'] == 'openrouter'
        assert msg_events[-1]['execution_generation'] == receipt['execution_generation']
        assert msg_events[-1]['revision'] == receipt['revision']
    finally:
        await viewer.close()
        runner.session_store.close_all_db_handles()


@pytest.mark.asyncio
async def test_title_republish_racing_a_model_pick_never_repaints_the_old_model(tmp_path, monkeypatch):
    """The auto-title thread's snapshot must not straddle a model pick. Desktop's idle fence takes any
    session.info naming the current generation, so a snapshot read before the pick and stamped after it
    lands last and puts the old model back on every viewer."""
    import asyncio
    import threading
    from dataclasses import asdict, replace
    import json

    from gateway import session_local, session_mutations
    from gateway.run_turn_runner import TurnRunner
    from gateway.session_contract import SessionRef
    from gateway.session_policy import restore_policy
    from tests.gateway.test_session_create_tier_cwd import _owner
    from tests.waiting import async_wait_until

    monkeypatch.chdir(tmp_path)
    store, runner, authority, owner = _owner(
        tmp_path / 'owner', monkeypatch, {'model': {'default': 'base-model', 'provider': 'openrouter'}},
    )

    async def prepare(auth, live, payload, prepared):
        old = restore_policy(prepared['snapshot']['receipt']['policy'])
        config = old.config()
        config['model'] = {'default': 'switched-model', 'provider': 'claude-code'}
        return {**prepared, 'policy': asdict(replace(old, model='switched-model', config_json=json.dumps(config)))}

    monkeypatch.setattr('gateway.session_mutation_model.prepare_model', prepare)
    loop_thread = threading.current_thread()
    built, release, calls = threading.Event(), threading.Event(), []
    real_info = session_local.local_session_info

    def snapshot(authority, ref):
        calls.append(None)
        info = {**real_info(authority, ref), 'probe_call': len(calls)}
        built.set()
        if threading.current_thread() is not loop_thread:  # the pick commits and publishes meanwhile
            release.wait(10)
        return info

    try:
        created = await owner.dispatch({'id': 'c1', 'method': 'session.create', 'params': {
            'request_id': 'c1', 'source': 'gui', 'cwd': str(tmp_path), 'toolsets': [],
        }})
        ref = SessionRef(authority.profile_id, created['result']['session_id'])
        handle = authority._handle(ref)
        monkeypatch.setattr(session_local, 'local_session_info', snapshot)
        turn = object.__new__(TurnRunner)
        turn._ctx = SimpleNamespace(source=SimpleNamespace(platform=Platform.LOCAL), session_key='k',
                                    _loop_for_step=asyncio.get_running_loop())
        turn._approval_owner = (authority, ref.session_id, handle.execution_generation)
        turn._runner = runner
        agent = SimpleNamespace(session_id=ref.session_id)
        turn._attach_session_title_callback(agent, turn._ctx)
        authority.persistence.sessions.set_title(ref.session_id, 'Renamed')
        titler = threading.Thread(target=agent._on_session_title, args=('Renamed', 'llm'), daemon=True)
        titler.start()
        assert await asyncio.to_thread(built.wait, 10)
        await session_mutations.mutate_session(authority, owner.actor, ref, {
            'session_id': ref.session_id, 'request_id': 'pick-claude', 'operation': 'model',
            'payload': {'model': 'switched-model', 'provider': 'claude-code'},
            'expected_revision': handle.revision, 'expected_generation': handle.execution_generation,
        })
        release.set()
        await asyncio.to_thread(titler.join, 10)
        stream = authority.sessions[ref.session_id].event_stream

        def probed():
            return [e['payload'] for e in stream.since(stream.watermark()[0], 0)['events']
                    if e.get('type') == 'session.info' and 'probe_call' in e.get('payload', {})]

        await async_wait_until(lambda: any(p['probe_call'] == 1 for p in probed()), what='the title republish')
        assert probed()[-1]['model'] == 'switched-model'
    finally:
        release.set()
        await owner.close()
        store.close_all_db_handles()
