"""Opposing transport/runtime policy and restart controls, no vendor credentials."""
import asyncio
from copy import deepcopy
import json


async def multiplex_probe(runner, authority, primary, state, mode, peer):
    from agent.secret_scope import set_multiplex_active, get_secret
    from hermes_constants import get_hermes_home
    from run_agent import AIAgent

    # Observe the actual worker entry without replacing its implementation or auth.
    worker_scopes = []
    original = AIAgent.run_conversation

    def observed(self, *args, **kwargs):
        worker_scopes.append((str(get_hermes_home()), get_secret('OPENAI_API_KEY')))
        return original(self, *args, **kwargs)

    AIAgent.run_conversation = observed
    from gateway.config import Platform
    from gateway.platforms.event import MessageEvent
    from gateway.profile_routing import ProfileRoute
    from hermes_state_runtime import admit_session_input, list_session_admissions

    transport = state / 'profiles' / 'transport'
    transport.mkdir(parents=True, exist_ok=True)
    (state / '.env').write_text('TELEGRAM_ALLOWED_USERS=runtime-only\nOPENAI_API_KEY=runtime-fixture-key\n')
    (transport / '.env').write_text('TELEGRAM_ALLOWED_USERS=fixture-user\nOPENAI_API_KEY=transport-decoy-key\n')
    (transport / 'config.yaml').write_text((state / 'config.yaml').read_text())
    runner.config.multiplex_profiles = True
    runner.config._runtime_profile_homes = [('default', state), ('transport', transport)]
    route = ProfileRoute(name='to-runtime', platform='telegram', profile='default', chat_id='mux-chat',
                         bot_profile='transport')
    runner.config.profile_routes = [route]
    set_multiplex_active(True)
    adapter = type(primary)()
    adapter.config.token = 'secondary-connector-fixture'
    adapter.gateway_runner = runner
    runner._profile_adapters = {'transport': {Platform.TELEGRAM: adapter}}
    runner._configure_profile_adapter(adapter, 'transport', Platform.TELEGRAM)
    source = adapter.build_source(chat_id='mux-chat', chat_type='dm', user_id='fixture-user')

    def rows(sid):
        return list_session_admissions(authority.db, session_id=sid, pending_only=False)

    if mode.startswith('recover'):
        saved = json.loads((state / 'multiplex.json').read_text())
        sid = saved['sid']
        before = rows(sid)
        rejected = {}
        if mode == 'recover':
            # Each mutation is fixture-owned; restoration precedes the positive control.
            for case in ('removed-adapter', 'ambiguous-adapter', 'removed-home', 'ambiguous-route',
                         'removed-route', 'changed-credential', 'revoked-sender'):
                old_token = adapter.config.token
                if case == 'removed-adapter':
                    runner._profile_adapters = {}
                elif case == 'ambiguous-adapter':
                    runner._profile_adapters['alias'] = {Platform.TELEGRAM: adapter}
                elif case == 'removed-home':
                    runner.config._runtime_profile_homes = [('default', state)]
                elif case == 'ambiguous-route':
                    runner.config.profile_routes = [route, route]
                elif case == 'removed-route':
                    runner.config.profile_routes = []
                elif case == 'changed-credential':
                    adapter.config.token = 'new-connector'
                elif case == 'revoked-sender':
                    (transport / '.env').write_text('TELEGRAM_ALLOWED_USERS=revoked\n')
                result = await authority.recover_native_sessions([(sid, source, adapter)])
                assert result[sid] != 'ready', (case, result)
                assert rows(sid) == before, (case, rows(sid))
                rejected[case] = result[sid]
                runner._profile_adapters = {'transport': {Platform.TELEGRAM: adapter}}
                runner.config._runtime_profile_homes = [('default', state), ('transport', transport)]
                runner.config.profile_routes = [route]
                adapter.config.token = old_token
                (transport / '.env').write_text('TELEGRAM_ALLOWED_USERS=fixture-user\nOPENAI_API_KEY=transport-decoy-key\n')
        result = await authority.recover_native_sessions([(sid, source, adapter)])
        assert result[sid] == 'ready', result
        await authority.sessions[sid].task
        assert all(row['status'] == 'terminal' and row['outcome'] == 'completed' for row in rows(sid)), rows(sid)
        expected_calls = 1 if mode == 'recover' else 0
        assert len(peer.requests) == expected_calls, peer.requests
        assert worker_scopes == [(str(state), 'runtime-fixture-key')] * expected_calls, worker_scopes
        print(json.dumps({'mode': mode, 'rejected': rejected, 'model_calls': len(peer.requests),
                          'rows': rows(sid), 'deliveries': adapter.deliveries}))
        return

    event = MessageEvent(text='MULTIPLEX_NATIVE', source=source, message_id='mux-1')
    await adapter.handle_message(event)
    async with asyncio.timeout(15):
        while adapter._active_sessions:
            await asyncio.sleep(0.01)
    entry = runner.session_store.get_or_create_session(source)
    committed = rows(entry.session_id)
    assert committed and committed[0]['outcome'] == 'completed', (committed, adapter.deliveries)
    provenance = committed[0]['payload']['native_text_v1']['provenance']
    assert provenance['transport_home'] == str(transport.resolve()), provenance
    assert provenance['runtime_home'] == str(state.resolve()), provenance
    assert worker_scopes == [(str(state), 'runtime-fixture-key')], worker_scopes
    assert len(adapter.deliveries) and not primary.deliveries, (adapter.deliveries, primary.deliveries)
    # Runtime's allowed sender is explicitly NOT allowed by this transport.
    denied = MessageEvent(text='WRONG_TRANSPORT_SENDER', message_id='deny', source=adapter.build_source(
        chat_id='mux-chat', chat_type='dm', user_id='runtime-only'))
    await adapter.handle_message(denied)
    async with asyncio.timeout(10):
        while adapter._active_sessions:
            await asyncio.sleep(0.01)
    assert rows(entry.session_id) == committed
    if mode == 'multiplex':
        sessions_before = set(authority.sessions)
        runner.config.profile_routes.append(ProfileRoute(name='unowned-runtime', platform='telegram',
                                                         profile='transport', chat_id='unowned-runtime', bot_profile='transport'))
        unowned = adapter.build_source(chat_id='unowned-runtime', chat_type='dm', user_id='fixture-user')
        await adapter.handle_message(MessageEvent(text='NO_SECONDARY_DB_OWNER', source=unowned, message_id='unowned'))
        async with asyncio.timeout(10):
            while adapter._active_sessions:
                await asyncio.sleep(0.01)
        assert set(authority.sessions) == sessions_before, 'launch DB impersonated secondary runtime'
        runner.config.profile_routes = [route]
        await adapter.handle_message(MessageEvent(text='BLOCK_FIFO', source=source, message_id='busy-1'))
        assert await asyncio.to_thread(peer.blocked.wait, 5), 'model gate not reached'
        queued = MessageEvent(text='MULTIPLEX_QUEUED', source=source, message_id='busy-2')
        task = asyncio.create_task(adapter.handle_message(queued))
        try:
            async with asyncio.timeout(3):
                while not any(row['request_id'] == 'busy-2' for row in rows(entry.session_id)):
                    await asyncio.sleep(0.01)
            assert not adapter._pending_messages, 'callback wrapper restored adapter-local queue'
            (state / '.env').write_text('TELEGRAM_ALLOWED_USERS=runtime-only\nOPENAI_API_KEY=runtime-fresh-key\n')
        finally:
            peer.release.set()
            await task
            async with asyncio.timeout(10):
                while adapter._active_sessions:
                    await asyncio.sleep(0.01)
        assert all(row['outcome'] == 'completed' for row in rows(entry.session_id))
        assert worker_scopes == [(str(state), 'runtime-fixture-key')] * 2 + [(str(state), 'runtime-fresh-key')], worker_scopes
    if mode == 'capture':
        # A queued storage fixture derived from the actual committed callback envelope.
        # This is restart reauthorization evidence, not a native post-ACK kill claim.
        payload = deepcopy(committed[0]['payload'])
        payload['text'] = 'RESTART_SOURCE_REAUTH'
        admit_session_input(authority.db, epoch=authority.epoch, principal_id=committed[0]['principal_id'],
                            session_id=entry.session_id, request_id='restart-fixture', payload=payload)
        (state / 'multiplex.json').write_text(json.dumps({'sid': entry.session_id}))
    print(json.dumps({'mode': mode, 'model_calls': len(peer.requests), 'rows': rows(entry.session_id),
                      'deliveries': adapter.deliveries, 'worker_scopes': worker_scopes}))
