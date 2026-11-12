"""A model's queued speed is committed with its destination, never the running route."""
import asyncio
from contextlib import asynccontextmanager
import json
import threading
from types import SimpleNamespace

import pytest

from tests.gateway.test_session_mutation_model_busy import _owner, _run_turn, _running_model
from tests.waiting import async_wait_until


@asynccontextmanager
async def _busy_owner(tmp_path, monkeypatch, *, guarded=False, initial_tier=None, entitled=False):
    from gateway.session_contract import Submission
    from gateway.session_policy import policy_for_source
    from gateway.session_state import SERVICE_TIER_UNSET
    from hermes_cli import model_switch

    monkeypatch.chdir(tmp_path)
    profile = tmp_path / 'owned'
    profile.mkdir()
    (profile / '.env').write_text('OPENAI_API_KEY=test-key\n')
    started, release = threading.Event(), asyncio.Event()
    state, seen, tiers, events = {}, [], {}, []

    async def handle(event):
        authority, ref = state['authority'], state['ref']
        live = authority.sessions[ref.session_id]
        policy = policy_for_source(authority.runner, live.source)
        seen.append((event.text, _running_model(authority, ref),
                     (policy.current_reasoning_config(authority.runner, live.route) or {}).get('effort'),
                     policy.current_service_tier(authority.runner, live.route)))
        if event.text == 'RUNNING':
            started.set()
            await release.wait()
        return 'ok'

    store, db, authority, owner, ref = _owner(tmp_path, monkeypatch, handle)
    state.update(authority=authority, ref=ref)
    live = authority.sessions[ref.session_id]
    if initial_tier is not None:
        tiers[live.route] = initial_tier
    authority.runner._peek_session_state = lambda route: SimpleNamespace(conversation=SimpleNamespace(
        service_tier_override=tiers.get(route, SERVICE_TIER_UNSET), reasoning_override=None))

    def set_tier(route, tier, clear=False):
        if clear:
            tiers.pop(route, None)
        else:
            tiers[route] = tier
    authority.runner._set_session_service_tier_override = set_tier
    authority.runner._set_session_reasoning_override = lambda *a: None

    def resolve(**kwargs):
        model = kwargs['raw_input']
        return model_switch.ModelSwitchResult(success=model != 'gone', new_model=model, target_provider='openai-api',
                                             base_url='https://api.openai.com/v1')
    monkeypatch.setattr(model_switch, 'switch_model', resolve)
    if entitled is not None:
        monkeypatch.setattr('agent.codex_service_tiers.ultrafast_offered', lambda *a, **k: entitled)
    publish = live.event_stream.publish
    monkeypatch.setattr(live.event_stream, 'publish', lambda sid, payload, **k: (
        events.append((k.get('event_type'), payload)), publish(sid, payload, **k))[1])
    if guarded:
        from agent.usage_anchor import USAGE_ANCHOR_MODEL_CONFIG_KEY, capture_usage_anchor
        monkeypatch.setattr('hermes_cli.model_selection_guards._context_cache_threshold', lambda: 1000)
        db.append_message(ref.session_id, role='user', content='hello')
        db.append_message(ref.session_id, role='assistant', content='hi')
        db.patch_session_model_config(ref.session_id, {USAGE_ANCHOR_MODEL_CONFIG_KEY: capture_usage_anchor(
            250_000, 10, db.get_messages_as_conversation(ref.session_id))})

    async def pick(request_id, payload, operation='model'):
        row = db.get_session(ref.session_id)
        return await owner.dispatch({'id': request_id, 'method': 'session.mutate', 'params': {
            'session_id': ref.session_id, 'request_id': request_id, 'expected_revision': row['runtime_revision'],
            'expected_generation': row['runtime_generation'], 'operation': operation, 'payload': payload}})

    async def fast(value):
        return await owner.dispatch({'id': 'fast', 'method': 'config.set', 'params': {
            'session_id': ref.session_id, 'key': 'fast', 'value': value}})

    async def next_turn():
        release.set()
        await _run_turn(authority, owner, db, ref, 'NEXT')

    try:
        first = await authority.submit(owner.actor, Submission('running', ref, {'text': 'RUNNING'}, 'queue'))
        await async_wait_until(started.is_set, timeout=10, what='the first turn is running')
        yield SimpleNamespace(store=store, db=db, authority=authority, owner=owner, ref=ref, live=live,
                              seen=seen, tiers=tiers, events=events, release=release, first=first, state=state,
                              pick=pick, fast=fast, next_turn=next_turn)
    finally:
        release.set()
        await owner.close()
        store.close_all_db_handles()


def _pending(h):
    from hermes_state_mutation_prepared import DEFERRED_PICK_PREFIX
    with h.db._read_ctx() as conn:
        row = conn.execute('SELECT value FROM state_meta WHERE key=?',
                           (DEFERRED_PICK_PREFIX + h.ref.session_id,)).fetchone()
    return json.loads(row[0]) if row else None


@pytest.mark.asyncio
@pytest.mark.parametrize('order, words, expected', [
    ('before', ['fast'], 'priority'),
    ('after', ['fast'], 'priority'),
    ('after', ['fast', 'normal'], None),
    ('after', ['toggle', 'toggle'], None),
    ('after', ['ultrafast'], 'ultrafast'),
], ids=['reasoning-first', 'reasoning-last', 'fast-off', 'toggle-twice', 'ultrafast'])
async def test_model_reasoning_and_speed_land_together_without_changing_the_running_turn(
        tmp_path, monkeypatch, order, words, expected):
    async with _busy_owner(tmp_path, monkeypatch, entitled=True) as h:
        assert (await h.pick('model', {'model': 'gpt-6.1-sol', 'provider': 'openai-api'}))['result']['status'] == 'deferred'
        if order == 'before':
            assert (await h.pick('level', {'effort': 'high'}, 'reasoning'))['result']['status'] == 'deferred'
        for word in words:
            reply = await h.fast(word)
            assert reply.get('result', {}).get('deferred') is True, reply
        if order == 'after':
            assert (await h.pick('level', {'effort': 'high'}, 'reasoning'))['result']['status'] == 'deferred'
        assert _running_model(h.authority, h.ref) == 'frozen'
        assert not h.tiers
        assert h.seen == [('RUNNING', 'frozen', None, None)]
        await h.next_turn()
        assert h.seen[-1] == ('NEXT', 'gpt-6.1-sol', 'high', expected)
        assert _pending(h) is None


@pytest.mark.asyncio
@pytest.mark.parametrize('kind, model, word, expected_model', [
    ('resolution', 'gone', 'fast', 'frozen'),
    ('unsupported', 'no-fast-model', 'fast', 'frozen'),
    ('entitlement', 'gpt-6.1-sol', 'ultrafast', 'frozen'),
    ('superseded', 'gpt-5.4', 'fast', 'no-fast-model'),
], ids=['resolver-refuses', 'fast-unavailable', 'ultrafast-unavailable', 'new-model-drops-speed'])
async def test_refused_or_replaced_destination_never_leaves_its_speed_on_the_old_model(
        tmp_path, monkeypatch, kind, model, word, expected_model):
    async with _busy_owner(tmp_path, monkeypatch) as h:
        assert (await h.pick('model', {'model': model, 'provider': 'openai-api'}))['result']['status'] == 'deferred'
        reply = await h.fast(word)
        assert reply.get('result', {}).get('deferred') is True, reply
        if kind == 'superseded':
            assert (await h.pick('new-model', {'model': expected_model, 'provider': 'openai-api'}))['result']['status'] == 'deferred'
        assert not h.tiers
        await h.next_turn()
        assert h.seen[-1] == ('NEXT', expected_model, None, None)
        assert _pending(h) is None
        if kind != 'superseded':
            assert any(t == 'session.updated' and p.get('status') == 'refused' for t, p in h.events)


@pytest.mark.asyncio
async def test_duplicate_fast_and_original_model_retry_preserve_the_latest_durable_choice(tmp_path, monkeypatch):
    async with _busy_owner(tmp_path, monkeypatch) as h:
        payload = {'model': 'gpt-5.4', 'provider': 'openai-api'}
        original = await h.pick('model', payload)
        assert original['result']['status'] == 'deferred'
        assert (await h.fast('fast'))['result']['deferred'] is True
        pending = _pending(h)
        assert (await h.fast('on'))['result']['deferred'] is True
        assert _pending(h) == pending
        # A retry uses the original fences: its durable receipt must win over current state.
        retry = await h.pick('model', payload)
        assert retry == original
        assert _pending(h) == pending
        assert (await h.fast('normal'))['result']['deferred'] is True
        assert _pending(h)['application_id'] != pending['application_id']
        await h.next_turn()
        assert h.seen[-1] == ('NEXT', 'gpt-5.4', None, None)


@pytest.mark.asyncio
@pytest.mark.parametrize('confirm_pick', ['unanswered', 'full', 'legacy'])
async def test_guarded_model_keeps_speed_pending_until_the_same_selection_is_confirmed(
        tmp_path, monkeypatch, confirm_pick):
    async with _busy_owner(tmp_path, monkeypatch, guarded=True) as h:
        assert (await h.pick('model', {'model': 'gpt-5.4', 'provider': 'openai-api'}))['result']['status'] == 'deferred'
        assert (await h.fast('fast'))['result']['deferred'] is True
        h.release.set()
        asks = lambda: [p for t, p in h.events if t == 'session.updated' and p.get('status') == 'confirmation_required']
        await async_wait_until(asks, timeout=10, what='the destination asks for consent')
        ask = asks()[-1]
        assert _running_model(h.authority, h.ref) == 'frozen'
        assert not h.tiers
        if confirm_pick != 'unanswered':
            payload = {**ask['payload'], 'confirm': ask['confirm']}
            if confirm_pick == 'legacy':
                payload.pop('service_tier')
            row = h.db.get_session(h.ref.session_id)
            request = {'id': 'confirmed', 'method': 'session.mutate', 'params': {
                'session_id': h.ref.session_id, 'request_id': 'confirmed',
                'expected_revision': row['runtime_revision'], 'expected_generation': row['runtime_generation'],
                'operation': 'model', 'payload': payload}}
            reply = await h.owner.dispatch(request)
            assert reply.get('result', {}).get('model') == 'gpt-5.4', reply
            assert reply['result']['service_tier'] == 'fast'
            if confirm_pick == 'legacy':
                assert 'service_tier' not in payload
        await _run_turn(h.authority, h.owner, h.db, h.ref, 'NEXT')
        expected = ('gpt-5.4', 'priority') if confirm_pick != 'unanswered' else ('frozen', None)
        assert (h.seen[-1][1], h.seen[-1][3]) == expected
        if confirm_pick == 'legacy':
            assert _pending(h) is None
            assert (await h.fast('normal'))['result']['value'] == 'normal'
            assert await h.owner.dispatch(request) == reply
            assert h.tiers[h.live.route] is None
            await _run_turn(h.authority, h.owner, h.db, h.ref, 'AFTER-REPLAY')
            assert h.seen[-1] == ('AFTER-REPLAY', 'gpt-5.4', None, None)


@pytest.mark.asyncio
@pytest.mark.parametrize('legacy', [False, True], ids=['full', 'legacy'])
async def test_changing_queued_speed_invalidates_the_previous_confirmation(tmp_path, monkeypatch, legacy):
    async with _busy_owner(tmp_path, monkeypatch, guarded=True) as h:
        assert (await h.pick('model', {'model': 'gpt-5.4', 'provider': 'openai-api'}))['result']['status'] == 'deferred'
        assert (await h.fast('fast'))['result']['deferred'] is True
        h.release.set()
        asks = lambda: [p for t, p in h.events if t == 'session.updated' and p.get('status') == 'confirmation_required']
        await async_wait_until(asks, timeout=10, what='the initial destination asks for consent')
        old = asks()[-1]
        assert (await h.fast('normal'))['result']['deferred'] is True
        await async_wait_until(lambda: len(asks()) >= 2, timeout=10, what='the changed speed asks again')
        current = asks()[-1]
        assert current['confirm'] != old['confirm']
        payload = {**old['payload'], 'confirm': old['confirm']}
        if legacy:
            payload.pop('service_tier')
        stale = await h.pick('stale-confirm', payload)
        assert stale.get('result', {}).get('status') == 'confirmation_required', stale
        assert _running_model(h.authority, h.ref) == 'frozen'
        assert not h.tiers
        confirmed = await h.pick('confirmed', {**current['payload'], 'confirm': current['confirm']})
        assert confirmed.get('result', {}).get('model') == 'gpt-5.4', confirmed
        await _run_turn(h.authority, h.owner, h.db, h.ref, 'NEXT')
        assert h.seen[-1] == ('NEXT', 'gpt-5.4', None, None)


@pytest.mark.asyncio
@pytest.mark.parametrize('choice', ['same-model', 'wrong-model', 'wrong-provider'])
async def test_new_or_mismatched_model_choices_do_not_inherit_pending_speed(tmp_path, monkeypatch, choice):
    async with _busy_owner(tmp_path, monkeypatch, guarded=True) as h:
        payload = {'model': 'gpt-5.4', 'provider': 'openai-api'}
        assert (await h.pick('model', payload))['result']['status'] == 'deferred'
        assert (await h.fast('fast'))['result']['deferred'] is True
        h.release.set()
        asks = lambda: [p for t, p in h.events if t == 'session.updated' and p.get('status') == 'confirmation_required']
        await async_wait_until(asks, timeout=10, what='the queued destination asks for consent')
        ask, pending = asks()[-1], _pending(h)
        selected = dict(payload)
        if choice != 'same-model':
            selected['model' if choice == 'wrong-model' else 'provider'] = (
                'gpt-6.1-sol' if choice == 'wrong-model' else 'other-provider')
            selected['confirm'] = ask['confirm']
        reply = await h.pick('new-choice', selected)
        assert reply.get('result', {}).get('status') == 'confirmation_required', reply
        assert reply['result']['confirm'] != ask['confirm']
        assert _pending(h) == pending
        assert _running_model(h.authority, h.ref) == 'frozen'
        assert not h.tiers
        # Fresh consent to this new choice commits only what the new request selected.
        selected['confirm'] = reply['result']['confirm']
        confirmed = await h.pick('new-confirmed', selected)
        assert confirmed.get('result', {}).get('model') == selected['model'], confirmed
        assert 'service_tier' not in confirmed['result']
        await _run_turn(h.authority, h.owner, h.db, h.ref, 'NEXT')
        assert h.seen[-1] == ('NEXT', selected['model'], None, None)


@pytest.mark.asyncio
async def test_pending_speed_survives_a_fresh_authority_and_empty_runner_state(tmp_path, monkeypatch):
    from gateway import session_deferred_pick
    from gateway.session_authority import SessionAuthority
    from gateway.session_controls import AuthorityConnection
    from gateway.session_local_recovery import restore_local_session
    import hermes_state_runtime as rt

    async def hold_pick(*args):
        pass

    async with _busy_owner(tmp_path, monkeypatch) as h:
        assert (await h.pick('model', {'model': 'gpt-5.4', 'provider': 'openai-api'}))['result']['status'] == 'deferred'
        assert (await h.fast('fast'))['result']['deferred'] is True
        pending = _pending(h)
        apply = session_deferred_pick.apply_deferred_pick
        monkeypatch.setattr(session_deferred_pick, 'apply_deferred_pick', hold_pick)
        h.release.set()
        await async_wait_until(
            lambda: rt.get_session_admission(h.db, admission_id=h.first.admission_id)['status'] == 'terminal',
            timeout=10, what='the old owner settled its running turn')
        if h.live.task is not None:
            await h.live.task
        await h.owner.close()
        runner = SimpleNamespace(**vars(h.authority.runner))
        runner.adapters = {}
        runner._adapter_for_source = lambda source: runner.adapters.get(source.platform)
        h.tiers.clear()
        restarted = SessionAuthority(runner, profile_id=h.authority.profile_id, instance_id='restarted', db=h.db,
                                     epoch=rt.begin_runtime_epoch(h.db, instance_id='restarted'))
        runner.session_authority = restarted
        owner = AuthorityConnection(restarted, object(), {'user_id': 'human'})
        ref = restore_local_session(restarted, h.ref.session_id)
        h.state.update(authority=restarted, ref=ref)
        assert _pending(h) == pending
        monkeypatch.setattr(session_deferred_pick, 'apply_deferred_pick', apply)
        try:
            await _run_turn(restarted, owner, h.db, ref, 'NEXT')
            assert h.seen[-1] == ('NEXT', 'gpt-5.4', None, 'priority')
            assert _pending(h) is None
        finally:
            await owner.close()


@pytest.mark.asyncio
async def test_a_following_session_durably_pins_the_picked_tier_over_profile_and_older_override(tmp_path, monkeypatch):
    from gateway.session_policy import policy_for_source, restore_policy
    from gateway.session_state import SERVICE_TIER_UNSET
    from hermes_cli import model_switch
    from hermes_state_local import local_receipt
    from tests.gateway.test_session_create_tier_cwd import _owner as configured_owner

    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv('OPENAI_API_KEY', 'test-key')
    config = {'model': {'default': 'gpt-5.4', 'provider': 'openai-api', 'base_url': 'https://api.openai.com/v1'},
              'agent': {'service_tier': 'fast'}}
    store, runner, authority, owner = configured_owner(tmp_path / 'home', monkeypatch, config)
    tiers = {}
    runner._peek_session_state = lambda route: SimpleNamespace(conversation=SimpleNamespace(
        service_tier_override=tiers.get(route, SERVICE_TIER_UNSET), reasoning_override=None))
    runner._set_session_service_tier_override = lambda route, tier, clear=False: (
        tiers.pop(route, None) if clear else tiers.__setitem__(route, tier))
    runner._resolve_session_agent_runtime = lambda **k: ('gpt-5.4', {'provider': 'openai-api', 'base_url': 'https://api.openai.com/v1'})
    monkeypatch.setattr(model_switch, 'switch_model', lambda **kw: model_switch.ModelSwitchResult(
        success=True, new_model=kw['raw_input'], target_provider='openai-api', base_url='https://api.openai.com/v1'))
    try:
        created = await owner.dispatch({'id': 'create', 'method': 'session.create', 'params': {
            'request_id': 'follow', 'source': 'gui', 'cwd': str(tmp_path), 'toolsets': [], 'follow_profile_config': True}})
        sid = created['result']['session_id']
        live = authority.sessions[sid]
        tiers[live.route] = 'ultrafast'
        row = authority.db.get_session(sid)
        reply = await owner.dispatch({'id': 'pick', 'method': 'session.mutate', 'params': {
            'session_id': sid, 'request_id': 'model', 'expected_revision': row['runtime_revision'],
            'expected_generation': row['runtime_generation'], 'operation': 'model',
            'payload': {'model': 'gpt-5.4', 'provider': 'openai-api', 'service_tier': 'normal'}}})
        assert reply.get('result', {}).get('model') == 'gpt-5.4', reply
        policy = policy_for_source(runner, live.source)
        assert policy.current_service_tier(runner, live.route) is None
        assert live.route not in tiers
        durable = restore_policy(local_receipt(authority.db, sid)['policy'])
        assert durable.follows_profile
        assert durable.current_service_tier(runner, live.route) is None
    finally:
        await owner.close()
        store.close_all_db_handles()


@pytest.mark.asyncio
@pytest.mark.parametrize('same_model', [False, True], ids=['queued-reset', 'already-default-reset'])
async def test_reset_keeps_an_explicit_tier_even_when_the_default_model_is_already_selected(
        tmp_path, monkeypatch, same_model):
    monkeypatch.setattr('gateway.session_mutation_model.configured_default', lambda authority: (
        'frozen' if same_model else 'gpt-5.4', '' if same_model else 'openai-api'))
    async with _busy_owner(tmp_path, monkeypatch, initial_tier='priority' if same_model else None) as h:
        if same_model:
            h.release.set()
            await _run_turn(h.authority, h.owner, h.db, h.ref, 'SETTLED')
            result = await h.pick('reset', {'model': 'reset', 'service_tier': 'normal'})
            assert result.get('result', {}).get('service_tier') == 'normal', result
            await _run_turn(h.authority, h.owner, h.db, h.ref, 'NEXT')
            assert h.seen[-1][3] is None
        else:
            assert (await h.pick('reset', {'model': 'reset'}))['result']['status'] == 'deferred'
            assert (await h.fast('fast'))['result']['deferred'] is True
            await h.next_turn()
            assert h.seen[-1] == ('NEXT', 'gpt-5.4', None, 'priority')


@pytest.mark.asyncio
async def test_fast_waiting_for_the_boundary_validates_the_newly_committed_route(tmp_path, monkeypatch):
    from gateway import session_tui_config
    import hermes_state_runtime as rt
    async with _busy_owner(tmp_path, monkeypatch) as h:
        assert (await h.pick('model', {'model': 'gpt-5.4', 'provider': 'openai-api'}))['result']['status'] == 'deferred'
        boundary, authorized = asyncio.Event(), asyncio.Event()
        claim, authorize = h.authority._claim_next, session_tui_config._authorize_write

        async def claim_at_boundary(*args):
            boundary.set()
            return await claim(*args)

        def authorize_fast(*args):
            policy = authorize(*args)
            authorized.set()
            return policy
        monkeypatch.setattr(h.authority, '_claim_next', claim_at_boundary)
        monkeypatch.setattr(session_tui_config, '_authorize_write', authorize_fast)
        await h.live.mutation_lock.acquire()
        task = None
        try:
            h.release.set()
            await async_wait_until(
                lambda: rt.get_session_admission(h.db, admission_id=h.first.admission_id)['status'] == 'terminal',
                timeout=10, what='the running turn settles before its boundary')
            await asyncio.wait_for(boundary.wait(), 10)
            task = asyncio.create_task(h.fast('fast'))
            await asyncio.wait_for(authorized.wait(), 10)
        finally:
            h.live.mutation_lock.release()
        try:
            reply = await task
            assert reply.get('result', {}).get('value') == 'fast', reply
            assert _running_model(h.authority, h.ref) == 'gpt-5.4'
            await _run_turn(h.authority, h.owner, h.db, h.ref, 'NEXT')
            assert h.seen[-1] == ('NEXT', 'gpt-5.4', None, 'priority')
        finally:
            if task is not None and not task.done():
                task.cancel()


@pytest.mark.asyncio
async def test_replaying_a_compound_model_receipt_preserves_a_later_speed_choice(tmp_path, monkeypatch):
    async with _busy_owner(tmp_path, monkeypatch) as h:
        h.release.set()
        await _run_turn(h.authority, h.owner, h.db, h.ref, 'SETTLED')
        row = h.db.get_session(h.ref.session_id)
        request = {'id': 'compound', 'method': 'session.mutate', 'params': {
            'session_id': h.ref.session_id, 'request_id': 'compound',
            'expected_revision': row['runtime_revision'], 'expected_generation': row['runtime_generation'],
            'operation': 'model', 'payload': {'model': 'gpt-5.4', 'provider': 'openai-api', 'service_tier': 'fast'}}}
        first = await h.owner.dispatch(request)
        assert first.get('result', {}).get('model') == 'gpt-5.4', first
        assert (await h.fast('normal'))['result']['value'] == 'normal'
        assert h.live.route in h.tiers and h.tiers[h.live.route] is None
        assert await h.owner.dispatch(request) == first
        assert h.live.route in h.tiers and h.tiers[h.live.route] is None
        await _run_turn(h.authority, h.owner, h.db, h.ref, 'NEXT')
        assert h.seen[-1] == ('NEXT', 'gpt-5.4', None, None)


@pytest.mark.asyncio
@pytest.mark.parametrize('selected_entitled', [False, True], ids=['selected-pool-denied', 'selected-pool-allowed'])
async def test_ultrafast_uses_the_destination_pools_account_instead_of_the_profiles_singleton(
        tmp_path, monkeypatch, selected_entitled):
    from agent import codex_service_tiers, model_metadata
    from gateway import session_worker_construct
    from hermes_cli import auth_codex, model_switch
    from tests.agent.test_codex_service_tiers import token
    base = 'https://chatgpt.com/backend-api/codex'
    tokens = {'pool': token('pool'), 'singleton': token('singleton')}
    monkeypatch.setattr(codex_service_tiers, '_catalogs', {})
    for account, entitled in [('pool', selected_entitled), ('singleton', not selected_entitled)]:
        tiers = ['priority', 'ultrafast'] if entitled else ['priority']
        key = model_metadata._codex_oauth_token_fingerprint(tokens[account], base)
        codex_service_tiers.remember_catalog(key, [{'slug': 'gpt-6.1-sol',
            'service_tiers': [{'id': tier} for tier in tiers]}])
    monkeypatch.setattr(auth_codex, 'resolve_codex_runtime_credentials', lambda **k: {'api_key': tokens['singleton']})
    resolved = []

    def route_runtime(policy, frozen, model, key, *, fallback=True):
        resolved.append((model, fallback))
        return {'provider': 'openai-codex', 'base_url': base, 'api_key': tokens['pool']}, None
    monkeypatch.setattr(session_worker_construct, 'route_runtime', route_runtime)
    async with _busy_owner(tmp_path, monkeypatch, entitled=None) as h:
        monkeypatch.setattr(model_switch, 'switch_model', lambda **kw: model_switch.ModelSwitchResult(
            success=True, new_model=kw['raw_input'], target_provider='openai-codex', base_url=base))
        assert (await h.pick('model', {'model': 'gpt-6.1-sol', 'provider': 'openai-codex'}))['result']['status'] == 'deferred'
        assert (await h.fast('ultrafast'))['result']['deferred'] is True
        await h.next_turn()
        assert resolved == [('gpt-6.1-sol', False)]
        expected = ('gpt-6.1-sol', 'ultrafast') if selected_entitled else ('frozen', None)
        assert (h.seen[-1][1], h.seen[-1][3]) == expected
