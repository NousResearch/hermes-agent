

import json


async def exercise_control_boundaries(tmp_path, monkeypatch):
    """Real tools waiters and TurnRunner callback, no alternative approval queue."""
    import asyncio
    from dataclasses import replace
    from types import SimpleNamespace
    from gateway.run_turn_runner import TurnRunner
    from gateway.session_authority import LiveSession, SessionAuthority
    from gateway.session_contract import Principal, SessionRef
    from hermes_state import SessionDB
    from hermes_state_runtime import (RuntimeStoreError, admit_session_input, begin_runtime_epoch,
                                     claim_session_input, settle_session_input)
    from tools.approval import list_gateway_approvals, resolve_gateway_approval
    from tools.approval_gateway_wait import _await_gateway_decision

    db = SessionDB(tmp_path / 'controls.db')
    db.create_session('s', source='telegram')
    epoch = begin_runtime_epoch(db, instance_id='controls')
    admit_session_input(db, epoch=epoch, principal_id='owner', session_id='s', request_id='first', payload={'text': 'first'})
    row = claim_session_input(db, epoch=epoch, session_id='s')
    authority = SessionAuthority(SimpleNamespace(), profile_id='p', instance_id='controls', db=db, epoch=epoch)
    live = authority.sessions['s'] = LiveSession(None, 'boundary-route')
    ref = SessionRef('p', 's')
    actor = Principal('operator', 'p', frozenset({'session:read', 'session:approve'}), 'a')
    snap = await authority.attach(actor, ref)

    class NativeCard:
        mode = 'sent'
        sends = 0
        def pause_typing_for_chat(self, chat_id):
            pass
        async def send_exec_approval(self, **kwargs):
            if self.mode == 'declined':
                return SimpleNamespace(success=False, error='fixture egress declined: target is not approved')
            return SimpleNamespace(success=True)
        async def send(self, *args, **kwargs):
            self.sends += 1
            return SimpleNamespace(success=True)

    adapter = NativeCard()
    ctx = SimpleNamespace(session_id='s', session_key='boundary-route', _status_adapter=adapter,
                          _status_chat_id='chat', _status_thread_metadata=None,
                          _loop_for_step=asyncio.get_running_loop(), stream_consumer_holder=[None])
    turn = TurnRunner(SimpleNamespace(session_authority=authority), ctx)

    async def wait_prompt():
        async with asyncio.timeout(5):
            while True:
                snapshot = await authority.attach(actor, ref)
                if snapshot.prompts:
                    return snapshot.prompts[-1]
                await asyncio.sleep(.01)

    tasks = []
    try:
        for n in range(2):
            task = asyncio.create_task(asyncio.to_thread(_await_gateway_decision, live.route,
                turn._approval_notify_sync, {'command': f'owned-command-{n}', 'description': 'fixture',
                'allow_session': False, 'allow_permanent': False, 'pattern_keys': [str(n)]}))
            tasks.append(task)
            async with asyncio.timeout(5):
                while len((await authority.attach(actor, ref)).prompts) < n + 1:
                    await asyncio.sleep(.01)
        prompts = (await authority.attach(actor, ref)).prompts
        import pytest
        for bad_actor, bad_ref, gen, reason in [
            (replace(actor, capabilities=frozenset({'session:read'})), ref, row['generation'], 'permission_denied'),
            (replace(actor, profile_id='foreign'), ref, row['generation'], 'profile_mismatch'),
            (actor, SessionRef('foreign', 's'), row['generation'], 'profile_mismatch'),
            (actor, SessionRef('p', 'missing'), row['generation'], 'not_found'),
            (replace(actor, transport_id='unattached'), ref, row['generation'], 'permission_denied'),
            (actor, ref, row['generation'] + 1, 'stale_generation'),
        ]:
            with pytest.raises(RuntimeStoreError) as exc:
                await authority.respond(bad_actor, bad_ref, gen, prompts[0]['prompt_id'], {'choice': 'once'})
            assert exc.value.reason == reason
        for answer in ({'choice': 'always'}, {'choice': 'session'}, {'answer': 'PRIVATE_SUDO_ANSWER'}, {'choice': []}):
            with pytest.raises(RuntimeStoreError) as exc:
                await authority.respond(actor, ref, row['generation'], prompts[0]['prompt_id'], answer)
            assert exc.value.reason == 'invalid_params'
        await authority.detach(actor, snap.subscription_id)
        assert len(list_gateway_approvals(live.route)) == 2
        await authority.attach(actor, ref)
        # Resolve second first: no FIFO consumption by a stale/native response.
        response = await authority.respond(actor, ref, row['generation'], prompts[1]['prompt_id'], {'choice': 'once'})
        assert response['status'] == 'resolved'
        assert (await tasks[1])['choice'] == 'once'
        assert not tasks[0].done()
        resolve_gateway_approval(live.route, 'deny', request_id=prompts[0]['prompt_id'])
        assert (await tasks[0])['choice'] == 'deny'
        assert not (await authority.attach(actor, ref)).prompts
        replay = live.event_stream.since(live.event_stream.epoch, 0)
        assert 'PRIVATE_SUDO_ANSWER' not in json.dumps(replay)

        # Native refusal cannot be converted into an observer approval path.
        adapter.mode = 'declined'
        before = live.event_stream.watermark()
        declined = await asyncio.to_thread(_await_gateway_decision, live.route, turn._approval_notify_sync,
                                          {'command': 'declined-fixture', 'pattern_keys': ['decline']})
        assert declined.get('notify_failed') and adapter.sends == 0
        assert live.event_stream.watermark() == before
        assert not list_gateway_approvals(live.route)
        adapter.mode = 'sent'
        from tools import approval_context
        with monkeypatch.context() as scoped:
            scoped.setattr(approval_context, '_get_approval_timeout', lambda: 0)
            timed_out = await asyncio.to_thread(_await_gateway_decision, live.route, turn._approval_notify_sync,
                                               {'command': 'timeout-fixture', 'pattern_keys': ['timeout']})
        assert timed_out['resolved'] is False and timed_out['choice'] is None
        assert not (await authority.attach(actor, ref)).prompts

        orphan = asyncio.create_task(asyncio.to_thread(_await_gateway_decision, live.route,
            turn._approval_notify_sync, {'command': 'orphaned-worker', 'pattern_keys': ['orphan']}))
        tasks.append(orphan)
        orphan_prompt = await wait_prompt()
        settle_session_input(db, epoch=epoch, admission_id=row['admission_id'], generation=row['generation'], outcome='completed')
        assert not (await authority.attach(actor, ref)).prompts, 'snapshot presents retired worker approval'
        with pytest.raises(RuntimeStoreError) as exc:
            await authority.respond(actor, ref, row['generation'], orphan_prompt['prompt_id'], {'choice': 'once'})
        assert exc.value.reason == 'stale_generation'
        resolve_gateway_approval(live.route, 'deny', request_id=orphan_prompt['prompt_id'])
        assert (await orphan)['choice'] == 'deny'
        assert not (await authority.attach(actor, ref)).prompts

        # A delayed old worker callback must not publish a fresh actionable card
        # after a new generation has claimed the session.
        admit_session_input(db, epoch=epoch, principal_id='owner', session_id='s', request_id='next', payload={'text': 'next'})
        newer = claim_session_input(db, epoch=epoch, session_id='s')
        from tools import approval_context
        monkeypatch.setattr(approval_context, "_get_approval_timeout", lambda: 0)
        before = live.event_stream.watermark()
        stale = await asyncio.to_thread(_await_gateway_decision, live.route, turn._approval_notify_sync,
                                      {'command': 'late-old-worker', 'pattern_keys': ['late']})
        assert stale.get('notify_failed'), 'retired worker registered a new shared approval'
        assert live.event_stream.watermark() == before, 'retired worker published an approval event'
        assert not (await authority.attach(actor, ref)).prompts
        settle_session_input(db, epoch=epoch, admission_id=newer['admission_id'], generation=newer['generation'], outcome='completed')
    finally:
        resolve_gateway_approval(live.route, 'deny', resolve_all=True)
        await asyncio.gather(*tasks)
        db.close()
