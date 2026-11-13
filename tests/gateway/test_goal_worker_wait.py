"""Persisted waits keep their barrier until the exact FIFO claim can execute."""
import pytest

from tests.gateway.test_session_end_delegations import _local, _admissions


@pytest.mark.asyncio
async def test_busy_worker_does_not_consume_queued_wait(tmp_path, monkeypatch):
    from gateway.run_goal_waits import poll_goal_waits
    from hermes_cli import goals
    from hermes_state_runtime import register_worker_execution
    runner, authority, ref, _actor = await _local(tmp_path, monkeypatch)
    try:
        monkeypatch.setattr(goals, '_get_session_db', lambda: authority.db)
        live = authority.sessions[ref.session_id]
        target = runner.session_store.lookup_by_session_key(live.route).session_id
        mgr = goals.GoalManager(target)
        mgr.set('do deferred work')
        mgr.wait_for_seconds(1)
        mgr.state.waiting_until = 1.0
        mgr._save()
        await poll_goal_waits(runner)
        assert len(_admissions(authority, ref)) == 1
        before = authority.db.get_meta('goal:' + target)
        register_worker_execution(authority.db, epoch=authority.epoch, execution_id='busy',
                                  session_id=ref.session_id, generation=0, kind='compute', adoption_secret='fixture')
        await authority._drain(ref)
        assert _admissions(authority, ref)[0]['status'] == 'queued'
        assert authority.db.get_meta('goal:' + target) == before, 'a blocked claim consumed the wait'
    finally:
        authority.db.close()


@pytest.mark.asyncio
@pytest.mark.parametrize('dependency,unrelated,silent,chained', [
    (True, False, False, False), (False, False, False, False), (True, True, False, False),
    (True, False, True, False), (True, False, False, True)])
async def test_real_completion_rejudges_only_its_worker_barrier(tmp_path, monkeypatch, dependency, unrelated, silent, chained):
    from types import SimpleNamespace
    from agent import auxiliary_client
    from gateway.session_results import execution_result
    from hermes_cli import goals
    from tools import async_delegation as delegation
    runner, authority, ref, _actor = await _local(tmp_path, monkeypatch)
    try:
        monkeypatch.setattr(goals, '_get_session_db', lambda: authority.db)
        live = authority.sessions[ref.session_id]
        target = runner.session_store.lookup_by_session_key(live.route).session_id
        event = {'type': 'async_delegation', 'delegation_id': 'owned-worker', 'session_key': live.route,
                 'parent_session_id': target, 'status': 'completed', 'summary': 'fixture work completed',
                 'dispatched_at': 1.0}
        delegation._persist_dispatch(event)
        manager = goals.GoalManager(target)
        manager.set('finish delegated work')
        judges, turns = [], []
        def judge(**kwargs):
            judges.append(kwargs)
            raw = ('{"verdict":"wait","wait_for_seconds":600,"wait_on_delegations":'
                   + ('true' if dependency else 'false') + '}') if len(judges) == 1 else '{"verdict":"done"}'
            if chained and len(judges) == 2:
                raw = '{"verdict":"continue"}'
            return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=raw))])
        monkeypatch.setattr(auxiliary_client, 'call_llm', judge)
        assert manager.evaluate_after_turn('waiting for delegated work')['verdict'] == 'wait'
        deadline = manager.state.waiting_until
        delegation._persist_completion(event, {'status': 'completed', 'summary': 'fixture work completed'})
        owned_event = event
        if unrelated:
            # The live count is still one, but that replacement was not in the judged snapshot.
            delegation._persist_dispatch(dict(event, delegation_id='replacement'))
            event = dict(event, delegation_id='unrelated-completion')
            delegation._persist_dispatch(event)
            delegation._persist_completion(event, {'status': 'completed', 'summary': 'unrelated work completed'})
        async def turn(event, source, key, generation):
            turns.append(event.text)
            text = '' if silent else 'delegated result checked'
            execution_result.get()['result'] = {'final_response': text, 'messages': [], 'failed': False}
            return text
        monkeypatch.setattr(runner, '_handle_message_with_agent', turn)
        assert await runner._deliver_async_delegation_group([event]) is True
        await authority._drain(ref)
        assert len(turns) == (2 if chained else 1)
        state = goals.GoalManager(target).state
        if dependency and not unrelated:
            assert len(judges) == (3 if chained else 2) and state.status == 'done', 'completion turn did not rejudge its worker barrier'
        else:
            assert len(judges) == 1 and state.waiting_until == deadline and state.status == 'active'
        if unrelated:
            assert await runner._deliver_async_delegation_group([owned_event]) is True
            await authority._drain(ref)
            assert len(turns) == len(judges) == 2 and goals.GoalManager(target).state.status == 'done'
        assert _admissions(authority, ref)[0]['outcome'] == 'completed'
        if chained:
            assert turns[1] == manager.next_continuation_prompt()
            assert [(r['status'], r['outcome']) for r in _admissions(authority, ref)] == [('terminal', 'completed')] * 2
    finally:
        authority.db.close()


@pytest.mark.asyncio
@pytest.mark.parametrize('control', ['stop', 'idle-exit'])
async def test_future_parked_goal_is_duty_until_explicit_stop(tmp_path, monkeypatch, control):
    from gateway.platforms.event import MessageEvent
    from gateway.run_idle_exit import _ledger_busy
    from hermes_cli import goals
    runner, authority, ref, _actor = await _local(tmp_path, monkeypatch)
    try:
        monkeypatch.setattr(goals, '_get_session_db', lambda: authority.db)
        live = authority.sessions[ref.session_id]
        target = runner.session_store.lookup_by_session_key(live.route).session_id
        mgr = goals.GoalManager(target)
        mgr.set('finish work later')
        mgr.wait_for_seconds(600)
        if control == 'stop':
            await runner._handle_message(MessageEvent(text='/stop', source=live.source))
            assert goals.GoalManager(target).state.status == 'paused', 'explicit Stop left autonomous goal duty active'
            assert _ledger_busy(runner) is None
        else:
            assert _ledger_busy(runner) is not None, 'idle exit forgot a persisted future goal timer'
            mgr.pause()
            assert _ledger_busy(runner) is None
    finally:
        authority.db.close()


@pytest.mark.asyncio
async def test_pause_after_postturn_admission_cancels_autonomous_followup(tmp_path, monkeypatch):
    from agent import auxiliary_client
    from types import SimpleNamespace
    from gateway.session_results import execution_result
    from hermes_cli import goals
    runner, authority, ref, _actor = await _local(tmp_path, monkeypatch)
    turns = []
    try:
        monkeypatch.setattr(goals, '_get_session_db', lambda: authority.db)
        live = authority.sessions[ref.session_id]
        entry = runner.session_store.lookup_by_session_key(live.route)
        goals.GoalManager(entry.session_id).set('finish work')
        monkeypatch.setattr(auxiliary_client, 'call_llm', lambda **k: SimpleNamespace(choices=[
            SimpleNamespace(message=SimpleNamespace(content='{"verdict":"continue"}'))]))
        async def turn(event, source, key, generation):
            turns.append(event.text)
            execution_result.get()['result'] = {'final_response': 'finished', 'messages': [], 'failed': False}
            return 'finished'
        monkeypatch.setattr(runner, '_handle_message_with_agent', turn)
        from tests.gateway.test_unviewed_completion_notes import _human
        _human(authority, ref, 'real-predecessor', 'working')
        original = authority.admit_automation
        async def admit_then_pause(*args, **kwargs):
            row = await original(*args, **kwargs)
            goals.GoalManager(entry.session_id).pause()
            return row
        monkeypatch.setattr(authority, 'admit_automation', admit_then_pause)
        await authority._drain(ref)
        assert len(_admissions(authority, ref)) == 1
        assert turns == ['working'], 'pause failed to fence the admitted postturn goal decision'
        assert _admissions(authority, ref)[0]['outcome'] == 'cancelled'
    finally:
        authority.db.close()


@pytest.mark.asyncio
@pytest.mark.parametrize('control', ['pause', 'clear', 'replace', 'reset', 'manual-wait', 'persisted-close', 'estop', 'human'])
async def test_reserved_goal_wake_rechecks_controls_at_claim(tmp_path, monkeypatch, control):
    from types import SimpleNamespace
    from agent import auxiliary_client, estop
    from gateway.run_goal_waits import poll_goal_waits
    from gateway.session_results import execution_result
    from gateway.session_viewer_close import _persist_close, _PendingClose
    from hermes_cli import goals
    from tests.gateway.test_unviewed_completion_notes import _human
    runner, authority, ref, _actor = await _local(tmp_path, monkeypatch)
    seen = []
    paused = [False]
    monkeypatch.setattr(estop, 'check_paused', lambda *args: paused[0])
    try:
        monkeypatch.setattr(goals, '_get_session_db', lambda: authority.db)
        live = authority.sessions[ref.session_id]
        target = runner.session_store.lookup_by_session_key(live.route).session_id
        mgr = goals.GoalManager(target)
        mgr.set('finish deferred work')
        mgr.wait_for_seconds(1)
        mgr.state.waiting_until = 1.0
        mgr._save()
        await poll_goal_waits(runner)
        assert len(_admissions(authority, ref)) == 1
        if control == 'reset':
            from dataclasses import replace
            from gateway.session_mutations import mutate_session
            row = authority.db.get_session(ref.session_id)
            actor = replace(_actor, capabilities=_actor.capabilities | {'session:control'})
            reset = await mutate_session(authority, actor, ref, dict(session_id=ref.session_id,
                request_id='reset', expected_revision=row['runtime_revision'],
                expected_generation=row['runtime_generation'], operation='reset', payload={}))
            assert reset['target_session_id'] != target
        else:
            {'pause': lambda: goals.GoalManager(target).pause(),
             'clear': lambda: goals.GoalManager(target).clear(),
             'replace': lambda: goals.GoalManager(target).set('new user goal'),
             'manual-wait': lambda: goals.GoalManager(target).wait_for_seconds(900),
             'persisted-close': lambda: _persist_close(authority, ref.session_id, _PendingClose('tui_close').record),
             'estop': lambda: paused.__setitem__(0, True),
             'human': lambda: _human(authority, ref, 'human', 'HUMAN_FIRST')}[control]()
        before = authority.db.get_meta('goal:' + target)
        async def turn(event, source, key, generation):
            seen.append(event.text)
            execution_result.get()['result'] = {'final_response': 'finished', 'messages': [], 'failed': False}
            return 'finished'
        monkeypatch.setattr(runner, '_handle_message_with_agent', turn)
        monkeypatch.setattr(auxiliary_client, 'call_llm', lambda **k: SimpleNamespace(choices=[
            SimpleNamespace(message=SimpleNamespace(content='{"verdict":"done"}'))]))
        await authority._drain(ref)
        if control == 'human':
            assert seen == ['HUMAN_FIRST']
            assert goals.GoalManager(target).state.status == 'done'
        else:
            assert seen == [] and authority.db.get_meta('goal:' + target) == before
        if control == 'estop':
            assert _admissions(authority, ref)[0]['status'] == 'queued'
            paused[0] = False
            await poll_goal_waits(runner)
            await authority._drain(ref)
            assert len(seen) == 1 and goals.GoalManager(target).state.status == 'done'
        else:
            assert _admissions(authority, ref)[0]['outcome'] == 'cancelled'
    finally:
        authority.db.close()


@pytest.mark.asyncio
@pytest.mark.parametrize('restart', [False, True])
@pytest.mark.parametrize('humans', [1, 31])
async def test_priority_yield_preserves_unchanged_wait_after_failed_human(tmp_path, monkeypatch, restart, humans):
    from types import SimpleNamespace
    from agent import auxiliary_client
    from gateway.config import GatewayConfig
    from gateway.run import GatewayRunner
    from gateway.run_goal_waits import poll_goal_waits
    from gateway.session_authority import initialize_session_authority
    from gateway.session_results import execution_result
    from hermes_cli import goals
    from hermes_state import SessionDB, AsyncSessionDB
    from tests.gateway.test_unviewed_completion_notes import _human
    runner, authority, ref, _actor = await _local(tmp_path, monkeypatch)
    seen, judges = [], []
    try:
        monkeypatch.setattr(goals, '_get_session_db', lambda: authority.db)
        live = authority.sessions[ref.session_id]
        target = runner.session_store.lookup_by_session_key(live.route).session_id
        mgr = goals.GoalManager(target)
        mgr.set('finish deferred work')
        mgr.wait_for_seconds(1)
        mgr.state.waiting_until = 1.0
        mgr._save()
        before = authority.db.get_meta('goal:' + target)
        await poll_goal_waits(runner)
        texts = ['HUMAN_FIRST'] + [f'HUMAN_{i}' for i in range(1, humans)]
        for i, text in enumerate(texts):
            _human(authority, ref, f'human-priority-{i}', text)

        async def turn(event, source, key, generation):
            seen.append(event.text)
            failed = event.text.startswith('HUMAN_')
            # Pause the fixture's drain at this real turn boundary, as shutdown does.
            runner._draining = seen == ['HUMAN_FIRST']
            execution_result.get()['result'] = {
                'final_response': 'temporary failure' if failed else 'finished',
                'messages': [], 'failed': failed}
            return execution_result.get()['result']['final_response']

        def judge(**kwargs):
            judges.append(kwargs)
            return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content='{"verdict":"done"}'))])

        monkeypatch.setattr(runner, '_handle_message_with_agent', turn)
        monkeypatch.setattr(auxiliary_client, 'call_llm', judge)
        await authority._drain(ref)
        assert seen == ['HUMAN_FIRST'] and judges == []
        assert authority.db.get_meta('goal:' + target) == before
        if restart:
            authority.db.close()
            runner = GatewayRunner(GatewayConfig(sessions_dir=tmp_path / 'sessions'))
            db = SessionDB(tmp_path / 'state.db')
            runner.session_store._db = db
            runner._session_db = AsyncSessionDB(db)
            authority = await initialize_session_authority(runner, profile_id='default', instance_id='restarted', db=db)
            monkeypatch.setattr(authority, '_schedule', lambda *_: None)
            monkeypatch.setattr(runner, '_handle_message_with_agent', turn)
        else:
            runner._draining = False
        await poll_goal_waits(runner)
        await authority._drain(ref)
        assert seen == texts + [mgr.next_continuation_prompt()], 'priority refusal stranded the unchanged wait'
        assert len(judges) == 1 and goals.GoalManager(target).state.status == 'done'
        from hermes_state_runtime import list_session_admissions
        rows = list_session_admissions(authority.db, session_id=ref.session_id, pending_only=False)
        assert [(r['status'], r['outcome']) for r in rows] == [
            ('terminal', 'cancelled')] + [('terminal', 'failed')] * humans + [('terminal', 'completed')]
        await poll_goal_waits(runner)
        await authority._drain(ref)
        assert len(seen) == humans + 1
    finally:
        authority.db.close()


@pytest.mark.asyncio
@pytest.mark.parametrize('matching', [False, True])
async def test_unrelated_automation_identity_is_not_worker_completion(tmp_path, monkeypatch, matching):
    import json
    from types import SimpleNamespace
    from agent import auxiliary_client
    from gateway.platforms.event import MessageEvent
    from hermes_cli import goals
    from hermes_cli.goals_workers import completion_ids
    from tools import async_delegation as delegation
    runner, authority, ref, _actor = await _local(tmp_path, monkeypatch)
    judges = []
    try:
        monkeypatch.setattr(goals, '_get_session_db', lambda: authority.db)
        live = authority.sessions[ref.session_id]
        target = runner.session_store.lookup_by_session_key(live.route).session_id
        worker = {'type': 'async_delegation', 'delegation_id': 'owned-worker', 'session_key': live.route,
                  'parent_session_id': target, 'status': 'running', 'dispatched_at': 1.0}
        delegation._persist_dispatch(worker)
        def judge(**kwargs):
            judges.append(kwargs)
            raw = '{"verdict":"wait","wait_for_seconds":600,"wait_on_delegations":true}' if len(judges) == 1 else '{"verdict":"done"}'
            return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=raw))])
        monkeypatch.setattr(auxiliary_client, 'call_llm', judge)
        mgr = goals.GoalManager(target)
        mgr.set('finish delegated work')
        assert mgr.evaluate_after_turn('waiting for work')['verdict'] == 'wait'
        before = authority.db.get_meta('goal:' + target)
        delegation._persist_completion(dict(worker, status='completed'), {'status': 'completed', 'summary': 'worker finished'})
        identities = ['plugin-wake', 'null', '{}', '["async_delegation",{}]', '["async_delegation","owned-worker"]']
        if matching:
            identities.append(json.dumps(['async_delegation', 'owned-worker', '']))
        event = MessageEvent(text='ordinary internal input', source=live.source, internal=True,
                             message_id='plugin-wake', metadata={'gateway_session_key': live.route,
                             'gateway_session_id': target, 'automation_identities': identities})
        await authority.admit_automation(runner._adapter_for_source(live.source), event, 'plugin-wake')
        row, _first = await authority._claim_next(ref, live)
        assert row['request_id'] == 'plugin-wake' and row['status'] == 'started'
        assert completion_ids(target) == ({'owned-worker'} if matching else set())
        decision = mgr.evaluate_after_turn('internal input processed')
        if matching:
            assert decision['verdict'] == 'done' and len(judges) == 2
        else:
            assert decision['verdict'] == 'waiting' and len(judges) == 1
            assert authority.db.get_meta('goal:' + target) == before
    finally:
        authority.db.close()


@pytest.mark.asyncio
@pytest.mark.parametrize('blocker', ['approval', 'unknown'])
async def test_ready_wait_cannot_bypass_unfinished_human_work(tmp_path, monkeypatch, blocker):
    from gateway.run_goal_waits import poll_goal_waits
    from gateway.session_authority import initialize_session_authority
    from hermes_cli import goals
    from tests.gateway.test_unviewed_completion_notes import _human
    runner, authority, ref, _actor = await _local(tmp_path, monkeypatch)
    try:
        monkeypatch.setattr(goals, '_get_session_db', lambda: authority.db)
        live = authority.sessions[ref.session_id]
        target = runner.session_store.lookup_by_session_key(live.route).session_id
        mgr = goals.GoalManager(target)
        mgr.set('finish deferred work')
        mgr.wait_for_seconds(1)
        mgr.state.waiting_until = 1.0
        mgr._save()
        before = authority.db.get_meta('goal:' + target)
        _human(authority, ref, 'unfinished-human', 'HUMAN')
        row, _ = await authority._claim_next(ref, live)
        assert row['status'] == 'started'
        if blocker == 'approval':
            authority.register_approval(ref.session_id, row['generation'], live.route,
                                       {'request_id': 'approval', 'command': 'fixture command'})
            assert live.controls.pending
        else:
            authority = await initialize_session_authority(runner, profile_id='default', instance_id='next', db=authority.db)
            from hermes_state_runtime import get_session_admission
            assert get_session_admission(authority.db, admission_id=row['admission_id'])['status'] == 'unknown'
        await poll_goal_waits(runner)
        assert _admissions(authority, ref) == []
        assert authority.db.get_meta('goal:' + target) == before
    finally:
        authority.db.close()
