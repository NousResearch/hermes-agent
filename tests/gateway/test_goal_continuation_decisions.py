"""Fresh host decisions survive admission refusal; descriptors cannot invent those decisions."""
from types import SimpleNamespace

import pytest


def _answer(raw):
    return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=raw))])


async def _owner(tmp_path, monkeypatch, kind):
    from tests.gateway.test_session_end_delegations import _local
    if kind != 'messaging':
        runner, authority, ref, _actor = await _local(tmp_path, monkeypatch)
        return runner, authority, ref
    from gateway.platforms.event import MessageEvent
    from hermes_state_runtime import claim_session_input, settle_session_input
    from tests.gateway.test_automation_compression_media import _discord_owner, _admit_human
    runner, _adapter, authority, source = await _discord_owner(tmp_path, monkeypatch)
    receipt = await _admit_human(runner, authority, MessageEvent(text='seed route', source=source, message_id='seed'))
    row = claim_session_input(authority.db, epoch=authority.epoch, session_id=receipt.ref.session_id)
    settle_session_input(authority.db, epoch=authority.epoch, admission_id=row['admission_id'],
                         generation=row['generation'], outcome='completed')
    return runner, authority, receipt.ref


async def _restart(runner, authority, monkeypatch):
    from gateway.run import GatewayRunner
    from gateway.session_authority import initialize_session_authority
    from gateway.config import Platform
    from gateway.session_local_recovery import restore_local_session
    from hermes_state import AsyncSessionDB, SessionDB
    path, config = authority.db.db_path, runner.config
    adapters = {platform: adapter for platform, adapter in runner.adapters.items() if platform != Platform.LOCAL}
    locals_ = [sid for sid, live in authority.sessions.items() if live.source.platform == Platform.LOCAL]
    authority.db.close()
    runner = GatewayRunner(config)
    db = SessionDB(path)
    runner.session_store._db = db
    runner._session_db = AsyncSessionDB(db)
    runner.adapters = adapters
    authority = await initialize_session_authority(runner, profile_id='default', instance_id='restarted', db=db)
    monkeypatch.setattr(authority, '_schedule', lambda *_: None)
    for adapter in adapters.values():
        runner._wire_adapter_handlers(adapter)
    for sid in locals_:
        restore_local_session(authority, sid)
    return runner, authority


@pytest.mark.asyncio
@pytest.mark.parametrize('interference,kind,restart', [
    ('none', 'local', False), ('human', 'local', False), ('estop', 'local', False),
    ('transient', 'local', False), ('human', 'local', True), ('estop', 'local', True),
    ('transient', 'local', True), ('human', 'messaging', False), ('estop', 'messaging', True),
    ('transient', 'worker', True),
])
async def test_postturn_continuation_survives_transient_admission_refusal(
        tmp_path, monkeypatch, interference, kind, restart):
    from agent import auxiliary_client, estop
    from gateway import session_goal_waits
    from gateway.run_goal_waits import poll_goal_waits
    from gateway.session_results import execution_result
    from hermes_cli import goals
    from hermes_state_runtime import RuntimeStoreError, list_session_admissions
    from tests.gateway.test_unviewed_completion_notes import _human

    runner, authority, ref = await _owner(tmp_path, monkeypatch, kind)
    seen, judges, paused, refusals = [], [], [False], []
    try:
        monkeypatch.setattr(goals, '_get_session_db', lambda: authority.db)
        monkeypatch.setattr(estop, 'check_paused', lambda *args: paused[0])
        live = authority.sessions[ref.session_id]
        target = runner.session_store.lookup_by_session_key(live.route).session_id
        manager = goals.GoalManager(target)
        manager.set('finish deferred work')
        if kind == 'worker':
            from tools import async_delegation as delegation
            worker = dict(type='async_delegation', delegation_id='owned-worker', session_key=live.route,
                          parent_session_id=target, status='completed', summary='worker finished', dispatched_at=1.0)
            delegation._persist_dispatch(worker)
            monkeypatch.setattr(auxiliary_client, 'call_llm', lambda **k: _answer(
                '{"verdict":"wait","wait_for_seconds":900,"wait_on_delegations":true}'))
            assert manager.evaluate_after_turn('waiting for worker')['verdict'] == 'wait'
            delegation._persist_completion(worker, dict(status='completed', summary='worker finished'))
        else:
            manager.wait_for_seconds(1)
            manager.state.waiting_until = 1.0
            manager._save()

        async def turn(event, source, key, generation):
            seen.append(event.text)
            failed = event.text == 'HUMAN_FAILED'
            if len(seen) == 1 and interference == 'human':
                if kind == 'messaging':
                    from gateway.platforms.event import MessageEvent
                    from tests.gateway.test_automation_compression_media import _admit_human
                    await _admit_human(runner, authority, MessageEvent(
                        text='HUMAN_FAILED', source=source, message_id='human-during-turn'))
                else:
                    _human(authority, ref, 'human-during-turn', 'HUMAN_FAILED')
            execution_result.get()['result'] = dict(final_response='temporary failure' if failed else 'work advanced',
                                                    messages=[], failed=failed)
            return execution_result.get()['result']['final_response']

        def judge(**kwargs):
            judges.append(kwargs)
            if len(judges) == 1 and interference == 'estop':
                paused[0] = True
            return _answer('{"verdict":"continue"}' if len(judges) == 1 else '{"verdict":"done"}')

        original = session_goal_waits.admit_wait
        async def transient(*args, **kwargs):
            if interference == 'transient' and kwargs.get('after') and not refusals:
                refusals.append(kwargs['after'])
                raise RuntimeStoreError('session_busy')
            return await original(*args, **kwargs)
        monkeypatch.setattr(session_goal_waits, 'admit_wait', transient)
        monkeypatch.setattr(runner, '_handle_message_with_agent', turn)
        monkeypatch.setattr(auxiliary_client, 'call_llm', judge)
        if kind == 'worker':
            assert await runner._deliver_async_delegation_group([worker]) is True
        else:
            await poll_goal_waits(runner)
        await authority._drain(ref)
        if interference != 'none':
            assert len(judges) == 1 and len(seen) == (2 if interference == 'human' else 1)
            assert goals.GoalManager(target).state.status == 'active'
        paused[0] = False
        if restart:
            runner, authority = await _restart(runner, authority, monkeypatch)
            monkeypatch.setattr(runner, '_handle_message_with_agent', turn)
        for _ in range(2):
            await poll_goal_waits(runner)
            await authority._drain(ref)
        state = goals.GoalManager(target).state
        rows = list_session_admissions(authority.db, session_id=ref.session_id, pending_only=False)
        assert state.status == 'done', 'a transient post-turn refusal lost the standing-goal continuation'
        assert len(judges) == 2
        assert len(seen) == (3 if interference == 'human' else 2)
        expected = [('terminal', 'completed')] * (3 if kind == 'messaging' else 2)
        if interference == 'human':
            expected.insert(-1, ('terminal', 'failed'))
        assert [(r['status'], r['outcome']) for r in rows] == expected
        if interference == 'transient':
            assert len(refusals) == 1
    finally:
        authority.db.close()


@pytest.mark.asyncio
@pytest.mark.parametrize('after', ['', 'not-a-real-predecessor', 'absent', 'completed', 'failed', 'unknown', 'foreign'])
async def test_after_descriptor_must_prove_a_postturn_decision(tmp_path, monkeypatch, after):
    from gateway.platforms.event import MessageEvent
    from hermes_cli import goals
    from hermes_cli.goals_waits import observed_token
    from hermes_state_runtime import RuntimeStoreError, admit_session_input, claim_session_input, settle_session_input
    from tests.gateway.test_session_end_delegations import _local
    from tests.gateway.test_unviewed_completion_notes import _human

    runner, authority, ref, _actor = await _local(tmp_path, monkeypatch)
    try:
        monkeypatch.setattr(goals, '_get_session_db', lambda: authority.db)
        live = authority.sessions[ref.session_id]
        target = runner.session_store.lookup_by_session_key(live.route).session_id
        manager = goals.GoalManager(target)
        manager.set('respect manual cooldown')
        manager.wait_for_seconds(900)
        evidence = {'target': target, 'token': observed_token(manager.state), 'after': after}
        if after == 'absent':
            evidence.pop('after')
        elif after in {'completed', 'failed', 'unknown', 'foreign'}:
            owner = ref.session_id
            if after == 'foreign':
                owner = 'foreign-session'
                authority.db.create_session(owner, source='cli')
                admit_session_input(authority.db, epoch=authority.epoch, session_id=owner,
                                    principal_id='foreign', request_id='foreign', payload={'text': 'unrelated'})
            else:
                _human(authority, ref, 'real-unrelated-row', 'unrelated')
            row = claim_session_input(authority.db, epoch=authority.epoch, session_id=owner)
            if after != 'unknown':
                settle_session_input(authority.db, epoch=authority.epoch, admission_id=row['admission_id'],
                                     generation=row['generation'], outcome='failed' if after == 'failed' else 'completed')
            else:
                runner, authority = await _restart(runner, authority, monkeypatch)
                live = authority.sessions[ref.session_id]
            evidence['after'] = row['admission_id']
        event = MessageEvent(text='premature autonomous wake', source=live.source, internal=True,
                             message_id='ordinary-producer', metadata={'gateway_session_key': live.route,
                             'gateway_session_id': target, 'goal_wait': evidence})
        before = authority.db.get_meta('goal:' + target)
        with pytest.raises(RuntimeStoreError) as caught:
            await authority.admit_automation(runner._adapter_for_source(live.source), event, 'ordinary-producer')
        assert caught.value.reason in {'invalid_params', 'admission_conflict', 'permission_denied'}
        assert authority.db.get_meta('goal:' + target) == before
    finally:
        authority.db.close()


@pytest.mark.asyncio
@pytest.mark.parametrize('control', ['pause', 'clear', 'replace', 'manual-wait', 'pause-resume'])
async def test_new_user_control_invalidates_outstanding_decision(tmp_path, monkeypatch, control):
    from agent import auxiliary_client, estop
    from gateway.run_goal_waits import poll_goal_waits
    from gateway.session_results import execution_result
    from hermes_cli import goals
    from tests.gateway.test_session_end_delegations import _local, _admissions
    runner, authority, ref, _actor = await _local(tmp_path, monkeypatch)
    turns, judges, paused = [], [], [False]
    try:
        monkeypatch.setattr(goals, '_get_session_db', lambda: authority.db)
        monkeypatch.setattr(estop, 'check_paused', lambda *args: paused[0])
        target = runner.session_store.lookup_by_session_key(authority.sessions[ref.session_id].route).session_id
        manager = goals.GoalManager(target)
        manager.set('finish work')
        manager.wait_for_seconds(1)
        manager.state.waiting_until = 1.0
        manager._save()
        async def turn(event, *args):
            turns.append(event.text)
            execution_result.get()['result'] = dict(final_response='advanced', messages=[], failed=False)
            return 'advanced'
        def judge(**kwargs):
            judges.append(kwargs)
            paused[0] = True
            return _answer('{"verdict":"continue"}')
        monkeypatch.setattr(runner, '_handle_message_with_agent', turn)
        monkeypatch.setattr(auxiliary_client, 'call_llm', judge)
        await poll_goal_waits(runner)
        await authority._drain(ref)
        assert len(turns) == len(judges) == 1
        manager = goals.GoalManager(target)
        if control == 'pause-resume':
            manager.pause()
            manager.resume(reset_budget=False)
        else:
            {'pause': manager.pause, 'clear': manager.clear, 'replace': lambda: manager.set('replacement'),
             'manual-wait': lambda: manager.wait_for_seconds(900)}[control]()
        before = authority.db.get_meta('goal:' + target)
        paused[0] = False
        for _ in range(2):
            await poll_goal_waits(runner)
            await authority._drain(ref)
        assert len(turns) == len(judges) == len(_admissions(authority, ref)) == 1
        assert authority.db.get_meta('goal:' + target) == before
    finally:
        authority.db.close()


@pytest.mark.asyncio
@pytest.mark.parametrize('outcome', ['failed', 'interrupted', 'unknown'])
async def test_unsettled_or_unsuccessful_predecessor_never_authorizes_successor(tmp_path, monkeypatch, outcome):
    import asyncio
    from agent import auxiliary_client, estop
    from gateway.run_goal_waits import poll_goal_waits
    from gateway.session_results import execution_result
    from hermes_cli import goals
    from tests.gateway.test_session_end_delegations import _local, _admissions
    runner, authority, ref, _actor = await _local(tmp_path, monkeypatch)
    seen, judges, paused = [], [], [False]
    try:
        monkeypatch.setattr(goals, '_get_session_db', lambda: authority.db)
        monkeypatch.setattr(estop, 'check_paused', lambda *args: paused[0])
        target = runner.session_store.lookup_by_session_key(authority.sessions[ref.session_id].route).session_id
        manager = goals.GoalManager(target)
        manager.set('finish work')
        manager.wait_for_seconds(1)
        manager.state.waiting_until = 1.0
        manager._save()
        async def turn(event, *args):
            seen.append(event.text)
            execution_result.get()['result'] = dict(final_response='advanced', messages=[], failed=False)
            return 'advanced'
        def judge(**kwargs):
            judges.append(kwargs)
            paused[0] = True
            return _answer('{"verdict":"continue"}')
        original = authority._execute_claimed
        async def uncertain(*args):
            response, _ = await original(*args)
            if outcome == 'unknown':
                raise asyncio.CancelledError()
            return response, outcome
        monkeypatch.setattr(runner, '_handle_message_with_agent', turn)
        monkeypatch.setattr(auxiliary_client, 'call_llm', judge)
        monkeypatch.setattr(authority, '_execute_claimed', uncertain)
        await poll_goal_waits(runner)
        if outcome == 'unknown':
            with pytest.raises(asyncio.CancelledError):
                await authority._drain(ref)
        else:
            await authority._drain(ref)
        assert len(seen) == len(judges) == 1
        paused[0] = False
        runner, authority = await _restart(runner, authority, monkeypatch)
        monkeypatch.setattr(runner, '_handle_message_with_agent', turn)
        for _ in range(2):
            await poll_goal_waits(runner)
            await authority._drain(ref)
        assert len(seen) == len(judges) == len(_admissions(authority, ref)) == 1
        assert goals.GoalManager(target).state.status == 'active'
        assert _admissions(authority, ref)[0]['outcome'] != 'completed'
    finally:
        authority.db.close()


@pytest.mark.asyncio
@pytest.mark.parametrize('mode', ['replay', 'yield-restart', 'forged-identity', 'forged-text'])
async def test_exact_host_decision_replay_and_priority_replacement(tmp_path, monkeypatch, mode):
    from dataclasses import replace
    from agent import auxiliary_client
    from gateway.run_goal_waits import poll_goal_waits
    from gateway.session_results import execution_result
    from hermes_cli import goals
    from hermes_state_runtime import RuntimeStoreError, list_session_admissions
    from tests.gateway.test_session_end_delegations import _local
    from tests.gateway.test_unviewed_completion_notes import _human
    runner, authority, ref, _actor = await _local(tmp_path, monkeypatch)
    turns, judges, captured, verified = [], [], [], []
    try:
        monkeypatch.setattr(goals, '_get_session_db', lambda: authority.db)
        live = authority.sessions[ref.session_id]
        target = runner.session_store.lookup_by_session_key(live.route).session_id
        manager = goals.GoalManager(target)
        manager.set('finish work')
        manager.wait_for_seconds(1)
        manager.state.waiting_until = 1.0
        manager._save()
        async def turn(event, *args):
            turns.append(event.text)
            execution_result.get()['result'] = dict(final_response='advanced', messages=[],
                                                    failed=event.text == 'HUMAN_FAILED')
            return 'advanced'
        def judge(**kwargs):
            judges.append(kwargs)
            return _answer('{"verdict":"continue"}' if len(judges) == 1 else '{"verdict":"done"}')
        original = authority.admit_automation
        async def intercept(adapter, event, identity, **kwargs):
            evidence = event.metadata.get('goal_wait')
            if evidence and 'after' in evidence and not captured:
                captured.append(event)
                if mode.startswith('forged'):
                    forged = replace(event, text='unrelated work') if mode == 'forged-text' else event
                    wrong_identity = identity + '-fabricated' if mode == 'forged-identity' else identity
                    with pytest.raises(RuntimeStoreError, match='admission_conflict'):
                        await original(adapter, forged, wrong_identity)
                accepted = await original(adapter, event, identity, **kwargs)
                replay = await original(adapter, event, identity, **kwargs)
                assert accepted.admission_id == replay.admission_id
                if mode == 'yield-restart':
                    _human(authority, ref, 'priority', 'HUMAN_FAILED')
                    runner._draining = True
                verified.append(True)
                return accepted
            return await original(adapter, event, identity, **kwargs)
        monkeypatch.setattr(authority, 'admit_automation', intercept)
        monkeypatch.setattr(runner, '_handle_message_with_agent', turn)
        monkeypatch.setattr(auxiliary_client, 'call_llm', judge)
        await poll_goal_waits(runner)
        await authority._drain(ref)
        if mode == 'yield-restart':
            runner, authority = await _restart(runner, authority, monkeypatch)
            monkeypatch.setattr(runner, '_handle_message_with_agent', turn)
        for _ in range(2):
            await poll_goal_waits(runner)
            await authority._drain(ref)
        assert goals.GoalManager(target).state.status == 'done'
        assert verified == [True], 'the real post-turn hook swallowed a producer assertion'
        assert len(judges) == 2 and len(turns) == (3 if mode == 'yield-restart' else 2)
        rows = list_session_admissions(authority.db, session_id=ref.session_id, pending_only=False)
        expected = [('terminal', 'completed')] * 2
        if mode == 'yield-restart':
            expected[1:1] = [('terminal', 'cancelled'), ('terminal', 'failed')]
        assert [(r['status'], r['outcome']) for r in rows] == expected
    finally:
        authority.db.close()


@pytest.mark.asyncio
@pytest.mark.parametrize('seam', ['during-judge', 'before-record'])
async def test_new_pause_resume_does_not_recreate_superseded_duty(tmp_path, monkeypatch, seam):
    from agent import auxiliary_client
    from gateway import session_goal_decisions
    from gateway.run_goal_waits import poll_goal_waits
    from gateway.session_results import execution_result
    from hermes_cli import goals
    from hermes_state_runtime import list_session_admissions
    from tests.gateway.test_session_end_delegations import _local

    runner, authority, ref, _actor = await _local(tmp_path, monkeypatch)
    turns, judges, control_bytes = [], [], []
    try:
        monkeypatch.setattr(goals, '_get_session_db', lambda: authority.db)
        live = authority.sessions[ref.session_id]
        target = runner.session_store.lookup_by_session_key(live.route).session_id
        manager = goals.GoalManager(target)
        manager.set('respect the newest control')
        manager.wait_for_seconds(1)
        manager.state.waiting_until = 1.0
        manager._save()

        def newer_control():
            before = authority.db.get_meta('goal:' + target)
            current = goals.GoalManager(target)
            current.pause()
            current.resume(reset_budget=False)
            after = authority.db.get_meta('goal:' + target)
            assert before == after, 'witness must exercise identical restored goal bytes'
            control_bytes.append(after)

        async def turn(event, *args):
            turns.append(event.text)
            execution_result.get()['result'] = dict(final_response='advanced', messages=[], failed=False)
            return 'advanced'

        def judge(**kwargs):
            judges.append(kwargs)
            if len(judges) == 1 and seam == 'during-judge':
                newer_control()
            raw = '{"verdict":"continue"}' if len(judges) == 1 else '{"verdict":"done"}'
            return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=raw))])

        original = session_goal_decisions.record
        async def record(*args, **kwargs):
            if seam == 'before-record' and not control_bytes:
                newer_control()
            return await original(*args, **kwargs)

        monkeypatch.setattr(runner, '_handle_message_with_agent', turn)
        monkeypatch.setattr(auxiliary_client, 'call_llm', judge)
        monkeypatch.setattr(session_goal_decisions, 'record', record)
        await poll_goal_waits(runner)
        await authority._drain(ref)
        for _ in range(2):
            await poll_goal_waits(runner)
            await authority._drain(ref)
        rows = list_session_admissions(authority.db, session_id=ref.session_id, pending_only=False)
        assert control_bytes, 'newer control was not reached'
        assert len(turns) == len(judges) == 1, 'a superseded evaluation/decision recreated autonomous duty after newer pause/resume'
        assert [(r['status'], r['outcome']) for r in rows] == [('terminal', 'completed')]
        assert authority.db.get_meta('goal:' + target) == control_bytes[0]
        assert authority.db.get_meta('goal-decision:' + target) is None
    finally:
        authority.db.close()


@pytest.mark.asyncio
@pytest.mark.parametrize('kind', ['local', 'messaging'])
async def test_completed_goal_turn_during_restart_wait_preserves_successor(tmp_path, monkeypatch, kind):
    from agent import auxiliary_client
    from gateway.drain_control import open_restart_wait, restart_wait_open
    from gateway.run_goal_waits import poll_goal_waits
    from gateway.session_results import execution_result
    from hermes_cli import goals
    from hermes_state_runtime import list_session_admissions

    runner, authority, ref = await _owner(tmp_path, monkeypatch, kind)
    turns, judges, entered = [], [], []
    try:
        monkeypatch.setattr(goals, '_get_session_db', lambda: authority.db)
        live = authority.sessions[ref.session_id]
        target = runner.session_store.lookup_by_session_key(live.route).session_id
        manager = goals.GoalManager(target)
        manager.set('finish after a graceful owner restart')
        manager.wait_for_seconds(1)
        manager.state.waiting_until = 1.0
        manager._save()

        async def turn(event, *args):
            turns.append(event.text)
            execution_result.get()['result'] = dict(final_response='advanced', messages=[], failed=False)
            return 'advanced'

        def judge(**kwargs):
            judges.append(kwargs)
            if len(judges) == 1:
                runner._draining = True
                runner._running = True
                runner.session_runtime_descriptor = {}
                open_restart_wait(runner)
                assert restart_wait_open(runner)
                entered.append(True)
            return _answer('{"verdict":"continue"}' if len(judges) == 1 else '{"verdict":"done"}')

        monkeypatch.setattr(runner, '_handle_message_with_agent', turn)
        monkeypatch.setattr(auxiliary_client, 'call_llm', judge)
        await poll_goal_waits(runner)
        await authority._drain(ref)
        rows = list_session_admissions(authority.db, session_id=ref.session_id, pending_only=False)
        assert entered == [True] and len(turns) == len(judges) == 1
        assert all((row['status'], row['outcome']) == ('terminal', 'completed') for row in rows)
        assert len(rows) == (2 if kind == 'messaging' else 1)
        state = goals.GoalManager(target).state
        assert state.status == 'active' and state.waiting_until == 0 and state.turns_used == 1
        runner, authority = await _restart(runner, authority, monkeypatch)
        monkeypatch.setattr(runner, '_handle_message_with_agent', turn)
        for _ in range(3):
            await poll_goal_waits(runner)
            if ref.session_id in authority.sessions:
                await authority._drain(ref)
        state = goals.GoalManager(target).state
        assert len(turns) == len(judges) == 2 and state.status == 'done', (
            'graceful restart lost the completed predecessor\'s exact continuation')
    finally:
        authority.db.close()


@pytest.mark.asyncio
@pytest.mark.parametrize('abort_record,restart', [(False, False), (True, False), (True, True)])
async def test_decision_record_storage_refusal_does_not_lose_successor(tmp_path, monkeypatch, abort_record, restart):
    import sqlite3
    from agent import auxiliary_client
    from gateway import session_goal_decisions
    from gateway.run_goal_waits import poll_goal_waits
    from gateway.session_results import execution_result
    from hermes_cli import goals
    from hermes_state_runtime import list_session_admissions

    runner, authority, ref = await _owner(tmp_path, monkeypatch, 'local')
    turns, judges, aborted = [], [], []
    try:
        monkeypatch.setattr(goals, '_get_session_db', lambda: authority.db)
        live = authority.sessions[ref.session_id]
        target = runner.session_store.lookup_by_session_key(live.route).session_id
        manager = goals.GoalManager(target)
        manager.set('finish despite one temporary durable-duty storage failure')
        manager.wait_for_seconds(1)
        manager.state.waiting_until = 1.0
        manager._save()

        async def turn(event, *args):
            turns.append(event.text)
            execution_result.get()['result'] = dict(final_response='advanced', messages=[], failed=False)
            return 'advanced'

        def judge(**kwargs):
            judges.append(kwargs)
            return _answer('{"verdict":"continue"}' if len(judges) == 1 else '{"verdict":"done"}')

        original_record = session_goal_decisions.record

        async def record(*args, **kwargs):
            if not abort_record or aborted:
                return await original_record(*args, **kwargs)
            db, execute = authority.db, authority.db._execute_write
            marker, public = db.get_meta('goal-control:' + target), db.get_meta('goal:' + target)
            assert db.get_meta('goal-decision:' + target) is None

            def fail_once(fn, *a, **k):
                def write(conn):
                    fn(conn)
                    duty = conn.execute('SELECT value FROM state_meta WHERE key=?',
                                        ('goal-decision:' + target,)).fetchone()
                    assert duty is not None, 'must reach the validated duty INSERT before abort'
                    aborted.append(duty[0])
                    raise sqlite3.OperationalError('one-shot duty I/O failure')
                return execute(write, *a, **k)

            with monkeypatch.context() as patch:
                patch.setattr(db, '_execute_write', fail_once)
                try:
                    return await original_record(*args, **kwargs)
                finally:
                    assert db.get_meta('goal:' + target) == public
                    assert db.get_meta('goal-control:' + target) == marker
                    assert db.get_meta('goal-decision:' + target) is None

        monkeypatch.setattr(session_goal_decisions, 'record', record)
        monkeypatch.setattr(runner, '_handle_message_with_agent', turn)
        monkeypatch.setattr(auxiliary_client, 'call_llm', judge)
        await poll_goal_waits(runner)
        await authority._drain(ref)
        if abort_record:
            rows = list_session_admissions(authority.db, session_id=ref.session_id, pending_only=False)
            assert len(aborted) == len(turns) == len(judges) == 1
            assert [(r['status'], r['outcome']) for r in rows] == [('terminal', 'completed')]
            assert goals.GoalManager(target).state.status == 'active'
            assert goals.GoalManager(target).state.waiting_until == 0
        if restart:
            runner, authority = await _restart(runner, authority, monkeypatch)
            monkeypatch.setattr(runner, '_handle_message_with_agent', turn)
        for _ in range(3):
            await poll_goal_waits(runner)
            await authority._drain(ref)
        assert goals.GoalManager(target).state.status == 'done' and len(turns) == len(judges) == 2, (
            'one refused exact-duty write stranded the completed predecessor')
    finally:
        authority.db.close()


@pytest.mark.asyncio
@pytest.mark.parametrize('retry', ['immediate', 'admission-restart', 'storage-restart'])
async def test_gate_failure_successor_preserves_actual_host_decision_prompt(tmp_path, monkeypatch, retry):
    import json
    import sqlite3
    from agent import auxiliary_client, estop
    from gateway import session_goal_decisions
    from gateway.run_goal_waits import poll_goal_waits
    from gateway.session_results import execution_result
    from hermes_cli import goals
    from hermes_state_runtime import list_session_admissions

    runner, authority, ref = await _owner(tmp_path, monkeypatch, 'local')
    turns, gates, decisions, aborted, paused = [], [], [], [], [False]
    try:
        monkeypatch.setattr(goals, '_get_session_db', lambda: authority.db)
        monkeypatch.setattr(estop, 'check_paused', lambda *args: paused[0])
        target = runner.session_store.lookup_by_session_key(authority.sessions[ref.session_id].route).session_id
        manager = goals.GoalManager(target)
        manager.set('finish the migration')
        manager.add_gate('fixture-only gate command')
        manager.wait_for_seconds(1)
        manager.state.waiting_until = 1.0
        manager._save()

        async def turn(event, *args):
            turns.append(event.text)
            execution_result.get()['result'] = dict(final_response='advanced', messages=[], failed=False)
            return 'advanced'

        def gate(*args, **kwargs):
            gates.append(True)
            if len(gates) == 1:
                paused[0] = retry != 'immediate'
                return False, 1, 'specific failing invariant: fixture_migration'
            return True, 0, 'passed'

        evaluate = goals.GoalManager.evaluate_after_turn
        def capture(self, *args, **kwargs):
            decision = evaluate(self, *args, **kwargs)
            decisions.append(decision)
            return decision

        persist = session_goal_decisions.persist
        def fail_record(conn, duty):
            persist(conn, duty)
            if retry == 'storage-restart' and not aborted:
                inserted = conn.execute('SELECT value FROM state_meta WHERE key=?',
                                        ('goal-decision:' + target,)).fetchone()
                assert json.loads(inserted[0]) == duty, 'validated INSERT must precede failure'
                aborted.append(duty)
                raise sqlite3.OperationalError('one-shot gate-duty storage refusal')

        monkeypatch.setattr(session_goal_decisions, 'persist', fail_record)
        monkeypatch.setattr(goals.GoalManager, 'evaluate_after_turn', capture)
        monkeypatch.setattr(goals, 'run_gate', gate)
        monkeypatch.setattr(runner, '_handle_message_with_agent', turn)
        monkeypatch.setattr(auxiliary_client, 'call_llm', lambda **kwargs: _answer('{"verdict":"done"}'))
        await poll_goal_waits(runner)
        await authority._drain(ref)
        assert decisions[0]['verdict'] == 'gate_failed' and decisions[0]['should_continue']
        expected = decisions[0]['continuation_prompt']
        assert 'fixture-only gate command' in expected and 'specific failing invariant: fixture_migration' in expected
        if retry != 'immediate':
            assert len(turns) == len(gates) == len(decisions) == 1
            assert len(aborted) == (1 if retry == 'storage-restart' else 0)
            rows = list_session_admissions(authority.db, session_id=ref.session_id, pending_only=False)
            assert [(r['status'], r['outcome']) for r in rows] == [('terminal', 'completed')]
            assert json.loads(authority.db.get_meta('goal-decision:' + target))['prompt'] == expected
            paused[0] = False
            runner, authority = await _restart(runner, authority, monkeypatch)
            monkeypatch.setattr(runner, '_handle_message_with_agent', turn)
            for _ in range(2):
                await poll_goal_waits(runner)
                await authority._drain(ref)
        assert len(turns) == len(gates) == len(decisions) == 2
        assert goals.GoalManager(target).state.status == 'done'
        assert turns[1] == expected, 'canonical successor discarded the actual gate-failure decision prompt'
    finally:
        authority.db.close()


@pytest.mark.asyncio
@pytest.mark.parametrize('failures', [2, 4])
async def test_decision_duty_retries_only_storage_and_exhaustion_fences_unknown(tmp_path, monkeypatch, failures):
    import json
    import sqlite3
    from agent import auxiliary_client
    from gateway import session_goal_decisions, session_settlement_recovery
    from gateway.run_goal_waits import poll_goal_waits
    from gateway.session_results import admission_result, execution_result
    from hermes_cli import goals
    from hermes_state_runtime import list_session_admissions

    runner, authority, ref = await _owner(tmp_path, monkeypatch, 'local')
    turns, judges, aborted, snapshots = [], [], [], []
    try:
        monkeypatch.setattr(goals, '_get_session_db', lambda: authority.db)
        monkeypatch.setattr(session_settlement_recovery, '_SETTLE_RETRY_DELAYS_S', (0.0, 0.0))
        target = runner.session_store.lookup_by_session_key(authority.sessions[ref.session_id].route).session_id
        manager = goals.GoalManager(target)
        manager.set('preserve intent or report uncertainty without repeating model work')
        manager.wait_for_seconds(1)
        manager.state.waiting_until = 1.0
        manager._save()

        async def turn(event, *args):
            turns.append(event.text)
            execution_result.get()['result'] = dict(final_response='advanced', messages=[], failed=False)
            return 'advanced'

        def judge(**kwargs):
            judges.append(kwargs)
            return _answer('{"verdict":"continue"}' if len(judges) == 1 else '{"verdict":"done"}')

        persist = session_goal_decisions.persist
        def refuse(conn, duty):
            if not snapshots:
                snapshots.extend(conn.execute('SELECT value FROM state_meta WHERE key=?', (key + target,)).fetchone()[0]
                                 for key in ('goal:', 'goal-control:'))
            persist(conn, duty)
            if len(aborted) < failures:
                inserted = conn.execute('SELECT value FROM state_meta WHERE key=?',
                                        ('goal-decision:' + target,)).fetchone()
                predecessor = conn.execute('SELECT status,outcome FROM session_admissions WHERE admission_id=?',
                                           (duty['after'],)).fetchone()
                assert json.loads(inserted[0]) == duty and tuple(predecessor) == ('started', None)
                aborted.append(duty)
                raise sqlite3.OperationalError('validated duty write unavailable')

        monkeypatch.setattr(session_goal_decisions, 'persist', refuse)
        monkeypatch.setattr(runner, '_handle_message_with_agent', turn)
        monkeypatch.setattr(auxiliary_client, 'call_llm', judge)
        await poll_goal_waits(runner)
        await authority._drain(ref)
        rows = list_session_admissions(authority.db, session_id=ref.session_id, pending_only=False)
        assert len(turns) == len(judges) == 1 and len(aborted) == failures
        assert authority.db.get_meta('goal:' + target) == snapshots[0]
        assert authority.db.get_meta('goal-control:' + target) == snapshots[1]
        if failures == 4:
            assert [(r['status'], r['outcome']) for r in rows] == [('unknown', None)]
            assert authority.db.get_meta('goal-decision:' + target) is None
            assert admission_result(authority.db, rows[0]['admission_id']) is None
            assert rows[0]['admission_id'] in authority.pending_results
        else:
            assert [(r['status'], r['outcome']) for r in rows] == [('terminal', 'completed')]
            assert authority.db.get_meta('goal-decision:' + target) is not None
            saved = admission_result(authority.db, rows[0]['admission_id'])
            assert 'goal_continuation_write' not in saved
            json.dumps(saved)  # no host callback/authority escapes into stored model output
        monkeypatch.setattr(session_goal_decisions, 'persist', persist)
        runner, authority = await _restart(runner, authority, monkeypatch)
        monkeypatch.setattr(runner, '_handle_message_with_agent', turn)
        for _ in range(2):
            await poll_goal_waits(runner)
            await authority._drain(ref)
        rows = list_session_admissions(authority.db, session_id=ref.session_id, pending_only=False)
        if failures == 4:
            assert len(turns) == len(judges) == 1
            assert all(r['outcome'] != 'completed' for r in rows)
            assert authority.db.get_meta('goal-decision:' + target) is None
        else:
            assert len(turns) == len(judges) == 2 and goals.GoalManager(target).state.status == 'done'
    finally:
        authority.db.close()


@pytest.mark.asyncio
@pytest.mark.parametrize('kind,failures', [('local', 2), ('local', 4), ('messaging', 4)])
async def test_same_owner_unattended_resolution_preserves_exact_duty(tmp_path, monkeypatch, kind, failures):
    import asyncio
    import json
    import sqlite3
    from types import MethodType
    from agent import auxiliary_client
    from gateway import session_goal_decisions, session_settlement_recovery
    from gateway.platforms.event import MessageEvent
    from gateway.run_goal_waits import poll_goal_waits
    from gateway.session_authority import SessionAuthority
    from gateway.session_contract import SessionActor, Submission
    from gateway.session_results import admission_result, execution_result
    from hermes_cli import goals
    from hermes_state_runtime import list_session_admissions
    from tests.gateway.test_automation_compression_media import _admit_human

    schedule = SessionAuthority._schedule
    runner, authority, ref = await _owner(tmp_path, monkeypatch, kind)
    turns, judges, aborted, resolutions, snapshots = [], [], [], [], []
    live = authority.sessions[ref.session_id]
    try:
        monkeypatch.setattr(goals, '_get_session_db', lambda: authority.db)
        monkeypatch.setattr(session_settlement_recovery, '_SETTLE_RETRY_DELAYS_S', (0.0, 0.0))
        target = runner.session_store.lookup_by_session_key(live.route).session_id
        manager = goals.GoalManager(target)
        manager.set('finish after storage recovers without repeating predecessor inference')
        manager.wait_for_seconds(1)
        manager.state.waiting_until = 1.0
        manager._save()

        async def turn(event, *_):
            turns.append(event.text)
            failed = event.text == 'HUMAN_TRANSIENT_FAILURE'
            execution_result.get()['result'] = dict(final_response='temporary failure' if failed else 'advanced',
                                                    messages=[], failed=failed)
            return execution_result.get()['result']['final_response']

        def judge(**kwargs):
            judges.append(kwargs)
            return _answer('{"verdict":"continue"}' if len(judges) == 1 else '{"verdict":"done"}')

        persist = session_goal_decisions.persist
        def refuse(conn, duty):
            if not snapshots:
                snapshots.extend(conn.execute('SELECT value FROM state_meta WHERE key=?', (prefix + target,)).fetchone()[0]
                                 for prefix in ('goal:', 'goal-control:'))
            persist(conn, duty)
            if len(aborted) < failures:
                inserted = conn.execute('SELECT value FROM state_meta WHERE key=?', ('goal-decision:' + target,)).fetchone()
                predecessor = conn.execute('SELECT status,outcome FROM session_admissions WHERE admission_id=?',
                                           (duty['after'],)).fetchone()
                assert json.loads(inserted[0]) == duty and tuple(predecessor) == ('started', None)
                aborted.append(duty.copy())
                raise sqlite3.OperationalError('validated duty INSERT rollback')

        original_resolution = authority.resolve_unknown_as_owner
        async def resolve(*args, **kwargs):
            admission_id = args[1]
            pending = authority.pending_results.get(admission_id)
            callback = callable((pending or {}).get('goal_continuation_write'))
            result = await original_resolution(*args, **kwargs)
            rows = list_session_admissions(authority.db, session_id=ref.session_id, pending_only=False)
            row = next(row for row in rows if row['admission_id'] == admission_id)
            resolutions.append(dict(callback_before=callback, status=row['status'], outcome=row['outcome'],
                                    duty_after=authority.db.get_meta('goal-decision:' + target)))
            return result

        monkeypatch.setattr(session_goal_decisions, 'persist', refuse)
        monkeypatch.setattr(authority, 'resolve_unknown_as_owner', resolve)
        monkeypatch.setattr(runner, '_handle_message_with_agent', turn)
        monkeypatch.setattr(auxiliary_client, 'call_llm', judge)
        await poll_goal_waits(runner)
        await authority._drain(ref)
        rows = list_session_admissions(authority.db, session_id=ref.session_id, pending_only=False)
        predecessor = rows[-1]
        assert len(turns) == len(judges) == 1 and len(aborted) == failures
        assert all(duty == aborted[0] for duty in aborted)
        assert authority.db.get_meta('goal:' + target) == snapshots[0]
        assert authority.db.get_meta('goal-control:' + target) == snapshots[1]
        if failures == 4:
            assert (predecessor['status'], predecessor['outcome']) == ('unknown', None)
            assert admission_result(authority.db, predecessor['admission_id']) is None
            assert authority.db.get_meta('goal-decision:' + target) is None
            assert callable(authority.pending_results[predecessor['admission_id']]['goal_continuation_write'])
        else:
            assert (predecessor['status'], predecessor['outcome']) == ('terminal', 'completed')
            assert authority.db.get_meta('goal-decision:' + target) is not None

        monkeypatch.setattr(session_goal_decisions, 'persist', persist)
        monkeypatch.setattr(authority, '_schedule', MethodType(schedule, authority))
        if kind == 'local':
            actor = SessionActor('test-owner', 'default', frozenset({'session:submit', 'session:read'}), 'fixture')
            await authority.submit(actor, Submission('human-after-storage', ref,
                                   {'text': 'HUMAN_TRANSIENT_FAILURE'}, 'queue'))
        else:
            await _admit_human(runner, authority, MessageEvent(text='HUMAN_TRANSIENT_FAILURE', source=live.source,
                                                            message_id='human-after-storage'))
        await asyncio.wait_for(live.task, timeout=20)
        assert turns[1] == 'HUMAN_TRANSIENT_FAILURE'
        if failures == 4:
            assert len(resolutions) == 1 and resolutions[0]['callback_before'] is True
            assert (resolutions[0]['status'], resolutions[0]['outcome']) == ('terminal', 'completed')
            assert predecessor['admission_id'] not in authority.pending_results
            assert admission_result(authority.db, predecessor['admission_id'])['result']['final_response'] == 'advanced'
        else:
            assert resolutions == []
        for _ in range(3):
            await poll_goal_waits(runner)
            if live.task is not None:
                await asyncio.wait_for(live.task, timeout=20)
        state = goals.GoalManager(target).state
        assert state.status == 'done' and len(turns) == 3 and len(judges) == 2, (
            'same-owner resolution committed a completed answer without its exact continuation duty')
        assert turns[2] == aborted[0]['prompt']
        assert authority.db.get_meta('goal-decision:' + target) is None
    finally:
        if live.task is not None and not live.task.done():
            live.task.cancel()
            await asyncio.gather(live.task, return_exceptions=True)
        authority.db.close()
