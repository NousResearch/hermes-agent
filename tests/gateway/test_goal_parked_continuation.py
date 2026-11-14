"""An overdue manual goal wait resumes an unviewed gateway-owned LOCAL session."""

import asyncio
from pathlib import Path
from types import SimpleNamespace

import pytest

from tests.gateway.test_session_end_delegations import _admissions, _local


@pytest.mark.asyncio
async def test_overdue_parked_goal_resumes_once_without_a_viewer(tmp_path, monkeypatch):
    from agent import auxiliary_client
    from gateway import run_goals
    from gateway.config import Platform
    from gateway.session_results import execution_result
    from hermes_cli import goals
    from hermes_constants import reset_hermes_home_override, set_hermes_home_override

    monkeypatch.setenv('HOME', str(tmp_path))
    monkeypatch.setenv('HERMES_HOME', str(tmp_path))
    monkeypatch.setattr(Path, 'home', lambda: tmp_path)
    token = set_hermes_home_override(tmp_path)
    runner = authority = ticker = None
    clock = SimpleNamespace(now=1700000000.0)
    module_clock = SimpleNamespace(time=lambda: clock.now)
    monkeypatch.setattr(goals, 'time', module_clock)
    monkeypatch.setattr(run_goals, 'time', module_clock)
    entered, release = asyncio.Queue(), asyncio.Queue()
    turns, judges = [], []

    async def controlled_poll_delay(_seconds):
        # Reaching the next delay acknowledges completion of the entire real scan.
        entered.put_nowait(None)
        await release.get()

    # Replace only this module's asyncio binding, never asyncio.sleep process-wide.
    monkeypatch.setattr(run_goals, 'asyncio', SimpleNamespace(
        sleep=controlled_poll_delay, create_task=asyncio.create_task, to_thread=asyncio.to_thread))

    def judge(**kwargs):
        judges.append(kwargs)
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(
            content='{"verdict":"done","reason":"Timer elapsed and work completed"}'))])

    monkeypatch.setattr(auxiliary_client, 'call_llm', judge)

    async def turn(event, source, quick_key, run_generation):
        turns.append((event.text, source, quick_key))
        execution_result.get()['result'] = {
            'final_response': 'The deferred work is complete.', 'messages': [],
            'failed': False, 'interrupted': False,
        }
        return 'The deferred work is complete.'

    async def poll():
        release.put_nowait(None)
        await asyncio.wait_for(entered.get(), timeout=10)
        # Scheduling is suppressed by _local: execution must cross the real FIFO drain.
        await authority._drain(ref)

    try:
        runner, authority, ref, _actor = await _local(tmp_path, monkeypatch)
        live = authority.sessions[ref.session_id]
        assert live.source.platform == Platform.LOCAL and not live.subscribers
        target = runner.session_store.lookup_by_session_key(live.route).session_id
        # The managers and FIFO share the fixture's owner, not a second state.db writer.
        monkeypatch.setattr(goals, '_get_session_db', lambda: authority.db)
        assert goals._get_session_db() is authority.db
        mgr = goals.GoalManager(target)
        mgr.set('Complete the deferred work after the manual timer')
        mgr.wait_for_seconds(60, reason='manual timer')
        persisted = goals.GoalManager(target).state
        assert persisted.status == 'active' and persisted.waiting_until == clock.now + 60
        assert authority.db.get_session(target) is not None
        expected_prompt = mgr.next_continuation_prompt()
        monkeypatch.setattr(runner, '_handle_message_with_agent', turn)

        runner._start_heartbeat_poller()
        ticker = runner._heartbeat_poll_task
        await asyncio.wait_for(entered.get(), timeout=10)
        await poll()
        assert turns == [] and judges == [] and _admissions(authority, ref) == []

        clock.now = persisted.waiting_until + 1
        await poll()
        assert len(turns) == 1, (
            'overdue parked goal must execute one canonical continuation without a viewer; '
            f'got {len(turns)} agent executions after the gateway poll and authority drain'
        )
        assert turns == [(expected_prompt, live.source, live.route)]
        assert len(judges) == 1 and judges[0]['task'] == 'goal_judge'
        settled = goals.GoalManager(target).state
        assert settled.status == 'done' and settled.last_verdict == 'done'
        assert settled.turns_used == 1 and settled.consecutive_parse_failures == 0
        rows = _admissions(authority, ref)
        assert [(r['status'], r['outcome']) for r in rows] == [('terminal', 'completed')]
        await poll()
        assert len(turns) == len(judges) == len(_admissions(authority, ref)) == 1
        assert not live.subscribers
    finally:
        if ticker is not None:
            ticker.cancel()
            await asyncio.gather(ticker, return_exceptions=True)
        if authority is not None:
            authority.db.close()
        reset_hermes_home_override(token)


@pytest.mark.asyncio
@pytest.mark.parametrize('reserved', [False, True])
async def test_restart_rediscovers_the_same_persisted_wait(tmp_path, monkeypatch, reserved):
    from agent import auxiliary_client
    from gateway.config import GatewayConfig
    from gateway.run import GatewayRunner
    from gateway.run_goal_waits import poll_goal_waits
    from gateway.session_authority import initialize_session_authority
    from gateway.session_results import execution_result
    from hermes_cli import goals
    from hermes_state import SessionDB, AsyncSessionDB
    runner, authority, ref, _actor = await _local(tmp_path, monkeypatch)
    turns, judges = [], []
    try:
        monkeypatch.setattr(goals, '_get_session_db', lambda: authority.db)
        route = authority.sessions[ref.session_id].route
        target = runner.session_store.lookup_by_session_key(route).session_id
        mgr = goals.GoalManager(target)
        mgr.set('resume after owner restart')
        mgr.wait_for_seconds(1)
        mgr.state.waiting_until = 1.0
        mgr._save()
        if reserved:
            await poll_goal_waits(runner)
            await poll_goal_waits(runner)
            assert len(_admissions(authority, ref)) == 1
        raw = authority.db.get_meta('goal:' + target)
        authority.db.close()
        runner = GatewayRunner(GatewayConfig(sessions_dir=tmp_path / 'sessions'))
        db = SessionDB(tmp_path / 'state.db')
        runner.session_store._db = db
        runner._session_db = AsyncSessionDB(db)
        authority = await initialize_session_authority(runner, profile_id='default', instance_id='restarted', db=db)
        async def turn(event, source, key, generation):
            turns.append(event.text)
            execution_result.get()['result'] = {'final_response': 'finished after restart', 'messages': [], 'failed': False}
            return 'finished after restart'
        def judge(**kwargs):
            judges.append(kwargs)
            return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content='{"verdict":"done"}'))])
        monkeypatch.setattr(runner, '_handle_message_with_agent', turn)
        monkeypatch.setattr(auxiliary_client, 'call_llm', judge)
        await poll_goal_waits(runner)
        assert authority.db.get_meta('goal:' + target) == raw
        assert len(_admissions(authority, ref)) == 1
        await authority._drain(ref)
        assert len(turns) == len(judges) == 1
        assert goals.GoalManager(target).state.status == 'done'
        assert _admissions(authority, ref)[0]['outcome'] == 'completed'
    finally:
        authority.db.close()


@pytest.mark.asyncio
@pytest.mark.parametrize('stage', ['scanned', 'admitted'])
async def test_profile_retirement_and_recreation_fence_parked_wake(tmp_path, monkeypatch, stage):
    import shutil
    from agent.secret_scope import current_secret_scope_home
    from gateway import run, run_goal_waits
    from gateway.run import _profile_runtime_scope
    from gateway.run_runtime import initialize_gateway_runtime, release_profile_home, reserve_profile_home
    from gateway.run_runtime import serve_profile_runtime, unserve_profile_runtime
    from gateway.runtime_ownership import process_ownership
    from gateway.session_authorities import active_authority, async_owner_scope
    from gateway.session_authority import SessionAuthority
    from gateway.session_contract import SessionActor
    from gateway.session_local import create_local_session
    from hermes_cli import goals
    from hermes_constants import get_hermes_home
    from hermes_state_registry import close_all_under
    from hermes_state_runtime import RuntimeStoreError, list_session_admissions
    from tools.terminal_scope import get_terminal_scope
    from tests.gateway.test_session_authorities_multiplex import _reserve_homes, _runner

    root, homes = _reserve_homes(tmp_path, monkeypatch, names=('alpha',))
    home = homes[1][1]
    monkeypatch.setattr(run, '_load_gateway_config', lambda: {'platform_toolsets': {'gui': []}})
    monkeypatch.setattr(SessionAuthority, '_schedule', lambda *_: None)
    process_ownership.reserve([h for _, h in homes])
    runner, old, polling = None, None, None
    release, scanned = asyncio.Event(), asyncio.Event()
    scopes, turns = [], []
    try:
        runner = _runner(root, homes)
        await initialize_gateway_runtime(runner)
        old = runner.session_authorities.for_home(home)
        monkeypatch.setattr(goals, '_get_session_db', lambda: getattr(active_authority(runner), 'db', None))
        with _profile_runtime_scope(home, {}):
            actor = SessionActor('owner', old.profile_id, frozenset({'session:create'}), 'fixture')
            ref = create_local_session(old, actor, {'request_id': 'desktop', 'source': 'gui',
                'cwd': str(tmp_path), 'model': 'fixture', 'toolsets': []})
            target = runner.session_store.lookup_by_session_key(old.sessions[ref.session_id].route).session_id
            mgr = goals.GoalManager(target)
            mgr.set('old profile deferred work')
            mgr.wait_for_seconds(1)
            mgr.state.waiting_until = 1.0
            mgr._save()
        original = runner._run_in_executor_with_context

        async def intercept(func, *args):
            result = await original(func, *args)
            if func is run_goal_waits.candidates and args[0] is old:
                scopes.append((get_hermes_home(), current_secret_scope_home(), get_terminal_scope() is not None))
                if stage == 'scanned':
                    scanned.set()
                    await release.wait()
            return result

        async def unexpected_turn(*args, **kwargs):
            turns.append(args)
            raise AssertionError('retired goal dispatched a model turn')

        monkeypatch.setattr(runner, '_run_in_executor_with_context', intercept)
        monkeypatch.setattr(runner, '_handle_message_with_agent', unexpected_turn)
        if stage == 'scanned':
            polling = asyncio.create_task(run_goal_waits.poll_goal_waits(runner))
            await asyncio.wait_for(scanned.wait(), timeout=10)
        else:
            await run_goal_waits.poll_goal_waits(runner)
            assert len(_admissions(old, ref)) == 1
        assert scopes == [(home, str(home), True)]
        assert await unserve_profile_runtime(runner, home)
        async with async_owner_scope(old):
            assert active_authority(runner) is None, 'removed profile fell back to launch owner'
            if stage == 'admitted':
                await old._drain(ref)
                assert _admissions(old, ref)[0]['status'] == 'queued'
        close_all_under(home)
        release_profile_home(runner, home)
        shutil.rmtree(home)
        home.mkdir()
        (home / 'config.yaml').write_text('{}')
        reserve_profile_home(runner, 'alpha', home)
        replacement = await serve_profile_runtime(runner, 'alpha', home)
        assert replacement is not old
        release.set()
        if polling is not None:
            await asyncio.wait_for(polling, timeout=10)
        with pytest.raises(RuntimeStoreError, match='runtime_draining|runtime_retired'):
            old._require_admission_open()
        await run_goal_waits.poll_goal_waits(runner)
        for authority in (replacement, runner.session_authorities.launch):
            assert authority.db.get_meta('goal:' + target) is None
            assert list_session_admissions(authority.db, session_id=ref.session_id, pending_only=False) == []
        assert turns == []
    finally:
        release.set()
        if polling is not None:
            await asyncio.gather(polling, return_exceptions=True)
        if runner is not None:
            runner.session_store.close_all_db_handles()
            runner.close_all_session_db_handles()
            for authority in runner.session_authorities:
                authority.db.close()
        for _, h in homes:
            process_ownership.release(h)
