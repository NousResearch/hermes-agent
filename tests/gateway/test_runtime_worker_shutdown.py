"""Shutdown counts managed claims and retains profile ownership until actual threads stop."""
import asyncio
import threading
from types import SimpleNamespace

import pytest

from gateway import run_runtime
from gateway.run_shutdown import GatewayShutdownMixin
from gateway.session_authorities import SessionAuthorities


@pytest.mark.asyncio
async def test_managed_claims_participate_in_zero_and_nonzero_drain(tmp_path):
    release = asyncio.Event()
    task = asyncio.create_task(release.wait())
    stopped = []
    live = SimpleNamespace(task=task, route='route', event_stream=SimpleNamespace(execution={'execution_generation': 2}))
    authority = SimpleNamespace(sessions={'s': live}, _managed_workers={
        's': SimpleNamespace(control=lambda frame: stopped.append(frame), closed=threading.Event())})
    runner = SimpleNamespace(session_authority=authority, _running_agents={},
        _running_agent_count=lambda: 0, _active_cron_job_count=lambda: 0, _active_api_run_count=lambda: 0,
        _active_deferred_agent_worker_count=lambda: 0, _snapshot_running_agents=dict,
        _update_runtime_status=lambda *a: None)
    runner._drain_work_counts = GatewayShutdownMixin._drain_work_counts.__get__(runner)
    authority.runner = runner
    authority.pending_stops = {}
    authority.adopted = {}
    try:
        assert GatewayShutdownMixin._active_work_count(runner) == 1
        _, timed_out = await GatewayShutdownMixin._drain_active_agents(runner, timeout=0)
        assert timed_out is True
        run_runtime.stop_managed_turns(runner)
        assert stopped == [{'type': 'stop'}]
        drain = asyncio.create_task(GatewayShutdownMixin._drain_active_agents(runner, timeout=5))
        await asyncio.sleep(0)
        assert not drain.done()
        authority._managed_workers['s'].closed.set()
        release.set()
        _, timed_out = await asyncio.wait_for(drain, 5)
        assert timed_out is False
    finally:
        release.set()
        await asyncio.gather(task, return_exceptions=True)


@pytest.mark.asyncio
async def test_profile_timeout_keeps_authority_until_physical_worker_exits(tmp_path, monkeypatch):
    started, release, done, stopped = (threading.Event() for _ in range(4))
    def turn():
        started.set()
        try:
            assert release.wait(10)
        finally:
            done.set()
    task = asyncio.create_task(asyncio.to_thread(turn))
    assert await asyncio.to_thread(started.wait, 5)
    launch, beta = tmp_path, tmp_path / 'profiles' / 'beta'
    beta.mkdir(parents=True)
    registry = SessionAuthorities(launch)
    registry.add(launch, SimpleNamespace(profile_id=str(launch)))
    agent = SimpleNamespace(hard_interrupt=lambda message: stopped.set())
    live = SimpleNamespace(task=task, route='beta-route', event_stream=SimpleNamespace(execution={'execution_generation': 2}))
    authority = SimpleNamespace(sessions={'s': live}, profile_id=str(beta), pending_stops={}, adopted={},
        waiters={}, native_waiters=set(), agent=lambda ref: agent, _turn_workers={1: (SimpleNamespace(worker_done=done), [agent])})
    registry.add(beta, authority, name='beta')
    runner = SimpleNamespace(session_authorities=registry, session_runtime_descriptor={},
                             session_ticket_store=SimpleNamespace(profile_ids=registry.profile_ids()))
    unbound = []
    monkeypatch.setattr('gateway.session_cron.unbind_owner', lambda a: unbound.append(a))
    authority.runner = runner
    monkeypatch.setattr(run_runtime, 'TURN_SETTLE_SECONDS', 0)
    try:
        assert await run_runtime.unserve_profile_runtime(runner, beta) is False
        assert stopped.is_set() and not done.is_set()
        assert registry.for_home(beta) is authority
        assert not unbound
        # Kept for its live writer, but it refuses all work: no longer published or ticketed.
        assert str(beta) not in runner.session_ticket_store.profile_ids
        release.set()
        await asyncio.wait_for(task, 5)
        await run_runtime.unserve_profile_runtime(runner, beta)
        assert registry.for_home(beta) is None
        assert str(beta) not in runner.session_ticket_store.profile_ids
        assert unbound == [authority]
    finally:
        release.set()
        await asyncio.gather(task, return_exceptions=True)


@pytest.mark.asyncio
async def test_whole_runtime_settle_logs_late_work_and_finishes_shutdown(monkeypatch):
    """A writer that outlives TURN_SETTLE_SECONDS must not abort the stop sequence: the rest of
    shutdown (tickets, API, adapters, flush, exit state) still runs and the watchdog stays armed."""
    late, wedged = asyncio.Event(), threading.Event()
    task = asyncio.create_task(late.wait())
    agent = SimpleNamespace(hard_interrupt=lambda message=None: None)
    live = SimpleNamespace(task=task, route='r', event_stream=SimpleNamespace(execution={}))
    authority = SimpleNamespace(sessions={'s': live}, _managed_workers={}, pending_stops={}, adopted={}, profile_id='p',
                                _turn_workers={1: (SimpleNamespace(worker_done=wedged), [agent])})
    revoked, stopped = [], []
    runner = SimpleNamespace(session_authority=authority, _running_agents={},
                             session_ticket_store=SimpleNamespace(revoke=lambda: revoked.append(True)),
                             session_api=SimpleNamespace())
    authority.runner = runner
    async def stop_api(handle):
        stopped.append(handle)
    monkeypatch.setattr('gateway.run_api.stop_gateway_api', stop_api)
    monkeypatch.setattr(run_runtime, 'TURN_SETTLE_SECONDS', 0.05)
    try:
        await asyncio.wait_for(run_runtime.settle_gateway_runtime(runner), 5)
        assert revoked == [True] and stopped == [runner.session_api]
        assert not task.done(), 'late work is left for recovery, not cancelled into a false settlement'
    finally:
        late.set()
        await asyncio.gather(task, return_exceptions=True)


@pytest.mark.asyncio
async def test_unserve_keeps_ownership_without_aborting_the_reconcile(tmp_path, monkeypatch):
    """A profile whose writer outlived its Stop is unrouted but keeps its reservation and store
    handles; the reconcile is not aborted, so the other removed profile is still unserved."""
    from gateway.run_profile_reconcile import GatewayProfileReconcileMixin
    stuck, gone = tmp_path / 'stuck', tmp_path / 'gone'
    async def unserve(runner, home):
        return home != stuck
    released, closed = [], []
    monkeypatch.setattr(run_runtime, 'unserve_profile_runtime', unserve)
    monkeypatch.setattr(run_runtime, 'release_profile_home', lambda runner, home: released.append(home))
    monkeypatch.setattr('hermes_state_registry.close_all_under', lambda home: closed.append(home))
    runner = SimpleNamespace(session_authorities=object(), _profile_failed_platforms={}, _profile_adapters={},
        _served_profile_homes={'stuck': stuck, 'gone': gone}, _served_profile_signatures={}, _agent_cache={},
        _evict_cached_agent=lambda key: None, served_profile_names=lambda: sorted(runner._served_profile_homes),
        _live_resource_claims=lambda active: {}, _record_served_profiles=lambda active, homes: None)
    monkeypatch.setattr('hermes_cli.profiles.profiles_to_serve', lambda **kwargs: [])
    runner._unserve_profile = GatewayProfileReconcileMixin._unserve_profile.__get__(runner)
    result = await GatewayProfileReconcileMixin._apply_profile_changes(
        runner, {}, [], ['stuck', 'gone'], [], reason='watcher')
    assert result['removed'] == ['stuck', 'gone']
    assert released == [gone] and closed == [gone], 'ownership of the stuck profile was released'
    assert runner._served_profile_homes == {}


@pytest.mark.asyncio
async def test_persisted_work_count_returns_to_zero_when_the_drain_ends():
    """gateway_state.json's active_agents is written at the agent-slot release, while the owner's
    drain (counted until it ends) is still live; the drain's end must write it again, or the
    persisted count stays at 1 after every messaging turn and the gateway never reads idle."""
    from gateway.session_authority import SessionAuthority
    from gateway.session_runtime_workers import uncounted_runtime_work

    persisted, release = [], asyncio.Event()
    authority = SessionAuthority.__new__(SessionAuthority)
    runner = SimpleNamespace(_running_agents={}, session_authority=authority,
                             _persist_active_agents=lambda: persisted.append(uncounted_runtime_work(runner)))
    live = SimpleNamespace(task=None, route='route', event_stream=SimpleNamespace(execution={'execution_generation': 1}))
    authority.runner, authority.sessions, authority._managed_workers = runner, {'s': live}, {}

    async def drain(_ref):
        await release.wait()

    authority._drain = drain
    authority._schedule(SimpleNamespace(session_id='s'))
    await asyncio.sleep(0)
    assert uncounted_runtime_work(runner) == 1
    release.set()
    await live.task
    await asyncio.sleep(0)
    assert persisted and persisted[-1] == 0


@pytest.mark.asyncio
async def test_side_agent_is_owner_work_until_its_thread_unwinds(monkeypatch):
    """/bg (like /btw and preview restart) runs on a detached thread (``session_tui_side._spawn``).
    The restart wait, stop drain, SessionDB close gate and idle exit must count it, and retirement
    must interrupt and join it, or a planned restart exits under it and its result never arrives."""
    import contextlib
    from gateway import session_tui_side
    from gateway.session_contract import SessionRef
    from gateway.session_runtime_workers import join_authority_work, uncounted_runtime_work

    started, release, persisted, interrupts = threading.Event(), threading.Event(), [], []
    def run_conversation(**_):
        started.set()
        release.wait(10)
        return {'final_response': 'done'}
    agent = SimpleNamespace(run_conversation=run_conversation, close=lambda: None,
                            hard_interrupt=lambda message=None: interrupts.append(message))
    authority = SimpleNamespace(sessions={}, pending_stops={}, adopted={})
    runner = SimpleNamespace(_running_agents={}, session_authority=authority,
                             _persist_active_agents=lambda: persisted.append(uncounted_runtime_work(runner)))
    authority.runner = runner
    monkeypatch.setattr(session_tui_side, '_admit', lambda *_: (None, None, 'task', 'bg_x'))
    monkeypatch.setattr(session_tui_side, '_side_scope', lambda *_: contextlib.nullcontext())
    monkeypatch.setattr(session_tui_side, '_side_agent', lambda *_: agent)
    try:
        await session_tui_side.background(SimpleNamespace(authority=authority), SessionRef('p', 's'), {})
        assert await asyncio.to_thread(started.wait, 5)
        thread = next(t for t in threading.enumerate() if t.name == 'side-agent-bg_x')
        assert uncounted_runtime_work(runner) == 1 and persisted == [1]
        with pytest.raises(TimeoutError):  # retirement keeps ownership while it runs
            await join_authority_work(authority, .2)
        assert set(interrupts) == {'Profile runtime is stopping'}  # the /bg agent, once built
        release.set()
        await join_authority_work(authority, 5)
        await asyncio.to_thread(thread.join, 5)
        await asyncio.sleep(0)  # the thread's recount runs on the owner loop
        assert uncounted_runtime_work(runner) == 0 and persisted[-1] == 0
    finally:
        release.set()


@pytest.mark.asyncio
@pytest.mark.parametrize('exit_type', [asyncio.CancelledError, KeyboardInterrupt, SystemExit])
async def test_side_agent_thread_killed_by_a_base_exception_releases_owner_work(monkeypatch, exit_type):
    """The agent re-raises cancellation and exit (``agent/turn_facade.py``). A side-agent thread
    that dies that way must still leave the ledger, or idle exit, retirement, the persisted count
    and the SessionDB close gate stay blocked on work that no longer exists."""
    from gateway import session_tui_side
    from gateway.session_contract import SessionRef
    from gateway.session_runtime_workers import join_authority_work, uncounted_runtime_work

    started, threads, failures, persisted = threading.Event(), [], [], []
    authority = SimpleNamespace(sessions={}, pending_stops={}, adopted={})
    runner = SimpleNamespace(_running_agents={}, session_authority=authority,
                             _persist_active_agents=lambda: persisted.append(uncounted_runtime_work(runner)))
    authority.runner = runner
    monkeypatch.setattr(threading, 'excepthook', lambda args: failures.append(args.exc_type))

    def body():
        threads.append(threading.current_thread())
        started.set()
        raise exit_type()

    session_tui_side._spawn(authority, SessionRef('p', 's'), 'bg_abort', 'background.complete', body)
    assert await asyncio.to_thread(started.wait, 5)
    await asyncio.to_thread(threads[0].join, 5)
    await asyncio.sleep(0)  # the thread's recount runs on the owner loop
    assert not threads[0].is_alive() and failures == [exit_type]
    assert uncounted_runtime_work(runner) == 0 and (not persisted or persisted[-1] == 0)
    await join_authority_work(authority, 0)


@pytest.mark.asyncio
@pytest.mark.parametrize('path', ['fork', 'oneshot'])
async def test_btw_side_question_does_not_hold_profile_retirement(monkeypatch, path):
    """``/btw`` is counted owner work (``_spawn``), so retirement must be able to stop it too.

    Its fork is interrupted through the holder; the one-shot fallback (``agent/side_question.py``,
    ``timeout=180.0``) is an uncancellable aux call that never mutates anything, so retirement
    abandons it instead of keeping the profile's reservation for the full settle window."""
    import contextlib
    from gateway import session_tui_side
    from gateway.session_contract import SessionRef
    from gateway.session_runtime_workers import join_authority_work, uncounted_runtime_work
    from agent import side_question

    started, release, interrupts, events = threading.Event(), threading.Event(), [], []

    class Fork:
        def hard_interrupt(self, message=None):
            interrupts.append(message)
            release.set()

    def answer(question, history, *, parent_agent=None, main_runtime=None, on_fork=None, should_stop=None):
        if path == 'fork':
            on_fork(Fork())
            started.set()
            release.wait(30)  # interrupted through the holder
            on_fork(None)  # _answer_via_fork's teardown
            return 'interrupted'
        started.set()
        release.wait(30)  # the 180 s one-shot aux call, which nothing can interrupt
        return 'late answer'

    live = SimpleNamespace(task=None, event_stream=SimpleNamespace(execution={},
        publish=lambda sid, payload, event_type: events.append((event_type, payload))))
    authority = SimpleNamespace(sessions={'s': live}, pending_stops={}, adopted={},
                                agent=lambda ref: SimpleNamespace(_session_messages=[{'role': 'user', 'content': 'hi'}]))
    runner = SimpleNamespace(_running_agents={}, session_authority=authority, _persist_active_agents=lambda: None)
    authority.runner = runner
    monkeypatch.setattr(session_tui_side, '_admit', lambda *_: (None, None, 'why?', 'btw_x'))
    monkeypatch.setattr(session_tui_side, '_side_scope', lambda *_: contextlib.nullcontext())
    monkeypatch.setattr(session_tui_side, '_resolve_runtime', lambda *_: ('m', {}))
    monkeypatch.setattr(side_question, 'answer_side_question', answer)
    try:
        await session_tui_side.btw(SimpleNamespace(authority=authority), SessionRef('p', 's'), {})
        assert await asyncio.to_thread(started.wait, 5)
        thread = next(t for t in threading.enumerate() if t.name == 'side-agent-btw_x')
        assert uncounted_runtime_work(runner) == 1
        await join_authority_work(authority, 5)
        assert uncounted_runtime_work(runner) == 0
        if path == 'fork':
            assert interrupts and set(interrupts) == {'Profile runtime is stopping'}
        release.set()
        await asyncio.to_thread(thread.join, 5)
        assert not thread.is_alive() and events == []  # a stopped /btw publishes no late answer
    finally:
        release.set()


def test_shutdown_interrupt_reaches_counted_side_agents():
    """The stop drain waits for counted side agents (/bg, /btw, preview restart), so the timed-out
    drain's interrupt must reach them too, or a long /bg holds the whole drain and is then
    abandoned mid-tool when the process exits."""
    from gateway.run_shutdown import GatewayShutdownMixin
    interrupts = []
    agent = SimpleNamespace(hard_interrupt=lambda message=None, **_: interrupts.append(message))
    authority = SimpleNamespace(sessions={}, pending_stops={}, adopted={},
                                _side_agents={object(): [agent], object(): [None], object(): None})
    runner = SimpleNamespace(_running_agents={}, session_authority=authority,
                             _interrupt_api_server_runs=lambda reason: 0,
                             _interrupt_deferred_agent_workers=lambda reason: 0)
    GatewayShutdownMixin._interrupt_running_agents(runner, 'Gateway shutting down')
    assert interrupts == ['Gateway shutting down']
