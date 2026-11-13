"""``gateway.sessiondb_loop_guard`` (config.yaml): SessionDB storage calls made on an event-loop thread are flagged.

The guard is off (nothing wrapped) unless installed; ``raise`` fails the call on a loop thread and
leaves worker threads alone. The composed test runs a real owner through a cron fire and an inbound
messaging turn with the guard recording, and pins the loop-side population: the per-poll and
per-claim reads moved off the loop never come back, and any on-loop call site not named below fails
the test, so a new one is a reviewed decision, not a silent regression.
"""
import asyncio
import os
import threading

import pytest

import hermes_state_loop_guard as guard

# Loop-side reads that stay on the loop on purpose (``(module, function)`` of the caller), each because
# it is paired, in one loop step, with loop-only state:
KEPT_ON_LOOP = {
    # The pending/handle snapshot published under ``event_stream.lock`` with the stream's own state.
    ('gateway/session_authority.py', '_handle'), ('gateway/session_authority.py', '_pending_rows'),
    # Restores mutate ``authority.sessions`` and the SessionStore's entries, which only the loop mutates.
    ('gateway/session_local_recovery.py', 'owned_receipt'), ('gateway/session_local_recovery.py', '_checked_entry'),
    ('gateway/session_local_recovery.py', '_with_untracked_compression'),
    ('gateway/session_local_recovery.py', 'restore_local_session'),
    # Compression-lineage lookups used by the synchronous route helpers.
    ('gateway/session_authority.py', 'logical_owner'), ('gateway/session_authority.py', 'physical_target'),
    # Registered in the same step as the delivery waiter: an await here lets a settled drain hand the
    # reply to that waiter and the read then drop it (``admit_message``).
    ('gateway/session_ingress.py', 'admit_message'),
    # A cron fire's exact-retry check before its off-loop creation commit (``_create``).
    ('gateway/session_cron.py', '_create'),
    # The routing index's one-time load in a fresh process.
    ('gateway/session_persistence.py', '_load_routing_rows_locked'),
}
# The hot paths: a running cron job's poll (every <= 2 s per job) and the FIFO claim's reads (every
# turn). Their direct reads (the innermost caller of the store) must never be on the loop again.
MOVED_OFF_LOOP = {('gateway/session_cron.py', 'operation'), ('gateway/session_cron.py', '_read_admission'),
                  ('gateway/session_authority.py', '_claim_next'), ('gateway/session_results.py', 'admission_result')}
HOT_READS = {('hermes_state_runtime.py', 'get_session_admission'), ('hermes_state_runtime.py', 'list_session_admissions'),
             ('gateway/session_results.py', 'admission_result')}


@pytest.fixture
def recording():
    guard.install('record')
    guard.clear()
    try:
        yield guard
    finally:
        guard.uninstall()


def _site(record):
    root = os.path.dirname(os.path.abspath(guard.__file__)) + os.sep
    path = record['site'].rsplit(':', 1)[0]
    return path.removeprefix(root), record['function']


def test_guard_is_inert_until_installed(tmp_path):
    from hermes_state import SessionDB
    guard.uninstall()
    assert not getattr(SessionDB, '_loop_guard_installed', False)
    db = SessionDB(tmp_path / 'state.db')

    async def on_loop():
        return db.get_session('missing')
    try:
        assert asyncio.run(on_loop()) is None
        assert guard.recorded() == []
    finally:
        db.close()


def test_raise_mode_refuses_on_the_loop_thread_and_allows_workers(tmp_path):
    from hermes_state import SessionDB
    db = SessionDB(tmp_path / 'state.db')
    guard.install('raise')

    async def probe():
        with pytest.raises(guard.LoopSessionDBCall, match='event-loop thread'):
            db.get_session('missing')
        with pytest.raises(guard.LoopSessionDBCall):
            db.create_session('s', source='cli')
        assert await asyncio.to_thread(db.get_session, 'missing') is None  # a worker is fine
        await asyncio.to_thread(db.create_session, 's', source='cli')
    try:
        asyncio.run(probe())
        assert db.get_session('s') is not None  # no loop on this thread: allowed
    finally:
        guard.uninstall()
        db.close()


def test_gateway_startup_installs_the_guard_from_config_yaml(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from gateway import run
    from gateway.run_startup_liveness import start_loop_liveness_guards
    (tmp_path / 'config.yaml').write_text('gateway:\n  sessiondb_loop_guard: warn\n', encoding='utf-8')
    monkeypatch.setattr(run, '_hermes_home', tmp_path)
    owner = SimpleNamespace(config=SimpleNamespace(loop_watchdog=False))
    loop = asyncio.new_event_loop()
    try:
        start_loop_liveness_guards(owner, loop)
        from hermes_state import SessionDB
        assert SessionDB._loop_guard_installed and guard._state['mode'] == 'warn'
        guard.uninstall()
        (tmp_path / 'config.yaml').write_text('gateway: {}\n', encoding='utf-8')
        start_loop_liveness_guards(owner, loop)
        assert not getattr(SessionDB, '_loop_guard_installed', False)  # unset: nothing wrapped
    finally:
        guard.uninstall()
        loop.close()


@pytest.mark.asyncio
async def test_cron_run_and_inbound_turn_touch_sessiondb_on_the_loop_only_where_named(
        tmp_path, monkeypatch, recording):
    from cron import jobs, scheduler
    from cron.scheduler_authority import run_canonical_job
    from gateway import run
    from gateway.config import GatewayConfig, Platform, PlatformConfig
    from gateway.platforms.base import SendResult
    from gateway.platforms.event import MessageEvent
    from gateway.run import GatewayRunner
    from gateway.session_authority import initialize_session_authority
    from gateway.session_ingress_context import register_transport_home
    from tests.gateway.test_command_bypass_active_session import _StubAdapter

    monkeypatch.setenv('HERMES_HOME', str(tmp_path))
    monkeypatch.setenv('TELEGRAM_ALLOWED_USERS', '42')
    monkeypatch.setattr('gateway.run._hermes_home', tmp_path)
    monkeypatch.setattr('hermes_state.DEFAULT_DB_PATH', tmp_path / 'state.db')
    monkeypatch.setattr(run, '_load_gateway_config', lambda *a: {'model': {}, 'platform_toolsets': {'cli': []}})
    monkeypatch.setattr(jobs, 'get_job', lambda jid: {'id': jid, 'prompt': 'p', 'model': 'm', 'name': 'job'})
    release = threading.Event()

    def run_job(job, *, extra_prompt, execution_id, cancel_event):
        release.wait(20)
        return (True, 'doc', 'answer', None)
    monkeypatch.setattr(scheduler, 'run_job', run_job)
    runner = GatewayRunner(GatewayConfig(multiplex_profiles=False, sessions_dir=tmp_path / 'sessions'))
    adapter = _StubAdapter(PlatformConfig(enabled=True, token='fixture', typing_indicator=False), Platform.TELEGRAM)
    sent = []

    async def send(chat_id, content, **kwargs):
        sent.append(content)
        return SendResult(success=True, message_id=str(len(sent)))
    monkeypatch.setattr(adapter, 'send', send)

    async def run_agent(**kwargs):
        return {'final_response': 'pong', 'messages': [], 'tools': [], 'history_offset': 0, 'last_prompt_tokens': 0}
    runner._run_agent = run_agent
    adapter.gateway_runner = runner
    runner.adapters = {Platform.TELEGRAM: adapter}
    register_transport_home(runner, None, tmp_path)
    await initialize_session_authority(runner, profile_id=str(tmp_path), instance_id='owner')
    runner._wire_adapter_handlers(adapter)
    recording.clear()  # boot is not under test
    try:
        firer = asyncio.create_task(asyncio.to_thread(run_canonical_job, {'id': 'job'}, execution_id='fire'))
        await asyncio.sleep(1.5)  # health: allow HX013 -- the firer polls the running job's status meanwhile
        source = adapter.build_source(chat_id='42', chat_type='dm', user_id='42')
        await adapter.handle_message(MessageEvent(text='ping', source=source, message_id='m1'))
        for _ in range(200):
            if 'pong' in sent:
                break
            await asyncio.sleep(0.05)  # health: allow HX013 -- waits for the turn's reply
        release.set()
        assert await asyncio.wait_for(firer, 20) == (True, 'doc', 'answer', None)
        records = recording.recorded()
    finally:
        release.set()
        await asyncio.to_thread(runner.session_store.close_all_db_handles)
    assert 'pong' in sent, sent
    sites = {_site(record) for record in records}

    def hot_caller(record):
        """The frame that called a moved ledger read, if this record is one."""
        frames = [_site({'site': f'{f.filename}:{f.lineno}', 'function': f.name}) for f in record['stack']]
        for inner, outer in zip(reversed(frames), list(reversed(frames))[1:]):
            if inner in HOT_READS:
                return outer
        return None
    regressed = sorted({caller for caller in map(hot_caller, records) if caller in MOVED_OFF_LOOP})
    assert not regressed, f'a cron poll or FIFO claim read SessionDB on the owner loop again from {regressed}'
    unexpected = sorted(sites - KEPT_ON_LOOP)
    assert not unexpected, f'new SessionDB call sites on the owner loop: {unexpected}'
