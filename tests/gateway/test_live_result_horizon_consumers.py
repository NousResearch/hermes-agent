"""Consumers of settled results once a live session's result passed the retry horizon (LC-9).

Cron journals, receipt accounting, ACP, hosted-room and A2A handoffs, and the background pass
itself: it never holds a settled turn's completion and works in bounded, key-range batches.
"""
import asyncio
import json
import threading
import time
from types import SimpleNamespace

import pytest

from hermes_state_mutation_retirement import RETRY_HORIZON_S
from tests.gateway.test_live_result_horizon import HOUR, PAD, _expire, _maintenance, _stored, _turn


@pytest.fixture
def clock(monkeypatch):
    offset = [0.0]
    real = time.time
    monkeypatch.setattr(time, 'time', lambda: real() + offset[0])
    return offset


@pytest.mark.asyncio
async def test_unreconciled_cron_result_is_never_reduced(api, owner, clock, monkeypatch):
    """A scheduler down for a week still recovers the run's own result, never a failed verdict."""
    from cron import jobs
    from cron.scheduler_authority import journal_path
    from gateway import run, session_cron
    from hermes_state_runtime import claim_session_input, settle_session_input
    owner.runner.adapters = {}
    monkeypatch.setattr(owner, '_schedule', lambda ref: None)
    monkeypatch.setattr(jobs, 'get_job', lambda jid: {'id': jid, 'prompt': 'report', 'model': 'm'})
    monkeypatch.setattr(run, '_load_gateway_config', lambda: {'model': {}, 'platform_toolsets': {'cli': []}})
    params = {'job_id': 'job', 'request_id': 'fire', 'extra_prompt': None}
    receipt = await session_cron.operation(owner, 'submit', params)
    journal = journal_path('job', 'fire')
    journal.parent.mkdir(parents=True, exist_ok=True)
    journal.write_text(json.dumps({'params': params, 'receipt': receipt}))
    row = claim_session_input(owner.db, epoch=owner.epoch, session_id=receipt['session_id'])
    expected = [True, 'full report ' + PAD, 'delivery answer', None]
    settle_session_input(owner.db, epoch=owner.epoch, admission_id=row['admission_id'],
                         generation=row['generation'], outcome='completed',
                         result={'result': {'completed': True, 'final_response': expected[2], 'cron_result': expected,
                                            'cron_job_flags': {'quota_hold': True}}, 'usage': {}})
    await _turn(api, owner, 'stamp', 'stamp', {'final_response': 'stamp', 'completed': True})
    await _expire(api, owner, clock)
    recovered = await session_cron.operation(owner, 'recover', params)
    assert recovered['result'] == expected and recovered['job_flags'] == {'quota_hold': True}


@pytest.mark.asyncio
async def test_usage_reads_the_last_turn_accounting_after_the_horizon(api, owner, clock, capsys):
    from gateway.session_contract import Principal
    from gateway.session_controls import AuthorityConnection
    from hermes_cli.gateway_chat_commands import _usage
    _, ref, row = await _turn(api, owner, 'chat', 'work', {
        'completed': True, 'final_response': 'answer ' + PAD, 'model': 'used-model', 'input_tokens': 123,
        'output_tokens': 17, 'total_tokens': 140, 'api_calls': 3, 'estimated_cost_usd': 0.0123})
    connection = SimpleNamespace(authority=owner, actor=Principal(
        'api', owner.profile_id, frozenset({'session:submit'}), 'probe'))

    async def rpc(method, **params):
        if method == 'slash.exec':
            return {'output': 'summary'}
        return await AuthorityConnection.receipt(connection, ref, params)
    view = SimpleNamespace(client=SimpleNamespace(rpc=rpc), session_id=ref.session_id,
                           completions={row['admission_id']: {}}, model='fixture')
    await _usage(view, '')
    fresh = capsys.readouterr().out
    assert 'Input tokens: 123' in fresh and 'Model: used-model' in fresh
    assert 'result_reclaimed' not in await rpc('prompt.receipt', session_id=ref.session_id,
                                               admission_id=row['admission_id'], include_result=True)
    await _expire(api, owner, clock)
    receipt = await rpc('prompt.receipt', session_id=ref.session_id, admission_id=row['admission_id'],
                        include_result=True)
    assert receipt['result_reclaimed'] is True and 'messages' not in receipt['result']
    await _usage(view, '')
    assert capsys.readouterr().out == fresh


@pytest.mark.asyncio
async def test_hosted_rooms_settle_a_reduced_turn_with_its_reply(api, owner, clock, monkeypatch):
    """Local rooms read ``_terminal`` (history, results, callbacks); remote rooms poll the runs
    projection through ``PeerRunsHTTPClient``. A week later both still settle with the reply."""
    from gateway.platforms.api_server_authority_runs import run_projection
    from gateway.session_hosted_rpc import HostedRoomAuthorityRPC
    from hermes_state_runtime import get_session_admission
    from tui_gateway import hosted_room_driver as driver
    from tui_gateway.hosted_room_peer_http import PeerRunsHTTPClient
    reply = 'room reply ' + PAD
    _, _, row = await _turn(api, owner, 'room', 'peer-run', {'final_response': reply, 'completed': True})
    aid = row['admission_id']
    task = SimpleNamespace(task_id='task')
    commits = []
    monkeypatch.setattr(driver.state, 'settle_task', lambda *a, **kw: commits.append(kw) or {})
    runtime = SimpleNamespace(db_path=None, clock=time.time, _publish=lambda *a: None, wakeup=lambda: None)
    peer = SimpleNamespace(_observation_receipt=lambda **kw: {'task_id': 'task', 'execution_generation': 1},
                           _poll_receipt=lambda *a, **kw: run_projection(api, 'peer-run'))

    def readers():
        hosted = SimpleNamespace(authority=owner, callbacks={
            aid: lambda receipt: driver.HostedRoomRuntime._on_terminal(runtime, None, None, receipt)})
        local = HostedRoomAuthorityRPC._terminal(hosted, get_session_admission(owner.db, admission_id=aid), task, 1)
        remote = driver._find_terminal_receipt(PeerRunsHTTPClient.history(
            peer, room_id='room', profile='member', session_id='room', grant='fixture'), task, 1)
        return (local['status'], local['text']), remote.result['text'], commits.pop()
    fresh = readers()
    assert fresh[:2] == (('settled', reply), reply) and (fresh[2]['status'], fresh[2]['result']['text']) == ('settled', reply)
    await _expire(api, owner, clock)
    assert 'reclaimed_at' in _stored(owner)[aid]
    assert readers() == fresh


@pytest.mark.asyncio
async def test_acp_receipt_recovery_delivers_a_reduced_turns_reply(api, owner, clock):
    """An editor whose submit acknowledgement was lost recovers the turn from ``prompt.receipt``."""
    from acp_adapter.gateway_server import GatewayACPAgent
    from gateway.session_contract import Principal
    from gateway.session_controls import AuthorityConnection
    _, ref, row = await _turn(api, owner, 'acp', 'lost-ack', {'completed': True, 'final_response': 'original answer'})
    connection = SimpleNamespace(authority=owner, actor=Principal(
        'api', owner.profile_id, frozenset({'session:read', 'session:submit'}), 'test'))

    async def rpc(method, **params):
        assert method == 'prompt.receipt'
        return await AuthorityConnection.receipt(connection, ref, params)
    delivered = []

    async def update(**kwargs):
        delivered.append(kwargs['update'].content.text)
    agent = GatewayACPAgent()
    agent._conn = SimpleNamespace(session_update=update)
    await _expire(api, owner, clock)
    await agent._settled_receipt(SimpleNamespace(rpc=rpc), ref.session_id, ref.session_id,
                                 {'admission_id': row['admission_id'], 'outcome': 'completed'})
    assert delivered == ['original answer']
    assert agent._terminals[row['admission_id']] == {'outcome': 'completed'}


async def _forwarded_turn(tmp_path, monkeypatch, reply):
    """A real owner and dispatcher connection with one settled A2A-forwarded turn of its own."""
    from gateway.config import GatewayConfig
    from gateway.session import SessionStore
    from gateway.session_authority import initialize_session_authority
    from gateway.session_controls import AuthorityConnection
    from gateway import run
    from hermes_state_runtime import claim_session_input, settle_session_input
    monkeypatch.setenv('HERMES_HOME', str(tmp_path))
    monkeypatch.setattr(run, '_load_gateway_config', lambda: {'model': {'default': 'fixture'}, 'platform_toolsets': {'cli': []}})
    monkeypatch.setattr(run, '_resolve_gateway_model', lambda config: 'fixture')
    store = SessionStore(tmp_path / 'sessions', GatewayConfig())
    runner = run.GatewayRunner.__new__(run.GatewayRunner)
    runner.adapters, runner.session_store, runner._session_db, runner._draining = {}, store, store._db, False
    authority = await initialize_session_authority(runner, profile_id='target', instance_id='test')
    authority._schedule = lambda ref: None
    conn = AuthorityConnection(authority, SimpleNamespace(write=lambda frame: None), {'user_id': 'producer'})
    params = dict(agent='dev', tenant='team', peer='alice', context_id='ctx', input_id='first', text='hello')
    first = (await conn.dispatch({'id': 1, 'method': 'a2a.forward', 'params': params}))['result']
    row = claim_session_input(authority.db, epoch=authority.epoch, session_id=first['session_id'])
    settle_session_input(authority.db, epoch=authority.epoch, admission_id=row['admission_id'],
                         generation=row['generation'], outcome='completed',
                         result={'result': {'completed': True, 'final_response': reply}, 'usage': {}})
    return authority, conn, params, first


async def _reduce(authority, clock):
    from gateway.session_results import reclaim_expired_results
    clock[0] += RETRY_HORIZON_S + HOUR
    await reclaim_expired_results(authority)


@pytest.mark.asyncio
async def test_a2a_reforward_of_a_reduced_input_returns_its_reply(tmp_path, monkeypatch, clock):
    authority, conn, params, first = await _forwarded_turn(tmp_path, monkeypatch, 'answer ' + PAD)
    replay = await conn.dispatch({'id': 2, 'method': 'a2a.forward', 'params': dict(params)})
    assert replay['result']['result']['final_response'] == 'answer ' + PAD
    await _reduce(authority, clock)
    again = await conn.dispatch({'id': 3, 'method': 'a2a.forward', 'params': dict(params)})
    assert again['result']['admission_id'] == first['admission_id']
    assert again['result']['result']['final_response'] == 'answer ' + PAD
    assert 'reclaimed_at' in _stored(SimpleNamespace(db=authority.db))[first['admission_id']]
    await conn.close()


@pytest.mark.asyncio
async def test_receipt_dispatch_of_a_reduced_turn_matches_the_wire_contract(tmp_path, monkeypatch, clock):
    """``prompt.receipt`` through the dispatcher, which checks each result against its contract
    (a violation raises under the test runner): ``result_reclaimed`` is a declared field."""
    authority, conn, _, first = await _forwarded_turn(tmp_path, monkeypatch, 'answer')
    request = {'id': 4, 'method': 'prompt.receipt', 'params': {
        'session_id': first['session_id'], 'admission_id': first['admission_id'], 'include_result': True}}
    fresh = (await conn.dispatch(request))['result']
    assert fresh['result']['final_response'] == 'answer' and 'result_reclaimed' not in fresh
    await _reduce(authority, clock)
    reduced = await conn.dispatch(request)
    assert 'result' in reduced, reduced
    assert reduced['result']['result_reclaimed'] is True and reduced['result']['result']['final_response'] == 'answer'
    await conn.close()


@pytest.mark.asyncio
async def test_reclaim_never_holds_a_settled_turns_completion(api, owner, monkeypatch):
    """The pass is background work: a settled turn's waiter resolves while the pass it spawned is
    still held in its batch, and the FIFO continues once that (bounded) batch commits."""
    import hermes_state_terminal as terminal
    from gateway.session_api_turn import admit_api_turn
    from gateway.session_results import execution_result
    entered, release = threading.Event(), threading.Event()
    backfill = terminal.backfill_result_horizon

    def held(conn, **kwargs):
        entered.set()
        assert release.wait(10)
        return backfill(conn, **kwargs)
    monkeypatch.setattr(terminal, 'backfill_result_horizon', held)

    async def handle(event):
        execution_result.get()['result'] = {'final_response': 'done ' + event.text, 'completed': True}
        return 'done ' + event.text
    owner.runner._handle_message = handle
    monkeypatch.setattr(owner, '_schedule', lambda ref: None)
    rows = [admit_api_turn(api, session_id='s', request_id=name, user_message=name, conversation_history=[])
            for name in ('head', 'next')]
    waiters = [owner.waiters.setdefault(row[2]['admission_id'], asyncio.get_running_loop().create_future())
               for row in rows]
    drain = asyncio.create_task(owner._drain(rows[0][1]))
    try:
        assert await asyncio.wait_for(asyncio.shield(waiters[0]), 5) == 'done head'
        assert await asyncio.to_thread(entered.wait, 5) and not release.is_set()
    finally:
        release.set()
    assert await asyncio.wait_for(asyncio.shield(waiters[1]), 5) == 'done next'
    await asyncio.wait_for(drain, 5)
    await _maintenance(owner)


@pytest.mark.asyncio
async def test_reclaim_works_in_bounded_batches_and_spreads_the_upgrade_wave(api, owner, clock, monkeypatch):
    import hermes_state_terminal as terminal
    from gateway import session_results
    from hermes_state_runtime import admit_session_input, claim_session_input, settle_session_input
    monkeypatch.setattr(session_results, '_RECLAIM_BATCH', 100)
    monkeypatch.setattr(session_results, '_RECLAIM_BATCHES', 3)
    batches = []
    for name in ('backfill_result_horizon', 'reclaim_settled_results'):
        step = getattr(terminal, name)

        def counted(conn, _step=step, _name=name, **kwargs):
            done = _step(conn, **kwargs)
            batches.append((_name, done))
            return done
        monkeypatch.setattr(terminal, name, counted)
    owner.db.create_session('old', source='api_server')
    for index in range(350):
        admit_session_input(owner.db, epoch=owner.epoch, principal_id='api', session_id='old',
                            request_id=str(index), payload={'text': str(index)})
        row = claim_session_input(owner.db, epoch=owner.epoch, session_id='old')
        settle_session_input(owner.db, epoch=owner.epoch, admission_id=row['admission_id'], generation=row['generation'],
                             outcome='completed', result={'result': {'final_response': PAD, 'completed': True}, 'usage': {}})
    # Results settled before this release: no horizon stamp yet.
    owner.db._execute_write(lambda conn: conn.execute(
        "DELETE FROM state_meta WHERE key GLOB 'gateway.admission.result_horizon.*'"))

    def reduced():
        return sum('reclaimed_at' in raw for raw in _stored(owner).values())
    await session_results.reclaim_expired_results(owner)  # the first pass after the upgrade
    clock[0] = HOUR
    await session_results.reclaim_expired_results(owner)
    assert reduced() == 0, 'an upgrade reclaimed results at once'
    clock[0] = RETRY_HORIZON_S + 2 * HOUR
    await session_results.reclaim_expired_results(owner)
    assert reduced() == 300  # one pass is bounded; the rest of the wave waits for the next one
    clock[0] += HOUR
    await session_results.reclaim_expired_results(owner)
    assert reduced() == 350
    assert batches and max(done for _, done in batches) <= 100


def test_reclaim_examines_a_bounded_number_of_stamps(owner):
    """The expiry is in the stamp key: a batch is a key-range scan, whatever is not yet due."""
    from hermes_state_terminal import reclaim_settled_results, stamp_result_horizon
    value = {'result': {'completed': True, 'final_response': 'x'}}
    owner.db._execute_write(lambda conn: [stamp_result_horizon(conn, f'{i:032x}', 'completed', value)
                                          for i in range(100_000)])

    def measure(conn):
        ticks = [0]

        def progress():
            ticks[0] += 1
            return 0
        conn.set_progress_handler(progress, 100)
        try:
            assert reclaim_settled_results(conn, now=time.time() + HOUR, limit=200) == 0
        finally:
            conn.set_progress_handler(None, 0)
        return ticks[0] * 100
    assert owner.db._execute_write(measure) < 20_000


@pytest.mark.asyncio
@pytest.mark.parametrize('aid', ['aaaaaaaaaaaa4aaa8aaaaaaaaaaaaaaa', '00000000000040008000000000000001'])
async def test_stamps_from_an_earlier_draft_format_are_inert(api, owner, clock, monkeypatch, aid):
    """An unreleased draft of this change wrote ``result_horizon.v1.<admission>`` stamps (expiry in
    the value) and a ``v1`` backfill marker. They neither break the pass nor keep a result whole."""
    from uuid import UUID
    import hermes_state_runtime as runtime
    from gateway.session_results import reclaim_expired_results
    monkeypatch.setattr(runtime, 'uuid', SimpleNamespace(uuid4=lambda: UUID(hex=aid)))
    await _turn(api, owner, 'legacy', 'old', {'completed': True, 'final_response': 'answer',
                                              'messages': [{'role': 'assistant', 'content': 'answer'}]})

    def draft_install(conn):
        conn.execute("DELETE FROM state_meta WHERE key GLOB 'gateway.admission.result_horizon.*'")
        conn.executemany('INSERT INTO state_meta(key,value) VALUES(?,?)', [
            ('gateway.admission.result_horizon.v1.' + aid, json.dumps(time.time())),
            ('gateway.admission.result_horizon.backfill.v1', 'null')])
    owner.db._execute_write(draft_install)
    await reclaim_expired_results(owner)  # first sight for this release
    clock[0] += RETRY_HORIZON_S + HOUR
    await reclaim_expired_results(owner)
    assert 'reclaimed_at' in json.loads(_stored(owner)[aid])


def test_backfill_examines_a_bounded_number_of_keys(owner):
    import hermes_state_terminal as terminal

    def seed(conn):
        # Retired results have no live admission but still lie on the backfill's path.
        conn.executemany('INSERT INTO state_meta(key,value) VALUES(?,?)', (
            (terminal.RESULT_PREFIX + f'{i:032x}', '{"result":{}}') for i in range(100_000)))
        conn.execute('INSERT INTO state_meta(key,value) VALUES(?,?)', (terminal._BACKFILL_KEY, json.dumps(f'{99000:032x}')))
    owner.db._execute_write(seed)

    def measure(conn):
        ticks = [0]

        def progress():
            ticks[0] += 1
            return 0
        conn.set_progress_handler(progress, 100)
        try:
            assert terminal.backfill_result_horizon(conn, now=time.time(), limit=200) == 200
        finally:
            conn.set_progress_handler(None, 0)
        return ticks[0] * 100
    assert owner.db._execute_write(measure) < 20_000
