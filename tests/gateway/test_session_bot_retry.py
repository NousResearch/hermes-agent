"""The owner retries a transiently failed Bot DM delivery exactly once, never unknown execution.

``gateway/session_bot.py`` admits a Bot Chat delivery into the authority FIFO. When that admission
SETTLES ``failed`` with a committed result whose error classifies transient
(``tools.bot_failure_reasons.result_retry_action``), one retry is admitted under the derived
identity ``bot:<id>:retry``; the delivery's receipt follows it. A repeated ``deliver`` of the same
id or an owner restart (``recover_bot_deliveries``) re-reads that admission instead of minting
another, and an unknown/non-transient admission is never replayed.
"""
import asyncio
from types import SimpleNamespace

import pytest
import pytest_asyncio

from gateway.session_contract import Principal
from hermes_state_runtime import list_session_admissions

KEY = 'c' * 32


@pytest_asyncio.fixture
async def bot(tmp_path, monkeypatch):
    from gateway import run
    from gateway.config import GatewayConfig
    from gateway.session import SessionStore
    from gateway.session_authority import initialize_session_authority
    from gateway.session_local import create_local_session
    from gateway.session_local_title import title_new_session
    from hermes_state import SessionDB

    monkeypatch.setenv('HERMES_HOME', str(tmp_path))
    monkeypatch.setattr(run, '_load_gateway_config', lambda: {'platform_toolsets': {'cli': []}})
    store = SessionStore(tmp_path / 'sessions', GatewayConfig())
    store._db = SessionDB(db_path=tmp_path / 'state.db')
    runner = SimpleNamespace(session_store=store, _session_db=store._db, adapters={}, _draining=False,
                             _evict_cached_agent=lambda route: None)
    runner._adapter_for_source = lambda source: runner.adapters.get(source.platform)
    authority = await initialize_session_authority(runner, profile_id='default', instance_id='fixture')
    owner = Principal('uid:1000', 'default', frozenset({'session:create', 'session:read', 'session:submit'}), 'native')
    chat = create_local_session(authority, owner, {'request_id': 'bot', 'source': 'gui', 'cwd': str(tmp_path),
                                                    'model': 'fixture', 'toolsets': []})
    title_new_session(authority, chat, 'Bot Chat')
    errors = []

    async def execute(authority, ref, row):
        error = errors.pop(0) if errors else None
        if error is None:
            return 'pong'
        authority.pending_results[row['admission_id']] = {
            'result': {'final_response': '', 'messages': [], 'failed': True, 'error': error}, 'usage': {}}
        return ''

    monkeypatch.setattr('gateway.session_finite.execute_finite_admission', execute)
    try:
        yield SimpleNamespace(authority=authority, chat=chat.session_id, errors=errors, home=tmp_path,
                              connection=SimpleNamespace(authority=authority, actor=owner))
    finally:
        store._db.close()


def _admissions(bot):
    return list_session_admissions(bot.authority.db, session_id=bot.chat, pending_only=False)


def _quiet(bot):
    """Nothing left that could admit another row: no receipt task, mailbox operation or drain runs."""
    authority = bot.authority
    return (not any(not task.done() for task in getattr(authority, '_bot_receipt_tasks', ()))
            and not getattr(authority, '_bot_mailbox_operations', None)
            and all(live.task is None or live.task.done() for live in authority.sessions.values()))


async def _settle_quietly(bot):
    from tests.waiting import async_wait_until
    await async_wait_until(lambda: _quiet(bot), timeout=10, what='receipt tasks and drains to go idle')


async def _settled(bot, count):
    from gateway.session_bot import _result
    from tools.bot_live_delivery import read_delivery_result
    for _ in range(200):
        await asyncio.sleep(0.02)
        rows = _admissions(bot)
        if len(rows) >= count and all(r['status'] == 'terminal' for r in rows):
            await asyncio.sleep(0.05)  # let the receipt task observe the settle
            return _result(bot.authority, (await asyncio.to_thread(read_delivery_result, bot.home, KEY)))
    raise AssertionError(f'admissions never settled: {_admissions(bot)}')


@pytest.mark.asyncio
async def test_transient_failure_retries_exactly_once_and_the_receipt_follows_the_retry(bot):
    from gateway.session_bot import deliver, recover_bot_deliveries
    params = dict(id=KEY, profile='default', message='ping')
    bot.errors[:] = ['Error code: 429 - rate limit exceeded']
    first = await deliver(bot.connection, params)
    receipt = await _settled(bot, 2)
    assert receipt['status'] == 'settled' and receipt['reply'] == 'pong', receipt
    assert receipt['admission_id'] == first['admission_id'] and receipt['retry_admission_id']
    assert [r['request_id'] for r in _admissions(bot)] == ['bot:' + KEY, 'bot:' + KEY + ':retry']
    # A repeated delivery of the same id and an owner restart re-read the one retry; never a third.
    again = await deliver(bot.connection, params)
    assert again['status'] == 'settled' and again['reply'] == 'pong'
    await recover_bot_deliveries(bot.authority)
    await _settle_quietly(bot)
    assert len(_admissions(bot)) == 2


@pytest.mark.asyncio
async def test_retry_that_fails_again_and_non_transient_failures_are_never_replayed(bot):
    from gateway.session_bot import deliver, recover_bot_deliveries
    # The retry itself fails transiently: the sender sees that failure, no second retry.
    bot.errors[:] = ['HTTP 503 server error', 'HTTP 503 server error']
    await deliver(bot.connection, dict(id=KEY, profile='default', message='ping'))
    receipt = await _settled(bot, 2)
    assert receipt['status'] == 'failed' and receipt['retry_admission_id']
    # The relay's Desktop forwards ``reason`` to the sender verbatim (relay.ts postReply).
    assert receipt['reason'] == 'provider_server_error' and '503' in receipt['error'], receipt
    await deliver(bot.connection, dict(id=KEY, profile='default', message='ping'))
    await recover_bot_deliveries(bot.authority)
    await _settle_quietly(bot)
    assert len(_admissions(bot)) == 2
    # Auth failures do not classify transient: settled failed, never retried.
    other = 'd' * 32
    bot.errors[:] = ['Error code: 401 - invalid api key']
    await deliver(bot.connection, dict(id=other, profile='default', message='auth'))
    for _ in range(100):
        await asyncio.sleep(0.02)
        if all(r['status'] == 'terminal' for r in _admissions(bot)) and len(_admissions(bot)) == 3:
            break
    await recover_bot_deliveries(bot.authority)
    await _settle_quietly(bot)
    assert [r['request_id'] for r in _admissions(bot)][2:] == ['bot:' + other]
    from gateway.session_bot import _result
    from tools.bot_live_delivery import read_delivery_result
    auth = _result(bot.authority, (await asyncio.to_thread(read_delivery_result, bot.home, other)))
    assert (auth['status'], auth['reason']) == ('failed', 'provider_auth_or_access'), auth


@pytest.mark.asyncio
async def test_context_overflow_retries_once_unless_the_failed_dm_is_still_an_open_tail(bot):
    """Main re-ran an overflowed delivery once (compress-then-resume: the re-run's own preflight
    compaction does the compress). The owner retry keeps that for an overflow that left no open DM
    row, and refuses (recorded, typed) when the DM is still the durable tail: the owner execution has
    no adoption seam, and a second copy would merge into the unanswered one."""
    from gateway.session_bot import deliver
    from tools.bot_live_delivery import read_delivery_result
    bot.errors[:] = ["This model's maximum context length is 200000 tokens"]
    await deliver(bot.connection, dict(id=KEY, profile='default', message='ping'))
    receipt = await _settled(bot, 2)
    assert (receipt['status'], receipt['reply']) == ('settled', 'pong'), receipt

    other = 'e' * 32
    bot.errors[:] = ["This model's maximum context length is 200000 tokens"]
    bot.authority.db.append_message(bot.chat, 'user', content='overflowed dm')
    await deliver(bot.connection, dict(id=other, profile='default', message='overflowed dm'))
    for _ in range(100):
        await asyncio.sleep(0.02)
        record = (await asyncio.to_thread(read_delivery_result, bot.home, other))
        if (record.get('retry') or {}).get('refused'):
            break
    assert record['retry']['refused'] == 'open_user_tail' and len(_admissions(bot)) == 3
    assert (record['status'], record['reason']) == ('failed', 'context_overflow'), record


@pytest.mark.asyncio
async def test_peer_dm_waiter_follows_the_retry_admission_to_its_answer(bot):
    """``hermes peer dm`` waits on the delivery, not on the original admission id: once that
    fails transiently its future never fires again, and the answer is the retry's."""
    from gateway.platforms.api_server_bot_chat import await_peer_receipt
    from gateway import session_bot
    from gateway.session_bot import deliver
    from tests.waiting import async_wait_until
    gate = asyncio.Event()
    import gateway.session_finite as finite
    execute = finite.execute_finite_admission
    wait_admission, followed = session_bot.peer_wait_admission, []

    def peer_wait_admission(authority, record):
        live = wait_admission(authority, record)
        followed.append(live)
        return live

    async def held(authority, ref, row):
        if row['request_id'].endswith(':retry'):
            await gate.wait()
        return await execute(authority, ref, row)

    finite.execute_finite_admission = held
    session_bot.peer_wait_admission = peer_wait_admission
    try:
        bot.errors[:] = ['Error code: 429 - rate limit exceeded']
        first = await deliver(bot.connection, dict(id=KEY, profile='default', message='ping'))
        waiting = asyncio.create_task(await_peer_receipt(bot.authority, first, 20))
        for _ in range(200):
            await asyncio.sleep(0.02)
            if len(_admissions(bot)) == 2:
                break
        retry_id = _admissions(bot)[1]['admission_id']
        # The waiter re-read the delivery after the original failed and now follows the held retry.
        await async_wait_until(lambda: (retry_id, False) in followed or waiting.done(), timeout=10,
                               what='the peer waiter to follow the retry admission')
        assert not waiting.done(), waiting.result()
        gate.set()
        receipt = await asyncio.wait_for(waiting, 5)
    finally:
        finite.execute_finite_admission = execute
        session_bot.peer_wait_admission = wait_admission
    assert (receipt['status'], receipt['reply']) == ('settled', 'pong'), receipt
    assert receipt['retry_admission_id']


@pytest.mark.asyncio
async def test_receipt_file_stays_pending_while_a_draining_owner_leaves_the_retry_to_its_successor(bot):
    """Senders wait on the published receipt file, not an authority RPC. An original that failed
    transiently while the owner drains is interim there (``claimed``), never the final ``failed``:
    the restarted owner's recovery admits the one retry and the same wait gets its answer."""
    from gateway.session_bot import deliver, recover_bot_deliveries
    from tools.bot_live_delivery import await_delivery_async
    import gateway.session_finite as finite
    started, gate = asyncio.Event(), asyncio.Event()
    execute = finite.execute_finite_admission

    async def held(authority, ref, row):
        if not row['request_id'].endswith(':retry'):
            started.set()
            await gate.wait()
        return await execute(authority, ref, row)

    finite.execute_finite_admission = held
    try:
        bot.errors[:] = ['Error code: 429 - rate limit exceeded']
        await deliver(bot.connection, dict(id=KEY, profile='default', message='ping'))
        await asyncio.wait_for(started.wait(), 5)
        bot.authority.runner._draining = True  # shutdown starts while the original runs
        gate.set()
        for _ in range(200):
            await asyncio.sleep(0.02)
            if _admissions(bot)[0]['status'] == 'terminal':
                break
        await _settle_quietly(bot)  # the original's receipt task has published its interim state
        interim = await await_delivery_async(bot.home, KEY, 0.2)
        assert interim['status'] == 'claimed' and len(_admissions(bot)) == 1, interim
        waiting = asyncio.create_task(await_delivery_async(bot.home, KEY, 10))
        bot.authority.runner._draining = False  # the successor owner's startup recovery
        await recover_bot_deliveries(bot.authority)
        receipt = await asyncio.wait_for(waiting, 10)
    finally:
        finite.execute_finite_admission = execute
    assert (receipt['status'], receipt['reply']) == ('settled', 'pong'), receipt
    assert [r['request_id'] for r in _admissions(bot)] == ['bot:' + KEY, 'bot:' + KEY + ':retry']


@pytest.mark.asyncio
async def test_mailbox_file_lock_held_elsewhere_never_stalls_the_owner_loop(bot):
    """The cross-process mailbox lock guards one receipt write, taken off the event loop: a DM
    runner (another process) holding it delays only that write, never every other session, socket
    and delivery on the owner's loop, and never across an admission await."""
    import sys
    import threading
    import time
    from gateway.session_bot import deliver
    from hermes_cli.active_sessions import _FileLock
    from tools.bot_live_delivery import _locked, _root
    with _locked(bot.home):
        pass
    from tests.waiting import async_wait_until
    held, release = threading.Event(), threading.Event()

    def foreign_holder():
        with _FileLock(_root(bot.home) / '.lock'):
            held.set()
            release.wait(10)  # the test releases it as soon as the loop has been measured
    holder = threading.Thread(target=foreign_holder, daemon=True)
    holder.start()
    assert held.wait(5)

    def queued_behind_holder():
        # Another thread (the off-loop receipt write) is parked in flock behind the holder.
        for ident, frame in sys._current_frames().items():
            if ident in (holder.ident, threading.get_ident()):
                continue
            while frame is not None:
                if frame.f_code.co_name == '_flock':
                    return True
                frame = frame.f_back
        return False
    delivering = asyncio.create_task(deliver(bot.connection, dict(id=KEY, profile='default', message='ping')))
    started = time.monotonic()
    # The loop keeps polling while the write waits off it; a write on the loop would block every poll.
    await async_wait_until(queued_behind_holder, timeout=10, what='the mailbox write to wait off the loop')
    loop_stall = time.monotonic() - started
    release.set()
    receipt = await asyncio.wait_for(delivering, 5)
    assert loop_stall < 0.5, f'event loop blocked {loop_stall:.2f}s behind a foreign mailbox lock holder'
    assert receipt['status'] in {'queued', 'claimed', 'settled'}, receipt
