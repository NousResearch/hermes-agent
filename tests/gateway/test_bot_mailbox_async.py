"""Bot receipt polling is local and mailbox serialization never parks the event loop."""
import asyncio
import threading

import pytest

pytest_plugins = ('tests.gateway.test_session_bot_retry',)


def test_canonical_poll_reads_owner_projection_without_handshake(tmp_path, monkeypatch):
    from tools import bot_live_delivery as mailbox
    key = 'a' * 32
    record = dict(delivery_id=key, admission_id='admitted', status='queued', message='hello',
                  profile_home=str(tmp_path), session_id='s', principal_id='p', reply='')
    with mailbox._locked(tmp_path) as root:
        mailbox._write(root / f'{key}.json', record)
    monkeypatch.setattr(mailbox, 'authority_delivery', lambda *a, **k: pytest.fail('poll dialed authority'))
    assert mailbox.read_delivery_result(tmp_path, key)['status'] == 'queued'
    with mailbox._locked(tmp_path) as root:
        mailbox._write(root / f'{key}.json', dict(record, status='settled', reply='owner reply'))
    assert mailbox.read_delivery_result(tmp_path, key)['reply'] == 'owner reply'


@pytest.mark.asyncio
async def test_admission_wait_releases_file_lock_and_concurrent_deliver_keeps_loop_live(bot, monkeypatch):
    from gateway.session_bot import deliver
    from tools.bot_live_delivery import _locked
    entered, release, ticked = asyncio.Event(), asyncio.Event(), asyncio.Event()
    original = bot.authority.admit_automation
    async def held(*args, **kwargs):
        entered.set()
        await release.wait()
        return await original(*args, **kwargs)
    monkeypatch.setattr(bot.authority, 'admit_automation', held)
    first = asyncio.create_task(deliver(bot.connection, dict(id='a' * 32, profile='default', message='first')))
    second = None
    acquired = threading.Event()
    def inspect():
        with _locked(bot.home): acquired.set()
    inspecting = None
    try:
        await asyncio.wait_for(entered.wait(), 5)
        inspecting = asyncio.create_task(asyncio.to_thread(inspect))
        assert await asyncio.to_thread(acquired.wait, 3), 'file lock remained held across admission await'
        second = asyncio.create_task(deliver(bot.connection, dict(id='b' * 32, profile='default', message='second')))
        asyncio.get_running_loop().call_soon(ticked.set)
        await asyncio.wait_for(ticked.wait(), 5)
        assert not second.done()
        release.set()
        responses = await asyncio.wait_for(asyncio.gather(first, second), 10)
        assert responses[0]['admission_id'] != responses[1]['admission_id']
    finally:
        release.set()
        await asyncio.gather(first, *([second] if second else []), *([inspecting] if inspecting else []), return_exceptions=True)
        tasks = list(getattr(bot.authority, '_bot_receipt_tasks', ()))
        if tasks: await asyncio.wait_for(asyncio.gather(*tasks, return_exceptions=True), 10)


@pytest.mark.asyncio
async def test_paused_receipt_rearms_on_resume_without_polling_rpc(bot, monkeypatch):
    from gateway.session_bot import deliver
    from gateway.session_contract import SessionRef
    from tools.bot_live_delivery import read_delivery_result
    schedule = bot.authority._schedule
    monkeypatch.setattr(bot.authority, '_schedule', lambda ref: None)
    key = 'c' * 32
    receipt = await deliver(bot.connection, dict(id=key, profile='default', message='paused'))
    ref = SessionRef(bot.authority.profile_id, receipt['session_id'])
    bot.authority._pause(ref, 'unknown_execution')
    await asyncio.wait_for(asyncio.gather(*list(bot.authority._bot_receipt_tasks)), 5)
    assert (await asyncio.to_thread(read_delivery_result, bot.home, key))['status'] == 'queued'
    monkeypatch.setattr(bot.authority, '_schedule', schedule)
    schedule(ref)
    await bot.authority.sessions[ref.session_id].task
    async with asyncio.timeout(5):
        while (await asyncio.to_thread(read_delivery_result, bot.home, key))['status'] != 'settled':
            await asyncio.sleep(.01)
    assert (await asyncio.to_thread(read_delivery_result, bot.home, key))['reply'] == 'pong'


@pytest.mark.asyncio
async def test_adopted_result_during_unknown_projection_is_not_lost(bot, monkeypatch):
    from gateway import session_bot
    from gateway.session_contract import SessionRef
    from hermes_state_runtime import (begin_runtime_epoch, claim_session_input, recover_session_inputs,
        register_worker_execution, adopt_worker_execution, settle_session_input)
    from tools.bot_live_delivery import read_delivery_result
    monkeypatch.setattr(bot.authority, '_schedule', lambda ref: None)
    key = 'd' * 32
    receipt = await session_bot.deliver(bot.connection, dict(id=key, profile='default', message='unknown'))
    ref = SessionRef(bot.authority.profile_id, receipt['session_id'])
    row = claim_session_input(bot.authority.db, epoch=bot.authority.epoch, session_id=ref.session_id)
    assignment = dict(execution_id='recovered-worker', session_id=ref.session_id, generation=row['generation'])
    register_worker_execution(bot.authority.db, epoch=bot.authority.epoch, **assignment,
                              kind='compute', adoption_secret='owned-worker')
    bot.authority.epoch = begin_runtime_epoch(bot.authority.db, instance_id='recovery')
    recover_session_inputs(bot.authority.db, epoch=bot.authority.epoch)
    bot.authority._pause(ref, 'unknown_execution')
    await asyncio.wait_for(asyncio.gather(*list(bot.authority._bot_receipt_tasks)), 5)
    entered, release = asyncio.Event(), asyncio.Event()
    original = session_bot.write_receipt
    async def held(home, record):
        if record['status'] == 'ambiguous' and not entered.is_set():
            entered.set()
            await release.wait()
        await original(home, record)
    monkeypatch.setattr(session_bot, 'write_receipt', held)
    bot.authority._publish_pending(ref)
    try:
        await asyncio.wait_for(entered.wait(), 5)
        adopt_worker_execution(bot.authority.db, epoch=bot.authority.epoch, **assignment, adoption_secret='owned-worker')
        settle_session_input(bot.authority.db, epoch=bot.authority.epoch, admission_id=row['admission_id'],
            generation=row['generation'], outcome='completed', result={'result': {'final_response': 'adopted reply'}, 'usage': {}})
        bot.authority._publish_pending(ref)
        release.set()
        async with asyncio.timeout(5):
            while (await asyncio.to_thread(read_delivery_result, bot.home, key))['status'] == 'ambiguous':
                await asyncio.sleep(.01)
        assert (await asyncio.to_thread(read_delivery_result, bot.home, key))['status'] == 'settled'
        assert (await asyncio.to_thread(read_delivery_result, bot.home, key))['reply'] == 'adopted reply'
    finally:
        release.set()
        tasks = list(getattr(bot.authority, '_bot_receipt_tasks', ()))
        if tasks: await asyncio.wait_for(asyncio.gather(*tasks, return_exceptions=True), 10)


@pytest.mark.asyncio
async def test_drain_defers_retry_without_publishing_interim_failure(bot, monkeypatch):
    from gateway.session_bot import deliver, recover_bot_deliveries
    from tools.bot_live_delivery import read_delivery_result
    import gateway.session_finite as finite
    execute = finite.execute_finite_admission
    async def drain_after_failure(authority, ref, row):
        result = await execute(authority, ref, row)
        authority.runner._draining = True
        return result
    monkeypatch.setattr(finite, 'execute_finite_admission', drain_after_failure)
    bot.errors[:] = ['Error code: 429 - rate limit exceeded']
    key = 'e' * 32
    await deliver(bot.connection, dict(id=key, profile='default', message='retry after restart'))
    tasks = list(getattr(bot.authority, '_bot_receipt_tasks', ()))
    if tasks: await asyncio.wait_for(asyncio.gather(*tasks, return_exceptions=True), 10)
    assert (await asyncio.to_thread(read_delivery_result, bot.home, key))['status'] == 'claimed'
    from gateway import session_bot
    read = session_bot.read_receipt
    async def drain_during_read(home, key):
        result = await read(home, key)
        bot.authority.runner._draining = True
        return result
    bot.authority.runner._draining = False
    monkeypatch.setattr(session_bot, 'read_receipt', drain_during_read)
    await deliver(bot.connection, dict(id=key, profile='default', message='retry after restart'))
    assert (await asyncio.to_thread(read_delivery_result, bot.home, key))['status'] == 'claimed'
    monkeypatch.setattr(session_bot, 'read_receipt', read)
    monkeypatch.setattr(finite, 'execute_finite_admission', execute)
    bot.authority.runner._draining = False
    await recover_bot_deliveries(bot.authority)
    async with asyncio.timeout(5):
        while (await asyncio.to_thread(read_delivery_result, bot.home, key))['status'] != 'settled':
            await asyncio.sleep(.01)
    assert (await asyncio.to_thread(read_delivery_result, bot.home, key))['retry_admission_id']
