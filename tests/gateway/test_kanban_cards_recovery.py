"""Durable process recovery and subscription-generation contracts; no network."""
import asyncio
from contextlib import closing
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

if __name__ == '__main__':
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

spec = importlib.util.spec_from_file_location('card_harness', Path(__file__).with_name('test_kanban_cards_integration.py'))
assert spec is not None and spec.loader is not None
h = importlib.util.module_from_spec(spec)
spec.loader.exec_module(h)
rig = h.rig


async def child(home, mode):
    home.mkdir(exist_ok=True)
    os.environ['HERMES_HOME'] = str(home)
    path = home/'board.db'
    os.environ['HERMES_KANBAN_DB'] = str(path)
    with closing(h.kbc.connect(path)) as c:
        row = c.execute('SELECT id FROM tasks').fetchone()
        if row:
            tid = row[0]
            task = h.kb.get_task(c, tid)
            assert task is not None
            assert h.kb.edit_task(c, tid, title=task.title + ' reopened')
            # Deterministic lease-clock advance after the old process has died.
            c.execute('UPDATE kanban_delivery_receipts SET lease_expires_at=0 WHERE state="pending"')
        else:
            tid = h.kb.create_task(c, title='synthetic restart task')
            h.notify.add_notify_sub(c, task_id=tid, platform='telegram', chat_id='-100', thread_id='7', notifier_profile='default')
    h.config(home, task_id=tid)
    manager = h.get_plugin_manager(); manager.discover_and_load()
    bot = h.Bot()
    if mode == 'unknown': bot.failure = h.TimedOut('synthetic uncertainty')
    if mode == 'crash':
        async def accepted_but_local_ack_lost(**kw):
            print('PAUSED_AFTER_REMOTE_ACCEPT', flush=True)
            await asyncio.Event().wait()
        bot.send_message = accepted_but_local_ack_lost
    adapter = h.TelegramAdapter(h.PlatformConfig(enabled=True, token='123:synthetic', typing_indicator=False))
    adapter._bot = bot
    runner = h.Runner(home, adapter)
    r = SimpleNamespace(home=home,path=path,manager=manager,bot=bot,adapter=adapter,runner=runner,tid=tid)
    # Exercise the actual watcher, stopping its clock after one production tick.
    real_sleep = asyncio.sleep
    async def clock(delay):
        if delay == 5: return
        runner._running = False
        await real_sleep(0)
    asyncio.sleep = clock
    try:
        await runner._kanban_notifier_watcher(interval=1)
    finally:
        asyncio.sleep = real_sleep
    sources = tuple(manager._task_card_registration.sources)
    if sources: await asyncio.gather(*(s.task for s in sources))
    manager.unload()
    receipt = h.receipt(r)
    print(json.dumps(dict(pid=os.getpid(), sent=bot.sent, edited=bot.edited,
                         state=receipt.state, id=receipt.destination_message_id,
                         delivered=receipt.delivered_revision, desired=receipt.desired_revision)), flush=True)


def launch(home, mode):
    result = subprocess.run([sys.executable, str(Path(__file__).resolve()), str(home), mode],
                            capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stdout + result.stderr
    return json.loads(result.stdout.splitlines()[-1])


@pytest.mark.parametrize('first', ['known', 'unknown', 'crash'])
def test_actual_subprocess_stop_reopen_watcher_and_board(tmp_path, first):
    home = tmp_path/'restart'
    if first == 'crash':
        process = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), str(home), first],
                                   stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        try:
            assert process.stdout is not None
            assert process.stdout.readline().strip() == 'PAUSED_AFTER_REMOTE_ACCEPT'
        finally:
            process.terminate()
            process.communicate(timeout=10)
        old_pid = process.pid
        assert process.returncode != 0
    else:
        old = launch(home, first)
        old_pid = old['pid']
        assert len(old['sent']) == 1
    new = launch(home, 'reopen')
    assert new['pid'] != old_pid
    assert not new['sent']
    if first == 'known':
        assert len(new['edited']) == 1 and new['edited'][0]['message_id'] == 701
        assert new['state'] == 'sent' and new['delivered'] == new['desired']
    else:
        assert not new['edited'] and new['state'] == 'unknown'
        again = launch(home, 'reopen')
        assert again['state'] == 'unknown' and not again['sent']


@pytest.mark.asyncio
async def test_host_reconciliation_after_lost_local_ack_and_duplicate_evidence(rig):
    rig.bot.failure = h.TimedOut('acknowledgement lost')
    await h.tick(rig)
    old = h.receipt(rig)
    revision = h.advance(rig)
    await h.tick(rig)
    assert len(rig.bot.sent) == 1
    # Independent exact transport evidence is a host input, NEVER a plugin API.
    with closing(h.kbc.connect(rig.path)) as c:
        row = c.execute('SELECT * FROM kanban_delivery_receipts WHERE id=?',(old.id,)).fetchone()
        evidence = dict(owner_id=row['attempt_owner_id'],owner_epoch=old.owner_epoch,
                        attempt_id=old.attempt_id,desired_revision=old.attempted_revision,
                        delivered_revision=old.attempted_revision,state='sent',message_id='701',destination_profile='default')
        one = h.receipts.reconcile_delivery_outcome(c, old.id, **evidence)
        two = h.receipts.reconcile_delivery_outcome(c, old.id, **evidence)
        assert one == two and two.delivered_revision == old.attempted_revision and two.desired_revision == revision
        with pytest.raises(h.receipts.DeliveryReceiptLeaseLost):
            h.receipts.reconcile_delivery_outcome(c, old.id, **dict(evidence,message_id='999'))
    rig.bot.failure = None
    await h.tick(rig)
    assert len(rig.bot.sent) == 1 and rig.bot.edited[-1]['message_id'] == 701
    with closing(h.kbc.connect(rig.path)) as c:
        with pytest.raises(h.receipts.DeliveryReceiptLeaseLost):
            h.receipts.reconcile_delivery_outcome(c, old.id, **evidence)


@pytest.mark.asyncio
@pytest.mark.parametrize('change', ['profile', 'topic'])
async def test_subscription_a_b_a_generation_fences_paused_writer(rig, change):
    # Hold before network admission, then revoke/recreate the exact route.
    lock = rig.adapter._chat_send_lock('-100')
    await lock.__aenter__()
    await h.tick(rig,False)
    await asyncio.sleep(0.05)
    source = next(iter(rig.manager._task_card_registration.sources))
    old_token = source.data['binding_token']
    with closing(h.kbc.connect(rig.path)) as c:
        c.execute('DELETE FROM kanban_notify_subs WHERE task_id=?',(rig.tid,))
        h.notify.add_notify_sub(c,task_id=rig.tid,platform='telegram',chat_id='-100',
                               thread_id='8' if change=='topic' else '7',
                               notifier_profile='b' if change=='profile' else 'default')
        c.execute('DELETE FROM kanban_notify_subs WHERE task_id=?',(rig.tid,))
        h.notify.add_notify_sub(c,task_id=rig.tid,platform='telegram',chat_id='-100',thread_id='7',notifier_profile='default')
        assert c.execute('SELECT binding_token FROM kanban_notify_subs').fetchone()[0] != old_token
    await lock.__aexit__(None,None,None)
    await asyncio.wait_for(source.task,5)
    assert not rig.bot.sent and not source.admitted()
    with closing(h.kbc.connect(rig.path)) as c: c.execute('UPDATE kanban_delivery_receipts SET retry_at=0')
    await h.tick(rig)
    assert len(rig.bot.sent)==1 and rig.bot.sent[0]['message_thread_id']==7


@pytest.mark.asyncio
async def test_multiple_chat_topic_receipts_and_unsubscribed_task(rig):
    with closing(h.kbc.connect(rig.path)) as c:
        other = h.kb.create_task(c,title='must not be exposed')
        for chat,thread in [('-100','8'),('-200','7')]:
            h.notify.add_notify_sub(c,task_id=rig.tid,platform='telegram',chat_id=chat,thread_id=thread,notifier_profile='default')
    await h.tick(rig)
    assert {(m['chat_id'],m['message_thread_id']) for m in rig.bot.sent} == {(-100,7),(-100,8),(-200,7)}
    with closing(h.kbc.connect(rig.path)) as c:
        assert len(h.receipts.list_delivery_receipts(c,task_id=rig.tid))==3
        assert not h.receipts.list_delivery_receipts(c,task_id=other)
    await h.tick(rig)
    assert len(rig.bot.sent)==3


@pytest.mark.asyncio
async def test_known_unknown_edit_retries_same_id_and_deleted_budget_exhausts(rig):
    await h.tick(rig)
    rig.bot.edit_failure=h.TimedOut('ambiguous edit')
    h.advance(rig); await h.tick(rig)
    assert h.receipt(rig).state=='unknown'
    rig.bot.edit_failure=None
    await h.tick(rig)
    assert len(rig.bot.sent)==1 and rig.bot.edited[-1]['message_id']==701
    rig.bot.edit_failure=h.BadRequest('Message to edit not found')
    h.advance(rig); await h.tick(rig); await h.tick(rig)
    assert len(rig.bot.sent)==2 and h.receipt(rig).destination_message_id=='702'
    h.advance(rig); await h.tick(rig); await h.tick(rig)
    assert len(rig.bot.sent)==2 and h.receipt(rig).state=='deleted'


if __name__ == '__main__':
    asyncio.run(child(Path(sys.argv[1]),sys.argv[2]))
