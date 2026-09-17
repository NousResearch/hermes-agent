"""Salt R3 composed R4 probes (B2/B3), ported as regression cells."""
import json, threading
import pytest
from cron import executions, scheduler, bot_chat_delivery, jobs
from tools import bot_live_delivery as mailbox
from tests.cron.test_mixed_delivery_retention import setup
from tests.cron.test_r4_salt_recovery_probes import restart, fail_bot_store


def test_real_scheduler_manifest_park_races_active_drain(setup,monkeypatch):
    home,cfg,run,sent=setup
    entered=threading.Event();release=threading.Event();threads=[];errors=[]
    # Consumer edge only, not a real model launch; receipt claim/completion is production.
    def consume(job,content,profile,*,deferred):
        entered.set();assert release.wait(10);return None
    from cron import scheduler_delivery
    original_delivery=scheduler_delivery._deliver_to_bot_chat
    def route(job,content,profile,**kwargs):
        if kwargs.get('deferred') is not None:return consume(job,content,profile,**kwargs)
        return original_delivery(job,content,profile,**kwargs)
    monkeypatch.setattr('cron.scheduler_delivery._deliver_to_bot_chat',route)
    orig_record=executions.record_delivery_manifest
    def fail(eid,manifest):
        if 'bot' not in manifest:return orig_record(eid,manifest)
        monkeypatch.setattr(mailbox,'find_canonical_owner',lambda h:None)
        def drain():
            try:bot_chat_delivery.drain()
            except Exception as e:errors.append(e)
        t=threading.Thread(target=drain);threads.append(t);t.start()
        assert entered.wait(10)
        raise OSError('recorder unavailable while receipt consumer active')
    monkeypatch.setattr(executions,'record_delivery_manifest',fail)
    original_attach=executions.journal_manifest
    def attach(eid,manifest):
        original_attach(eid,manifest)
        assert executions.journaled_manifests()[eid]==manifest
        release.set()
        for t in threads:t.join(10)
        assert not errors and all(not t.is_alive() for t in threads)
    monkeypatch.setattr(executions,'journal_manifest',attach)
    job,before=run(False)
    monkeypatch.setattr(executions,'record_delivery_manifest',orig_record)
    records=bot_chat_delivery._records(bot_chat_delivery._root())
    restart(home)
    after=executions.get_execution(before['id'])
    print('COMPOSED_RACE',json.dumps({'receipt':records[0][1],'execution':after,'queued':jobs.get_job(job['id']).get('last_delivery_queued'),'native_sends':len(sent)}))
    assert after['delivery_outcome']=='delivered'
    assert not jobs.get_job(job['id']).get('last_delivery_queued')


def test_external_store_fault_recovers_after_bot_failure_without_false_delivery(setup,monkeypatch):
    home,cfg,run,sent=setup
    original=fail_bot_store(monkeypatch)
    job,before=run(True);scheduler.drain_delivery_queue({},None)
    mid=executions.get_execution(before['id'])
    restart(home,fault=True)
    # Drive the pending receipt to an actual failed attempt through production drain.
    monkeypatch.setattr(mailbox,'find_canonical_owner',lambda h:None)
    monkeypatch.setattr('cron.scheduler_delivery._deliver_to_bot_chat',lambda *a,**k:'consumer failed')
    bot_chat_delivery.drain()
    monkeypatch.setattr(executions,'_store_manifest',original)
    restart(home)
    after=executions.get_execution(before['id'])
    receipt=bot_chat_delivery._records(bot_chat_delivery._root())[0][1]
    print('FAULT_RECOVERY',json.dumps({'mid':mid,'after':after,'receipt_status':receipt['status'],'queued':jobs.get_job(job['id']).get('last_delivery_queued'),'native_sends':len(sent)}))
    assert after['delivery_outcome']=='unknown'


def test_pruned_row_cannot_be_reconstructed_from_parked_manifest(setup,monkeypatch):
    home,cfg,run,sent=setup
    original=fail_bot_store(monkeypatch)
    job,before=run(True);scheduler.drain_delivery_queue({},None)
    monkeypatch.setattr(executions,'MAX_TERMINAL_EXECUTIONS',0)
    with executions._transaction() as conn:executions._prune_unlocked(conn)
    # Fixed contract: a journaled (outstanding-manifest) run is fenced from retention.
    assert executions.get_execution(before['id']) is not None
    monkeypatch.setattr(executions,'_store_manifest',original)
    cfg(True);bot_chat_delivery.drain();restart(home)
    receipt=bot_chat_delivery._records(bot_chat_delivery._root())[0][1]
    print('PRUNED_REPLAY',json.dumps({'execution':executions.get_execution(before['id']),'receipt_status':receipt['status'],'journaled':before['id'] in executions.journaled_manifests(),'job':jobs.get_job(job['id'])}))
    assert executions.get_execution(before['id']) is not None
