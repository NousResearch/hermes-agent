"""Legacy migration, identity and packaged executable capability contract."""
from contextlib import closing
from dataclasses import replace
from types import SimpleNamespace

import pytest
from hermes_cli import kanban_db as kb, kanban_db_connect as kbc, kanban_db_surface as s

# Frozen accepted Slice1 table, not reconstructed by removing new columns.
LEGACY = '''CREATE TABLE kanban_delivery_receipts (
 id INTEGER PRIMARY KEY AUTOINCREMENT, task_id TEXT NOT NULL, task_incarnation INTEGER NOT NULL,
 desired_revision INTEGER NOT NULL, delivered_revision INTEGER, platform TEXT NOT NULL,
 chat_id TEXT NOT NULL, thread_id TEXT NOT NULL DEFAULT '', notifier_profile TEXT,
 routing_metadata TEXT, destination_message_id TEXT, destination_profile TEXT,
 renderer_version TEXT, renderer_hash TEXT, owner_epoch INTEGER NOT NULL DEFAULT 0,
 owner_id TEXT, lease_expires_at INTEGER, attempt_id TEXT, attempt_count INTEGER NOT NULL DEFAULT 0,
 state TEXT NOT NULL DEFAULT 'pending', retry_disposition TEXT, last_error TEXT,
 replacement_budget INTEGER NOT NULL DEFAULT 1, created_at INTEGER NOT NULL, updated_at INTEGER NOT NULL,
 UNIQUE (task_id, task_incarnation, platform, chat_id, thread_id),
 CHECK (state IN ('pending', 'sent', 'unknown', 'failed', 'deleted')))'''


def _plugin_scope():
    return {
        'routes': [dict(profile='default', platform='telegram', chat_id='-100', thread_id='7')],
        'task_resources': [dict(board='default', task_id='t_12345678')],
    }


def test_legacy_receipts_migrate_once_without_guessing_attempt_revision(tmp_path):
    path=tmp_path/'legacy.db'
    with closing(kbc.connect(path)) as c:
        tid=kb.create_task(c,title='legacy synthetic')
        rev=s.get_task_source(c,tid).current_revision
        c.execute('DROP TABLE kanban_delivery_receipts'); c.execute(LEGACY)
        for chat,state,msg,attempt in [('1','sent','701','known'),('2','pending',None,'lost'),('3','pending',None,None)]:
            c.execute('INSERT INTO kanban_delivery_receipts(task_id,task_incarnation,desired_revision,delivered_revision,platform,chat_id,notifier_profile,destination_message_id,state,attempt_id,created_at,updated_at) VALUES(?,?,?,?,?,?,?,?,?,?,0,0)',
                      (tid,rev,rev,rev if msg else None,'telegram',chat,'a',msg,state,attempt))
    kb._INITIALIZED_PATHS.discard(path.resolve())
    kb.init_db(db_path=path)
    with closing(kbc.connect(path)) as c:
        rows=s.list_delivery_receipts(c,task_id=tid)
        assert [(r.state,r.destination_message_id,r.attempted_revision) for r in rows]==[('sent','701',None),('unknown',None,None),('pending',None,None)]
        before=[tuple(r) for r in c.execute('SELECT * FROM kanban_delivery_receipts ORDER BY id')]
        s.migrate_delivery_receipts(c)
        assert before==[tuple(r) for r in c.execute('SELECT * FROM kanban_delivery_receipts ORDER BY id')]
        with pytest.raises(s.DeliveryReceiptUnknown): s.claim_delivery_receipt(c,rows[1].id,owner_id='new')
        assert s.claim_delivery_receipt(c,rows[2].id,owner_id='new').attempt_id
        kb.add_comment(c,tid,'synthetic','new')
        new=s.get_task_source(c,tid).current_revision
        kept=s.ensure_delivery_receipt(c,task_id=tid,platform='telegram',chat_id='1',notifier_profile='a',desired_revision=new)
        assert s.claim_delivery_receipt(c,kept.id,owner_id='new').receipt.destination_message_id=='701'


def test_board_profile_and_source_incarnation_isolation(tmp_path,monkeypatch):
    monkeypatch.setattr(kb,'_new_task_id',lambda:'t_shared')
    ids=[]
    for board in ('alpha','beta'):
        with closing(kbc.connect(tmp_path/(board+'.db'))) as c:
            tid=kb.create_task(c,title='synthetic')
            rev=s.get_task_source(c,tid).current_revision
            a=s.ensure_delivery_receipt(c,task_id=tid,platform='telegram',chat_id='-100',thread_id='7',notifier_profile='a',desired_revision=rev)
            b=s.ensure_delivery_receipt(c,task_id=tid,platform='telegram',chat_id='-100',thread_id='7',notifier_profile='b',desired_revision=rev)
            assert a.id!=b.id
            lease=s.claim_delivery_receipt(c,a.id,owner_id='a')
            s.reconcile_delivery_outcome(c,a.id,owner_id='a',owner_epoch=lease.owner_epoch,attempt_id=lease.attempt_id,desired_revision=rev,state='sent',message_id=board,destination_profile='a',delivered_revision=rev)
            assert s.get_delivery_receipt(c,b.id).destination_message_id is None
            # Canonical delete/recreate uses the same external task ID, new created event.
            c.execute('DELETE FROM tasks WHERE id=?',(tid,))
            kb.create_task(c,title='new incarnation')
            new=s.get_task_source(c,tid)
            assert new.task_incarnation!=a.task_incarnation
            with pytest.raises(s.DeliveryReceiptIdentityError): s.claim_delivery_receipt(c,a.id,owner_id='stale')
            fresh=s.ensure_delivery_receipt(c,task_id=tid,platform='telegram',chat_id='-100',thread_id='7',notifier_profile='a',desired_revision=new.current_revision)
            assert fresh.id!=a.id and fresh.destination_message_id is None
            ids.append(s.get_delivery_receipt(c,a.id).destination_message_id)
    assert ids==['alpha','beta']


@pytest.mark.parametrize('settings', [{},{'enabled':True},{'enabled':'true','durable_cards':True}])
def test_package_default_cards_off(settings):
    from hermes_telegram_experience import register
    calls=[]
    config={**settings, 'scope': _plugin_scope()}
    ctx=SimpleNamespace(live_todo_capability=2,register_live_todo=lambda f,**kw:calls.append('todo'),
                        get_config=lambda key,default:config.get(key,default))
    register(ctx)
    assert calls==(['todo'] if settings.get('enabled') is True else [])


def test_missing_card_capability_fails_before_partial_registration():
    from hermes_telegram_experience import register
    calls=[]
    settings={'enabled':True,'durable_cards':True,'scope':_plugin_scope()}
    ctx=SimpleNamespace(live_todo_capability=2,register_live_todo=lambda f,**kw:calls.append('todo'),
                        get_config=lambda key,default: settings.get(key,default))
    with pytest.raises(RuntimeError,match='task_card capability 2'): register(ctx)
    assert not calls


def test_plugin_renderer_bounded_sanitized_status_and_pure():
    from gateway.kanban_surfaces import TaskCardSnapshot
    from hermes_telegram_experience.cards import render_task_card
    snapshot=TaskCardSnapshot('a','board','t_test',1,2,'review','secret=synthetic https://example.invalid /Users/synthetic/file '+('😀'*5000),'worker',1)
    text=render_task_card(snapshot)
    assert text.endswith('\nIn review')
    assert 'example.invalid' not in text and '/Users/' not in text and 'secret=synthetic' not in text
    assert len(text.encode('utf-16-le'))//2<=4096
    assert render_task_card(replace(snapshot,status='blocked')).endswith('\nWaiting')
    assert 'Completed' in render_task_card(replace(snapshot,status='done'))
    assert snapshot.revision==2 and snapshot.status=='review'
