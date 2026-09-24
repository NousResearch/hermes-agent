"""Installed native loader/watcher/PTB dispatcher + canonical task transaction.
Only network socket/HTTP/update origin and synthetic configuration are fixtures.
"""
import asyncio
from contextlib import closing
import importlib
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace
from urllib.parse import parse_qs

import pytest
import pytest_asyncio
pytest.importorskip(
    "hermes_telegram_experience",
    reason="optional external-plugin integration; exercised by plugin repository CI",
)

# tests/gateway/conftest.py installs a Telegram MagicMock when the optional SDK
# has not been imported yet. This installed-plugin contract needs the real SDK
# dispatcher, so replace only that process-local test double before importing it.
mocked_telegram = sys.modules.get('telegram')
if mocked_telegram is not None and getattr(mocked_telegram, '__file__', None) is None:
    for module_name in tuple(sys.modules):
        if module_name == 'telegram' or module_name.startswith('telegram.'):
            sys.modules.pop(module_name, None)
    importlib.invalidate_caches()
from telegram import Update
from gateway.config import Platform, PlatformConfig
from gateway.run import GatewayRunner
from hermes_cli import kanban_db as kb, kanban_db_connect as kbc, kanban_db_notify as notify
from hermes_cli import kanban_db_surface as receipts
from hermes_cli.plugins import get_plugin_manager
from plugins.platforms.telegram import adapter as am, telegram_network, transport_admission as wire

spec = importlib.util.spec_from_file_location('decision_socket_fixture', Path(__file__).parents[1]/'plugins/test_telegram_live_todo_repair.py')
f = importlib.util.module_from_spec(spec); spec.loader.exec_module(f)

class Socket(f.SocketWriter):
    def respond(self):
        if self.responded or self.reader.at_eof(): return
        self.responded = True
        if b'/getMe ' in self.buffer:
            result = dict(id=123, is_bot=True, first_name='Synthetic', username='synthetic_bot')
        elif b'/answerCallbackQuery ' in self.buffer:
            result = True
        else:
            result = dict(message_id=701, date=1, chat={'id':-100,'type':'supergroup'},text='synthetic')
        body = json.dumps(dict(ok=True,result=result)).encode()
        self.reader.feed_data(b'HTTP/1.1 200 OK\r\nConnection: close\r\nContent-Type: application/json\r\nContent-Length: '+str(len(body)).encode()+b'\r\n\r\n'+body)


def configure(r, grants=True):
    import yaml
    grant=dict(id='synthetic-grant-v1',actor=42,profile='default',board='default',task_id=r.tid,actions=['unblock_needs_input'])
    scope=dict(routes=[dict(profile='default',platform='telegram',chat_id='-100',thread_id='7')],task_resources=[dict(board='default',task_id=r.tid)])
    config=dict(plugins=dict(enabled=['hermes-telegram-experience'],entries={'hermes-telegram-experience':dict(settings=dict(enabled=True,durable_cards=True,decisions=True,scope=scope))}),kanban=dict(decision_grants=[grant] if grants else []))
    (r.home/'config.yaml').write_text(yaml.safe_dump(config))

@pytest_asyncio.fixture
async def rig(tmp_path, monkeypatch):
    home=tmp_path/'.hermes';home.mkdir(exist_ok=True);path=home/'board.db'
    monkeypatch.setattr(Path,'home',lambda:tmp_path)
    monkeypatch.setattr('hermes_constants.get_default_hermes_root',lambda:home)
    for k,v in dict(HERMES_HOME=str(home),HERMES_KANBAN_HOME=str(home),HERMES_KANBAN_DB=str(path),TELEGRAM_ALLOWED_USERS='42,43',HERMES_TELEGRAM_DISABLE_FALLBACK_IPS='true').items():monkeypatch.setenv(k,v)
    with closing(kbc.connect(path)) as c:
        existing=c.execute('SELECT id FROM tasks ORDER BY id LIMIT 1').fetchone()
        if existing:
            tid=existing[0]
        else:
            tid=kb.create_task(c,title='Synthetic input decision')
            kb.block_task(c,tid,reason='Synthetic input needed',kind='needs_input')
            notify.add_notify_sub(c,task_id=tid,platform='telegram',chat_id='-100',thread_id='7',chat_type='group',notifier_profile='default')
    r=SimpleNamespace(home=home,path=path,tid=tid,sockets=[],planned=[])
    configure(r)
    r.manager=get_plugin_manager();r.manager.discover_and_load(force=True)
    assert r.manager._task_card_registration.active
    import hermes_telegram_experience
    import sys
    assert hermes_telegram_experience.__file__ is not None
    assert Path(hermes_telegram_experience.__file__).is_relative_to(Path(sys.prefix))
    assert 'site-packages' in hermes_telegram_experience.__file__
    r.adapter=am.TelegramAdapter(PlatformConfig(enabled=True,token='123:synthetic',typing_indicator=False))
    monkeypatch.setattr(am,'resolve_proxy_url',lambda *a,**k:None)
    monkeypatch.setattr(telegram_network,'_resolve_proxy_url',lambda *a,**k:None)
    async def connect(*args,**kwargs):
        sock=r.planned.pop(0) if r.planned else Socket();r.sockets.append(sock)
        return sock.reader,sock
    monkeypatch.setattr(wire.asyncio,'open_connection',connect)
    general,updates=await r.adapter._build_ptb_requests()
    r.app=am.Application.builder().token('123:synthetic').request(general).get_updates_request(updates).build()
    r.adapter._bot=r.app.bot
    r.runner=GatewayRunner.__new__(GatewayRunner)
    r.runner.config=SimpleNamespace(multiplex_profiles=True,profile_routes=[])
    r.runner.adapters={Platform.TELEGRAM:r.adapter};r.runner._profile_adapters={}
    r.runner._primary_profile_name=r.runner._kanban_notifier_profile='default'
    r.runner._profile_failed_platforms={};r.runner._kanban_dispatcher_lock_handle=object()
    r.runner._running=True
    r.adapter.gateway_runner=r.runner
    r.adapter.set_authorization_check(r.runner._make_adapter_auth_check(Platform.TELEGRAM))
    r.adapter._wire_plugin_handlers(r.app);r.adapter._register_handlers(r.app)
    await r.app.initialize()
    yield r
    for sock in r.sockets:sock.release.set();sock.respond()
    r.manager.unload()
    for s in tuple(r.adapter._live_todo_sources):await s.finish()
    await r.app.shutdown()

async def tick(r, monkeypatch, drain=True):
    real_sleep=asyncio.sleep
    async def initial(delay):
        if delay != 5: await real_sleep(delay)
    async def end(interval):r.runner._running=False
    r.runner._running=True
    with monkeypatch.context() as clock:
        clock.setattr(asyncio,'sleep',initial)
        clock.setattr(r.runner,'_sleep_between_ticks',end)
        await r.runner._kanban_notifier_watcher(interval=1)
    if drain:
        tasks=[s.task for s in tuple(r.manager._task_card_registration.sources)]
        if tasks:await asyncio.wait_for(asyncio.gather(*tasks),8)


def readback(r):
    with closing(kbc.connect(r.path)) as c:
        return (kb.get_task(c,r.tid).status,
                [dict(x) for x in c.execute("SELECT * FROM task_events WHERE task_id=? AND kind='task_decision'",(r.tid,))],
                [dict(x) for x in c.execute('SELECT * FROM kanban_action_records')])

def callback(r, token, **changes):
    data=dict(update_id=1,callback_query=dict(id='synthetic-query',chat_instance='synthetic-chat',data='hte:d:'+token,
        **{'from':dict(id=42,is_bot=False,first_name='Synthetic actor')},
        message=dict(message_id=701,date=1,chat=dict(id=-100,type='supergroup',is_forum=True),message_thread_id=7,
                     **{'from':dict(id=123,is_bot=True,first_name='Synthetic bot')})))
    q=data['callback_query']
    if 'actor' in changes:q['from']['id']=changes['actor']
    if 'chat' in changes:q['message']['chat']['id']=changes['chat']
    if 'topic' in changes:q['message']['message_thread_id']=changes['topic']
    if 'message' in changes:q['message']['message_id']=changes['message']
    if changes.get('copied'):q['message']['forward_origin']=dict(type='hidden_user',date=1,sender_user_name='Copied')
    if changes.get('inaccessible'):q['message']['date']=0
    return Update.de_json(data,r.app.bot)


def answers(r):
    return [parse_qs(s.buffer.split(b'\r\n\r\n',1)[1].decode()).get('text',[''])[0]
            for s in r.sockets if b'/answerCallbackQuery ' in s.buffer]

@pytest.mark.asyncio
async def test_native_positive_duplicate_and_coalesced_keyboard(rig,monkeypatch):
    r=rig;await tick(r,monkeypatch)
    status,audit,rows=readback(r)
    assert status=='blocked' and not audit and len(rows)==1
    token=rows[0]['token']
    with closing(kbc.connect(r.path)) as c:
        receipt=receipts.list_delivery_receipts(c,task_id=r.tid)[0]
        assert receipt.state=='sent' and receipt.renderer_hash and receipt.control_hash
    await tick(r,monkeypatch)
    assert len([s for s in r.sockets if b'/editMessageText ' in s.buffer])==1
    await r.app.process_update(callback(r,token))
    assert readback(r)[0]=='ready' and len(readback(r)[1])==1
    assert 'Not completed' in answers(r)[-1]
    await r.app.process_update(callback(r,token))
    assert len(readback(r)[1])==1 and readback(r)[2][0]['state']=='completed'
    assert 'Not completed' in answers(r)[-1]

@pytest.mark.asyncio
@pytest.mark.parametrize('case',['actor','permitted_actor','chat','topic','message','copied','inaccessible','malformed','forged','expired','future','stale_revision','revoked','wrong_resource','wrong_profile','wrong_board','incarnation','state','unsupported','missing_grant'])
async def test_native_denials(rig,monkeypatch,case):
    r=rig;await tick(r,monkeypatch);token=readback(r)[2][0]['token'];changes={}
    mapping=dict(actor=('actor',99),permitted_actor=('actor',43),chat=('chat',-200),topic=('topic',8),message=('message',702),copied=('copied',True),inaccessible=('inaccessible',True))
    if case in mapping:changes=dict([mapping[case]])
    if case=='malformed':token='x:42:default'
    if case=='forged':token='x'*24
    if case in ('revoked','missing_grant'):configure(r,False)
    if case in ('wrong_resource','wrong_profile','wrong_board'):
        import yaml
        config=yaml.safe_load((r.home/'config.yaml').read_text())
        key=dict(wrong_resource='task_id',wrong_profile='profile',wrong_board='board')[case]
        config['kanban']['decision_grants'][0][key]='wrong'
        (r.home/'config.yaml').write_text(yaml.safe_dump(config))
    with closing(kbc.connect(r.path)) as c:
        if case=='expired':c.execute('UPDATE kanban_action_records SET expires_at=0')
        if case=='future':c.execute('UPDATE kanban_action_records SET created_at=expires_at+1')
        if case=='stale_revision':kb.add_comment(c,r.tid,'synthetic','new revision')
        if case=='incarnation':c.execute('UPDATE kanban_action_records SET task_incarnation=task_incarnation+1')
        if case=='state':kb.unblock_task(c,r.tid)
        if case=='unsupported':c.execute("UPDATE kanban_action_records SET action_kind='complete'")
    await r.app.process_update(callback(r,token,**changes))
    assert not readback(r)[1]
    assert readback(r)[0]==('ready' if case=='state' else 'blocked')
    assert 'No change' in answers(r)[-1]

@pytest.mark.asyncio
@pytest.mark.parametrize('change',['revoke','disable','replacement'])
async def test_recheck_after_await_before_mutation(rig,monkeypatch,change):
    r=rig;await tick(r,monkeypatch);token=readback(r)[2][0]['token']
    service=r.manager._task_card_registration.decisions
    await service.mutation_lock.acquire()
    pending=asyncio.create_task(r.app.process_update(callback(r,token)))
    await asyncio.sleep(0)
    if change=='revoke':configure(r,False)
    if change=='disable':r.manager.unload('hermes-telegram-experience')
    if change=='replacement':r.adapter._fence_live_todo_transport()
    service.mutation_lock.release();await pending
    assert readback(r)[0]=='blocked' and not readback(r)[1]

@pytest.mark.asyncio
async def test_simultaneous_duplicate_and_fault_rollback(rig,monkeypatch):
    r=rig;await tick(r,monkeypatch);token=readback(r)[2][0]['token']
    real=kb._append_event
    def fault(conn,tid,kind,*a,**kw):
        if kind=='task_decision':raise RuntimeError('injected transaction fault')
        return real(conn,tid,kind,*a,**kw)
    with monkeypatch.context() as m:
        m.setattr(kb,'_append_event',fault)
        await r.app.process_update(callback(r,token))
    assert readback(r)[0]=='blocked' and not readback(r)[1]
    assert readback(r)[2][0]['state']=='pending'
    await asyncio.gather(*(r.app.process_update(callback(r,token)) for _ in range(4)))
    assert readback(r)[0]=='ready' and len(readback(r)[1])==1


@pytest.mark.asyncio
async def test_native_handler_lifecycle_and_core_callback(rig,monkeypatch):
    r=rig;await tick(r,monkeypatch);old=readback(r)[2][0]['token']
    r.adapter._wire_plugin_handlers(r.app);r.adapter._wire_plugin_handlers(r.app)
    assert len(r.app.handlers[-20])==1
    r.manager.unload('hermes-telegram-experience')
    assert not r.app.handlers.get(-20)
    disabled=readback(r)
    writes=len(wire_payloads(r,'editMessageText'))
    await r.app.process_update(callback(r,old))
    assert readback(r)==disabled
    assert len(wire_payloads(r,'editMessageText'))==writes
    r.manager.discover_and_load(force=True);r.adapter._wire_plugin_handlers(r.app)
    assert len(r.app.handlers[-20])==1
    await tick(r,monkeypatch)
    await r.app.process_update(callback(r,old))
    assert readback(r)[0]=='blocked' and 'No change' in answers(r)[-1]
    q=callback(r,old).to_dict();q['callback_query']['data']='ea:deny:invalid'
    await r.app.process_update(Update.de_json(q,r.app.bot))
    assert answers(r)[-1]=='Invalid approval data.'
    assert not readback(r)[1]


@pytest.mark.asyncio
async def test_grant_revoke_restore_cannot_revive_pending_control(rig,monkeypatch):
    r=rig;await tick(r,monkeypatch);old=readback(r)[2][0]['token']
    configure(r,False)
    await r.app.process_update(callback(r,old))
    configure(r,True)
    await r.app.process_update(callback(r,old))
    assert readback(r)[0]=='blocked' and not readback(r)[1]


@pytest.mark.asyncio
async def test_missing_grant_and_unknown_edit_do_not_mint(rig,monkeypatch):
    r=rig;configure(r,False);await tick(r,monkeypatch)
    assert not readback(r)[2]
    configure(r,True)
    with closing(kbc.connect(r.path)) as c:
        kb.add_comment(c,r.tid,'synthetic','new snapshot')
    sock=Socket('receipt');r.planned.append(sock)
    await tick(r,monkeypatch,False)
    await asyncio.wait_for(sock.entered.wait(),8)
    source=next(iter(r.manager._task_card_registration.sources))
    source.close();sock.reader.feed_eof()
    await source.task
    assert not readback(r)[2]
    with closing(kbc.connect(r.path)) as c:
        assert receipts.list_delivery_receipts(c,task_id=r.tid)[0].state=='unknown'


@pytest.mark.asyncio
async def test_unknown_initial_create_cannot_issue_controls(rig,monkeypatch):
    r=rig;sock=Socket('receipt');r.planned.append(sock)
    await tick(r,monkeypatch,False)
    await asyncio.wait_for(sock.entered.wait(),8)
    source=next(iter(r.manager._task_card_registration.sources))
    source.close();sock.reader.feed_eof();await source.task
    assert not readback(r)[2]
    await tick(r,monkeypatch)
    assert not readback(r)[2]


@pytest.mark.asyncio
@pytest.mark.parametrize('kind',['complete','approve','drop','kill','release','command','grant'])
async def test_no_completion_kill_release_or_expansion(rig,monkeypatch,kind):
    r=rig;await tick(r,monkeypatch);token=readback(r)[2][0]['token']
    with closing(kbc.connect(r.path)) as c:
        c.execute('UPDATE kanban_action_records SET action_kind=?',(kind,))
    await r.app.process_update(callback(r,token))
    assert readback(r)[0]=='blocked' and not readback(r)[1]


@pytest.mark.asyncio
async def test_sqlite_fault_rolls_back_transition_claim_and_audit(rig,monkeypatch):
    r=rig;await tick(r,monkeypatch);token=readback(r)[2][0]['token']
    with closing(kbc.connect(r.path)) as c:
        c.execute("CREATE TRIGGER injected_fault BEFORE INSERT ON task_events WHEN NEW.kind='task_decision' BEGIN SELECT RAISE(ABORT,'synthetic DB fault'); END")
    await r.app.process_update(callback(r,token))
    assert readback(r)[0]=='blocked' and not readback(r)[1]
    assert readback(r)[2][0]['state']=='pending'
    with closing(kbc.connect(r.path)) as c:
        assert not c.execute("SELECT 1 FROM task_events WHERE kind='unblocked'").fetchone()
        c.execute('DROP TRIGGER injected_fault')
    await r.app.process_update(callback(r,token))
    assert readback(r)[0]=='ready' and len(readback(r)[1])==1


@pytest.mark.asyncio
async def test_competing_generation_choices_only_current_can_win(rig,monkeypatch):
    r=rig;await tick(r,monkeypatch);old=readback(r)[2][0]['token']
    r.manager.unload('hermes-telegram-experience')
    r.manager.discover_and_load(force=True);r.adapter._wire_plugin_handlers(r.app)
    await tick(r,monkeypatch);new=readback(r)[2][-1]['token']
    assert old!=new
    await asyncio.gather(r.app.process_update(callback(r,old)),r.app.process_update(callback(r,new)))
    assert readback(r)[0]=='ready' and len(readback(r)[1])==1
    assert sorted(x['state'] for x in readback(r)[2])==['completed','rejected']


@pytest.mark.asyncio
async def test_keyboard_attempt_keeps_revision_while_new_demand_coalesces(rig,monkeypatch):
    r=rig;paused=Socket('receipt');r.planned.extend([Socket(),paused])
    await tick(r,monkeypatch,False)
    await asyncio.wait_for(paused.entered.wait(),8)
    source=next(iter(r.manager._task_card_registration.sources))
    old=readback(r)[2][0]['token']
    with closing(kbc.connect(r.path)) as c:kb.add_comment(c,r.tid,'synthetic','new committed demand')
    await tick(r,monkeypatch,False)
    paused.respond();await source.task
    assert len([s for s in r.sockets if b'/sendMessage ' in s.buffer])==1
    with closing(kbc.connect(r.path)) as c:
        receipt=receipts.list_delivery_receipts(c,task_id=r.tid)[0]
        assert receipt.delivered_revision==receipt.desired_revision
        assert receipt.destination_message_id=='701'
    await r.app.process_update(callback(r,old))
    assert readback(r)[0]=='blocked'
    new=readback(r)[2][-1]['token'];assert old!=new
    await r.app.process_update(callback(r,new))
    assert readback(r)[0]=='ready' and len(readback(r)[1])==1


@pytest.mark.asyncio
@pytest.mark.parametrize('outcome',['unknown','not_modified'])
async def test_unconfirmed_keyboard_never_authorizes_action(rig,monkeypatch,outcome):
    r=rig;paused=Socket('receipt');r.planned.extend([Socket(),paused])
    await tick(r,monkeypatch,False)
    await asyncio.wait_for(paused.entered.wait(),8)
    source=next(iter(r.manager._task_card_registration.sources))
    token=readback(r)[2][0]['token']
    await r.app.process_update(callback(r,token))
    assert readback(r)[0]=='blocked'
    if outcome=='unknown':paused.reader.feed_eof()
    else:
        paused.responded=True
        body=json.dumps(dict(ok=False,error_code=400,description='Bad Request: message is not modified')).encode()
        paused.reader.feed_data(b'HTTP/1.1 400 Bad Request\r\nConnection: close\r\nContent-Type: application/json\r\nContent-Length: '+str(len(body)).encode()+b'\r\n\r\n'+body)
    await source.task
    await r.app.process_update(callback(r,token))
    assert readback(r)[0]=='blocked' and not readback(r)[1]


@pytest.mark.asyncio
async def test_committed_duplicate_still_requires_current_actor_route_and_grant(rig,monkeypatch):
    r=rig;await tick(r,monkeypatch);token=readback(r)[2][0]['token']
    await r.app.process_update(callback(r,token))
    for changes in ({'actor':43},{'message':702},{'chat':-200},{'topic':8},{'copied':True}):
        await r.app.process_update(callback(r,token,**changes))
        assert 'No change' in answers(r)[-1]
    configure(r,False)
    await r.app.process_update(callback(r,token))
    assert 'No change' in answers(r)[-1]
    assert readback(r)[0]=='ready' and len(readback(r)[1])==1


def wire_payloads(r, method):
    return [parse_qs(s.buffer.split(b'\r\n\r\n', 1)[1].decode())
            for s in r.sockets if ('/'+method+' ').encode() in s.buffer]


@pytest.mark.asyncio
@pytest.mark.parametrize('change', ['disabled', 'expired', 'revoked'])
async def test_keyboard_removal_uses_confirmed_same_id_transport(rig, monkeypatch, change):
    import yaml
    r=rig;await tick(r,monkeypatch)
    old=readback(r)[2][0]['token']
    assert 'reply_markup' not in wire_payloads(r, 'sendMessage')[0]
    attached=wire_payloads(r, 'editMessageText')[-1]
    assert json.loads(attached['reply_markup'][0])['inline_keyboard'][0][0]['callback_data']=='hte:d:'+old
    if change=='expired':
        with closing(kbc.connect(r.path)) as c:
            c.execute('UPDATE kanban_action_records SET expires_at=0')
    elif change=='revoked':configure(r,False)
    else:
        cfg=yaml.safe_load((r.home/'config.yaml').read_text())
        cfg['plugins']['entries']['hermes-telegram-experience']['settings']['decisions']=False
        (r.home/'config.yaml').write_text(yaml.safe_dump(cfg))
        r.manager.unload('hermes-telegram-experience')
        r.manager.discover_and_load(force=True);r.adapter._wire_plugin_handlers(r.app)
    await tick(r,monkeypatch)
    removed=wire_payloads(r, 'editMessageText')[-1]
    assert removed['message_id']==['701']
    assert json.loads(removed['reply_markup'][0])=={'inline_keyboard':[]}
    assert len(wire_payloads(r,'sendMessage'))==1
    with closing(kbc.connect(r.path)) as c:
        receipt=receipts.list_delivery_receipts(c,task_id=r.tid)[0]
        assert receipt.state=='sent' and receipt.renderer_hash and receipt.control_hash is None
    await r.app.process_update(callback(r,old))
    assert readback(r)[0]=='blocked' and not readback(r)[1]
    before=len(wire_payloads(r,'editMessageText'))
    await tick(r,monkeypatch)
    assert len(wire_payloads(r,'editMessageText'))==before


@pytest.mark.asyncio
async def test_decision_disabled_wire_omits_markup_on_create_and_edit(rig, monkeypatch):
    import yaml
    r=rig
    cfg=yaml.safe_load((r.home/'config.yaml').read_text())
    cfg['plugins']['entries']['hermes-telegram-experience']['settings']['decisions']=False
    (r.home/'config.yaml').write_text(yaml.safe_dump(cfg))
    r.manager.unload('hermes-telegram-experience');r.manager.discover_and_load(force=True)
    r.adapter._wire_plugin_handlers(r.app)
    await tick(r,monkeypatch)
    with closing(kbc.connect(r.path)) as c:
        assert kb.edit_task(c, r.tid, title='Synthetic input decision revised')
    await tick(r,monkeypatch)
    assert all('reply_markup' not in p for method in ('sendMessage','editMessageText') for p in wire_payloads(r,method))
    assert len(wire_payloads(r,'sendMessage'))==len(wire_payloads(r,'editMessageText'))==1
    assert not readback(r)[2]


@pytest.mark.asyncio
@pytest.mark.parametrize('unsupported_first', [True, False])
async def test_supported_competes_with_unsupported_choice(rig, monkeypatch, unsupported_first):
    from hermes_cli import kanban_db_actions as actions
    r=rig;await tick(r,monkeypatch);record=readback(r)[2][0]
    # Synthetic unsupported server record is not an additional shipped action.
    with closing(kbc.connect(r.path)) as c:
        fields={key:record[key] for key in ('task_id','task_incarnation','expected_revision',
            'expected_task_status','board_identity','profile','telegram_principal',
            'origin_chat_id','origin_thread_id','origin_message_id','expires_at','conflict_key')}
        bad=actions.issue_action(c,**fields,action_kind='complete',
            action_payload=json.loads(record['action_payload']),idempotency_key='synthetic-unsupported')
    tokens=[bad.token,record['token']] if unsupported_first else [record['token'],bad.token]
    await asyncio.gather(*(r.app.process_update(callback(r,t)) for t in tokens))
    status,audit,rows=readback(r)
    assert status=='ready' and len(audit)==1
    assert sum(x['state']=='completed' for x in rows)==1
    assert next(x for x in rows if x['token']==bad.token)['state']!='completed'
    with closing(kbc.connect(r.path)) as c:
        assert c.execute("SELECT count(*) FROM task_events WHERE kind='unblocked'").fetchone()[0]==1


@pytest.mark.asyncio
async def test_malformed_current_policy_does_not_mint_action(rig, monkeypatch):
    r = rig
    (r.home / 'config.yaml').write_text('kanban: [unterminated')
    await tick(r, monkeypatch)
    status, audit, rows = readback(r)
    assert (status, audit, rows) == ('blocked', [], [])


@pytest.mark.asyncio
async def test_malformed_policy_revokes_action_through_same_valid_repair(rig, monkeypatch):
    r = rig
    await tick(r, monkeypatch)
    before = readback(r)
    token = before[2][0]['token']
    (r.home / 'config.yaml').write_text('kanban: [unterminated')
    await r.app.process_update(callback(r, token))
    assert readback(r) == before
    configure(r, True)
    await r.app.process_update(callback(r, token))
    # Repairing the same grant is a new policy epoch; it cannot revive the old record.
    assert readback(r) == before
