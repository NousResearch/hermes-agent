"""Offline real discord.py wire regressions, adapted from the independent BLOCK probe."""
import pytest
import pytest_asyncio
import asyncio, queue, re, weakref
from types import SimpleNamespace
from unittest.mock import AsyncMock
discord = pytest.importorskip("discord")
if not isinstance(getattr(discord, "__version__", None), str):
    pytest.skip("Requires isolated real discord.py, not gateway/conftest's fallback", allow_module_level=True)
from gateway.run_turn import GatewayTurnMixin
from gateway.config import Platform, PlatformConfig
from gateway.session import SessionSource
from gateway.turn_context import TurnContext
from gateway.run_turn_runner import TurnRunner
from gateway.delegated_child_progress import DelegatedChildProgress as DCP
from gateway.stream_consumer import GatewayStreamConsumer
from plugins.platforms.discord.adapter import DiscordAdapter, _build_allowed_mentions
from tools.delegate_tool_progress import _build_child_progress_callback


class Wire:
    def __init__(self):
        self.sends, self.edits, self.chat = [], [], {}
        self.edit_attempts = 0
        self.fail_edit = False
        self.fail_send_at = None
        self.send_error = TimeoutError
        self.edit_error = None
        self.reject_refs = 0
        self.attempts = []
    def message(self, channel, payload, mid):
        return dict(id=str(mid), channel_id=str(channel), content=payload.get('content',''),
            timestamp='2026-09-29T00:00:00+00:00', edited_timestamp=None, tts=False,
            mention_everyone=False, mentions=[], mention_roles=[], attachments=[], embeds=[],
            pinned=False, type=0, flags=payload.get('flags',0), author=dict(id='10', username='probe', discriminator='0000', avatar=None, bot=True))
    async def send_message(self, channel, *, params):
        p = params.payload.copy()
        self.attempts.append(p)
        if self.reject_refs and p.get('message_reference'):
            self.reject_refs -= 1
            raise discord.NotFound(SimpleNamespace(status=404, reason='Not Found'),
                                   {'code':10008, 'message':'Unknown Message'})
        mid = str(1000 + len(self.sends))
        self.sends.append((str(channel), mid, p))
        self.chat[mid] = p
        if len(self.sends) == self.fail_send_at:
            raise self.send_error('response lost after server accepted message')
        return self.message(channel, p, mid)
    async def edit_message(self, channel, mid, *, params):
        self.edit_attempts += 1
        if self.fail_edit:
            raise ConnectionResetError('Connection reset by peer')
        p = params.payload.copy()
        self.edits.append((str(channel), str(mid), p))
        if self.edit_error is not None:
            raise self.edit_error
        self.chat[str(mid)] = {**self.chat.get(str(mid), {}), **p}
        return self.message(channel, p, mid)
    def text(self):
        return '\n'.join(p['content'] for p in self.chat.values())
    def bodies(self):
        return ''.join(re.findall(r'```\n(.*?)\n```', self.text(), re.S))

clients, turns = [], []
def adapter(channel_id='555'):
    a = DiscordAdapter(PlatformConfig(enabled=True, token='offline-review'))
    c = discord.Client(intents=discord.Intents.none(), allowed_mentions=_build_allowed_mentions())
    w = Wire()
    c._connection.http = w
    ch = discord.DMChannel(me=None, state=c._connection, data={'id':channel_id,'type':1,'recipients':[]})
    a._client = c
    a._resolve_channel = AsyncMock(return_value=ch)
    clients.append(c)
    return a, w

def turn(a, *, verbose=False, current=None, consumer=None, chat='555', thread='555'):
    src = SessionSource(platform=Platform.DISCORD, chat_id=chat, thread_id=thread)
    src._transport_adapter_ref = weakref.ref(a)
    ctx = TurnContext(source=src, _run_still_current=current or (lambda:False),
        progress_mode='verbose' if verbose else 'all', tool_progress_enabled=True,
        progress_queue=queue.Queue(), _loop_for_step=asyncio.get_running_loop(),
        _progress_metadata={'thread_id':thread,'non_conversational':True},
        stream_consumer_holder=[consumer] if consumer else [])
    runner = SimpleNamespace(_delivery_adapter_for=lambda s:a, _retain_background_task=lambda t:t)
    t = TurnRunner(runner, ctx)
    turns.append(t)
    return t

def relay(t, name='writer', model='claude-opus-5-5'):
    return _build_child_progress_callback(0, name+' goal',
        SimpleNamespace(tool_progress_callback=t.progress_callback, _delegate_spinner=None, session_id='parent'),
        subagent_id=name, depth=0, model=model, session_ref={'delegation_id':name,'session_id':'child-'+name})
async def event(cb, *args, **kw):
    await asyncio.to_thread(cb, *args, **kw)
async def flush(seconds=.12):
    await asyncio.sleep(seconds)
async def stop(t):
    task = t._child_progress._task if t._child_progress else None
    if task and not task.done():
        task.cancel()
        try: await task
        except asyncio.CancelledError: pass

def command(tag='A'):
    return 'python check.py ' + ' '.join('--case='+tag+f'{i:04d}' for i in range(300)) + ' --required-tail=TAIL_'+tag


@pytest_asyncio.fixture(autouse=True)
async def cleanup(monkeypatch):
    monkeypatch.setattr(DCP, 'EDIT_INTERVAL', .01)
    yield
    for t in turns: await stop(t)
    for c in clients: await c.close()
    turns.clear(); clients.clear()

@pytest.mark.asyncio
async def test_lossless_burst_and_updates():
    a,w=adapter(); t=turn(a,verbose=True); cb=relay(t)
    await event(cb,'subagent.start')
    await event(cb,'tool.started','terminal','run',{'command':command()})
    await flush()
    assert w.bodies()==command()
    old=list(w.sends)
    await event(cb,'tool.started','read_file','AFTER_LONG.py',{'path':'AFTER_LONG.py'})
    await flush()
    assert 'AFTER_LONG.py' in w.text()
    def burst():
        for i in range(12): cb('tool.started','terminal','run',{'command':command(str(i))})
        cb('subagent.complete',status='completed',preview='PRIVATE_REPLY')
    await asyncio.to_thread(burst)
    await flush(2)
    assert w.bodies()==command()+''.join(command(str(i)) for i in range(12))
    assert 'PRIVATE_REPLY' not in w.text()
    assert len(w.sends)>8
    assert max(len(p['content']) for p in w.chat.values()) <= 2000
    assert not any(mid in [s[1] for s in old[1:]] for _,mid,_ in w.edits)

@pytest.mark.asyncio
@pytest.mark.parametrize("error", [TimeoutError, ConnectionResetError])
async def test_ambiguous_continuation_stops_without_replay(error):
    a,w=adapter(); w.fail_send_at=2; w.send_error=error; t=turn(a,verbose=True); cb=relay(t)
    await event(cb,'subagent.start')
    await event(cb,'tool.started','terminal','run',{'command':command()})
    await flush()
    await event(relay(t,'other'),'subagent.start'); await flush()
    assert len(w.sends)==2
    assert t._child_progress._dead

@pytest.mark.asyncio
async def test_progress_wire_flags_overflow_and_exact_code():
    a,w=adapter(); metadata={'progress':True,'non_conversational':True}
    code='curl https://example.com/path <@123> @everyone --output out.txt'
    r=await a.send('555','```\n'+code+'\n```',reply_to='123',metadata=metadata)
    assert w.bodies()==code
    await a.edit_message('555',r.message_id,'```\n'+code+'\n```',metadata=metadata)
    overflow='\n'.join('status '+str(i)+' <@123> https://example.com/'+str(i)+' '+'x'*110 for i in range(30))
    r=await a.edit_message('555',r.message_id,overflow,finalize=True,metadata=metadata)
    assert r.success
    for _,_,p in w.sends+w.edits:
        assert p['allowed_mentions']['parse']==[]
        assert not p['allowed_mentions'].get('replied_user',False)
        assert p.get('flags',0)&4
    await a.send('555','ordinary <@123>',reply_to='123')
    assert 'users' in w.sends[-1][2]['allowed_mentions']['parse']
    await a.send('555','approval <@123>',metadata={'non_conversational':True})
    assert 'users' in w.sends[-1][2]['allowed_mentions']['parse']

@pytest.mark.asyncio
async def test_actual_native_lane_cleanup_handover():
    a,w=adapter(); current=[True]
    consumer=GatewayStreamConsumer(a,'555',run_still_current=lambda:current[0])
    consumer_task=asyncio.create_task(consumer.run())
    t=turn(a,verbose=True,current=lambda:current[0],consumer=consumer); cb=relay(t)
    t._ctx._cleanup_progress=True
    t.progress_callback('tool.started','read_file','PARENT.py',{'path':'PARENT.py'})
    parent_task=asyncio.create_task(t.send_progress_messages())
    await event(cb,'subagent.start')
    code='curl https://example.com/path --output out.txt'
    await event(cb,'tool.started','terminal','run',{'command':code})
    await flush(.5)
    assert not consumer.accepts_tool_progress  # actual Discord capability
    assert len(w.sends)==1
    assert 'PARENT.py' in w.text() and 'writer' in w.text()
    assert w.bodies()==code
    # Actual turn cleanup cancels the progress task, not the generation predicate.
    async def finish_stream(task):
        consumer.finish()
        await task
    cleanup_runner = SimpleNamespace(_draining=False, _await_stream_task=finish_stream)
    tracking = asyncio.create_task(asyncio.sleep(60))
    t._ctx.session_key = None
    await GatewayTurnMixin._run_agent_cleanup_turn_tasks(
        cleanup_runner, t._ctx, progress_task=parent_task, log_task=None,
        interrupt_monitor=None, _notify_task=None, tracking_task=tracking,
        stream_task=consumer_task,
    )
    assert current[0]  # predicate alone does NOT describe turn lifetime
    assert t._child_progress._msg_id not in t._ctx._cleanup_msg_ids
    assert not t._child_progress._native_live()
    before=w.bodies()
    await event(cb,'tool.started','read_file','LATE.py',{'path':'LATE.py'})
    await flush()
    assert 'LATE.py' in w.text() and w.bodies()==before
    await asyncio.wait_for(consumer_task,3)
    # No event is needed to leave the native lane, and a newer owner cannot steal routing.
    b,wb=adapter('888'); t2=turn(b,verbose=True,current=lambda:False,chat='777',thread='888')
    await event(relay(t2,'new'),'subagent.start')
    await event(cb,'tool.started','read_file','OLD_ROUTE.py',{'path':'OLD_ROUTE.py'})
    await flush()
    assert 'OLD_ROUTE.py' in w.text() and 'OLD_ROUTE.py' not in wb.text()
    assert all(chat=='555' for chat,_,_ in w.sends+w.edits)
    assert all(chat=='888' for chat,_,_ in wb.sends+wb.edits)
    assert all(call.args==('888',) for call in b._resolve_channel.await_args_list)
    assert t._child_progress.adapter is a and t2._child_progress.adapter is b


@pytest.mark.asyncio
async def test_reference_rejection_policy_and_ambiguous_overflow():
    a,w=adapter(); metadata={'progress':True}
    w.reject_refs=1
    r=await a.send('555','<@123> https://example.com',reply_to='999',metadata=metadata)
    assert r.success and len(w.attempts)==2
    w.reject_refs=1
    text='\n'.join('row '+str(i)+' <@123> '+'x'*200 for i in range(30))
    r=await a.edit_message('555',r.message_id,text,finalize=True,metadata=metadata)
    assert r.success
    assert any(not p.get('message_reference') for p in w.attempts)
    for p in w.attempts+[p for _,_,p in w.edits]:
        assert p['allowed_mentions']['parse']==[] and p['flags']&4
        assert not p['allowed_mentions'].get('replied_user',False)
    # A continuation POST accepted before reset is never retried without its reference.
    w.fail_send_at=len(w.sends)+1
    before=len(w.attempts)
    r=await a.edit_message('555',r.message_id,text+'END',finalize=True,metadata=metadata)
    assert not r.success and r.error_kind=='ambiguous'
    assert len(w.attempts)==before+1


@pytest.mark.asyncio
async def test_typed_bounded_edit_recovery(monkeypatch):
    a,w=adapter(); t=turn(a); cb=relay(t)
    await event(cb,'subagent.start'); await flush()
    owner=t._child_progress
    # Drive the owner directly, without its background wake racing a retry assertion.
    owner._task.cancel()
    try: await owner._task
    except asyncio.CancelledError: pass
    owner._dead=False  # resume only this isolated direct retry probe
    w.edits.clear()
    sleeps=[]
    async def no_wait(delay): sleeps.append(delay)
    monkeypatch.setattr(asyncio,'sleep',no_wait)
    w.edit_error=ConnectionResetError('reset')
    owner.add_activity('next')
    await owner._deliver_chunks()
    assert owner._dead and len(w.edits)==3 and len(w.sends)==1
    assert any(delay>=.49 for delay in sleeps) and any(delay>=1.99 for delay in sleeps)
    # Text alone is not typed transport evidence.
    w.edit_error=RuntimeError('connection reset')
    r=await a.edit_message('555',owner._msg_id,'unknown',metadata={'progress':True})
    assert not r.retryable and r.error_kind=='unknown'
    w.edit_error=discord.HTTPException(SimpleNamespace(status=429,reason='rate limited'), 'slow down')
    w.edit_error.retry_after=999
    r=await a.edit_message('555',owner._msg_id,'limited',metadata={'progress':True})
    assert r.retryable and r.error_kind=='rate_limited' and r.retry_after==999
    owner._dead=False; owner._transport_retries=0; sleeps.clear(); w.edits.clear()
    await owner._deliver_chunks()
    assert owner._dead and len(w.edits)==3
    assert len(sleeps)>=2 and all(delay<=30 for delay in sleeps)
    assert any(delay>29 for delay in sleeps)


@pytest.mark.asyncio
async def test_pending_cleanup_drains_without_new_event(monkeypatch):
    a,w=adapter(); t=turn(a,verbose=True,current=lambda:True); cb=relay(t)
    # Queue work before the native task starts and end while it has an unacked tail.
    await event(cb,'subagent.start')
    code=command('pending')
    await event(cb,'tool.started','terminal','run',{'command':code})
    task=asyncio.create_task(t.send_progress_messages())
    while not w.sends: await asyncio.sleep(.005)
    task.cancel(); await task
    await flush(.5)
    assert not t._child_progress._native_live()
    assert w.bodies()==code
    before=w.bodies()
    t.end_progress_turn(); await flush()
    assert w.bodies()==before


@pytest.mark.asyncio
async def test_multiple_children_burst_whitespace_and_utf16():
    a,w=adapter(); t=turn(a,verbose=True); callbacks=[relay(t,'one'),relay(t,'two')]
    commands=['printf "😀 <@123> https://example.com"\n'+'x'*2400+'  \n',command('two')]
    def burst():
        for cb in callbacks: cb('subagent.start')
        for cb,code in zip(callbacks,commands): cb('tool.started','terminal','run',{'command':code})
        for cb in callbacks: cb('subagent.complete',status='completed')
    await asyncio.to_thread(burst); await flush(.6)
    assert w.bodies()==''.join(commands)
    # Readable per-lane labels replace raw subagent ids in tool lines.
    assert '[Opus 1]' in w.text() and '[Opus 2]' in w.text()
    assert '[one]' not in w.text() and '[two]' not in w.text()
    assert w.text().count('✅')==2
    assert max(len(p['content'].encode('utf-16-le'))//2 for p in w.chat.values())<=2000


@pytest.mark.asyncio
async def test_progress_posts_are_standalone_but_ordinary_replies_still_reference():
    """Head and every overflow continuation are plain posts in the thread, never replies to
    the previous bubble; the quiet wire policy is unchanged; ordinary answers still reply."""
    a,w=adapter(); t=turn(a,verbose=True); cb=relay(t,'sa-0-40fb0793')
    await event(cb,'subagent.start')
    await event(cb,'tool.started','terminal','run',{'command':command('ref')})
    await event(cb,'tool.started','read_file','AFTER.py',{'path':'AFTER.py'})
    await flush(.5)
    assert len(w.sends)>=2  # a head plus at least one overflow continuation
    assert w.bodies()==command('ref') and 'AFTER.py' in w.text()
    for _,_,p in w.sends:
        assert 'message_reference' not in p
        assert p['allowed_mentions']['parse']==[] and p.get('flags',0)&4
    assert all(chat=='555' for chat,_,_ in w.sends+w.edits)
    # (The fixture's goal text is "<name> goal"; only the raw id *label* must be gone.)
    assert '[Opus 1] ' in w.text() and '[sa-0-40fb0793]' not in w.text()
    # Negative control: a normal (non-progress) reply keeps its reference.
    await a.send('555','ordinary answer',reply_to='123')
    assert str(w.sends[-1][2]['message_reference']['message_id'])=='123'


@pytest.mark.asyncio
async def test_verbose_delimiters_and_nonterminal_arguments():
    a,w=adapter(); t=turn(a,verbose=True); cb=relay(t)
    code='printf "``` https://example.com <@123>"\n'+'z'*4000+'  '
    await event(cb,'subagent.start')
    await event(cb,'tool.started','terminal','run',{'command':code})
    await event(cb,'tool.started','read_file','file.py',{'path':'file.py','offset':71,'limit':123})
    await flush(.5)
    bodies=re.findall(r'^````\n(.*?)\n````$',w.text(),re.M|re.S)
    assert ''.join(bodies)==code
    assert '"offset": 71' in w.text() and '"limit": 123' in w.text()
    assert max(len(p['content']) for p in w.chat.values())<=2000
