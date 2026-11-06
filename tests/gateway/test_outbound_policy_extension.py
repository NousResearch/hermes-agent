from types import SimpleNamespace
from unittest.mock import Mock
import asyncio
import pytest
from gateway import outbound_policy as p
from hermes_cli.plugins import PluginContext, PluginManager, PluginManifest

@pytest.mark.parametrize('mode',['missing','raises','invalid','hold','slow','changed'])
def test_required_denies_before_transport(monkeypatch, mode):
    manager=object(); calls=[]
    class Provider:
        def evaluate(self,e):
            if mode=='raises': raise RuntimeError('unavailable')
            if mode=='slow':
                import time; time.sleep(.05)
            if mode=='invalid': return True
            return p.OutboundDecision('hold' if mode=='hold' else 'allow',e.destination,reason='draft-ref')
    provider=Provider(); snapshots=[0]
    def snapshot(platform):
        if mode=='missing': raise p.OutboundDenied('required absent')
        snapshots[0]+=1
        return manager, (('required','example',object() if mode=='changed' and snapshots[0]>1 else provider),)
    monkeypatch.setattr(p,'_snapshot',snapshot)
    monkeypatch.setattr(p,'POLICY_TIMEOUT_SECONDS',.005)
    class Adapter:
        platform='telegram'
        async def send(self,chat_id,content,metadata=None): calls.append((chat_id,content))
    p.guard_adapter_methods(Adapter)
    with pytest.raises(p.OutboundDenied): asyncio.run(Adapter().send('peer','body'))
    assert not calls

def test_no_required_policies_preserves_send(monkeypatch):
    manager=object();monkeypatch.setattr(p,'_snapshot',lambda _: (manager,()))
    assert p.enforce_outbound('telegram','peer','body')[0]=='peer'

def test_native_registration_disposes():
    manager=PluginManager();ctx=PluginContext(PluginManifest(name='example',key='example'),manager)
    provider=SimpleNamespace(evaluate=lambda e:p.OutboundDecision('allow',e.destination))
    handle=ctx.register_outbound_policy('required',provider)
    assert manager._outbound_policies['required']==('example',provider)
    with pytest.raises(ValueError):ctx.register_outbound_policy('required',provider)
    handle.dispose();assert not manager._outbound_policies

def test_metadata_frozen():
    frozen=p._freeze({'nested':[{'key':'value'}]})
    with pytest.raises(TypeError):frozen['nested'][0]['key']='changed'

def test_transport_call_sites_present():
    from pathlib import Path
    root=Path(__file__).resolve().parents[2]
    assert 'guard_adapter_methods(cls)' in (root/'gateway/platforms/base.py').read_text()
    assert '_outbound_bot_call(self._bot' in (root/'plugins/platforms/telegram/adapter.py').read_text()
    assert '_OutboundBot(_unwrapped_telegram_bot(token))' in (root/'tools/send_message_senders.py').read_text()


def _native_helper(name):
    import ast
    from pathlib import Path
    source = Path(__file__).resolve().parents[2]/'plugins/platforms/telegram/adapter.py'
    tree = ast.parse(source.read_text())
    node = next(n for n in tree.body if isinstance(n, ast.AsyncFunctionDef) and n.name == name)
    namespace = {}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(source), 'exec'), namespace)
    return namespace[name]


def test_native_reroute_drops_original_topic(monkeypatch):
    from unittest.mock import AsyncMock
    manager = object()
    provider = SimpleNamespace(evaluate=lambda e: p.OutboundDecision('allow', 'review-peer'))
    monkeypatch.setattr(p, '_snapshot', lambda _: (manager, (('required', 'example', provider),)))
    send = AsyncMock()
    bot = SimpleNamespace(send_message=send)
    asyncio.run(_native_helper('_outbound_bot_call')(bot, 'send_message',
        chat_id='original-peer', text='body', message_thread_id='old-topic',
        direct_messages_topic_id='old-topic', reply_to_message_id='old-message'))
    kwargs = send.call_args.kwargs
    assert kwargs['chat_id'] == 'review-peer'
    assert not {'message_thread_id', 'direct_messages_topic_id', 'reply_to_message_id'} & kwargs.keys()


def test_native_edit_cannot_retarget(monkeypatch):
    from unittest.mock import AsyncMock
    manager = object()
    provider = SimpleNamespace(evaluate=lambda e: p.OutboundDecision('allow', 'other-peer'))
    monkeypatch.setattr(p, '_snapshot', lambda _: (manager, (('required', 'example', provider),)))
    send = AsyncMock()
    with pytest.raises(p.OutboundDenied):
        asyncio.run(_native_helper('_outbound_bot_call')(SimpleNamespace(edit_message_text=send),
            'edit_message_text', chat_id='original-peer', text='body'))
    send.assert_not_called()


def test_media_generation_change_denies_before_native_transport(monkeypatch):
    from unittest.mock import AsyncMock
    manager = object()
    old = (manager, (('required', 'example', object()),))
    new = (manager, (('required', 'example', object()),))
    monkeypatch.setattr(p, '_snapshot', lambda _: new)
    token = p._PIN.set(('telegram', old))
    send = AsyncMock()
    try:
        with pytest.raises(p.OutboundDenied):
            asyncio.run(p.guarded_transport_call('telegram', send, chat_id='peer', photo='local-file'))
    finally:
        p._PIN.reset(token)
    send.assert_not_called()
