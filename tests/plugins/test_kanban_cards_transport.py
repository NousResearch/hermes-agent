"""Installed card consumer through real PTB/httpx/httpcore; fake socket only."""
import asyncio
from contextlib import closing
import importlib.util
import json
from pathlib import Path

import pytest
from plugins.platforms.telegram import adapter as adapter_module
from plugins.platforms.telegram import telegram_network, transport_admission as wire


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module

h = load('cards_sdk_harness', Path(__file__).parents[1]/'gateway/test_kanban_cards_integration.py')
r = load('cards_socket_harness', Path(__file__).parents[1]/'plugins/test_telegram_live_todo_repair.py')
rig = h.rig


class Socket(r.SocketWriter):
    def respond(self):
        if self.responded: return
        self.responded = True
        body = json.dumps({'ok':True,'result':{'message_id':701,'date':1,
                           'chat':{'id':-100,'type':'supergroup'},'text':'synthetic'}}).encode()
        self.reader.feed_data(b'HTTP/1.1 200 OK\r\nConnection: close\r\nContent-Type: application/json\r\nContent-Length: '
                             +str(len(body)).encode()+b'\r\n\r\n'+body)


@pytest.mark.asyncio
@pytest.mark.parametrize('pause', ['connect','tls','drain','receipt'])
@pytest.mark.parametrize('fence', ['unload','adapter','expired','aba'])
async def test_card_exact_admission_and_late_receipt(rig,monkeypatch,pause,fence):
    monkeypatch.setattr(adapter_module,'resolve_proxy_url',lambda *a,**k:None)
    monkeypatch.setattr(telegram_network,'_resolve_proxy_url',lambda *a,**k:None)
    monkeypatch.setenv('HERMES_TELEGRAM_DISABLE_FALLBACK_IPS','true')
    first=Socket(pause)
    sockets=[]
    async def connect(*args,**kwargs):
        writer=first if not sockets else Socket()
        sockets.append(writer)
        if writer is first and pause=='connect':
            writer.entered.set(); await writer.release.wait()
        return writer.reader,writer
    monkeypatch.setattr(wire.asyncio,'open_connection',connect)
    general,updates=await rig.adapter._build_ptb_requests()
    rig.adapter._bot=(adapter_module.Application.builder().token('123:synthetic')
                      .request(general).get_updates_request(updates).build()).bot
    try:
        await h.tick(rig,False)
        await asyncio.wait_for(first.entered.wait(),5)
        source=next(iter(rig.manager._task_card_registration.sources))
        old=h.receipt(rig); new=h.advance(rig)
        await h.tick(rig,False)
        if fence=='unload': rig.manager.unload('hermes-telegram-experience')
        elif fence=='adapter': rig.adapter._fence_live_todo_transport()
        elif fence=='expired':
            with closing(h.kbc.connect(rig.path)) as c: c.execute('UPDATE kanban_delivery_receipts SET lease_expires_at=0')
        else:
            with closing(h.kbc.connect(rig.path)) as c:
                for profile in ('b','default'):
                    c.execute('DELETE FROM kanban_notify_subs WHERE task_id=?',(rig.tid,))
                    h.notify.add_notify_sub(c,task_id=rig.tid,platform='telegram',chat_id='-100',thread_id='7',notifier_profile=profile)
        first.release.set()
        if pause=='receipt': first.respond()
        await asyncio.wait_for(source.task,5)
        settled=h.receipt(rig)
        if pause=='receipt':
            assert settled.state=='sent' and settled.destination_message_id=='701'
            assert settled.delivered_revision==old.attempted_revision and settled.desired_revision==new
        else:
            assert not first.writes and settled.state=='failed'
        if fence=='unload': rig.manager.discover_and_load(force=True)
        with closing(h.kbc.connect(rig.path)) as c: c.execute('UPDATE kanban_delivery_receipts SET retry_at=0')
        await h.tick(rig)
        assert h.receipt(rig).delivered_revision==new
        payloads=[b''.join(s.writes) for s in sockets if s.writes]
        assert sum(b'/sendMessage' in p for p in payloads)==1
        if pause=='receipt':
            assert sum(b'/editMessageText' in p for p in payloads)==1
            assert b'message_id=701' in payloads[-1]
    finally:
        first.release.set()
        await general.shutdown(); await updates.shutdown()
