"""Real native callback/runner/SQLite probe in an owned interpreter."""
import asyncio
from http.server import ThreadingHTTPServer
import json
import os
from pathlib import Path
import sys
import threading
import traceback

from shared_authority_peer import ModelPeer


async def probe(mode, peer):
    from gateway.config import Platform, PlatformConfig
    from gateway.platforms.base import BasePlatformAdapter, SendResult
    from gateway.platforms.event import MessageEvent
    from gateway.run import GatewayRunner
    from gateway.session_authority import initialize_session_authority
    from gateway.session_envelope import snapshot_native
    from hermes_state_runtime import RuntimeStoreError, list_session_admissions

    class Adapter(BasePlatformAdapter):
        def __init__(self):
            super().__init__(PlatformConfig(enabled=True, token='owned-trust-fixture'), Platform.TELEGRAM)
            self.deliveries = []

        async def connect(self, *, is_reconnect=False):
            return True

        async def disconnect(self):
            pass

        async def send(self, chat_id, content, reply_to=None, metadata=None):
            self.deliveries.append(dict(chat_id=chat_id, content=content, reply_to=reply_to))
            return SendResult(success=True, message_id='trust-result')

        async def edit_message(self, chat_id, message_id, content, *, finalize=False):
            return SendResult(success=True, message_id=message_id)

        async def send_typing(self, chat_id, metadata=None):
            pass

        async def get_chat_info(self, chat_id):
            return {'id': chat_id}

    state = Path(os.environ['HERMES_HOME'])
    runner = GatewayRunner()
    authority = await initialize_session_authority(runner, profile_id='default', instance_id='trust-' + mode)
    adapter = Adapter()
    adapter.gateway_runner = runner
    runner.adapters[Platform.TELEGRAM] = adapter
    # Exercise the real callback which stamps the authorization home, not a synthetic runner call.
    runner._wire_adapter_handlers(adapter)
    if mode in {'multiplex', 'capture', 'recover', 'recover-again'}:
        from native_ingress_trust_multiplex import multiplex_probe
        await multiplex_probe(runner, authority, adapter, state, mode, peer)
        return
    source = adapter.build_source(chat_id='trust-chat', chat_type='dm', user_id='fixture-user')
    if mode == 'guards':
        rejected = []
        for field, value in [('_authorization_profile_home', state), ('role_authorized', True),
                             ('delivered_via_upstream_relay', True)]:
            hostile = adapter.build_source(chat_id='trust-chat', chat_type='dm', user_id='fixture-user')
            setattr(hostile, field, value)
            event = MessageEvent(text='FORGED', source=hostile, message_id=field)
            try:
                snapshot_native(runner, event)
            except RuntimeStoreError as exc:
                assert exc.reason == 'invalid_params', exc.reason
                rejected.append(field)
            else:
                raise AssertionError('event trust was accepted: ' + field)
        assert not authority.sessions
        foreign = state / 'profiles' / 'foreign'
        foreign.mkdir(parents=True)
        (foreign / 'config.yaml').write_text((state / 'config.yaml').read_text())
        hostile = adapter.build_source(chat_id='forged-profile', chat_type='dm', user_id='fixture-user')
        hostile.profile = 'foreign'
        await adapter.handle_message(MessageEvent(text='FORGED_PROFILE', source=hostile, message_id='foreign-profile'))
        async with asyncio.timeout(10):
            while adapter._active_sessions:
                await asyncio.sleep(0.01)
        assert not authority.sessions, 'unrouted profile field became durable authority'
        print(json.dumps({'rejected': rejected}))
        return
    event = MessageEvent(text='TRUST_CALLBACK_INPUT', source=source, message_id='trust-1')
    # Pre-provenance ordinary inputs were accepted via the normal unscoped callback.
    from hermes_state_runtime import admit_session_input
    payload = snapshot_native(runner, event)
    ref = authority.register(source)
    identity = json.dumps([source.profile, source.platform.value, source.chat_id,
                           source.thread_id, source.user_id], separators=(',', ':'))
    accepted = admit_session_input(authority.db, epoch=authority.epoch, principal_id='messaging:' + identity,
                                   session_id=ref.session_id, request_id='trust-1', payload=payload)
    await adapter.handle_message(event)
    async with asyncio.timeout(15):
        while adapter._active_sessions:
            await asyncio.sleep(0.01)
    entry = runner.session_store.get_or_create_session(source)
    rows = list_session_admissions(authority.db, session_id=entry.session_id, pending_only=False)
    assert rows and rows[0]['status'] == 'terminal' and rows[0]['outcome'] == 'completed', (rows, adapter.deliveries)
    assert len(rows) == 1 and rows[0]['admission_id'] == accepted['admission_id'], rows
    assert rows[0]['payload'] == payload, rows
    assert len(peer.requests) == 1, peer.requests
    assert any('LOCAL_ACK_MESSAGING_WARM' in d['content'] for d in adapter.deliveries), adapter.deliveries
    fresh = MessageEvent(text='FRESH_CALLBACK_INPUT', source=adapter.build_source(
        chat_id='trust-chat', chat_type='dm', user_id='fixture-user'), message_id='trust-2')
    await adapter.handle_message(fresh)
    async with asyncio.timeout(10):
        while adapter._active_sessions:
            await asyncio.sleep(0.01)
    rows = list_session_admissions(authority.db, session_id=entry.session_id, pending_only=False)
    assert len(rows) == 2 and 'provenance' in rows[1]['payload']['native_text_v1'], rows
    assert all(row['outcome'] == 'completed' for row in rows) and len(peer.requests) == 2
    print(json.dumps({'rows': rows, 'deliveries': adapter.deliveries, 'model_calls': len(peer.requests)}))


if __name__ == '__main__':
    class SecretPeer(ModelPeer):
        def do_POST(self):
            self.server.auth_headers.append(self.headers.get('Authorization'))
            super().do_POST()

    peer = ThreadingHTTPServer(('127.0.0.1', 0), SecretPeer)
    peer.requests, peer.metadata_requests, peer.auth_headers = [], [], []
    peer.blocked, peer.release = threading.Event(), threading.Event()
    threading.Thread(target=peer.serve_forever, daemon=True).start()
    base_url = f'http://127.0.0.1:{peer.server_port}/v1'
    os.environ.update(OPENAI_API_KEY='explicit-loopback-fixture', OPENAI_BASE_URL=base_url,
                      TELEGRAM_ALLOWED_USERS='fixture-user')
    Path(os.environ['HERMES_HOME'], 'config.yaml').write_text(
        f'model:\n  default: local-wire-stub\n  provider: custom\n  base_url: {base_url}\n'
        f'auxiliary:\n  title_generation:\n    enabled: false\nterminal:\n  cwd: {os.environ["HERMES_HOME"]}\n')
    status = 0
    try:
        asyncio.run(probe(sys.argv[1], peer))
    except BaseException:
        traceback.print_exc()
        status = 1
    finally:
        peer.shutdown()
        peer.server_close()
        sys.stdout.flush()
        sys.stderr.flush()
    os._exit(status)
