"""Owned temp-profile native admission/recovery with real loopback inference."""
import asyncio
from dataclasses import replace
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
    from gateway.platforms.base import BasePlatformAdapter, SendResult, cleanup_document_cache
    from gateway.platforms.event import MessageEvent, MessageType
    from gateway.run import GatewayRunner
    from gateway.session import SessionSource
    from gateway.session_authority import initialize_session_authority
    from gateway.session_envelope import restore_native, snapshot_native
    from hermes_state_runtime import RuntimeStoreError, claim_session_input, list_session_admissions

    class Adapter(BasePlatformAdapter):
        def __init__(self):
            super().__init__(PlatformConfig(enabled=True, token='owned-fixture'), Platform.TELEGRAM)
            self.deliveries = []

        async def connect(self, *, is_reconnect=False):
            return True

        async def disconnect(self):
            pass

        async def send(self, chat_id, content, reply_to=None, metadata=None):
            self.deliveries.append(dict(chat_id=chat_id, content=content, reply_to=reply_to, metadata=metadata))
            return SendResult(success=True, message_id='sent-fixture')

        async def edit_message(self, chat_id, message_id, content, *, finalize=False):
            return SendResult(success=True, message_id=message_id)

        async def send_typing(self, chat_id, metadata=None):
            pass

        async def get_chat_info(self, chat_id):
            return {'id': chat_id}

    state = Path(os.environ['HERMES_HOME'])
    runner = GatewayRunner()
    authority = await initialize_session_authority(runner, profile_id='default', instance_id='media-' + mode)
    adapter = Adapter()
    runner.adapters[Platform.TELEGRAM] = adapter
    adapter.set_message_handler(runner._handle_message)
    source = SessionSource(platform=Platform.TELEGRAM, chat_id='media-chat', thread_id='topic-7',
                           chat_type='dm', user_id='fixture-user')

    def rows(sid):
        return list_session_admissions(authority.db, session_id=sid, pending_only=False)

    if mode == 'capture':
        # Ordinary document preprocessing must consume original bytes after restart.
        original = state / 'mutable-note.txt'
        original.write_text('ORIGINAL_DOCUMENT_BYTES')
        event = MessageEvent(text='MEDIA_RECOVERY_INPUT', message_type=MessageType.DOCUMENT,
                             source=source, message_id='media-1', media_urls=[str(original)],
                             media_types=['text/plain'], media_text_inlined=[False],
                             channel_context='ORIGINAL_CHANNEL_CONTEXT', channel_prompt='ORIGINAL_CHANNEL_PROMPT',
                             reply_to_message_id='quoted-1', reply_to_text='ORIGINAL_QUOTE',
                             reply_to_author_id='quoted-user', reply_to_author_name='Quoted User')
        # A cache alias must never redirect publication outside managed media.
        import hashlib
        from gateway.platforms.base import get_document_cache_dir
        root = get_document_cache_dir() / 'native-inputs'
        root.mkdir(parents=True, exist_ok=True)
        escaped = state / 'escaped'
        escaped.mkdir()
        digest_dir = root / hashlib.sha256(original.read_bytes()).hexdigest()
        digest_dir.symlink_to(escaped, target_is_directory=True)
        try:
            try:
                snapshot_native(runner, event)
            except RuntimeStoreError as exc:
                assert exc.reason == 'invalid_params', exc.reason
            else:
                raise AssertionError('media publication followed a cache-directory symlink')
            assert not list(escaped.iterdir())
            assert not event._gateway_accepted
        finally:
            digest_dir.unlink()
        receipt = await authority.admit_native(event)
        assert event._gateway_accepted
        saved = rows(receipt.ref.session_id)[0]
        restored = restore_native(saved['payload'])
        retained = Path(restored.media_urls[0])
        assert retained != original, 'accepted attachment still aliases adapter temp file'
        original.write_text('MUTATED_TEMP_BYTES')
        original.unlink()
        event.media_urls[:] = ['/missing/mutated']
        event.media_types[:] = ['application/evil']
        event.text = 'MUTATED_EVENT'
        event.channel_context = 'MUTATED_CONTEXT'
        event.source.chat_id = 'mutated-route'
        os.utime(retained, (1, 1))
        cleanup_document_cache(max_age_hours=0)
        assert retained.read_text() == 'ORIGINAL_DOCUMENT_BYTES'
        # A separate started row becomes unknown; neither it nor its file is reclaimed.
        unknown_event = replace(restored, source=replace(restored.source, chat_id='unknown-media'), message_id='unknown-1')
        unknown = await authority.admit_native(unknown_event)
        assert claim_session_input(authority.db, epoch=authority.epoch, session_id=unknown.ref.session_id)
        evidence = dict(sid=receipt.ref.session_id, saved=saved, unknown_sid=unknown.ref.session_id,
                        retained=str(retained), unknown=rows(unknown.ref.session_id))
        (state / 'accepted.json').write_text(json.dumps(evidence))
        threading.Event().wait(60)
        raise AssertionError('parent failed to kill owner')

    if mode == 'recover':
        before = json.loads((state / 'accepted.json').read_text())
        sid = before['sid']
        initial = rows(sid)
        binding = (sid, source, adapter)
        os.environ['TELEGRAM_ALLOWED_USERS'] = 'foreign'
        assert (await authority.recover_native_sessions([binding]))[sid] == 'permission_denied'
        assert rows(sid) == initial
        os.environ['TELEGRAM_ALLOWED_USERS'] = 'fixture-user'
        retained = Path(before['retained'])
        original_bytes = retained.read_bytes()
        retained.write_bytes(b'CORRUPTED_RETAINED_BLOB')
        try:
            assert (await authority.recover_native_sessions([binding]))[sid] == 'storage_unavailable'
            assert rows(sid) == initial
            assert not peer.requests
        finally:
            retained.write_bytes(original_bytes)
        result = await authority.recover_native_sessions([binding, (
            before['unknown_sid'], replace(source, chat_id='unknown-media'), adapter)])
        assert result[before['unknown_sid']] == 'unknown_execution', result
        task = authority.sessions[sid].task
        if task:
            await asyncio.wait_for(task, 35)
        after = rows(sid)
        assert after[0]['status'] == 'terminal' and after[0]['outcome'] == 'completed', after
        assert after[0]['payload'] == before['saved']['payload']
        assert rows(before['unknown_sid'])[0]['status'] == 'unknown'
        for row in [after[0], rows(before['unknown_sid'])[0]]:
            assert Path(restore_native(row['payload']).media_urls[0]).read_text() == 'ORIGINAL_DOCUMENT_BYTES'
        if initial[0]['status'] != 'terminal':
            assert len(peer.requests) == 1, peer.requests
            wire = json.dumps(peer.requests[0]['messages'])
            # False explicitly means a cached-file reference, not implicit text inlining.
            assert str(retained) in wire, wire
            for text in ('MEDIA_RECOVERY_INPUT', 'ORIGINAL_CHANNEL_CONTEXT',
                         'ORIGINAL_CHANNEL_PROMPT', 'ORIGINAL_QUOTE'):
                assert text in wire, (text, wire)
            assert 'MUTATED_' not in wire
            delivery = next(d for d in adapter.deliveries if 'LOCAL_ACK' in d['content'])
            assert delivery['chat_id'] == source.chat_id and delivery['reply_to'] == 'media-1', delivery
            assert delivery['metadata']['thread_id'] == 'topic-7', delivery
        else:
            assert not peer.requests and not adapter.deliveries
        (state / 'recovered.json').write_text(json.dumps(dict(model_calls=len(peer.requests),
            deliveries=adapter.deliveries, recovery=result, retained=before['retained'], rows=after)))
        return

    # A committed text row from before media support must keep its retry identity.
    from hermes_state_runtime import admit_session_input, cancel_session_input
    old_event = MessageEvent(text='legacy native input', source=source, message_id='legacy-text')
    ref = authority.register(source)
    encoded_source = source.to_dict()
    encoded_source['is_bot'] = source.is_bot
    legacy = {'text': old_event.text, 'native_text_v1': {
        'source': encoded_source, 'route': runner.session_store._generate_session_key(source),
        'timestamp': old_event.timestamp.isoformat(), 'event': {
            key: getattr(old_event, key) for key in (
                'user_id', 'user_name', 'message_id', 'platform_update_id',
                'reply_to_message_id', 'reply_to_text', 'reply_to_author_id',
                'reply_to_author_name', 'reply_to_is_own_message', 'allow_gateway_control')}}}
    principal = 'messaging:' + json.dumps([source.profile, source.platform.value, source.chat_id,
                                          source.thread_id, source.user_id], separators=(',', ':'))
    old_row = admit_session_input(authority.db, epoch=authority.epoch, principal_id=principal,
                                 session_id=ref.session_id, request_id=old_event.message_id, payload=legacy)
    cancel_session_input(authority.db, epoch=authority.epoch, admission_id=old_row['admission_id'])
    retried = await authority.admit_native(old_event)
    assert retried.admission_id == old_row['admission_id'] and retried.status == 'terminal'
    assert rows(ref.session_id)[0]['payload'] == legacy

    # Closed owner context: lists are copied, public source codec never carries trust.
    event = MessageEvent(text='context', source=source, auto_skill=['first', 'second'],
                         channel_prompt='channel policy', channel_context='backfilled history')
    payload = snapshot_native(runner, event)
    event.auto_skill.append('mutated')
    restored = restore_native(payload)
    assert restored.auto_skill == ['first', 'second']
    assert restored.channel_prompt == 'channel policy' and restored.channel_context == 'backfilled history'
    assert not restored.source.role_authorized and not restored.source.delivered_via_upstream_relay
    # Every ordinary native media type captures local files without changing semantics.
    for kind, suffix, mime in (
        (MessageType.PHOTO, '.png', 'image/png'), (MessageType.VOICE, '.ogg', 'audio/ogg'),
        (MessageType.AUDIO, '.mp3', 'audio/mpeg'), (MessageType.VIDEO, '.mp4', 'video/mp4'),
        (MessageType.STICKER, '.webp', 'image/webp'), (MessageType.DOCUMENT, '.txt', 'text/plain'),
    ):
        path = state / ('attachment' + suffix)
        path.write_bytes(b'owned-media-' + kind.value.encode())
        media_event = replace(event, message_type=kind, media_urls=[str(path)],
                              media_types=[mime], media_text_inlined=[False])
        media_payload = snapshot_native(runner, media_event)
        path.write_bytes(b'changed')
        captured = restore_native(media_payload)
        assert captured.message_type == kind and captured.media_types == [mime]
        assert captured.media_text_inlined == [False]
        assert Path(captured.media_urls[0]).read_bytes() == b'owned-media-' + kind.value.encode()
    denied = []
    cases = [replace(event, metadata={key: True}) for key in
             ('internal', 'whatsapp_from_owner', 'gateway_session_id', '_authorization_profile_home', 'arbitrary')]
    cases += [replace(event, source=replace(source, role_authorized=True)),
              replace(event, source=replace(source, delivered_via_upstream_relay=True)),
              replace(event, internal=True), replace(event, prompt_response={'prompt_id': 'forged'}),
              replace(event, media_urls=['https://example.invalid/not-captured']),
              replace(event, channel_prompt={'untyped': True})]
    foreign_home = replace(source)
    foreign_home._authorization_profile_home = state / 'foreign-home'
    cases.append(replace(event, source=foreign_home))
    for case in cases:
        try:
            await authority.admit_native(case)
        except RuntimeStoreError as exc:
            assert exc.reason == 'invalid_params', exc.reason
            denied.append(exc.reason)
        else:
            raise AssertionError('unsupported trust/context accepted')
        assert not case._gateway_accepted
    runner.config.multiplex_profiles = True
    try:
        snapshot_native(runner, event)
    except RuntimeStoreError as exc:
        assert exc.reason == 'invalid_params'
    else:
        raise AssertionError('multiplex accepted without transport-home provenance')
    runner.config.multiplex_profiles = False
    missing = replace(event, source=replace(source, user_id='foreign'), media_urls=['/missing/private'])
    try:
        await authority.admit_native(missing)
    except RuntimeStoreError as exc:
        assert exc.reason == 'permission_denied', exc.reason
    else:
        raise AssertionError('unauthorized media was accepted')
    assert not missing._gateway_accepted
    (state / 'guards.json').write_text(json.dumps(dict(context=payload, rejected=denied)))


if __name__ == '__main__':
    peer = ThreadingHTTPServer(('127.0.0.1', 0), ModelPeer)
    peer.requests, peer.metadata_requests = [], []
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
