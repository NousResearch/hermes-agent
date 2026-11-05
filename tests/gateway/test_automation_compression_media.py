"""Producer completions and heartbeats keep entering a messaging conversation after compression
moved its route to a child and after a prior input's retained media was released."""
import asyncio

import pytest


async def _discord_owner(tmp_path, monkeypatch):
    from gateway.config import GatewayConfig, Platform, PlatformConfig
    from gateway.run import GatewayRunner
    from gateway.session_authority import initialize_session_authority, SessionAuthority
    from plugins.platforms.discord.adapter import DiscordAdapter

    monkeypatch.setattr(SessionAuthority, '_schedule', lambda self, ref: None)
    monkeypatch.setenv('DISCORD_ALLOWED_USERS', '42')
    runner = GatewayRunner(GatewayConfig())
    adapter = DiscordAdapter(PlatformConfig(enabled=True, token='fixture-token', typing_indicator=False))
    runner.adapters = {Platform.DISCORD: adapter}
    authority = await initialize_session_authority(runner, profile_id='default', instance_id='owner')
    runner._wire_adapter_handlers(adapter)
    return runner, adapter, authority, adapter.build_source(chat_id='42', chat_type='dm', user_id='42')


async def _admit_human(runner, authority, event):
    from gateway.session_ingress_context import native_callback
    from hermes_constants import get_hermes_home
    with native_callback(runner, event, get_hermes_home()):
        return await authority.admit_native(event)


@pytest.mark.asyncio
async def test_completion_after_compression_admits_to_the_logical_root(tmp_path, monkeypatch):
    from gateway.platforms.event import MessageEvent
    from persistence.conversation.lineage import CompressionChild
    from persistence.conversation.messages import NewMessage
    from tests.fixtures.storage_seed import seed_messages
    from tests.gateway.test_completion_admission import pending

    runner, _, authority, source = await _discord_owner(tmp_path, monkeypatch)
    receipt = await _admit_human(runner, authority, MessageEvent(text='human', source=source, message_id='human'))
    key = runner.session_store._generate_session_key(source)
    root = receipt.ref.session_id
    db = authority.persistence
    seed_messages(db, root, {'role': 'user', 'content': 'human'})
    child = root + '-child'
    db.lineage.publish_compression_child(root, CompressionChild(session_id=child, source='discord'),
                                         [NewMessage.from_live({'role': 'user', 'content': 'summary'})],
                                         require_compression_lease=False)
    assert runner.session_store.advance_compression_session(key, root, child) is not None
    event = dict(pending(key, 'after-compression'), parent_session_id=child)
    assert await runner._deliver_async_delegation_group([event]) is True
    rows = db.admissions.list(session_id=root, pending_only=False)
    assert next(r.request_id for r in rows) == 'human' and len(rows) == 2, rows
    assert 'after-compression' in rows[-1].payload['text']
    assert db.admissions.list(session_id=child, pending_only=False) == []
    # A lost-ACK redelivery is the same admission, found under the root's ledger.
    from gateway.session_automation import completion_admission
    assert completion_admission(runner, event)['admission_id'] == rows[-1].admission_id
    await asyncio.sleep(0)


@pytest.mark.asyncio
async def test_text_completion_after_a_released_photo_input_is_admitted(tmp_path, monkeypatch):
    from gateway.platforms.base import get_image_cache_dir
    from gateway.platforms.event import MessageEvent, MessageType
    from gateway.session_ingress_media import release_admission_media
    from gateway.session_persistence import authority_persistence
    from tests.gateway.test_completion_admission import pending

    runner, _, authority, source = await _discord_owner(tmp_path, monkeypatch)
    photo = get_image_cache_dir() / 'photo.png'
    photo.write_bytes(b'\x89PNG\r\n\x1a\n' + b'\x00' * 32)
    receipt = await _admit_human(runner, authority, MessageEvent(
        text='look', source=source, message_id='photo', message_type=MessageType.PHOTO,
        media_urls=[str(photo)], media_types=['image/png']))
    sid = receipt.ref.session_id
    started = authority.persistence.admissions.claim(epoch=authority.epoch, session_id=sid)
    authority.persistence.admissions.settle(epoch=authority.epoch, admission_id=started.admission_id,
                                            generation=started.generation, outcome='completed')
    assert release_admission_media(authority_persistence(authority), started.admission_id) == 1  # what _drain does
    key = runner.session_store._generate_session_key(source)
    assert await runner._deliver_async_delegation_group([dict(pending(key, 'after-photo'), parent_session_id=sid)])
    rows = authority.persistence.admissions.list(session_id=sid, pending_only=False)
    assert len(rows) == 2 and 'after-photo' in rows[-1].payload['text'], rows
    assert 'media' not in rows[-1].payload['native_text_v1']
    await asyncio.sleep(0)
