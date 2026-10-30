"""A restored messaging route without an origin still admits its completions (#146696)."""
import json

import pytest


@pytest.mark.asyncio
@pytest.mark.parametrize('platform', ['telegram', 'discord'])
@pytest.mark.parametrize('compressed', [False, True], ids=['direct', 'compressed'])
@pytest.mark.parametrize('missing_origin', [False, True], ids=['with-origin', 'null-origin'])
async def test_completion_uses_logical_owner_after_route_restore(tmp_path, monkeypatch, platform, compressed, missing_origin):
    from gateway.config import GatewayConfig, Platform, PlatformConfig
    from gateway.platforms.event import MessageEvent
    from gateway.run import GatewayRunner
    from gateway.session import SessionEntry
    from gateway.session_authority import SessionAuthority, initialize_session_authority
    from gateway.session_automation import completion_admission
    from gateway.session_ingress_context import native_callback
    from hermes_constants import get_hermes_home
    from hermes_state_runtime import list_session_admissions
    from plugins.platforms.discord.adapter import DiscordAdapter
    from plugins.platforms.telegram.adapter import TelegramAdapter
    from tests.gateway.test_completion_admission import pending
    from tools import async_delegation as delegation

    # Admit through the real adapter and SQLite FIFO, stopping before a model turn.
    monkeypatch.setattr(SessionAuthority, '_schedule', lambda self, ref: None)
    monkeypatch.setenv(platform.upper() + '_ALLOWED_USERS', '42')
    runner = GatewayRunner(GatewayConfig())
    adapter_cls = TelegramAdapter if platform == 'telegram' else DiscordAdapter
    adapter = adapter_cls(PlatformConfig(enabled=True, token='fixture-token', typing_indicator=False))
    runner.adapters = {Platform(platform): adapter}
    authority = await initialize_session_authority(runner, profile_id='default', instance_id='restore')
    runner._wire_adapter_handlers(adapter)
    source = adapter.build_source(chat_id='-10042' if platform == 'telegram' else '42',
                                  chat_type='group' if platform == 'telegram' else 'dm',
                                  thread_id='7' if platform == 'telegram' else None, user_id='42')
    human = MessageEvent(text='human', source=source, message_id='human')
    with native_callback(runner, human, get_hermes_home()):
        receipt = await authority.admit_native(human)
    key = runner.session_store._generate_session_key(source)
    # run_turn_prepare caches the inbound source before a turn can start a delegation.
    runner._cache_session_source(key, source)
    root = receipt.ref.session_id
    target = root
    if compressed:
        target = root + '-child'
        authority.db.append_message(root, 'user', 'human')
        authority.db.publish_compression_child(parent_session_id=root, child_session_id=target, source=platform,
                                              messages=[{'role': 'user', 'content': 'summary'}], require_compression_lease=False)
        assert runner.session_store.advance_compression_session(key, root, target) is not None

    store = runner.session_store
    entry = store.lookup_by_session_key(key)
    if missing_origin:
        entry.origin = None
    store._save_entry(key)
    stored = authority.db.load_gateway_routing_entries(scope=store._routing_scope())[key]
    store._entries[key] = SessionEntry.from_dict(json.loads(stored))
    event = dict(pending(key, 'restored-completion'), parent_session_id=target)

    assert completion_admission(runner, event) is None
    assert await runner._deliver_async_delegation_group([event]) is True
    rows = list_session_admissions(authority.db, session_id=root, pending_only=False)
    assert len(rows) == 2 and 'restored-completion' in rows[-1]['payload']['text']
    assert delegation.get_durable_delegation(event['delegation_id'])['delivery_state'] == 'delivered'

    # A lost acknowledgement must find the committed input under the logical root.
    if missing_origin:
        store._entries[key].origin = None
    runner._completion_deliveries_delivered.clear()
    assert completion_admission(runner, event)['admission_id'] == rows[-1]['admission_id']
    assert await runner._deliver_async_delegation_group([event]) is None
    assert len(list_session_admissions(authority.db, session_id=root, pending_only=False)) == 2
    if compressed:
        assert list_session_admissions(authority.db, session_id=target, pending_only=False) == []
