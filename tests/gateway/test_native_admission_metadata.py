"""Native admission keeps main's inbound contract: real adapter events run a turn (N78-N80)."""
import ast
from copy import deepcopy
from dataclasses import replace
from datetime import datetime
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from gateway import shutdown_flush as spool
from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.platforms.base import SendResult
from gateway.platforms.event import MessageEvent
from gateway.session import SessionSource
from gateway.run import GatewayRunner
from gateway.session_authority import initialize_session_authority
from gateway.session_envelope import (
    _native_metadata, _snapshot_native, _validate_native, restore_native, snapshot_native,
)
from gateway.session_results import execution_result
from hermes_state_runtime import RuntimeStoreError
from tests.waiting import async_wait_until

REPO = Path(__file__).resolve().parents[2]
CHAT = '15551234567@s.whatsapp.net'
VCARD = 'BEGIN:VCARD\nPHOTO;BASE64:' + 'A' * 20000 + '\nEND:VCARD'
# (bridge payload overrides, expected turn text, expected restored metadata)
BRIDGE_SHAPES = [
    ({'body': 'hello', 'nativeType': 'conversation'}, 'hello', {'whatsapp_native_type': 'conversation'}),
    ({'body': 'quoted', 'nativeType': 'extendedTextMessage'}, 'quoted',
     {'whatsapp_native_type': 'extendedTextMessage'}),
    ({'body': '/tmp/report.csv is broken', 'nativeType': 'conversation'}, '/tmp/report.csv is broken',
     {'whatsapp_native_type': 'conversation'}),
    ({'body': '📍 here', 'mediaType': 'location', 'nativeType': 'locationMessage',
      'nativeMetadata': {'location': {'latitude': 1.5, 'longitude': 2.5, 'live': False}}}, '📍 here',
     {'whatsapp_native_type': 'locationMessage',
      'whatsapp_native': {'location': {'latitude': 1.5, 'longitude': 2.5, 'live': False}}}),
    ({'body': '👍', 'mediaType': 'reaction', 'nativeType': 'reactionMessage',
      'nativeMetadata': {'reaction': {'text': '👍', 'messageId': 'x', 'remoteJid': '1', 'participant': ''}}}, '👍',
     {'whatsapp_native_type': 'reactionMessage',
      'whatsapp_native': {'reaction': {'text': '👍', 'messageId': 'x', 'remoteJid': '1', 'participant': ''}}}),
    ({'body': 'Poll: lunch?', 'mediaType': 'poll', 'nativeType': 'pollCreationMessage',
      'nativeMetadata': {'poll': {'question': 'lunch?', 'options': ['a', 'b'], 'selectableCount': 1}}},
     'Poll: lunch?', {'whatsapp_native_type': 'pollCreationMessage',
                      'whatsapp_native': {'poll': {'question': 'lunch?', 'options': ['a', 'b'],
                                                   'selectableCount': 1}}}),
    # A vCard with a photo exceeds the bound: the bulky native block is dropped, the turn still runs.
    ({'body': 'Contact: Ann', 'mediaType': 'contact', 'nativeType': 'contactMessage',
      'nativeMetadata': {'contact': {'displayName': 'Ann', 'vcard': VCARD}}}, 'Contact: Ann',
     {'whatsapp_native_type': 'contactMessage'}),
    # WHATSAPP_FORWARD_OWNER_MESSAGES owner forward: main runs it as a turn, so does the owner.
    ({'body': 'owner typed', 'nativeType': 'conversation', 'fromOwner': True}, '[owner reply] owner typed',
     {'whatsapp_native_type': 'conversation', 'whatsapp_from_owner': True}),
]


@pytest.mark.asyncio
async def test_real_whatsapp_bridge_events_run_a_turn_through_native_admission(tmp_path, monkeypatch):
    from plugins.platforms.whatsapp.adapter import WhatsAppAdapter
    monkeypatch.setenv('WHATSAPP_ALLOWED_USERS', '15551234567')
    runner = GatewayRunner(GatewayConfig(sessions_dir=tmp_path / 'sessions'))
    adapter = WhatsAppAdapter(PlatformConfig(enabled=True, extra={
        'text_batch_delay_seconds': 0, 'session_path': str(tmp_path / 'wa')}))
    adapter.gateway_runner, runner.adapters = runner, {Platform.WHATSAPP: adapter}
    authority = await initialize_session_authority(runner, profile_id='default', instance_id='wa')
    runner._wire_adapter_handlers(adapter)
    sent, turns = [], []

    async def send(chat_id, content, **_):
        sent.append(content)
        return SendResult(success=True, message_id='s')

    async def turn(event, *_args, **_kwargs):
        turns.append((event.text, event.metadata))
        execution_result.get()['result'] = {'final_response': 'ok', 'messages': []}
        return 'ok'

    monkeypatch.setattr(adapter, 'send', send)
    monkeypatch.setattr(runner, '_handle_message_with_agent', turn)
    try:
        for index, (shape, text, metadata) in enumerate(BRIDGE_SHAPES):
            event = await adapter._build_message_event(
                {'messageId': f'm{index}', 'chatId': CHAT, 'senderId': CHAT, **shape})
            adapter._enqueue_text_event(event)
            await async_wait_until(lambda: len(sent) > index and not adapter._background_tasks,
                                   timeout=20, what=f'reply to {shape["nativeType"]}')
            assert sent[index] == 'ok', sent
            # Execution carries exactly the obligation id the adapter spooled (none when the
            # adapter did not spool, e.g. command-shaped text) beside the adapter's own metadata.
            spooled = event.metadata.get('_hermes_durable_inbound_id')
            assert turns[index][1].pop('_hermes_durable_inbound_id', None) == spooled, turns
            assert turns[index] == (text, metadata), turns
    finally:
        authority.db.close()


@pytest.mark.asyncio
async def test_native_codec_closed_metadata_contract_and_reply_expected(tmp_path, monkeypatch):
    from plugins.platforms.whatsapp.adapter import WhatsAppAdapter
    monkeypatch.setenv('WHATSAPP_ALLOWED_USERS', '15551234567')
    runner = GatewayRunner(GatewayConfig(sessions_dir=tmp_path / 'sessions'))
    adapter = WhatsAppAdapter(PlatformConfig(enabled=True, extra={'session_path': str(tmp_path / 'wa')}))
    runner.adapters = {Platform.WHATSAPP: adapter}
    authority = await initialize_session_authority(runner, profile_id='default', instance_id='codec')
    try:
        event = MessageEvent(text='chatter', source=adapter.build_source(chat_id=CHAT, user_id=CHAT, chat_type='group'),
                             message_id='c1')
        # Slack-style silence policy survives the envelope (N80); unknown stays absent (old digest).
        assert 'reply_expected' not in snapshot_native(runner, event)['native_text_v1']['event']
        for value in (False, True):
            assert restore_native(snapshot_native(runner, replace(event, reply_expected=value))).reply_expected is value
        refused = [{'whatsapp_native_type': 'x', 'internal': True}, {'gateway_session_id': 's'},
                   {'_authorization_profile_home': '/x'}, {'arbitrary': 1}, {'whatsapp_native_type': 'x' * 65},
                   {'whatsapp_native': {'gateway_session_id': 's'}}, {'whatsapp_from_owner': 'yes'},
                   {'whatsapp_native': {'poll': {1, 2}}}]
        for metadata in refused:
            with pytest.raises(RuntimeStoreError, match='invalid_params'):
                snapshot_native(runner, replace(event, metadata=metadata))
        for bad in (replace(event, reply_expected='no'), replace(event, text='/new now')):
            with pytest.raises(RuntimeStoreError, match='invalid_params'):
                snapshot_native(runner, bad)
        # MSG-2: an adapter's absent media list (None) is no media at the envelope boundary, with the
        # same fingerprint as []; any other non-list still refuses.
        for name in ('media_urls', 'media_types', 'media_text_inlined'):
            assert snapshot_native(runner, replace(event, **{name: None})) == snapshot_native(runner, event)
            for bad in ('x', ('x',)):
                with pytest.raises(RuntimeStoreError, match='invalid_params'):
                    snapshot_native(runner, replace(event, **{name: bad}))
    finally:
        authority.db.close()
    # Closed producer matrix: no other adapter stamps MessageEvent.metadata on ordinary inbound,
    # so the WhatsApp keys above are the whole contract. A new producer must extend it here.
    producers = set()
    for path in [*REPO.glob('gateway/platforms/**/*.py'), *REPO.glob('plugins/platforms/**/*.py')]:
        for node in ast.walk(ast.parse(path.read_text(encoding='utf-8'))):
            if (isinstance(node, ast.Call) and getattr(node.func, 'id', getattr(node.func, 'attr', '')) == 'MessageEvent'
                    and any(k.arg == 'metadata' for k in node.keywords)):
                producers.add(path.relative_to(REPO).as_posix())
    assert producers == {'plugins/platforms/whatsapp/adapter.py'}, producers


@pytest.fixture
def world(tmp_path, monkeypatch):
    flush_dir = tmp_path / "flush"
    flush_dir.mkdir()
    monkeypatch.setattr(spool, "_get_flush_dir", lambda: flush_dir)
    adapter = object()
    key = "agent:main:telegram:dm:owner"
    runner = SimpleNamespace(
        config=SimpleNamespace(),
        _is_user_authorized_for_source=lambda *args, **kwargs: True,
        _adapter_for_source=lambda source: adapter,
        _adapter_flag=lambda *args: False,
        _adapter_profile_for_source=lambda source: None,
        session_store=SimpleNamespace(_generate_session_key=lambda source: key),
    )
    source = SessionSource(platform=Platform.TELEGRAM, chat_id="owner", user_id="owner", chat_type="dm")
    event = MessageEvent(text="ordinary owner text", source=source, message_id="message-1",
                         timestamp=datetime(2026, 1, 1))
    return runner, event, key, flush_dir


def test_spooled_telegram_metadata_is_accepted_without_mutating_event(world):
    runner, event, key, _ = world
    assert spool.record_durable_inbound_event(key, event)
    metadata = deepcopy(event.metadata)
    assert metadata["_hermes_durable_inbound_path"]
    assert metadata["_hermes_durable_inbound_id"]
    assert _native_metadata(event) == {}
    assert _validate_native(runner, event)["platform"] == "telegram"
    assert event.metadata == metadata


@pytest.mark.parametrize("durable", [False, True])
def test_foreign_metadata_still_refuses_native_admission(world, durable):
    runner, event, key, _ = world
    if durable:
        assert spool.record_durable_inbound_event(key, event)
    event.metadata["foreign_key"] = "not gateway bookkeeping"
    assert _native_metadata(event) is None
    with pytest.raises(RuntimeStoreError, match="invalid_params"):
        _validate_native(runner, event)


@pytest.mark.parametrize("durable", [False, True])
def test_whatsapp_metadata_projection_is_unchanged(world, durable):
    runner, event, key, _ = world
    event.source.platform = Platform.WHATSAPP
    expected = {"whatsapp_native_type": "locationMessage", "whatsapp_from_owner": True,
                "whatsapp_native": {"location": {"latitude": 1.5, "longitude": 2.5}}}
    event.metadata = deepcopy(expected)
    if durable:
        assert spool.record_durable_inbound_event(key, event)
    assert _native_metadata(event) == expected
    assert _validate_native(runner, event)["platform"] == "whatsapp"
    event.metadata["foreign_key"] = True
    assert _native_metadata(event) is None


def test_native_round_trip_keeps_obligation_until_terminal_acknowledgement(world):
    runner, event, key, flush_dir = world
    assert spool.record_durable_inbound_event(key, event)
    inbound_id = spool.durable_inbound_obligation_id(event)
    payload = json.loads(json.dumps(_snapshot_native(runner, event, None)))
    restored = restore_native(payload)
    assert spool.durable_inbound_obligation_id(restored) == inbound_id
    assert "_hermes_durable_inbound_path" not in restored.metadata
    assert restored.original_inputs()[0]["inbound_id"] == inbound_id
    assert _native_metadata(restored) == {}
    _validate_native(runner, restored)
    assert len(list(flush_dir.glob("inbound-*.json"))) == 1
    spool.acknowledge_durable_inbound_event(restored)
    assert list(flush_dir.glob("inbound-*.json")) == []


def test_unspooled_native_envelope_has_no_obligation(world):
    runner, event, _, _ = world
    payload = _snapshot_native(runner, event, None)
    assert "durable_inbound_id" not in payload["native_text_v1"]
    assert spool.durable_inbound_obligation_id(restore_native(payload)) == ""
