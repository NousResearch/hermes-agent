"""A relay automation row must re-verify against the source it stores.

Regression: ``snapshot_automation`` borrows the human envelope's provenance but drops
the platform reply anchor from the source (#52694, #123097). The relay seal binds the
whole source dict, so the borrowed seal no longer matches the stored source: the row's
re-verification raises ``profile_mismatch``, and because the automation row is the FIFO
head, the session pauses for good — every later message queues behind it and the chat
goes deaf until the session is rotated by hand.
"""

import pytest

from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.platforms.event import MessageEvent
from gateway.run import GatewayRunner
from gateway.session_authority import SessionAuthority, initialize_session_authority
from gateway.session import SessionSource
from hermes_state_runtime import get_session_admission


async def _authority(tmp_path, monkeypatch):
    monkeypatch.setenv('HERMES_HOME', str(tmp_path))
    monkeypatch.setattr(SessionAuthority, '_schedule', lambda self, ref: None)
    monkeypatch.setattr('gateway.session_bot._watch_reply', lambda *args: None)
    runner = GatewayRunner(GatewayConfig(sessions_dir=tmp_path / 'sessions'))
    from hermes_state import SessionDB
    db = SessionDB(tmp_path / 'state.db')
    runner.session_store._db = db
    runner._session_db = db
    authority = await initialize_session_authority(runner, profile_id='default', instance_id='relay-seal-fixture',
                                                  db=db)
    return runner, authority


def _relay_adapter(tmp_path):
    """A relay adapter whose transport passes the connector proof (``_relay_connector``)."""
    from gateway.relay.adapter import RelayAdapter
    from gateway.relay.descriptor import CapabilityDescriptor
    from gateway.relay.ws_transport import WebSocketRelayTransport

    descriptor = CapabilityDescriptor(1, 'relay', 'Fixture', 4096, False, True, False, 'plain', 'chars')
    transport = WebSocketRelayTransport('http://127.0.0.1:1/relay', 'relay', '',
                                        gateway_id='fixture-gateway', upgrade_secret='fixture-secret')
    transport._ws = object()  # the connector proof requires a live socket
    transport._descriptors_by_platform['relay'] = descriptor
    adapter = RelayAdapter(PlatformConfig(), descriptor, transport=transport)
    return adapter, transport


def _relay_source(message_id=None):
    return SessionSource(platform=Platform.RELAY, chat_id='hub', chat_type='group',
                          chat_name='agent-hub', user_id='alex', user_name='alex',
                          message_id=message_id, delivered_via_upstream_relay=True)


@pytest.mark.asyncio
async def test_relay_automation_row_reverifies_after_the_anchor_strip(tmp_path, monkeypatch):
    """The stored automation row must re-verify: its seal must bind the source it stores."""
    from gateway.session_envelope import restore_native
    from gateway.session_ingress_context import native_callback, register_transport_home, relay_callback

    runner, authority = await _authority(tmp_path, monkeypatch)
    adapter, transport = _relay_adapter(tmp_path)
    runner.adapters = {Platform.RELAY: adapter}
    runner._wire_adapter_handlers(adapter)
    register_transport_home(runner, None, tmp_path)
    try:
        human = MessageEvent(text='human', source=_relay_source('human-1'), message_id='human-1')
        with native_callback(runner, human, tmp_path), relay_callback(transport, human):
            await authority.admit_native(human)

        route = runner.session_store._generate_session_key(human.source)
        sid = runner.session_store.peek_session_id(route)
        # The wake's source carries no anchor: the automation envelope strips it.
        wake_source = _relay_source()
        event = MessageEvent(text='notice', source=wake_source, internal=True,
                             metadata={'gateway_session_key': route, 'gateway_session_id': sid})
        receipt = await authority.admit_automation(adapter, event, 'notice')
        row = get_session_admission(authority.db, admission_id=receipt.admission_id)
        assert row is not None
        envelope = row['payload']['native_text_v1']
        assert 'message_id' not in envelope['source']
        assert 'relay' in envelope['provenance']
        # The seal binds the STORED source: re-verifying the row must succeed.
        restored = restore_native(row['payload'], runner)
        assert restored.source.message_id is None
        assert restored.source.chat_id == 'hub'
    finally:
        authority.db.close()
