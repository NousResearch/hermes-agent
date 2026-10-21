"""A parked-profile notice sent through relay never seals another turn's open Slack stream.

On relay Slack the stream is the message: ``RelayAdapter.send`` seals a chat's single open draft with
the first send that carries neither a per-turn identity nor the interim marker. In a channel where
one user's turn is streaming, the notice to a second user whose message routes to a parked profile
must go out as its own message and leave that stream open.
"""
import asyncio
from types import SimpleNamespace

import pytest

from gateway.config import Platform, PlatformConfig
from gateway.parked_route_refusal import PARKED_ROUTE_NOTICE, refuse_parked_route
from gateway.relay.adapter import RelayAdapter
from gateway.relay.descriptor import CapabilityDescriptor


class _Transport:
    _identities = [('slack', 'hermes')]

    def __init__(self):
        self.frames = []

    def descriptor_for_platform(self, platform):
        return None

    async def send_outbound(self, frame, platform=None):
        self.frames.append(frame)
        return {'success': True, 'message_id': '9.9'}


@pytest.mark.asyncio
async def test_a_relay_notice_leaves_another_turns_open_stream_open():
    descriptor = CapabilityDescriptor(
        contract_version=1, platform='slack', label='Slack', max_message_length=4000,
        supports_draft_streaming=True, supports_edit=True, supports_threads=True, markdown_dialect='mrkdwn',
        len_unit='chars', supported_ops=('send', 'edit', 'draft', 'prompt'))
    adapter = RelayAdapter(PlatformConfig(enabled=True, extra={}), descriptor, transport=_Transport())
    adapter.gateway_runner = SimpleNamespace(_thread_metadata_for_source=lambda source: None)
    await adapter.send_draft('C', 7, "U1's answer, streaming")
    streaming = dict(adapter._open_draft_by_chat)
    assert streaming, "test setup: U1's draft did not open"
    source = SimpleNamespace(platform=Platform.SLACK, chat_id='C', user_id='U2')
    refuse_parked_route(adapter, source, 'beta', reply_to='200.2')
    await asyncio.gather(*list(adapter._background_tasks))
    assert adapter._open_draft_by_chat == streaming, "the notice sealed U1's stream"
    notices = [frame for frame in adapter._transport.frames if frame.get('content') == PARKED_ROUTE_NOTICE]
    assert [frame['op'] for frame in notices] == ['send'], adapter._transport.frames
