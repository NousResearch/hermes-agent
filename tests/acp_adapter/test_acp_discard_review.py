"""ACP ``/discard`` while a lost turn pauses the session (review #141357, GW-10).

Discard must not consume reply/approval sequence numbers the editor has not projected yet (the
pending-only refresh, not whole-snapshot adoption), and a paused prompt that is not ``/discard``
is refused before any media is staged or the retained retry identity is retired. Deterministic
protocol peers, real staging I/O."""
import asyncio
import base64

import acp
import pytest
from acp.schema import ImageContentBlock

from acp_adapter.gateway_commands import run_command
from acp_adapter.gateway_server import GatewayACPAgent
from hermes_cli.gateway_client import GatewayClientError
from tests.acp_adapter.test_acp_image_staging_boundary import PNG


class Peer:
    def __init__(self, snapshots):
        self.snapshots = iter(snapshots)
        self.events = asyncio.Queue()
        self.calls = []

    async def rpc(self, method, **params):
        self.calls.append(method)
        if method == 'session.resume':
            return next(self.snapshots)
        if method == 'prompt.resolve_unknown':
            return {'status': 'terminal', 'outcome': 'interrupted'}
        raise AssertionError((method, params))


class Editor:
    def __init__(self):
        self.messages = []
        self.marker = asyncio.Event()

    async def session_update(self, *, session_id, update):
        text = getattr(getattr(update, 'content', None), 'text', '')
        self.messages.append(text)
        if text == 'MARKER':
            self.marker.set()


@pytest.mark.asyncio
@pytest.mark.parametrize('route', ['existing_pending_refresh', 'new_discard'])
async def test_discard_preserves_buffered_reply(route):
    agent = GatewayACPAgent()
    old = {'pending': [{'admission_id': 'lost', 'status': 'unknown', 'execution_generation': 1}],
           'replay_epoch': 'owner', 'last_sequence': 1}
    fresh = {'pending': [], 'replay_epoch': 'owner', 'last_sequence': 4}
    agent._snapshots['s'] = dict(old)
    agent._gateway = Peer([dict(old), dict(fresh)] if route == 'new_discard' else [dict(fresh)])
    editor = agent._conn = Editor()
    if route == 'new_discard':
        await run_command(agent, 's', 'discard', 'lost')
    else:
        await agent._refresh_unknown('s')
    for seq, text in [(3, 'BUFFERED FOLLOWER ANSWER'), (5, 'MARKER')]:
        await agent._gateway.events.put({'params': {
            'session_id': 's', 'replay_epoch': 'owner', 'seq': seq,
            'type': 'message.delta', 'admission_id': 'follower', 'payload': {'text': text}}})
    pump = asyncio.create_task(agent._events())
    try:
        await asyncio.wait_for(editor.marker.wait(), 5)
        assert 'BUFFERED FOLLOWER ANSWER' in editor.messages, editor.messages
    finally:
        pump.cancel()
        await asyncio.gather(pump, return_exceptions=True)


@pytest.mark.asyncio
async def test_paused_image_refusal_does_not_stage_image(tmp_path):
    agent = GatewayACPAgent()
    agent._home = tmp_path
    unknown = {'pending': [{'admission_id': 'lost', 'status': 'unknown'}],
               'replay_epoch': 'owner', 'last_sequence': 1}
    agent._snapshots['s'] = dict(unknown)
    agent._gateway = Peer([dict(unknown)])
    agent._unacked['s'] = ('prior-retry', 'prior-id', ('prior-content', []))
    prompt = [acp.text_block('explain image'), ImageContentBlock(type='image',
              data=base64.b64encode(PNG).decode(), mime_type='image/png')]
    with pytest.raises(GatewayClientError, match='unknown_execution'):
        await agent.prompt(prompt, 's')
    staged = [str(p.relative_to(tmp_path)) for p in tmp_path.rglob('*.png')]
    assert not staged, staged
    assert 's' in agent._unacked
