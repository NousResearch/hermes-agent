"""ACP ``/retry`` on the gateway path: real authority, SQLite and ``AuthorityConnection.dispatch``
behind ``GatewayACPAgent.prompt`` (#140190 review).

An image turn's refusal is the command's reply, not a failed request; a rewind whose snapshot
refresh lost its connection is replayed by the next ``/retry``, never re-derived from the
already-shortened transcript (which would remove the turn before it too)."""
import asyncio
import contextlib
from types import SimpleNamespace

import acp
import pytest

from acp_adapter.gateway_server import GatewayACPAgent
from gateway.config import GatewayConfig
from gateway.session import SessionStore
from gateway.session_authority import initialize_session_authority
from gateway.session_controls import AuthorityConnection
from hermes_cli.gateway_client import GatewayClientError
from tests.acp_adapter.test_gateway_unacked_slash import SlashWire

IMAGE = {'type': 'image_url', 'image_url': {'url': 'data:image/png;base64,AA=='}}


class LossyResume(SlashWire):
    """Loses the ``session.resume`` right after a committed rewind (a dropped connection)."""

    def __init__(self, loop):
        super().__init__(loop)
        self.lose_ack, self.after_rewind, self.lost = False, False, 0

    async def rpc(self, method, _timeout=None, **params):
        if method == 'session.resume' and self.after_rewind and not self.lost:
            self.lost += 1
            raise GatewayClientError('Gateway disconnected')
        result = await super().rpc(method, **params)
        self.after_rewind = method == 'session.mutate' and params.get('operation') == 'rewind'
        return result


@contextlib.asynccontextmanager
async def _editor(tmp_path, monkeypatch, wire_class=SlashWire):
    monkeypatch.setenv('HERMES_HOME', str(tmp_path))
    store = SessionStore(tmp_path / 'sessions', GatewayConfig())
    executed = []

    async def answer(event):
        from gateway.session_results import execution_result
        executed.append(event.text)
        execution_result.get()['result'] = {'final_response': 'ACK_' + event.text, 'completed': True, 'messages': []}
        return 'ACK_' + event.text

    from gateway import run
    from gateway.session_local import create_local_session
    monkeypatch.setattr(run, '_load_gateway_config', lambda: {'platform_toolsets': {'acp': []}})
    runner = SimpleNamespace(_session_db=store._db, session_store=store, _draining=False, adapters={},
                             _handle_message=answer, _adapter_for_source=lambda source: None,
                             _cached_agent_for=lambda route: None, _evict_cached_agent=lambda route: None,
                             _resolve_session_agent_runtime=lambda **kwargs: ('frozen', {}))
    authority = await initialize_session_authority(runner, profile_id='p', instance_id='owner')
    wire = wire_class(asyncio.get_running_loop())
    wire.lose_ack = False
    wire.viewer = AuthorityConnection(authority, wire, {'user_id': 'editor', 'profile_id': 'p'})
    sid = create_local_session(authority, wire.viewer.actor, {'request_id': 'editor', 'source': 'acp',
                                                             'cwd': str(tmp_path), 'model': 'frozen',
                                                             'toolsets': []}).session_id
    agent = GatewayACPAgent()
    agent._gateway = wire
    updates = []

    async def session_update(session_id, update):
        updates.append(update.content.text)
    agent._conn = SimpleNamespace(session_update=session_update)
    agent._snapshots[sid] = await wire.rpc('session.resume', session_id=sid)
    agent._event_task = asyncio.create_task(agent._events())
    try:
        yield SimpleNamespace(agent=agent, db=authority.db, sid=sid, wire=wire, updates=updates, executed=executed)
    finally:
        agent._event_task.cancel()
        await wire.viewer.close()
        store.close_all_db_handles()


def _users(db, sid):
    return [m['content'] for m in db.get_messages(sid) if m['role'] == 'user']


@pytest.mark.asyncio
async def test_retry_of_an_image_turn_is_the_commands_reply_and_keeps_history(tmp_path, monkeypatch):
    async with _editor(tmp_path, monkeypatch) as e:
        e.db.append_message(e.sid, 'user', [{'type': 'text', 'text': 'what is this?'}, IMAGE])
        e.db.append_message(e.sid, 'assistant', 'a picture')
        response = await asyncio.wait_for(e.agent.prompt([acp.text_block('/retry')], e.sid), 5)
        assert response.stop_reason == 'end_turn'
        assert any(u.startswith('/retry: retry does not support media') for u in e.updates), e.updates
        assert len(e.db.get_messages(e.sid)) == 2 and not e.executed


@pytest.mark.asyncio
async def test_retry_after_a_lost_snapshot_refresh_replays_the_same_rewind(tmp_path, monkeypatch):
    async with _editor(tmp_path, monkeypatch, LossyResume) as e:
        for ask, reply in (('first ask', 'first reply'), ('second ask', 'second reply')):
            e.db.append_message(e.sid, 'user', ask)
            e.db.append_message(e.sid, 'assistant', reply)
        with pytest.raises(GatewayClientError, match='disconnected'):
            await asyncio.wait_for(e.agent.prompt([acp.text_block('/retry')], e.sid), 5)
        assert _users(e.db, e.sid) == ['first ask']  # the rewind committed before the refresh failed
        response = await asyncio.wait_for(e.agent.prompt([acp.text_block('/retry')], e.sid), 5)
        assert response.stop_reason == 'end_turn'
        # The repeat replays the original rewind: the earlier turn stays, the retried text is the same.
        assert e.executed == ['second ask'], e.executed
        assert _users(e.db, e.sid)[0] == 'first ask', _users(e.db, e.sid)
