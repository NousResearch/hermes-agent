"""Inert real authority and peers for canonical batch protocol tests (no model/provider)."""
import asyncio
from types import SimpleNamespace

from gateway.config import GatewayConfig, Platform
from gateway.session import SessionSource, SessionStore
from gateway.session_authority import LiveSession, initialize_session_authority
from tests.fixtures.storage_seed import seed_session
from tests.waiting import async_wait_until

QUESTIONS = [{'qid': 'q0', 'question': 'Colour?', 'choices': ['red', 'blue'], 'multi_select': True},
             {'qid': 'q1', 'question': 'Size?', 'multi_select': False}]


class Peer:
    def __init__(self):
        self.frames = []
        self.client_identity = None
        self.canonical_clarify_projection = False

    def write(self, frame):
        self.frames.append(frame)
        return True

    def close(self):
        pass


async def make_owner(tmp_path, monkeypatch, answer):
    monkeypatch.setenv('HERMES_HOME', str(tmp_path))
    store = SessionStore(tmp_path / 'sessions', GatewayConfig())
    runner = SimpleNamespace(_session_db=store.persistence, session_store=store, _draining=False,
                             _handle_message=answer, _adapter_for_source=lambda source: None)
    owner = await initialize_session_authority(runner, profile_id='p', instance_id='owner')
    seed_session(owner.persistence, 's', 'telegram')
    owner.sessions['s'] = LiveSession(SessionSource(platform=Platform.TELEGRAM, chat_id='c'), 's')
    return owner


async def rpc(connection, method, **params):
    return await connection.dispatch({'id': method, 'method': method, 'params': params})


async def opening(peer):
    await async_wait_until(lambda: any(f.get('params', {}).get('type') == 'clarify.batch.request' for f in peer.frames))
    return next(f['params']['payload'] for f in peer.frames if f.get('params', {}).get('type') == 'clarify.batch.request')
