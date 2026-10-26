"""A chat's /new (or /resume) wins over a turn that finishes on its previous conversation.

A Desktop turn runs on a Telegram chat's conversation A when the platform user sends ``/new``. The
reset publishes the chat's new conversation B on its route, then writes state.db (A ended, B's row
inserted). The finishing turn resolves the same route for its post-turn bookkeeping; if that lands
between the two halves, B has no row yet, and treating it as deleted hands the route back to A, so
the chat's next message continues the conversation the user just left. ``/resume`` has the same
window: the resumed row is reopened only after the route names it.

Real GatewayRunner, SessionAuthority, SessionStore and SessionDB; the commands run through their own
handlers. The window is held open at the real SQL statement that ends the predecessor row.
"""
import asyncio
import threading
import weakref
from pathlib import Path

import pytest

from gateway.config import GatewayConfig, Platform
from gateway.platforms.event import MessageEvent, MessageType
from gateway.run import GatewayRunner
from gateway.run_runtime import initialize_gateway_runtime
from gateway.runtime_ownership import process_ownership
from gateway.session import SessionSource
from gateway.session_ingress_context import register_transport_home
from tests.waiting import async_wait_until


class _Adapter:
    pass


@pytest.fixture
def served(tmp_path, monkeypatch):
    root = tmp_path / '.hermes'
    root.mkdir(mode=0o700)
    monkeypatch.setattr(Path, 'home', lambda: tmp_path)
    monkeypatch.setenv('HERMES_HOME', str(root))
    import hermes_state
    monkeypatch.setattr(hermes_state, 'DEFAULT_DB_PATH', hermes_state._IMPORT_DEFAULT_DB_PATH)
    config = GatewayConfig()
    config._runtime_profile_homes = (('default', root),)
    runner = GatewayRunner(config)
    process_ownership.reserve([root])
    yield runner, root
    for served_authority in list(getattr(runner, 'session_authorities', []) or []):
        served_authority.db.close()
    process_ownership.release(root)


async def _start(runner, root):
    await initialize_gateway_runtime(runner)
    register_transport_home(runner, None, root)
    adapter = _Adapter()
    runner.adapters = {Platform.TELEGRAM: adapter}
    runner._is_user_authorized_for_source = lambda *a, **k: True
    from gateway.session_authorities import active_authority
    return runner, active_authority(runner), adapter


def _chat(adapter):
    source = SessionSource(platform=Platform.TELEGRAM, chat_id='4242', chat_type='dm', user_id='u1',
                           user_name='probe')
    source._transport_adapter_ref = weakref.ref(adapter)
    return source


def _command(runner, source, text):
    event = MessageEvent(text=text, message_type=MessageType.TEXT, source=source, message_id='m-' + text)
    handler = runner._handle_reset_command if text.startswith('/new') else runner._handle_resume_command
    return handler(event)


class _HoldPredecessorEnd:
    """Parks the route transition at the SQL that ends the route's previous row: the route already
    names the new conversation, and nothing of it is in state.db yet."""

    def __init__(self, db):
        self.conn, self.reached, self.release = db._conn, threading.Event(), threading.Event()

    def _trace(self, sql):
        if 'SET ended_at' in sql and not self.reached.is_set():
            self.reached.set()
            self.release.wait(30)

    def __enter__(self):
        self.conn.set_trace_callback(self._trace)
        return self

    def __exit__(self, *exc):
        self.release.set()
        self.conn.set_trace_callback(None)


@pytest.mark.asyncio
@pytest.mark.parametrize('command', ['/new', '/resume First chat'])
async def test_a_turn_finishing_mid_transition_leaves_the_chat_on_the_conversation_it_named(served, command):
    runner, authority, adapter = await _start(*served)
    store, chat = runner.session_store, _chat(adapter)
    key = runner._session_key_for_source(chat)
    a = authority.register(chat).session_id  # the chat's first message
    authority.db.set_session_title(a, 'First chat')
    if command.startswith('/resume'):
        await _command(runner, chat, '/new')
        authority.register(chat)  # the chat talks in its second conversation, then resumes the first
    before = store.peek_session_id(key)

    with _HoldPredecessorEnd(authority.db) as hold:
        transition = asyncio.create_task(_command(runner, chat, command))
        await async_wait_until(hold.reached.is_set, what='the transition reaches state.db')
        named = store.lookup_by_session_key(key)
        assert named.session_id not in (None, before)
        stamp = named.updated_at
        # The Desktop turn on the previous conversation returns: its post-turn bookkeeping resolves
        # the chat's route while the transition is still writing state.db.
        finishing = asyncio.create_task(runner._run_post_turn_hooks(
            agent_result={'final_response': 'the answer on the old conversation'}, source=chat,
            is_internal=False))
        await async_wait_until(
            lambda: store.peek_session_id(key) != named.session_id or named.updated_at != stamp,
            what='the finishing turn resolved the route')
        hold.release.set()
        await transition
        await finishing

    label = {named.session_id: 'the conversation the command named', before: 'the one it left'}.get
    assert label(store.peek_session_id(key), 'a fresh one') == 'the conversation the command named'
    # Where the chat's next message lands.
    assert label(authority.register(chat).session_id, 'a fresh one') == 'the conversation the command named'
    assert authority.db.get_session(before)['end_reason'] is not None
