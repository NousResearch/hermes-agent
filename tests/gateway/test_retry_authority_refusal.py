"""Messaging /retry under a session authority answers the authority's admission refusals BEFORE
it rewrites the transcript (#139257).

The re-sent turn is admitted by the routed profile's authority (``admit_native``). A draining or
retiring runtime refuses it with ``runtime_draining`` and a full FIFO with ``queue_full``; both
landed after ``rewrite_transcript`` had already dropped the last exchange, so every refused /retry
lost a turn (and the next /retry ate the one before it). Real authority, real SQLite; only the
turn handler is replaced by a recorder.
"""
import weakref
from pathlib import Path

import pytest

from agent.i18n import t
from gateway.config import GatewayConfig, Platform
from gateway.platforms.event import MessageEvent, MessageType
from gateway.run import GatewayRunner
from gateway.run_runtime import initialize_gateway_runtime
from gateway.runtime_ownership import process_ownership
from gateway.session import SessionSource
from gateway.session_ingress_context import native_callback, register_transport_home


class _FakeAdapter:
    pass


@pytest.fixture
def served(tmp_path, monkeypatch):
    root = tmp_path / '.hermes'
    root.mkdir(mode=0o700)
    monkeypatch.setattr(Path, 'home', lambda: tmp_path)
    monkeypatch.setenv('HERMES_HOME', str(root))
    import hermes_state
    monkeypatch.setattr(hermes_state, 'DEFAULT_DB_PATH', hermes_state._IMPORT_DEFAULT_DB_PATH)
    homes = [('default', root)]
    config = GatewayConfig()
    config._runtime_profile_homes = tuple(homes)
    runner = GatewayRunner(config)
    process_ownership.reserve([home for _, home in homes])
    yield runner, root
    for authority in list(getattr(runner, 'session_authorities', []) or []):
        authority.db.close()
    for _, home in homes:
        process_ownership.release(home)


async def _exchange(runner, root):
    await initialize_gateway_runtime(runner)
    register_transport_home(runner, None, root)
    adapter = _FakeAdapter()
    runner.adapters = {Platform.DISCORD: adapter}
    runner._is_user_authorized_for_source = lambda *a, **k: True
    source = SessionSource(platform=Platform.DISCORD, chat_id='c1', chat_type='dm',
                           user_id='u1', user_name='probe')
    source._transport_adapter_ref = weakref.ref(adapter)
    entry = await runner.async_session_store.get_or_create_session(source)
    runner.session_store.append_to_transcript(entry.session_id, {'role': 'user', 'content': 'retry me'})
    runner.session_store.append_to_transcript(entry.session_id, {'role': 'assistant', 'content': 'old answer'})
    sent = []

    async def recorder(event):
        sent.append(event.text)
        return 'resent'
    runner._handle_message = recorder
    return source, entry.session_id, sent


async def _retry(runner, root, source):
    event = MessageEvent(text='/retry', message_type=MessageType.TEXT, source=source, message_id='m-retry')
    with native_callback(runner, event, root, None):
        return await runner._handle_retry_command(event)


def _contents(runner, sid):
    return [m.get('content') for m in runner.session_store.load_transcript(sid)]


@pytest.mark.asyncio
async def test_retry_into_a_draining_authority_keeps_the_exchange(served):
    runner, root = served
    source, sid, sent = await _exchange(runner, root)
    runner._draining = True

    reply = await _retry(runner, root, source)

    assert sent == [], 'the re-send ran into a draining authority'
    assert reply and '⏳' in reply
    assert _contents(runner, sid) == ['retry me', 'old answer']


@pytest.mark.asyncio
async def test_retry_into_a_full_fifo_keeps_the_exchange(served, monkeypatch):
    runner, root = served
    source, sid, sent = await _exchange(runner, root)
    import hermes_state_runtime
    monkeypatch.setattr(hermes_state_runtime, 'ADMISSION_QUEUE_MAX', 0)

    reply = await _retry(runner, root, source)

    assert sent == []
    assert reply and 'waiting' in reply
    assert _contents(runner, sid) == ['retry me', 'old answer']

async def _admitting_exchange(runner, root, rows=None):
    """Like ``_exchange`` but the real ``_handle_message`` runs, so the re-send reaches the
    authority's ``admit_native``; only the drain is held so the admitted row stays queued."""
    await initialize_gateway_runtime(runner)
    register_transport_home(runner, None, root)
    adapter = _FakeAdapter()
    runner.adapters = {Platform.DISCORD: adapter}
    runner._is_user_authorized_for_source = lambda *a, **k: True
    source = SessionSource(platform=Platform.DISCORD, chat_id='c1', chat_type='dm',
                           user_id='u1', user_name='probe')
    source._transport_adapter_ref = weakref.ref(adapter)
    entry = await runner.async_session_store.get_or_create_session(source)
    for row in rows or ({'role': 'user', 'content': 'retry me'}, {'role': 'assistant', 'content': 'old answer'}):
        runner.session_store.append_to_transcript(entry.session_id, row)
    from gateway.session_authorities import active_authority
    authority = active_authority(runner)
    authority._schedule = lambda ref: None
    return source, entry.session_id, authority


def _queued_texts(authority, sid):
    import hermes_state_runtime
    rows = hermes_state_runtime.list_session_admissions(authority.db, session_id=authority.logical_owner(sid))
    return [row['payload']['text'] for row in rows if row['status'] == 'queued']


@pytest.mark.asyncio
async def test_a_drain_that_begins_after_the_precheck_leaves_the_exchange(served, monkeypatch):
    """MSG-11's race (#139257 step 2): a drain that engages after the pre-check refuses the re-send
    inside its admission write, and the rewrite in that same write rolls back with it."""
    runner, root = served
    source, sid, authority = await _admitting_exchange(runner, root)
    original = type(runner)._retry_readmission_reply

    def precheck_then_drain(self, *args, **kwargs):
        reply = original(self, *args, **kwargs)
        self._draining = True  # the drain begins once the pre-check has passed
        return reply
    monkeypatch.setattr(type(runner), '_retry_readmission_reply', precheck_then_drain)

    reply = await _retry(runner, root, source)

    assert reply and '⏳' in reply
    assert _contents(runner, sid) == ['retry me', 'old answer']
    assert _queued_texts(authority, sid) == []


@pytest.mark.asyncio
async def test_a_turn_committed_after_the_read_refuses_the_retry_unchanged(served, monkeypatch):
    """The rewrite is fenced on the transcript it was computed from: a row committed between the
    read and the admission refuses both, so a newer exchange is never erased."""
    runner, root = served
    source, sid, authority = await _admitting_exchange(runner, root)
    original = type(runner)._retry_readmission_reply

    def precheck_then_append(self, *args, **kwargs):
        self.session_store.append_to_transcript(sid, {'role': 'user', 'content': 'newer ask'})
        return original(self, *args, **kwargs)
    monkeypatch.setattr(type(runner), '_retry_readmission_reply', precheck_then_append)

    reply = await _retry(runner, root, source)

    assert reply == t('gateway.retry.failed_unchanged')
    assert _contents(runner, sid) == ['retry me', 'old answer', 'newer ask']
    assert _queued_texts(authority, sid) == []


@pytest.mark.asyncio
async def test_an_admitted_retry_rewrites_in_its_own_admission(served):
    """A successful /retry still replaces the last exchange exactly once, in its admission."""
    runner, root = served
    source, sid, authority = await _admitting_exchange(runner, root, rows=(
        {'role': 'user', 'content': 'first'}, {'role': 'assistant', 'content': 'kept'},
        {'role': 'user', 'content': 'retry me'}, {'role': 'assistant', 'content': 'old answer'}))
    import asyncio
    task = asyncio.ensure_future(_retry(runner, root, source))
    for _ in range(200):
        if _queued_texts(authority, sid):
            break
        await asyncio.sleep(0.01)
    assert _queued_texts(authority, sid) == ['retry me']
    assert _contents(runner, sid) == ['first', 'kept']
    task.cancel()


def _carrier(ask='REAL ASK'):
    from agent.context_compressor import HISTORICAL_TASK_HEADING, SUMMARY_PREFIX, _SUMMARY_END_MARKER
    return {'role': 'user',
            'content': f'{SUMMARY_PREFIX}\n{HISTORICAL_TASK_HEADING}\nold task\n\n{_SUMMARY_END_MARKER}\n\n{ask}'}


@pytest.mark.asyncio
async def test_a_refused_carrier_retry_keeps_the_carrier(served, monkeypatch):
    """The compaction-carrier rewind takes the same path: refused in the write, nothing archived."""
    runner, root = served
    source, sid, authority = await _admitting_exchange(
        runner, root, rows=(_carrier(), {'role': 'assistant', 'content': 'failed answer'}))
    original = type(runner)._retry_readmission_reply

    def precheck_then_drain(self, *args, **kwargs):
        reply = original(self, *args, **kwargs)
        self._draining = True
        return reply
    monkeypatch.setattr(type(runner), '_retry_readmission_reply', precheck_then_drain)

    reply = await _retry(runner, root, source)

    assert reply and '⏳' in reply
    assert _contents(runner, sid) == [_carrier()['content'], 'failed answer']
    assert _queued_texts(authority, sid) == []


@pytest.mark.asyncio
async def test_an_admitted_carrier_retry_keeps_the_scaffold_and_resends_the_ask(served):
    runner, root = served
    source, sid, authority = await _admitting_exchange(
        runner, root, rows=(_carrier(), {'role': 'assistant', 'content': 'failed answer'}))
    import asyncio
    task = asyncio.ensure_future(_retry(runner, root, source))
    for _ in range(200):
        if _queued_texts(authority, sid):
            break
        await asyncio.sleep(0.01)
    assert _queued_texts(authority, sid) == ['REAL ASK']
    active = runner.session_store.load_transcript(sid)
    assert len(active) == 1 and active[0]['display_kind'] == 'hidden' and 'REAL ASK' not in active[0]['content']
    task.cancel()
