"""The canonical compress pins its viewers' compressing/compacted status frames (#149517).

The TUI refuses a rewind while a compaction runs, keyed on the ``status.update`` kinds the
in-process route already emits; the canonical ``session.mutate`` route must emit the same
kinds, or its viewers never learn a compress is running and a rewind read queues behind the
mutation on their socket until the client deadline drops it.
"""
import json
from types import SimpleNamespace

import pytest

from agent.context_compressor import ContextCompressor
from gateway.session_authority import SessionAuthority
from gateway.session_controls import AuthorityConnection
from gateway.session_local import create_local_session
import hermes_state_runtime as rt


def _world(tmp_path, monkeypatch):
    from gateway import run
    from gateway.config import GatewayConfig
    from gateway.session import SessionStore

    monkeypatch.setattr(run, '_load_gateway_config', dict)
    store = SessionStore(tmp_path / 'sessions', GatewayConfig())
    db = store._db
    epoch = rt.begin_runtime_epoch(db, instance_id='owner')
    runner = SimpleNamespace(session_store=store, _session_db=db, adapters={}, _draining=False,
        _evict_cached_agent=lambda route: None,
        _resolve_session_agent_runtime=lambda **k: ('frozen', {}))
    authority = SessionAuthority(runner, profile_id=str(tmp_path), instance_id='owner', db=db, epoch=epoch)
    owner = AuthorityConnection(authority, object(), {'user_id': 'human'})
    ref = create_local_session(authority, owner.actor, dict(request_id='owner', source='cli', cwd=str(tmp_path),
        model='frozen', toolsets=[]))
    for i in range(4):
        db.append_message(ref.session_id, 'user', f'question {i}')
        db.append_message(ref.session_id, 'assistant', f'answer {i}')
    return authority, owner, ref


def _mutate(owner, ref, before, payload=None, request_id='compress'):
    return owner.dispatch({'id': 1, 'method': 'session.mutate', 'params': {
        'session_id': ref.session_id, 'request_id': request_id, 'operation': 'compress',
        'expected_revision': before['runtime_revision'],
        'expected_generation': before['runtime_generation'],
        'payload': payload or {}}})


def _status_frames(authority, session_id, since):
    frames = authority.sessions[session_id].event_stream.since(*since)['events']
    return [(frame['type'], frame['payload'], frame['session_id'])
            for frame in frames if frame.get('type') == 'status.update']


@pytest.mark.asyncio
async def test_real_compress_pins_compressing_then_clears(tmp_path, monkeypatch):
    calls = []

    def fake_compress(self, messages, **kw):
        calls.append(len(messages))
        return [{**m, '_compressed_summary': 'SUMMARY'} for m in messages]

    monkeypatch.setattr(ContextCompressor, '__init__', lambda self, *a, **k: None)
    monkeypatch.setattr(ContextCompressor, 'compress', fake_compress)

    authority, owner, ref = _world(tmp_path, monkeypatch)
    before = authority.db.get_session(ref.session_id)
    watermark = authority.sessions[ref.session_id].event_stream.watermark()
    try:
        reply = await _mutate(owner, ref, before)
        assert reply.get('result', {}).get('operation') == 'compress', reply
        assert reply['result']['revision'] == before['runtime_revision'] + 1, reply
        assert len(calls) == 1, calls
        statuses = _status_frames(authority, ref.session_id, watermark)
        kinds = [payload['kind'] for _, payload, _ in statuses]
        assert kinds == ['compressing', 'compacted'], statuses
        assert statuses[0][1]['text'].startswith('⠋ compressing 8 messages'), statuses
        assert all(sid == ref.session_id for _, _, sid in statuses)
    finally:
        await owner.close()


@pytest.mark.asyncio
async def test_preview_publishes_no_status_frames(tmp_path, monkeypatch):
    def fake_compress(self, messages, **kw):
        raise AssertionError('a preview must not summarize')

    monkeypatch.setattr(ContextCompressor, '__init__', lambda self, *a, **k: None)
    monkeypatch.setattr(ContextCompressor, 'compress', fake_compress)

    authority, owner, ref = _world(tmp_path, monkeypatch)
    before = authority.db.get_session(ref.session_id)
    watermark = authority.sessions[ref.session_id].event_stream.watermark()
    try:
        reply = await _mutate(owner, ref, before, payload={'focus': '--preview'}, request_id='preview')
        assert reply.get('result', {}).get('status') == 'preview', reply
        assert _status_frames(authority, ref.session_id, watermark) == []
    finally:
        await owner.close()


@pytest.mark.asyncio
async def test_summary_failure_still_clears_the_pin(tmp_path, monkeypatch):
    def failing_compress(self, messages, **kw):
        raise RuntimeError('summary provider down')

    monkeypatch.setattr(ContextCompressor, '__init__', lambda self, *a, **k: None)
    monkeypatch.setattr(ContextCompressor, 'compress', failing_compress)

    authority, owner, ref = _world(tmp_path, monkeypatch)
    before = authority.db.get_session(ref.session_id)
    watermark = authority.sessions[ref.session_id].event_stream.watermark()
    refused = False
    reply = None
    try:
        try:
            reply = await _mutate(owner, ref, before)
            refused = 'error' in reply
        except RuntimeError:
            refused = True  # prepare_compress's summary error is not a RuntimeStoreError, so dispatch re-raises
        assert refused, f'expected a refused mutate, got {reply!r}'
        after = authority.db.get_session(ref.session_id)
        assert after['runtime_revision'] == before['runtime_revision']
        kinds = [payload['kind'] for _, payload, _ in _status_frames(authority, ref.session_id, watermark)]
        assert kinds == ['compressing', 'compacted'], kinds
    finally:
        await owner.close()
