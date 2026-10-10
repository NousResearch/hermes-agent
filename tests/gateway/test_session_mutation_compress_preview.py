"""``/compress --preview`` is read-only on the canonical mutation path, every surface.

Review of #106742 (F10): the ACP/CLI canonical path forwarded ``--preview`` as ``{'focus': '--preview'}``
and the authority summarized and prepared replacement history. The shared parser every native
surface uses must govern the payload, and a preview must not write a receipt, a summary or a revision.
"""
from types import SimpleNamespace

import pytest

from gateway.session_authority import SessionAuthority
from gateway.session_controls import AuthorityConnection
from gateway.session_local import create_local_session
from hermes_cli.gateway_client import GatewayClientError
from hermes_cli.gateway_mutations import slash_mutation
import hermes_state_runtime as rt


def test_slash_mutation_parses_compress_through_the_shared_parser():
    assert slash_mutation('/compress', '--preview') == ('compress', {'preview': True})
    assert slash_mutation('/compress', 'billing --dry-run') == ('compress', {'focus': 'billing', 'preview': True})
    assert slash_mutation('/compress', 'here 3') == ('compress', {'partial': True, 'keep_last': 3})
    assert slash_mutation('/compress', 'keep context') == ('compress', {'focus': 'keep context'})
    with pytest.raises(GatewayClientError, match='aggressive'):
        slash_mutation('/compress', '--aggressive')


@pytest.mark.asyncio
async def test_preview_mutation_reports_without_summarizing_or_writing(tmp_path, monkeypatch):
    from gateway import run
    from gateway.config import GatewayConfig
    from gateway.session import SessionStore
    from agent.context_compressor import ContextCompressor

    monkeypatch.setattr(run, '_load_gateway_config', dict)
    store = SessionStore(tmp_path / 'sessions', GatewayConfig())
    db = store._db
    epoch = rt.begin_runtime_epoch(db, instance_id='owner')
    summaries = []
    monkeypatch.setattr(ContextCompressor, '__init__', lambda self, *a, **k: None)
    monkeypatch.setattr(ContextCompressor, 'compress', lambda self, messages, **kw: summaries.append(kw) or messages)
    runner = SimpleNamespace(session_store=store, _session_db=db, adapters={}, _draining=False,
                             _evict_cached_agent=lambda route: None,
                             _resolve_session_agent_runtime=lambda **k: ('frozen', {}))
    authority = SessionAuthority(runner, profile_id='owned', instance_id='owner', db=db, epoch=epoch)
    owner = AuthorityConnection(authority, object(), {'user_id': 'human'})
    ref = create_local_session(authority, owner.actor, dict(request_id='preview', source='cli', cwd=str(tmp_path),
                                                              model='frozen', toolsets=[]))
    for i in range(4):
        db.append_message(ref.session_id, 'user', f'question {i}')
        db.append_message(ref.session_id, 'assistant', f'answer {i}')
    before = db.get_session(ref.session_id)
    watermark = authority.sessions[ref.session_id].event_stream.watermark()
    try:
        for raw in ('--preview', 'here 2 --preview'):
            operation, payload = slash_mutation('/compress', raw)
            reply = await owner.dispatch({'id': 1, 'method': 'session.mutate', 'params': {
                'session_id': ref.session_id, 'request_id': 'preview-' + raw, 'expected_revision': before['runtime_revision'],
                'expected_generation': before['runtime_generation'], 'operation': operation, 'payload': payload}})
            assert reply.get('result', {}).get('status') == 'preview', reply
            assert reply['result']['lines'][0].startswith('Preview'), reply
        assert summaries == [], 'a preview reached the summarizer'
        after = db.get_session(ref.session_id)
        assert (after['runtime_revision'], after['runtime_generation']) == (before['runtime_revision'], before['runtime_generation'])
        assert len(db.get_messages(ref.session_id)) == 8
        assert not db.list_meta_prefix('gateway.mutation.v1.'), 'a preview left a mutation receipt'
        assert authority.sessions[ref.session_id].event_stream.since(*watermark)['events'] == []
        refused = await owner.dispatch({'id': 2, 'method': 'session.mutate', 'params': {
            'session_id': ref.session_id, 'request_id': 'aggressive', 'expected_revision': before['runtime_revision'],
            'expected_generation': before['runtime_generation'], 'operation': operation,
            'payload': {'focus': '--aggressive'}}})
        assert refused['error']['message'] == 'unsupported_compress_options', refused
        assert summaries == []
    finally:
        await owner.close()


@pytest.mark.asyncio
async def test_canonical_compress_runs_the_live_agents_pre_compress_memory_hook(tmp_path, monkeypatch):
    """Canonical ``/compress`` gives the session's memory providers the same ``on_pre_compress``
    turn the in-process compressor does, and their insight reaches the summarizer."""
    from gateway import run
    from gateway.config import GatewayConfig
    from gateway.session import SessionStore
    from agent.context_compressor import ContextCompressor

    monkeypatch.setattr(run, '_load_gateway_config', dict)
    store = SessionStore(tmp_path / 'sessions', GatewayConfig())
    db = store._db
    epoch = rt.begin_runtime_epoch(db, instance_id='owner')
    hooked, summarized = [], []

    class Memory:
        def on_pre_compress(self, messages, **kwargs):
            hooked.append([m['content'] for m in messages])
            return 'PROVIDER_INSIGHT'

    def compress(self, messages, **kw):
        summarized.append(kw.get('memory_context'))
        return [{'role': 'user', 'content': 'summary', '_compressed_summary': True}]

    monkeypatch.setattr(ContextCompressor, '__init__', lambda self, *a, **k: None)
    monkeypatch.setattr(ContextCompressor, 'compress', compress)
    live_agent = SimpleNamespace(_memory_manager=Memory())
    runner = SimpleNamespace(session_store=store, _session_db=db, adapters={}, _draining=False,
                             _evict_cached_agent=lambda route: None, _cached_agent_for=lambda route: live_agent,
                             _resolve_session_agent_runtime=lambda **k: ('frozen', {}))
    authority = SessionAuthority(runner, profile_id='owned', instance_id='owner', db=db, epoch=epoch)
    owner = AuthorityConnection(authority, object(), {'user_id': 'human'})
    ref = create_local_session(authority, owner.actor, dict(request_id='hook', source='cli', cwd=str(tmp_path),
                                                              model='frozen', toolsets=[]))
    for i in range(4):
        db.append_message(ref.session_id, 'user', f'question {i}')
        db.append_message(ref.session_id, 'assistant', f'answer {i}')
    before = db.get_session(ref.session_id)
    try:
        await owner.dispatch({'id': 1, 'method': 'session.mutate', 'params': {
            'session_id': ref.session_id, 'request_id': 'compress', 'expected_revision': before['runtime_revision'],
            'expected_generation': before['runtime_generation'], 'operation': 'compress', 'payload': {}}})
        assert hooked and hooked[0][0] == 'question 0' and len(hooked[0]) == 8
        assert summarized and 'PROVIDER_INSIGHT' in summarized[0]
    finally:
        await owner.close()


@pytest.mark.asyncio
async def test_input_is_admitted_while_compress_summarizes_but_not_claimed_until_it_commits(tmp_path, monkeypatch):
    """The summarization runs outside the session mutation lock: a submit on the session being
    compressed is admitted at once (a 30 s client RPC timeout must not report it lost), while the
    FIFO claim still waits for the fenced compress commit."""
    import asyncio
    import threading
    from gateway import run
    from gateway.config import GatewayConfig
    from gateway.session import SessionStore
    from gateway.session_contract import Submission
    from agent.context_compressor import ContextCompressor

    monkeypatch.setattr(run, '_load_gateway_config', dict)
    store = SessionStore(tmp_path / 'sessions', GatewayConfig())
    db = store._db
    epoch = rt.begin_runtime_epoch(db, instance_id='owner')
    summarizing, release = threading.Event(), threading.Event()

    def compress(self, messages, **kw):
        summarizing.set()
        release.wait(10)
        return [{'role': 'user', 'content': 'summary', '_compressed_summary': True}]
    monkeypatch.setattr(ContextCompressor, '__init__', lambda self, *a, **k: None)
    monkeypatch.setattr(ContextCompressor, 'compress', compress)
    runner = SimpleNamespace(session_store=store, _session_db=db, adapters={}, _draining=False,
                             _evict_cached_agent=lambda route: None,
                             _resolve_session_agent_runtime=lambda **k: ('frozen', {}))
    authority = SessionAuthority(runner, profile_id='owned', instance_id='owner', db=db, epoch=epoch)
    monkeypatch.setattr(authority, '_schedule', lambda ref: None)
    owner = AuthorityConnection(authority, object(), {'user_id': 'human'})
    ref = create_local_session(authority, owner.actor, dict(request_id='busy', source='cli', cwd=str(tmp_path),
                                                              model='frozen', toolsets=[]))
    for i in range(4):
        db.append_message(ref.session_id, 'user', f'question {i}')
        db.append_message(ref.session_id, 'assistant', f'answer {i}')
    before = db.get_session(ref.session_id)
    try:
        compressing = asyncio.create_task(owner.dispatch({'id': 1, 'method': 'session.mutate', 'params': {
            'session_id': ref.session_id, 'request_id': 'compress', 'expected_revision': before['runtime_revision'],
            'expected_generation': before['runtime_generation'], 'operation': 'compress', 'payload': {}}}))
        assert await asyncio.to_thread(summarizing.wait, 10)
        receipt = await asyncio.wait_for(authority.submit(owner.actor, Submission(
            'during', ref, {'text': 'typed while compressing'}, 'queue')), 2)
        assert receipt.status == 'queued'
        claim = asyncio.create_task(authority._claim_next(ref, authority.sessions[ref.session_id]))
        await asyncio.sleep(0.2)
        assert not claim.done(), 'a turn was claimed under a compression still preparing its transcript'
        release.set()
        reply = await asyncio.wait_for(compressing, 10)
        assert reply.get('result', {}).get('operation') == 'compress', reply
        row, _ = await asyncio.wait_for(claim, 10)
        assert row['admission_id'] == receipt.admission_id
        assert row['generation'] == reply['result']['execution_generation'] + 1
    finally:
        release.set()
        await owner.close()
