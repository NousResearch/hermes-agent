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
from gateway.client import GatewayClientError
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

    monkeypatch.setattr(run, '_load_gateway_config', lambda: {})
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
