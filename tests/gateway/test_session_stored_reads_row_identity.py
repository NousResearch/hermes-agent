"""Regression: ``session.transcript`` publishes a message's durable row id, so a compaction that carries
the tail under new physical rows never makes a client append the same commentary again."""
import pytest

from tests.gateway.test_session_stored_reads import _append, _call, _compact_carrying_tail
from tests.gateway.test_session_stored_reads_bounds import _session


@pytest.mark.asyncio
async def test_carried_tail_keeps_its_row_ids_across_a_compaction(tmp_path, monkeypatch):
    from agent.context_compressor import SUMMARY_PREFIX
    with _session(tmp_path, monkeypatch) as (db, _authority, owner, ref, target):
        for role, text in (('user', 'question'), ('assistant', 'commentary'), ('user', 'follow-up'), ('assistant', 'answer')):
            _append(db, target, role, text)

        async def ids():
            reply = await _call(owner, 'session.transcript', session_id=ref.session_id, limit=20, order='latest')
            return {row['text']: row['row_id'] for row in reply['result']['rows'] if row.get('text')}
        before = await ids()
        _compact_carrying_tail(db, target, f'{SUMMARY_PREFIX}\n## Goal\none', tail_rows=2)
        after = await ids()
        for text in ('follow-up', 'answer'):
            assert after[text] == before[text], (text, before, after)
