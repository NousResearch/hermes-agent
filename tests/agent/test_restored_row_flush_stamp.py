"""A restored row the store skips is only stamped durable by the flush.

The flush marks a row whose ``message_uid`` it already saw durable in this session as restored; the store keeps its stored
row and inserts nothing. The live dict then carries the persisted marker and nothing from the stored row: no row id and
no snapshot, so a later edit of that dict is not addressed to the stored row.
"""

from __future__ import annotations

import os
from unittest.mock import patch

from agent.context_compressor import _DB_PERSISTED_MARKER
from agent.message_metadata import DB_ROW_SNAPSHOT
from tests.fixtures.storage_read import transcript_rows


def _agent(store):
    with patch.dict(os.environ, {"OPENROUTER_API_KEY": "test-key"}):
        from run_agent import AIAgent

        agent = AIAgent(api_key="test-key", base_url="https://openrouter.ai/api/v1", model="test/model", quiet_mode=True,
                        persistence=store, session_id="restored-stamp-001", skip_context_files=True, skip_memory=True)
    agent._ensure_db_session()
    return agent


def test_a_restored_row_the_store_skips_is_only_stamped_durable(tmp_path):
    from persistence_assembly import open_persistence, profile_scope

    store = open_persistence(profile_scope(home=tmp_path))
    try:
        agent = _agent(store)
        turn = [{"role": "user", "content": "first turn", "timestamp": 1000.0},
                {"role": "assistant", "content": "first reply", "timestamp": 1001.0}]
        agent._flush_messages_to_session_db(turn)
        assert all(isinstance(m.get("_row_id"), int) for m in turn)
        # The copies a restore hands back: the durable uid kept, every in-memory marker stripped.
        restored = [{key: value for key, value in m.items() if not key.startswith("_")} for m in turn]
        assert all(m.get("message_uid") for m in restored)
        agent._flush_messages_to_session_db(restored)
        assert len(transcript_rows(store, agent.session_id)) == 2
        for m in restored:
            assert m.get(_DB_PERSISTED_MARKER) is True
            assert "_row_id" not in m and DB_ROW_SNAPSHOT not in m
    finally:
        store.close()
