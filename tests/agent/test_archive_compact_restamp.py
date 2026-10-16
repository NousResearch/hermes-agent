"""In-place compaction must not resurrect its committed rows on an identity-less flush (#126021).

``compress()`` returns marker-swept copies (#57491) and ``archive_and_compact`` inserts them as the new live
set. If those dicts reached the next flush unstamped, an agent with no flush identity state (incremental
tool-progress persist with no history arg, a second agent instance on a multiplexed gateway) would re-INSERT
the whole compacted set as duplicate ``active=1`` rows with byte-identical content and timestamps.

Ported from Finn763's PR #126048 test to drive the production in-place commit (``compress_context``) instead of
the raw ``archive_and_compact`` call. Two layers hold the invariant: the commit's post-commit
``stamp_db_persisted_markers`` and the row id/digest ``_insert_message_rows`` stamps on every inserted dict.
"""

from __future__ import annotations

import os
from contextlib import closing
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from hermes_state import SessionDB
from run_agent import AIAgent


def _make_agent(db: SessionDB, sid: str) -> AIAgent:
    with patch.dict(os.environ, {"OPENROUTER_API_KEY": "test-key"}):
        agent = AIAgent(api_key="test-key", base_url="https://openrouter.ai/api/v1", model="test/model",
                        quiet_mode=True, session_db=db, session_id=sid, skip_context_files=True, skip_memory=True)
    agent.compression_in_place = True

    def _fake_compress(messages, current_tokens=None, focus_topic=None, force=False):
        # The real compress() contract: marker-swept fresh dicts, explicit timestamps kept.
        return [{"role": "user", "content": "[CONTEXT COMPACTION] summary", "timestamp": 1.5},
                {"role": "assistant", "content": "current answer", "timestamp": 2.5}]

    agent.context_compressor.compress = _fake_compress
    agent.context_compressor._last_compress_aborted = False
    agent.context_compressor._last_summary_error = None
    agent.context_compressor.compression_count = 1
    return agent


def _flush_only_agent(db: SessionDB, sid: str):
    """No flush identity state: the incremental-persist / multiplexed second-agent shape from #126048."""
    agent = SimpleNamespace(
        _session_db=db, _session_db_created=True, _persist_disabled=False, session_id=sid,
        _session_persist_lock=None, _flushed_db_message_ids=set(), _flushed_db_message_session_id=None,
        _last_flushed_db_idx=0, _persist_user_message_idx=None, _persist_user_message_override=None,
        _persist_user_message_timestamp=None, _pending_cli_user_message=None)
    agent._ensure_db_session = lambda: None
    agent._flush_messages_to_session_db = AIAgent._flush_messages_to_session_db.__get__(agent, AIAgent)
    agent._flush_messages_to_session_db_unlocked = (
        AIAgent._flush_messages_to_session_db_unlocked.__get__(agent, AIAgent))
    return agent


def _active_contents(db: SessionDB, sid: str) -> list:
    return [m.get("content") for m in db.get_messages(sid)]


def test_identityless_flush_after_in_place_commit_keeps_active_set_flat(tmp_path: Path) -> None:
    from agent.conversation_compression import compress_context

    sid = "restamp"
    with closing(SessionDB(db_path=tmp_path / "state.db")) as db:
        db.create_session(sid, "cli", model="test/model")
        for i in range(4):
            db.append_message(sid, "user" if i % 2 == 0 else "assistant", f"seed {i}")
        agent = _make_agent(db, sid)
        agent._session_db_created, agent._last_flushed_db_idx = True, 4
        live = db.get_messages_as_conversation(sid)

        compressed, _ = compress_context(agent, live, approx_tokens=100_000, system_message="sys")
        committed = _active_contents(db, sid)
        assert committed == ["[CONTEXT COMPACTION] summary", "current answer"]

        flusher = _flush_only_agent(db, sid)
        flusher._flush_messages_to_session_db(compressed)
        assert _active_contents(db, sid) == committed

        # Guard against over-skipping: a genuinely new tail still persists exactly once.
        compressed += [{"role": "user", "content": "next question"}, {"role": "assistant", "content": "next answer"}]
        flusher._flush_messages_to_session_db(compressed)
        assert _active_contents(db, sid) == committed + ["next question", "next answer"]
