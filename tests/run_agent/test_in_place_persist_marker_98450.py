"""Regression test for #98450: in-place batch compaction commit must stamp
_DB_PERSISTED_MARKER on the committed dicts.

``compress()`` returns marker-swept COPIES (``_strip_persistence_markers``,
#57491 — the sweep protects the ROTATION flush to a child session). The
in-place branch commits those copies durably via
``SessionDB.archive_and_compact()`` and hands the SAME dict instances back
as the live message list. Without a post-commit stamp, the next
``_persist_session`` → ``_flush_messages_to_session_db_unlocked`` walk sees
every compacted row as unpersisted and re-INSERTs the whole post-compaction
transcript — the live set doubles on every compaction (production:
~58K → ~512K tokens in two hours).

The healthy sibling ``ContextCompressor._sync_micro_compact_to_db`` already
stamps after its ``archive_and_compact`` call; both now share
``stamp_db_persisted_markers``.
"""

import os
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest


def _make_agent(session_db, session_id):
    with patch.dict(os.environ, {"OPENROUTER_API_KEY": "test-key"}):
        from run_agent import AIAgent

        agent = AIAgent(
            api_key="test-key",
            base_url="https://openrouter.ai/api/v1",
            model="test/model",
            quiet_mode=True,
            session_db=session_db,
            session_id=session_id,
            skip_context_files=True,
            skip_memory=True,
        )
    agent.compression_in_place = True

    def _fake_compress(messages, current_tokens=None, focus_topic=None, force=False):
        # Mirrors the real compress() contract: marker-swept fresh dicts.
        return [
            {"role": "user", "content": "[CONTEXT COMPACTION] summary of prior turns"},
            {"role": "assistant", "content": "recent reply 1"},
            {"role": "user", "content": "follow-up q"},
            {"role": "assistant", "content": "recent reply 2"},
        ]

    agent.context_compressor.compress = _fake_compress
    agent.context_compressor._last_compress_aborted = False
    agent.context_compressor._last_summary_error = None
    agent.context_compressor.compression_count = 1
    return agent


def _seed(db, sid, n=8):
    db.create_session(sid, "cli", model="test/model")
    for i in range(n):
        db.append_message(
            session_id=sid,
            role="user" if i % 2 == 0 else "assistant",
            content=f"seed msg {i}",
        )


def _row_counts(db, sid):
    total = db._conn.execute(
        "SELECT COUNT(*) FROM messages WHERE session_id = ?", (sid,)
    ).fetchone()[0]
    active = db._conn.execute(
        "SELECT COUNT(*) FROM messages WHERE session_id = ? AND active = 1", (sid,)
    ).fetchone()[0]
    return total, active


class TestInPlaceCommitPersistMarker:
    def test_post_commit_persist_does_not_reinsert_compacted_rows(self):
        """In-place commit → persist walk: row counts stay stable (#98450)."""
        from hermes_state import SessionDB
        from agent.conversation_compression import compress_context
        from agent.context_compressor import _DB_PERSISTED_MARKER

        with tempfile.TemporaryDirectory() as tmp:
            db = SessionDB(db_path=Path(tmp) / "t.db")
            sid = "20260830_120000_marker"
            _seed(db, sid, n=8)
            agent = _make_agent(db, sid)
            agent._session_db_created = True
            agent._last_flushed_db_idx = 8

            messages = [{"role": "user", "content": f"m{i}"} for i in range(8)]
            compressed, _sp = compress_context(
                agent, messages, approx_tokens=100_000, system_message="sys"
            )

            # ── Precondition: the compacted set genuinely went through
            # archive_and_compact — the 8 seeded rows were soft-archived
            # (kept on disk, active=0) and the 4 compacted dicts were
            # inserted as the new active set. Without this the "no
            # duplicates" assertion below would pass vacuously on a path
            # that never committed anything.
            total, active = _row_counts(db, sid)
            assert active == len(compressed) == 4
            assert total == 8 + len(compressed)  # archived seeds + active set
            archived = db._conn.execute(
                "SELECT COUNT(*) FROM messages WHERE session_id = ? AND active = 0",
                (sid,),
            ).fetchone()[0]
            assert archived == 8

            # ── The fix: every committed dict instance carries the marker.
            # These are the exact instances the caller keeps as the live
            # message list, so the stamp must be on THEM (compress() output
            # copies), not on some other collection.
            for msg in compressed:
                assert msg.get(_DB_PERSISTED_MARKER) is True, (
                    f"committed dict missing persistence marker: {msg.get('content')!r}"
                )

            # ── The symptom: run the post-compaction persist walk twice
            # (turn finalize + close safety-net in production). The flush
            # must skip the already-durable compacted rows: no re-INSERT,
            # counts stable, zero duplicate active contents.
            agent._persist_session(compressed, conversation_history=None)
            agent._persist_session(compressed, conversation_history=None)
            total2, active2 = _row_counts(db, sid)
            dup_groups = db._conn.execute(
                "SELECT content, COUNT(*) c FROM messages "
                "WHERE session_id = ? AND active = 1 "
                "GROUP BY content HAVING c > 1",
                (sid,),
            ).fetchall()
            assert (total2, active2) == (total, active), (
                f"post-commit persist re-INSERTed compacted rows: "
                f"total {total}->{total2}, active {active}->{active2}"
            )
            assert dup_groups == [], f"duplicate active rows: {dup_groups}"

    def test_stamp_helper_shared_by_micro_compact_sync(self):
        """The micro-compaction sync fulfils the same post-commit contract
        via the shared helper (class-of-bug guard, not a change detector:
        asserts the behavioral outcome — dicts stamped after a successful
        archive_and_compact — for the sibling call path)."""
        from hermes_state import SessionDB
        from agent.context_compressor import (
            ContextCompressor,
            _DB_PERSISTED_MARKER,
        )

        with tempfile.TemporaryDirectory() as tmp:
            db = SessionDB(db_path=Path(tmp) / "m.db")
            sid = "20260830_120001_micro0"
            _seed(db, sid, n=4)
            compressor = ContextCompressor.__new__(ContextCompressor)
            compressor._session_db = db
            compressor._session_id = sid
            compacted = [
                {"role": "user", "content": "u"},
                {"role": "assistant", "content": "micro summary"},
            ]
            compressor._sync_micro_compact_to_db(compacted)
            # Precondition: the commit really happened.
            total, active = _row_counts(db, sid)
            assert active == 2 and total == 6
            for msg in compacted:
                assert msg.get(_DB_PERSISTED_MARKER) is True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
