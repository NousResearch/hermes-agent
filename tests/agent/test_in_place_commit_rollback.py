"""Regression test for #99477: a FAILED in-place compaction commit must roll
the live transcript back to its pre-compression snapshot.

``compress()`` returns marker-swept copies (``_strip_persistence_markers``,
#57491) and only the post-commit ``stamp_db_persisted_markers`` (#98450) makes
them skippable by the append-only flush.  ``archive_and_compact()`` is atomic,
so when it raises — a concurrent queued-follow-up drain holding the write path,
a lost compression lease, ``database is locked`` — NOTHING was archived and
NOTHING was inserted: every pre-compaction row is still ``active = 1``.

Before this fix the rotation branch rolled the live list back but the in-place
branch (the DEFAULT, ``compression_in_place`` defaults to True) did not, so the
caller adopted the uncommitted, unstamped compacted list.  The next
``_persist_session`` walk then INSERTed the whole compacted transcript ON TOP of
the rows it was supposed to replace: the active set ended up holding the summary
AND the turns it summarized, the next resume reloaded both, the token count went
UP, preflight fired again, and every failed attempt appended another copy of the
protected head + tail (#99477: ~15 real turns stored as 3,814 rows, the first
user message repeated 893 times).
"""

import contextlib
import os
import tempfile
from pathlib import Path
from unittest.mock import patch


@contextlib.contextmanager
def _session_db(name):
    """Real SessionDB on a temp file, closed before the directory is removed.

    Windows holds the SQLite file open until the connection is closed, so
    letting TemporaryDirectory clean up first raises WinError 32 and masks the
    assertion result.
    """
    from hermes_state import SessionDB

    with tempfile.TemporaryDirectory() as tmp:
        db = SessionDB(db_path=Path(tmp) / name)
        try:
            yield db
        finally:
            db.close()


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
    agent._session_db_created = True

    def _fake_compress(messages, current_tokens=None, focus_topic=None, force=False):
        # Mirrors the real compress() contract: marker-swept fresh dicts.
        return [
            {"role": "user", "content": "[CONTEXT COMPACTION] summary of prior turns"},
            {"role": "assistant", "content": "kept reply 1"},
            {"role": "user", "content": "kept question"},
            {"role": "assistant", "content": "kept reply 2"},
        ]

    agent.context_compressor.compress = _fake_compress
    agent.context_compressor._last_compress_aborted = False
    agent.context_compressor._last_summary_error = None
    agent.context_compressor.compression_count = 1
    return agent


def _seed(db, sid, n=8):
    db.create_session(sid, "gateway", model="test/model")
    for i in range(n):
        db.append_message(
            session_id=sid,
            role="user" if i % 2 == 0 else "assistant",
            content=f"seed msg {i}",
        )


def _counts(db, sid):
    total = db._conn.execute(
        "SELECT COUNT(*) FROM messages WHERE session_id = ?", (sid,)
    ).fetchone()[0]
    active = db._conn.execute(
        "SELECT COUNT(*) FROM messages WHERE session_id = ? AND active = 1", (sid,)
    ).fetchone()[0]
    return total, active


class TestInPlaceCommitFailureRollback:
    def test_failed_commit_does_not_reinsert_the_transcript(self):
        """archive_and_compact raises → live list must return to the snapshot."""
        from hermes_state import SessionDB
        from agent.context_compressor import _DB_PERSISTED_MARKER
        from agent.conversation_compression import compress_context

        with _session_db("rollback.db") as db:
            sid = "20260831_120000_rollback"
            _seed(db, sid, n=8)
            agent = _make_agent(db, sid)

            messages = [
                {"role": "user" if i % 2 == 0 else "assistant", "content": f"m{i}"}
                for i in range(8)
            ]
            agent._flush_messages_to_session_db(messages)
            assert _counts(db, sid) == (16, 16)  # 8 seeded + 8 flushed

            # The realistic trigger: a concurrent queued-follow-up drain holds
            # the write path, so the atomic commit raises instead of landing.
            # Raised from the DB method itself, so the compression path's own
            # error handling runs — nothing is stubbed out around it.
            def _locked(*_a, **_kw):
                raise RuntimeError("database is locked (concurrent drain)")

            with patch.object(SessionDB, "archive_and_compact", _locked):
                compressed, _prompt = compress_context(
                    agent, messages, approx_tokens=900_000, system_message="sys"
                )

            # ── Precondition: the commit genuinely did not land. Without this
            # the assertions below would pass vacuously on a path that
            # actually compacted.
            assert _counts(db, sid) == (16, 16)

            # ── The fix: the returned transcript is the pre-compression
            # snapshot, not the uncommitted compacted set.
            assert [m["content"] for m in compressed] == [
                f"m{i}" for i in range(8)
            ], "failed in-place commit must not hand back the compacted set"
            assert all(
                m.get(_DB_PERSISTED_MARKER) for m in compressed
            ), "restored rows must keep their persistence marker"

            # ── The symptom: the post-compression persist walk must not
            # re-INSERT anything. Run it twice — multiple exit paths persist.
            agent._flush_messages_to_session_db(compressed)
            agent._flush_messages_to_session_db(compressed)
            assert _counts(db, sid) == (16, 16), (
                "a failed in-place compaction re-inserted the transcript"
            )

    def test_successful_commit_still_compacts_in_place(self):
        """The rollback must not fire when the commit landed (#98450 guard)."""
        from hermes_state import SessionDB
        from agent.context_compressor import _DB_PERSISTED_MARKER
        from agent.conversation_compression import compress_context

        with _session_db("committed.db") as db:
            sid = "20260831_120001_committed"
            _seed(db, sid, n=8)
            agent = _make_agent(db, sid)
            agent._last_flushed_db_idx = 8

            messages = [{"role": "user", "content": f"m{i}"} for i in range(8)]
            compressed, _prompt = compress_context(
                agent, messages, approx_tokens=900_000, system_message="sys"
            )

            total, active = _counts(db, sid)
            assert active == len(compressed) == 4, "in-place commit must compact"
            assert total == 12  # 8 soft-archived + 4 active
            assert all(m.get(_DB_PERSISTED_MARKER) for m in compressed)

            agent._flush_messages_to_session_db(compressed)
            assert _counts(db, sid) == (12, 4)
