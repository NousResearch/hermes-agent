"""Audit trail for compaction: `compaction_events` start/end bracketing.

Covers #104099. Every attempt that reaches the summary phase writes a ``start``
row on entry and an ``end`` row via the telemetry emitter; pre-lease sit-outs
(lock/breaker/cooldown/codex) stay log-only so they never leave an orphan.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from agent.conversation_compression import compress_context, _emit_compression_attempt_telemetry
from hermes_state import SessionDB


def _agent(tmp_path: Path, session_id: str, db: SessionDB | None = None):
    if db is None:
        db = SessionDB(db_path=tmp_path / "state.db")
        db.create_session(session_id, source="cli")
    with patch.dict(os.environ, {"OPENROUTER_API_KEY": "test-key"}):
        from run_agent import AIAgent

        agent = AIAgent(
            api_key="test-key",
            base_url="https://openrouter.ai/api/v1",
            model="test/model",
            quiet_mode=True,
            session_db=db,
            session_id=session_id,
            skip_context_files=True,
            skip_memory=True,
        )
    agent._compression_feasibility_checked = True
    agent.compression_in_place = True
    agent._cached_system_prompt = "sys"
    agent.context_compressor.threshold_tokens = 1000
    return db, agent


def _msgs(n=20):
    return [{"role": "user", "content": f"m{i} " + "x" * 200} for i in range(n)]


def _compress_small(messages, **kwargs):
    # Marker-swept shape compress() normally returns: short head+tail into one summary.
    return [{"role": "user", "content": "summary of earlier turns"}] + messages[-2:]


class TestSchema:
    def test_compaction_events_table_exists(self, tmp_path):
        db, _ = _agent(tmp_path, "s-schema")
        rows = db._read_all("SELECT name FROM sqlite_master WHERE type='table' AND name='compaction_events'")
        assert len(rows) == 1


class TestPairing:
    def test_committed_path_writes_start_and_end_and_no_orphan(self, tmp_path):
        db, agent = _agent(tmp_path, "s-pair")
        msgs = _msgs(20)
        with patch.object(agent.context_compressor, "compress", side_effect=_compress_small):
            out, _ = compress_context(agent, msgs, "sys", approx_tokens=50000, force=True)
            assert len(out) < len(msgs)
        starts = db._read_all("SELECT kind FROM compaction_events WHERE session_id=? ORDER BY id", ("s-pair",))
        kinds = [r[0] for r in starts]
        assert kinds.count("start") == 1
        assert kinds.count("end") == 1
        assert db.find_orphaned_compaction_starts("s-pair") == []
        assert db.find_orphaned_compaction_starts() == []

    def test_wiring_bite_emitter_persists_end(self, tmp_path):
        """Mutation-check: deleting the recorder call in the emitter must break pairing."""
        db, agent = _agent(tmp_path, "s-wiring")
        msgs = _msgs(20)
        with patch.object(agent.context_compressor, "compress", side_effect=_compress_small):
            # Normal pairing first.
            compress_context(agent, msgs, "sys", approx_tokens=50000, force=True)
        assert len(db._read_all("SELECT 1 FROM compaction_events WHERE kind='end' AND session_id='s-wiring'")) == 1
        # Sabotage the emitter's DB write (but let the start row through) -> orphan must appear.
        db2, agent2 = _agent(tmp_path, "s-wiring2")
        import agent.conversation_compression as cc

        with patch.object(cc, "_record_compaction_end"):
            with patch.object(agent2.context_compressor, "compress", side_effect=_compress_small):
                compress_context(agent2, _msgs(20), "sys", approx_tokens=50000, force=True)
            assert db2.find_orphaned_compaction_starts("s-wiring2") != []


class TestOrphans:
    def test_manual_start_is_orphan_until_end(self, tmp_path):
        db, _ = _agent(tmp_path, "s-orphan")
        assert db.find_orphaned_compaction_starts("s-orphan") == []
        db.record_compaction_event("s-orphan", "a1", "start", {"message_count": 3})
        orphans = db.find_orphaned_compaction_starts("s-orphan")
        assert len(orphans) == 1 and orphans[0]["attempt_id"] == "a1"
        # global scan sees it too
        assert any(r["attempt_id"] == "a1" for r in db.find_orphaned_compaction_starts())
        # scoping filters
        assert db.find_orphaned_compaction_starts("other") == []
        db.record_compaction_event("s-orphan", "a1", "end", {"commit_status": "committed"})
        assert db.find_orphaned_compaction_starts("s-orphan") == []

    def test_end_without_start_tolerated(self, tmp_path):
        db, agent = _agent(tmp_path, "s-no-start")
        # Emitter without a prior start must not create a row (silent no-op).
        agent._compression_attempt_id = "late123"
        agent.context_compressor._last_compression_telemetry = {
            "attempt_id": "late123",
            "session_id": "s-no-start",
            "trigger_source": "auto",
        }
        _emit_compression_attempt_telemetry(agent, started_at=time.monotonic(), commit_status="aborted", split_status="aborted", failure_class="no_progress")
        assert db._read_all("SELECT 1 FROM compaction_events WHERE session_id='s-no-start'") == []
        assert db.find_orphaned_compaction_starts("s-no-start") == []


class TestNoFalseOrphansOnEarlyExits:
    """Post-lease exits must never leave a false orphan (the #104122 defect)."""

    def test_breaker_under_lock_no_orphan(self, tmp_path):
        db, agent = _agent(tmp_path, "s-breaker")
        with patch("agent.conversation_compression._automatic_compression_gate_blocks", side_effect=[False, True]):
            compress_context(agent, _msgs(12), "sys", approx_tokens=99999, force=False)
        assert db.find_orphaned_compaction_starts("s-breaker") == []

    def test_cooldown_read_failed_no_orphan(self, tmp_path):
        db, agent = _agent(tmp_path, "s-cooldown")
        with patch("agent.conversation_compression._capture_authoritative_cooldown_under_lease", return_value=(False, {})):
            compress_context(agent, _msgs(12), "sys", approx_tokens=99999, force=False)
        assert db.find_orphaned_compaction_starts("s-cooldown") == []

    def test_parent_rotated_no_orphan(self, tmp_path):
        db, agent = _agent(tmp_path, "s-rot")
        with patch("agent.conversation_compression._session_was_rotated_by_compression", return_value=True):
            with patch("agent.conversation_compression._adopt_live_compression_child", return_value=None):
                compress_context(agent, _msgs(12), "sys", approx_tokens=99999, force=False)
        assert db.find_orphaned_compaction_starts("s-rot") == []

    def test_empty_transcript_no_orphan(self, tmp_path):
        db, agent = _agent(tmp_path, "s-empty")
        def _empty(messages, **kw):  # triggers _candidate_rejected empty branch (now emitting)
            return []
        with patch.object(agent.context_compressor, "compress", side_effect=_empty):
            compress_context(agent, _msgs(12), "sys", approx_tokens=99999, force=True)
        rows = db._read_all("SELECT kind FROM compaction_events WHERE session_id='s-empty' ORDER BY id")
        kinds = [r[0] for r in rows]
        # start + end (empty_transcript) paired
        assert kinds.count("start") == 1 and kinds.count("end") == 1
        assert db.find_orphaned_compaction_starts("s-empty") == []

    def test_no_progress_pairs(self, tmp_path):
        # A fence-cancelled dispatch returns the transcript unchanged; the
        # no_progress emit must still close the bracket (review gate #2 rebuttal).
        db, agent = _agent(tmp_path, "s-noprog")
        def _same(messages, **kw):
            return list(messages)
        with patch.object(agent.context_compressor, "compress", side_effect=_same):
            compress_context(agent, _msgs(12), "sys", approx_tokens=99999, force=True)
        rows = db._read_all("SELECT kind FROM compaction_events WHERE session_id='s-noprog' ORDER BY id")
        kinds = [r[0] for r in rows]
        assert kinds.count("start") == 1 and kinds.count("end") == 1
        end_row = db._read_one(
            "SELECT payload_json FROM compaction_events WHERE session_id='s-noprog' AND kind='end'")
        assert json.loads(end_row[0])["failure_class"] == "no_progress"
        assert db.find_orphaned_compaction_starts("s-noprog") == []


class TestPayloadAndGuards:
    def test_payload_bounding(self, tmp_path):
        db, _ = _agent(tmp_path, "s-cap")
        huge = {"attempt_id": "x", "session_id": "s-cap", "blob": "y" * 20000}
        assert db.record_compaction_event("s-cap", "x", "start", huge)
        row = db._read_one("SELECT payload_json FROM compaction_events WHERE session_id='s-cap' AND attempt_id='x'")
        payload = json.loads(row[0])
        assert payload.get("truncated") is True
        assert len(row[0].encode("utf-8")) <= 9000

    def test_exploding_store_never_raises(self, tmp_path):
        db, agent = _agent(tmp_path, "s-boom")
        orig = db._execute_write

        def _boom(fn, patience_s=None):
            raise Exception("boom")

        db._execute_write = _boom  # type: ignore[method-assign]
        # record helper is best-effort
        assert db.record_compaction_event("s-boom", "a1", "start", {}) is False
        assert db.find_orphaned_compaction_starts("s-boom") == []
        # compress must still succeed (audit never blocks)
        db._execute_write = orig  # restore for the real compaction DB writes
        with patch.object(agent.context_compressor, "compress", side_effect=_compress_small):
            out, _ = compress_context(agent, _msgs(12), "sys", approx_tokens=50000, force=True)
            assert len(out) > 0

    def test_no_op_guards(self, tmp_path):
        db, _ = _agent(tmp_path, "s-guard")
        assert db.record_compaction_event("", "a", "start", {}) is False
        assert db.record_compaction_event("s", "", "start", {}) is False
        assert db.record_compaction_event("s", "a", "bad_kind", {}) is False
        # no rows leaked
        assert db._read_all("SELECT 1 FROM compaction_events WHERE session_id='s-guard'") == []

    def test_reader_surface(self, tmp_path):
        db, agent = _agent(tmp_path, "s-read")
        with patch.object(agent.context_compressor, "compress", side_effect=_compress_small):
            compress_context(agent, _msgs(12), "sys", approx_tokens=50000, force=True)
        events = db.get_compaction_events("s-read", limit=10)
        assert any(e["kind"] == "start" for e in events)
        assert any(e["kind"] == "end" for e in events)

    def test_rotation_pairs_under_parent_session(self, tmp_path):
        """Rotation rebinds agent.session_id to the child; the pair must stay under the start sid."""
        db, agent = _agent(tmp_path, "s-rot-pair")
        agent.compression_in_place = False
        with patch.object(agent.context_compressor, "compress", side_effect=_compress_small):
            compress_context(agent, _msgs(20), "sys", approx_tokens=50000, force=True)
        rows = db._read_all(
            "SELECT session_id, kind FROM compaction_events WHERE attempt_id IN "
            "(SELECT attempt_id FROM compaction_events WHERE session_id='s-rot-pair') ORDER BY id"
        )
        kinds = [(r[0], r[1]) for r in rows]
        assert ("s-rot-pair", "start") in kinds
        assert ("s-rot-pair", "end") in kinds
        assert db.find_orphaned_compaction_starts() == []

    def test_getter_handles_missing_db_method(self, tmp_path):
        # Stores without the new method (legacy DB, test double) are a silent no-op, never AttributeError.
        from types import SimpleNamespace

        from agent.conversation_compression import _compaction_audit_db

        assert _compaction_audit_db(SimpleNamespace()) is None
        assert _compaction_audit_db(SimpleNamespace(_session_db=object())) is None

    def test_no_store_bound_stays_log_only(self, tmp_path):
        # No session DB at all: the attempt still compacts, telemetry stays log-only, nothing raises.
        _, agent = _agent(tmp_path, "s-nodb")
        agent._session_db = None
        with patch.object(agent.context_compressor, "compress", side_effect=_compress_small):
            out, _ = compress_context(agent, _msgs(20), "sys", approx_tokens=50000, force=True)
            assert len(out) < 20
