"""Compression rotation hardening — state-loss fixes at the compaction boundary.

When auto-compression rotates ``agent.session_id`` to a continuation child,
three pieces of state used to be lost or corrupted:

  * #33618 — a persistent ``/goal`` did not follow the rotation (``load_goal``
    is a flat per-session lookup with no lineage walk), so it silently died.
  * #33906/#33907 — if the child ``create_session`` raised, the outer handler
    only warned and let the agent continue on the NEW (un-indexed) id,
    producing an orphan session missing from state.db.
  * #27633 — the compaction-boundary ``on_session_start`` notification omitted
    the ``platform`` kwarg, so context-engine plugins saw ``source=unknown``
    for every message after the boundary.

These tests drive the real ``compress_context`` path against a real SessionDB.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from unittest.mock import MagicMock, patch

from hermes_state import SessionDB


_COMPRESSION_LINEAGE_PREFIX = "runtime_event=compression_lineage "


def _compression_lineage_events(caplog):
    events = []
    for record in caplog.records:
        message = record.getMessage()
        if _COMPRESSION_LINEAGE_PREFIX not in message:
            continue
        payload = message.split(_COMPRESSION_LINEAGE_PREFIX, 1)[1]
        events.append(json.loads(payload))
    return events


def _build_agent_with_db(db: SessionDB, session_id: str, platform: str = "telegram"):
    with patch.dict(os.environ, {"OPENROUTER_API_KEY": "test-key"}):
        from run_agent import AIAgent

        agent = AIAgent(
            api_key="test-key",
            base_url="https://openrouter.ai/api/v1",
            model="test/model",
            platform=platform,
            quiet_mode=True,
            session_db=db,
            session_id=session_id,
            skip_context_files=True,
            skip_memory=True,
        )

    compressor = MagicMock()
    compressor.compress.return_value = [
        {"role": "user", "content": "[CONTEXT COMPACTION] summary"},
        {"role": "user", "content": "tail"},
    ]
    compressor.compression_count = 1
    compressor.last_prompt_tokens = 0
    compressor.last_completion_tokens = 0
    compressor._last_summary_error = None
    compressor._last_compress_aborted = False
    compressor._last_summary_auth_failure = False
    compressor._last_aux_model_failure_model = None
    compressor._last_aux_model_failure_error = None
    agent.context_compressor = compressor
    # ROTATION fallback path — pin in_place=False so these keep covering fork
    # rotation regardless of the global default (flipped to True in #38763).
    agent.compression_in_place = False
    return agent


def _msgs(n=20):
    return [{"role": "user", "content": f"m{i}"} for i in range(n)]


class TestGoalMigratesOnRotation:
    def test_goal_follows_compression_rotation(self, tmp_path: Path):
        db = SessionDB(db_path=tmp_path / "state.db")
        parent = "PARENT_GOAL_ROT"
        db.create_session(parent, source="cli")
        agent = _build_agent_with_db(db, parent)

        # Set a persistent goal on the parent via the real persistence path.
        with patch.dict(os.environ, {"HERMES_HOME": str(tmp_path / ".hermes")}):
            (tmp_path / ".hermes").mkdir(exist_ok=True)
            import hermes_cli.goals as goals
            goals._DB_CACHE.clear()
            # Point the goal DB at the same state.db the agent uses.
            with patch.object(goals, "_get_session_db", return_value=db):
                goals.save_goal(parent, goals.GoalState(goal="finish the migration"))

                agent._compress_context(_msgs(), "sys", approx_tokens=120_000)
                child = agent.session_id
                assert child != parent  # rotation happened

                migrated = goals.load_goal(child)
                assert migrated is not None
                assert migrated.goal == "finish the migration"
            goals._DB_CACHE.clear()


class TestOrphanRollbackOnCreateFailure:
    def test_rolls_back_to_parent_when_child_create_fails(self, tmp_path: Path):
        db = SessionDB(db_path=tmp_path / "state.db")
        parent = "PARENT_ORPHAN_ROT"
        db.create_session(parent, source="cli")
        agent = _build_agent_with_db(db, parent)

        # Make the CHILD create_session raise, but let the initial parent
        # end_session/reopen work. We patch create_session to blow up.
        real_create = db.create_session

        def _boom(*a, **k):
            raise RuntimeError("FOREIGN KEY constraint failed")

        with patch.object(db, "create_session", side_effect=_boom):
            agent._compress_context(_msgs(), "sys", approx_tokens=120_000)

        # The live id must roll back to the still-indexed parent — NOT a
        # phantom child id that has no row in state.db.
        assert agent.session_id == parent
        assert db.get_session(parent) is not None
        _ = real_create  # silence unused


class TestPlatformForwardedAtBoundary:
    def test_on_session_start_receives_platform(self, tmp_path: Path):
        db = SessionDB(db_path=tmp_path / "state.db")
        parent = "PARENT_PLATFORM_ROT"
        db.create_session(parent, source="telegram")
        agent = _build_agent_with_db(db, parent, platform="telegram")

        agent._compress_context(_msgs(), "sys", approx_tokens=120_000)

        # The boundary notify must forward the platform so context-engine
        # plugins don't fall back to source=unknown (#27633).
        calls = [c for c in agent.context_compressor.on_session_start.call_args_list]
        assert calls, "on_session_start was not called at the boundary"
        kwargs = calls[-1].kwargs
        assert kwargs.get("platform") == "telegram"
        assert kwargs.get("boundary_reason") == "compression"


class TestCompressionLineageRuntimeEvents:
    def test_successful_rotation_logs_redacted_lineage_events(self, tmp_path: Path, caplog):
        db = SessionDB(db_path=tmp_path / "state.db")
        parent = "PARENT_LINEAGE_ROT"
        db.create_session(parent, source="cli")
        agent = _build_agent_with_db(db, parent)
        messages = _msgs()

        caplog.set_level(logging.INFO)
        agent._compress_context(messages, "SECRET_SYSTEM_PROMPT", approx_tokens=120_000)

        child = agent.session_id
        assert child != parent
        events = _compression_lineage_events(caplog)
        by_name = {event["event"]: event for event in events}

        assert by_name["compression_start"] == {
            "event": "compression_start",
            "message_count": len(messages),
            "reason": "compression",
            "session_id": parent,
            "status": "started",
            "token_estimate": 120_000,
        }
        assert by_name["compression_parent_ended"] == {
            "event": "compression_parent_ended",
            "message_count": len(messages),
            "parent_session_id": parent,
            "reason": "compression",
            "session_id": parent,
            "status": "ended",
            "token_estimate": 120_000,
        }
        assert by_name["compression_child_created"]["session_id"] == child
        assert by_name["compression_child_created"]["child_session_id"] == child
        assert by_name["compression_child_created"]["parent_session_id"] == parent
        assert by_name["compression_child_created"]["message_count"] == len(messages)
        assert by_name["compression_child_created"]["compressed_message_count"] == 2
        assert by_name["compression_child_created"]["reason"] == "compression"
        assert by_name["compression_child_created"]["status"] == "created"
        assert by_name["compression_child_created"]["token_estimate"] == 120_000

        allowed = {
            "event",
            "session_id",
            "parent_session_id",
            "child_session_id",
            "reason",
            "message_count",
            "compressed_message_count",
            "token_estimate",
            "status",
        }
        for event in events:
            assert set(event) <= allowed
        event_text = json.dumps(events, sort_keys=True)
        assert "[CONTEXT COMPACTION] summary" not in event_text
        assert "SECRET_SYSTEM_PROMPT" not in event_text
        assert "tail" not in event_text

    def test_aborted_compression_logs_abort_without_rotation(self, tmp_path: Path, caplog):
        db = SessionDB(db_path=tmp_path / "state.db")
        parent = "PARENT_ABORT_ROT"
        db.create_session(parent, source="cli")
        agent = _build_agent_with_db(db, parent)
        messages = _msgs()
        agent.context_compressor.compress.return_value = messages
        agent.context_compressor._last_compress_aborted = True
        agent.context_compressor._last_summary_error = (
            "provider said sk_live_never_log_me at https://private.invalid/path"
        )

        caplog.set_level(logging.INFO)
        returned_messages, _ = agent._compress_context(
            messages,
            "SECRET_SYSTEM_PROMPT",
            approx_tokens=120_000,
        )

        assert returned_messages is messages
        assert agent.session_id == parent
        events = _compression_lineage_events(caplog)
        by_name = {event["event"]: event for event in events}
        assert by_name["compression_abort"] == {
            "event": "compression_abort",
            "message_count": len(messages),
            "reason": "summary_failed",
            "session_id": parent,
            "status": "aborted",
            "token_estimate": 120_000,
        }
        assert "compression_parent_ended" not in by_name
        assert "compression_child_created" not in by_name
        event_text = json.dumps(events, sort_keys=True)
        assert "sk_live_never_log_me" not in event_text
        assert "private.invalid" not in event_text

    def test_resume_tip_resolution_is_quiet_by_default(self, tmp_path: Path, caplog):
        db = SessionDB(db_path=tmp_path / "state.db")
        parent = "PARENT_QUIET_TIP"
        child = "CHILD_QUIET_TIP"
        db.create_session(parent, source="cli")
        db.end_session(parent, "compression")
        db.create_session(child, source="cli", parent_session_id=parent)

        caplog.set_level(logging.INFO)
        assert db.get_compression_tip(parent) == child

        assert _compression_lineage_events(caplog) == []

    def test_resume_tip_resolution_logs_lineage_event_when_requested(self, tmp_path: Path, caplog):
        db = SessionDB(db_path=tmp_path / "state.db")
        parent = "PARENT_RESUME_TIP"
        child = "CHILD_RESUME_TIP"
        db.create_session(parent, source="cli")
        db.end_session(parent, "compression")
        db.create_session(child, source="cli", parent_session_id=parent)

        caplog.set_level(logging.INFO)
        assert db.get_compression_tip(parent, emit_lineage_event=True) == child

        events = _compression_lineage_events(caplog)
        assert events == [
            {
                "event": "compression_resume_tip_resolved",
                "child_session_id": child,
                "parent_session_id": parent,
                "reason": "compression_chain_tip",
                "session_id": child,
                "status": "resolved",
            }
        ]
