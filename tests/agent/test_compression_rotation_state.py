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

import os
from pathlib import Path
from unittest.mock import MagicMock, patch

from hermes_state import SessionDB


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


class TestTodoSnapshotMergedNotDuplicated:
    """Regression: the post-compression todo snapshot must not produce a
    second standalone user message when the compressed transcript already
    ends with a user message — that yielded consecutive user/user turns
    (a content-ordering violation some providers reject). Instead the
    snapshot is merged into the trailing user content with a blank-line
    separator, so the latest user turn keeps both the original text and
    the preserved task list.
    """

    def test_snapshot_merges_into_trailing_user(self, tmp_path: Path):
        db = SessionDB(db_path=tmp_path / "state.db")
        parent = "PARENT_TODO_MERGE"
        db.create_session(parent, source="cli")
        agent = _build_agent_with_db(db, parent)
        # Use a compressor transcript whose tail is a user message preceded by
        # a NON-user (assistant) message. The pre-existing shared mock ends
        # with two user messages, which would mask whether the snapshot is the
        # source of any consecutive user/user — here the ONLY way a
        # user/user pair can appear is if the snapshot appends a second
        # standalone user message next to the tail.
        agent.context_compressor.compress.return_value = [
            {"role": "assistant", "content": "earlier assistant turn"},
            {"role": "user", "content": "tail"},
        ]
        baseline_len = len(agent.context_compressor.compress.return_value)

        # Populate the todo store the way a real session would: an active
        # (pending) item, so format_for_injection() yields non-empty text.
        agent._todo_store.write(
            [{"id": "1", "content": "ship the migration", "status": "pending"}],
            merge=False,
        )
        snapshot = agent._todo_store.format_for_injection()
        assert snapshot  # sanity: the snapshot is non-empty

        compressed, _ = agent._compress_context(_msgs(), "sys", approx_tokens=120_000)
        child = agent.session_id
        assert child != parent  # rotation happened

        # The RETURNED compressed messages: the snapshot must NOT append a
        # second standalone user message (length unchanged) — it merges into
        # the trailing user content, which then carries both the original
        # tail text and the snapshot. No consecutive user/user is introduced
        # by the snapshot. In rotation mode the gateway caller persists this
        # returned transcript into the continuation session.
        assert len(compressed) == baseline_len, (
            "todo snapshot appended a standalone user message instead of merging"
        )
        last = compressed[-1]
        assert last["role"] == "user"
        assert "tail" in last["content"]
        assert snapshot in last["content"]
        for a, b in zip(compressed, compressed[1:]):
            assert not (a["role"] == "user" and b["role"] == "user"), (
                "consecutive user/user messages in compressed transcript "
                "caused by the todo snapshot"
            )

    def test_snapshot_merge_is_persisted_in_place(self, tmp_path: Path):
        db = SessionDB(db_path=tmp_path / "state.db")
        session = "PARENT_TODO_IN_PLACE"
        db.create_session(session, source="cli")
        agent = _build_agent_with_db(db, session)
        agent.compression_in_place = True
        agent.context_compressor.compress.return_value = [
            {"role": "assistant", "content": "earlier assistant turn"},
            {"role": "user", "content": "tail"},
        ]

        agent._todo_store.write(
            [{"id": "1", "content": "ship the migration", "status": "pending"}],
            merge=False,
        )
        snapshot = agent._todo_store.format_for_injection()
        assert snapshot

        compressed, _ = agent._compress_context(_msgs(), "sys", approx_tokens=120_000)
        assert agent.session_id == session  # in-place compaction kept the id

        # In-place mode writes the compacted transcript directly via
        # archive_and_compact(). The live DB transcript should therefore carry
        # the same merged tail and never persist a user/user pair caused by the
        # todo snapshot.
        live = db.get_messages(session)
        assert [(m["role"], m["content"]) for m in live] == [
            (m["role"], m["content"]) for m in compressed
        ]
        persisted_roles = [m["role"] for m in live]
        for a, b in zip(persisted_roles, persisted_roles[1:]):
            assert not (a == "user" and b == "user"), (
                "consecutive user/user messages persisted for live transcript "
                "caused by the todo snapshot"
            )
        persisted_last = live[-1]
        assert persisted_last["role"] == "user"
        assert "tail" in persisted_last["content"]
        assert snapshot in persisted_last["content"]
