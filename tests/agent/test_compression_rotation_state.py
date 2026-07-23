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
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from agent.context_compressor import ContextCompressor
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


def _bound_context_compressor(db: SessionDB, session_id: str) -> ContextCompressor:
    with patch(
        "agent.context_compressor.get_model_context_length",
        return_value=100_000,
    ):
        compressor = ContextCompressor(
            model="test/model",
            threshold_percent=0.85,
            protect_first_n=2,
            protect_last_n=2,
            quiet_mode=True,
        )
    compressor.bind_session_state(db, session_id)
    return compressor


@pytest.fixture
def refresh_state_db(tmp_path: Path):
    db = SessionDB(db_path=tmp_path / "state.db")
    try:
        yield db
    finally:
        db.close()


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


class TestWorkspaceMetadataFollowsRotation:
    def test_child_row_inherits_cwd_repo_and_origin_on_rotation(self, tmp_path: Path):
        """Behavioral #64709/#59527: drive the REAL compression rotation path
        and assert the child session row carries the parent's workspace and
        gateway-origin metadata, so the project sidebar entry and the peer
        routing mapping both survive the compaction boundary."""
        db = SessionDB(db_path=tmp_path / "state.db")
        parent = "PARENT_CWD_ROT"
        db.create_session(
            parent,
            source="telegram",
            user_id="u1",
            session_key="telegram:u1:c1",
            chat_id="c1",
            chat_type="private",
        )
        db.update_session_cwd(
            parent, "/work/repo", git_branch="main", git_repo_root="/work/repo"
        )
        agent = _build_agent_with_db(db, parent, platform="telegram")

        agent._compress_context(_msgs(), "sys", approx_tokens=120_000)
        child = agent.session_id
        assert child != parent  # rotation happened

        row = db.get_session(child)
        assert row is not None
        assert row["parent_session_id"] == parent
        # Workspace metadata (#64709): sidebar grouping keys must survive.
        assert row["cwd"] == "/work/repo"
        assert row["git_repo_root"] == "/work/repo"
        assert row["git_branch"] == "main"
        # Gateway origin metadata (#59527): routing keys must survive even if
        # the gateway never gets to re-record the peer (crash window).
        assert row["session_key"] == "telegram:u1:c1"
        assert row["chat_id"] == "c1"
        assert row["chat_type"] == "private"
        assert row["user_id"] == "u1"


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


class TestFallbackStreakFollowsRotation:
    def test_fallback_boundary_persists_on_child_session(self, tmp_path: Path):
        db = SessionDB(db_path=tmp_path / "state.db")
        parent = "PARENT_FALLBACK_ROT"
        db.create_session(parent, source="telegram")
        with patch(
            "agent.context_compressor.get_model_context_length",
            return_value=100_000,
        ):
            compressor = ContextCompressor(
                model="test/model",
                threshold_percent=0.85,
                protect_first_n=2,
                protect_last_n=2,
                quiet_mode=True,
            )
        compressor.bind_session_state(db, parent)

        # A fallback streak must survive the session-id rotation itself. The
        # boundary then records the just-completed fallback on the child row.
        compressor.record_completed_compaction(used_fallback=True)
        assert db.get_compression_fallback_streak(parent) == 1
        db.create_session(
            "CHILD_FALLBACK_ROT",
            source="telegram",
            parent_session_id=parent,
        )
        compressor.on_session_start(
            "CHILD_FALLBACK_ROT",
            session_db=db,
            boundary_reason="compression",
            old_session_id=parent,
        )
        assert compressor._fallback_compression_streak == 1

        compressor.record_completed_compaction(used_fallback=True)
        assert compressor._fallback_compression_streak == 2
        assert db.get_compression_fallback_streak("CHILD_FALLBACK_ROT") == 2

        resumed = ContextCompressor(
            model="test/model",
            threshold_percent=0.85,
            protect_first_n=2,
            protect_last_n=2,
            quiet_mode=True,
        )
        resumed.bind_session_state(db, "CHILD_FALLBACK_ROT")
        assert resumed._fallback_compression_streak == 2

    def test_real_rotation_records_fallback_after_lifecycle_rebind(self, tmp_path: Path):
        db = SessionDB(db_path=tmp_path / "state.db")
        parent = "PARENT_REAL_FALLBACK_ROT"
        db.create_session(parent, source="telegram")
        agent = _build_agent_with_db(db, parent, platform="telegram")

        with patch(
            "agent.context_compressor.get_model_context_length",
            return_value=100_000,
        ):
            compressor = ContextCompressor(
                model="test/model",
                threshold_percent=0.85,
                protect_first_n=2,
                protect_last_n=2,
                quiet_mode=True,
            )
        compressor.bind_session_state(db, parent)
        compressed = [
            {"role": "user", "content": "[CONTEXT COMPACTION] fallback"},
            {"role": "assistant", "content": "tail"},
        ]

        def _fallback_compress(*_args, **_kwargs):
            compressor._last_summary_error = "empty summary"
            compressor._last_summary_fallback_used = True
            compressor._last_compression_made_progress = True
            return compressed

        with patch.object(
            compressor,
            "compress",
            side_effect=_fallback_compress,
        ):
            compressor.compression_count = 1
            setattr(agent, "context_compressor", compressor)
            agent._compress_context(_msgs(), "sys", approx_tokens=120_000)
        child = getattr(agent, "session_id")

        assert child != parent
        assert compressor._fallback_compression_streak == 1
        assert db.get_compression_fallback_streak(child) == 1


class TestAutomaticCompressionStateRefreshAfterLock:
    def test_prebound_agent_rejects_parent_rotated_before_lock_acquisition(
        self,
        refresh_state_db: SessionDB,
    ):
        db = refresh_state_db
        parent_id = "STALE_ROTATED_PARENT"
        child_id = "CANONICAL_COMPRESSION_CHILD"
        db.create_session(parent_id, source="telegram")
        agent = _build_agent_with_db(db, parent_id, platform="telegram")
        compressor = _bound_context_compressor(db, parent_id)

        # A competing path completes rotation after this call's initial checks
        # but before it acquires the parent lock.
        real_acquire = db.try_acquire_compression_lock

        def _acquire_after_rotation(*args, **kwargs):
            db.end_session(parent_id, "compression")
            db.create_session(
                child_id,
                source="telegram",
                parent_session_id=parent_id,
            )
            return real_acquire(*args, **kwargs)

        db.try_acquire_compression_lock = _acquire_after_rotation
        agent.context_compressor = compressor
        agent.compression_in_place = False
        agent._compression_feasibility_checked = True
        messages = _msgs()

        with patch.object(
            compressor,
            "compress",
            side_effect=AssertionError("stale parent was compressed again"),
        ) as compress:
            returned, _ = agent._compress_context(
                messages,
                "sys",
                approx_tokens=120_000,
                force=True,
            )

        children = db._conn.execute(
            "SELECT id FROM sessions WHERE parent_session_id = ?",
            (parent_id,),
        ).fetchall()
        assert returned is messages
        assert agent.session_id == parent_id
        assert [row["id"] for row in children] == [child_id]
        compress.assert_not_called()
        assert db.get_compression_lock_holder(parent_id) is None

    def test_prebound_agent_reloads_persisted_streak_before_compressing(
        self,
        refresh_state_db: SessionDB,
    ):
        db = refresh_state_db
        session_id = "STALE_FALLBACK_BREAKER"
        db.create_session(session_id, source="telegram")
        db.set_compression_fallback_streak(session_id, 1)
        agent = _build_agent_with_db(db, session_id, platform="telegram")
        compressor = _bound_context_compressor(db, session_id)
        assert compressor._fallback_compression_streak == 1

        # A second agent finishes an in-place fallback boundary after this
        # call's initial gate but while it is acquiring the session lock.
        real_acquire = db.try_acquire_compression_lock

        def _acquire_after_fallback(*args, **kwargs):
            db.set_compression_fallback_streak(session_id, 2)
            return real_acquire(*args, **kwargs)

        db.try_acquire_compression_lock = _acquire_after_fallback
        agent.context_compressor = compressor
        agent.compression_in_place = True
        agent._compression_feasibility_checked = True
        messages = _msgs()

        with patch.object(
            compressor,
            "compress",
            side_effect=AssertionError("stale agent bypassed fallback breaker"),
        ) as compress:
            returned, _ = agent._compress_context(
                messages,
                "sys",
                approx_tokens=120_000,
            )

        assert returned is messages
        assert compressor._fallback_compression_streak == 2
        compress.assert_not_called()
        assert db.get_compression_lock_holder(session_id) is None

    def test_prebound_agent_reloads_persisted_cooldown_before_compressing(
        self,
        refresh_state_db: SessionDB,
    ):
        db = refresh_state_db
        session_id = "STALE_COMPRESSION_COOLDOWN"
        db.create_session(session_id, source="telegram")
        agent = _build_agent_with_db(db, session_id, platform="telegram")
        compressor = _bound_context_compressor(db, session_id)
        assert compressor.get_active_compression_failure_cooldown() is None

        # Another agent records a provider cooldown after this call's initial
        # gate but while it is acquiring the session lock.
        real_acquire = db.try_acquire_compression_lock

        def _acquire_after_cooldown(*args, **kwargs):
            db.record_compression_failure_cooldown(
                session_id,
                time.time() + 60,
                "rate limited",
            )
            return real_acquire(*args, **kwargs)

        db.try_acquire_compression_lock = _acquire_after_cooldown
        agent.context_compressor = compressor
        agent.compression_in_place = True
        agent._compression_feasibility_checked = True
        messages = _msgs()

        with patch.object(
            compressor,
            "compress",
            side_effect=AssertionError("stale agent bypassed compression cooldown"),
        ) as compress:
            returned, _ = agent._compress_context(
                messages,
                "sys",
                approx_tokens=120_000,
            )

        assert returned is messages
        assert compressor.get_active_compression_failure_cooldown() is not None
        compress.assert_not_called()
        assert db.get_compression_lock_holder(session_id) is None

    def test_prebound_agent_drops_stale_blocker_before_initial_gate(
        self,
        refresh_state_db: SessionDB,
    ):
        db = refresh_state_db
        session_id = "CLEARED_FALLBACK_BREAKER"
        db.create_session(session_id, source="telegram")
        db.set_compression_fallback_streak(session_id, 2)
        agent = _build_agent_with_db(db, session_id, platform="telegram")
        compressor = _bound_context_compressor(db, session_id)
        assert compressor._fallback_compression_streak == 2

        # A healthy boundary on another agent clears the durable breaker after
        # this compressor was bound. The initial gate must not remain stuck on
        # its stale in-memory snapshot.
        db.set_compression_fallback_streak(session_id, 0)
        agent.context_compressor = compressor
        agent.compression_in_place = True
        agent._compression_feasibility_checked = True
        messages = _msgs()

        with patch.object(compressor, "compress", return_value=messages) as compress:
            returned, _ = agent._compress_context(
                messages,
                "sys",
                approx_tokens=120_000,
            )

        assert returned is messages
        assert compressor._fallback_compression_streak == 0
        compress.assert_called_once()
        assert db.get_compression_lock_holder(session_id) is None

    def test_prebound_agent_drops_stale_cooldown_before_initial_gate(
        self,
        refresh_state_db: SessionDB,
    ):
        db = refresh_state_db
        session_id = "CLEARED_COMPRESSION_COOLDOWN"
        db.create_session(session_id, source="telegram")
        db.record_compression_failure_cooldown(
            session_id,
            time.time() + 60,
            "rate limited",
        )
        agent = _build_agent_with_db(db, session_id, platform="telegram")
        compressor = _bound_context_compressor(db, session_id)
        assert compressor.get_active_compression_failure_cooldown() is not None

        # A successful forced retry on another agent clears the durable row.
        # This prebound compressor must not keep honoring its stale local timer.
        db.clear_compression_failure_cooldown(session_id)
        agent.context_compressor = compressor
        agent.compression_in_place = True
        agent._compression_feasibility_checked = True
        messages = _msgs()

        with patch.object(compressor, "compress", return_value=messages) as compress:
            returned, _ = agent._compress_context(
                messages,
                "sys",
                approx_tokens=120_000,
            )

        assert returned is messages
        assert compressor.get_active_compression_failure_cooldown() is None
        compress.assert_called_once()
        assert db.get_compression_lock_holder(session_id) is None

    def test_force_still_bypasses_refreshed_persisted_breaker(
        self,
        refresh_state_db: SessionDB,
    ):
        db = refresh_state_db
        session_id = "FORCED_FALLBACK_RETRY"
        db.create_session(session_id, source="telegram")
        db.set_compression_fallback_streak(session_id, 2)
        agent = _build_agent_with_db(db, session_id, platform="telegram")
        compressor = _bound_context_compressor(db, session_id)
        agent.context_compressor = compressor
        agent.compression_in_place = True
        agent._compression_feasibility_checked = True
        messages = _msgs()

        with patch.object(compressor, "compress", return_value=messages) as compress:
            returned, _ = agent._compress_context(
                messages,
                "sys",
                approx_tokens=120_000,
                force=True,
            )

        assert returned is messages
        compress.assert_called_once_with(
            messages,
            current_tokens=120_000,
            focus_topic=None,
            force=True,
        )
        assert db.get_compression_lock_holder(session_id) is None


class TestGateLevelGuardRefresh:
    """The unblock direction must work from the should_compress() pre-gates.

    compress_context refreshes durable guards internally, but the automatic
    paths (preflight/turn gates) consult should_compress() first — if a stale
    in-memory fallback streak (which has no expiry timer) blocks there, the
    refresh inside compress_context is never reached and the agent stays
    blocked forever.
    """

    def test_should_compress_unblocks_after_another_agent_clears_streak(
        self,
        refresh_state_db: SessionDB,
    ):
        db = refresh_state_db
        session_id = "GATE_LEVEL_STREAK_CLEAR"
        db.create_session(session_id, source="telegram")
        db.set_compression_fallback_streak(session_id, 2)
        compressor = _bound_context_compressor(db, session_id)
        assert compressor._fallback_compression_streak == 2

        # Another agent's healthy boundary clears the durable breaker.
        db.set_compression_fallback_streak(session_id, 0)

        assert compressor.should_compress(10**9) is True
        assert compressor._fallback_compression_streak == 0

    def test_unblocked_gate_does_not_touch_the_db(
        self,
        refresh_state_db: SessionDB,
    ):
        db = refresh_state_db
        session_id = "GATE_LEVEL_HOT_PATH"
        db.create_session(session_id, source="telegram")
        compressor = _bound_context_compressor(db, session_id)

        with patch.object(
            compressor,
            "_refresh_durable_guards",
            side_effect=AssertionError("hot path must not refresh"),
        ):
            assert compressor._automatic_compression_blocked() is False


class TestCooldownPersistFailureIsNotAClearedRow:
    def test_refresh_keeps_local_cooldown_when_persist_failed(
        self,
        refresh_state_db: SessionDB,
    ):
        """An empty durable row is not evidence of a clear when OUR write failed.

        _record_compression_failure_cooldown sets the local timer first and
        persists best-effort. If that persist failed, a later refresh=True
        finding no DB row must keep the local cooldown (otherwise the #11529
        thrash guard silently re-opens), until it expires or a successful
        DB round-trip supersedes it.
        """
        db = refresh_state_db
        session_id = "PERSIST_FAILED_COOLDOWN"
        db.create_session(session_id, source="telegram")
        compressor = _bound_context_compressor(db, session_id)

        with patch.object(
            db,
            "record_compression_failure_cooldown",
            side_effect=Exception("disk full"),
        ):
            compressor._record_compression_failure_cooldown(60, "rate limited")
        assert compressor._cooldown_persist_failed is True

        state = compressor.get_active_compression_failure_cooldown(refresh=True)
        assert state is not None
        assert compressor._summary_failure_cooldown_until > 0
        assert compressor._automatic_compression_blocked() is True

        # Once a durable round-trip succeeds, the DB is authoritative again.
        compressor._record_compression_failure_cooldown(30, "retry later")
        assert compressor._cooldown_persist_failed is False
        db.clear_compression_failure_cooldown(session_id)
        assert compressor.get_active_compression_failure_cooldown(refresh=True) is None
        assert compressor._summary_failure_cooldown_until == 0.0

    def test_ineffective_count_only_block_skips_durable_refresh(
        self,
        refresh_state_db: SessionDB,
    ):
        """A block owed solely to the in-memory ineffective counter (which is
        not durable) must not re-read the DB on every gate check."""
        db = refresh_state_db
        session_id = "INEFFECTIVE_ONLY_BLOCK"
        db.create_session(session_id, source="telegram")
        compressor = _bound_context_compressor(db, session_id)
        compressor._ineffective_compression_count = 2

        with patch.object(
            compressor,
            "_refresh_durable_guards",
            side_effect=AssertionError("nothing durable to refresh"),
        ):
            assert compressor._automatic_compression_blocked() is True


class TestTodoSnapshotMergedNotDuplicated:
    """Todo snapshots preserve tail content without duplicate user turns."""

    def test_snapshot_merges_into_trailing_user(self, tmp_path: Path):
        db = SessionDB(db_path=tmp_path / "state.db")
        parent = "PARENT_TODO_MERGE"
        db.create_session(parent, source="cli")
        agent = _build_agent_with_db(db, parent, platform="cli")

        agent.context_compressor.compress.return_value = [
            {"role": "user", "content": "[CONTEXT COMPACTION] summary"},
            {"role": "assistant", "content": "acknowledged"},
            {"role": "user", "content": "tail"},
        ]
        agent._todo_store._todos = [
            {"id": "t1", "content": "task A", "status": "pending"}
        ]
        agent._todo_store.format_for_injection = (
            lambda: "## Current Tasks\n- [ ] task A"
        )

        compressed, _ = agent._compress_context(
            _msgs(), "sys", approx_tokens=120_000
        )

        assert len(compressed) == 3
        tail = compressed[-1]
        assert tail["role"] == "user"
        assert "tail" in tail["content"]
        assert "task A" in tail["content"]
        assert not any(
            previous.get("role") == current.get("role") == "user"
            for previous, current in zip(compressed, compressed[1:])
        )

    def test_multimodal_snapshot_merges_into_trailing_user_on_rotation(
        self, tmp_path: Path
    ):
        db = SessionDB(db_path=tmp_path / "state.db")
        parent = "PARENT_TODO_MULTIMODAL_ROTATION"
        db.create_session(parent, source="cli")
        agent = _build_agent_with_db(db, parent, platform="cli")

        original_parts = [
            {"type": "text", "text": "tail text"},
            {
                "type": "image_url",
                "image_url": {"url": "https://example.com/context.png"},
            },
        ]
        agent.context_compressor.compress.return_value = [
            {"role": "user", "content": "[CONTEXT COMPACTION] summary"},
            {"role": "assistant", "content": "acknowledged"},
            {"role": "user", "content": list(original_parts)},
        ]
        agent._todo_store._todos = [
            {"id": "t1", "content": "inspect image", "status": "pending"}
        ]
        agent._todo_store.format_for_injection = (
            lambda: "## Current Tasks\n- [ ] inspect image"
        )

        compressed, _ = agent._compress_context(
            _msgs(), "sys", approx_tokens=120_000
        )

        assert len(compressed) == 3
        tail = compressed[-1]
        assert tail["role"] == "user"
        assert isinstance(tail["content"], list)
        assert tail["content"][: len(original_parts)] == original_parts
        assert any(
            isinstance(part, dict) and "inspect image" in (part.get("text") or "")
            for part in tail["content"]
        )
        assert not any(
            previous.get("role") == current.get("role") == "user"
            for previous, current in zip(compressed, compressed[1:])
        )


    def test_snapshot_merge_is_persisted_in_place(self, tmp_path: Path):
        db = SessionDB(db_path=tmp_path / "state.db")
        parent = "PARENT_TODO_INPLACE"
        db.create_session(parent, source="cli")
        agent = _build_agent_with_db(db, parent, platform="cli")
        agent.compression_in_place = True
        agent.context_compressor.compress.return_value = [
            {"role": "user", "content": "[CONTEXT COMPACTION] summary"},
            {"role": "assistant", "content": "ok"},
            {"role": "user", "content": "last user msg"},
        ]
        agent._todo_store._todos = [
            {"id": "t1", "content": "do thing", "status": "in_progress"}
        ]
        agent._todo_store.format_for_injection = (
            lambda: "## Current Tasks\n- [ ] do thing"
        )

        agent._compress_context(_msgs(), "sys", approx_tokens=120_000)

        db_msgs = db.get_messages(agent.session_id)
        assert not any(
            previous.get("role") == current.get("role") == "user"
            for previous, current in zip(db_msgs, db_msgs[1:])
        )
        last_user = [message for message in db_msgs if message["role"] == "user"][-1]
        assert "last user msg" in last_user["content"]
        assert "do thing" in last_user["content"]

    def test_multimodal_snapshot_merge_is_persisted_in_place(self, tmp_path: Path):
        db = SessionDB(db_path=tmp_path / "state.db")
        parent = "PARENT_TODO_MULTIMODAL_INPLACE"
        db.create_session(parent, source="cli")
        agent = _build_agent_with_db(db, parent, platform="cli")
        agent.compression_in_place = True

        original_parts = [
            {"type": "text", "text": "last user msg"},
            {
                "type": "image_url",
                "image_url": {"url": "https://example.com/context.png"},
            },
        ]
        agent.context_compressor.compress.return_value = [
            {"role": "user", "content": "[CONTEXT COMPACTION] summary"},
            {"role": "assistant", "content": "ok"},
            {"role": "user", "content": list(original_parts)},
        ]
        agent._todo_store._todos = [
            {"id": "t1", "content": "inspect image", "status": "in_progress"}
        ]
        agent._todo_store.format_for_injection = (
            lambda: "## Current Tasks\n- [ ] inspect image"
        )

        compressed, _ = agent._compress_context(
            _msgs(), "sys", approx_tokens=120_000
        )

        assert len(compressed) == 3
        tail = compressed[-1]
        assert tail["role"] == "user"
        assert isinstance(tail["content"], list)
        assert tail["content"][: len(original_parts)] == original_parts
        assert any(
            isinstance(part, dict) and "inspect image" in (part.get("text") or "")
            for part in tail["content"]
        )
        assert not any(
            previous.get("role") == current.get("role") == "user"
            for previous, current in zip(compressed, compressed[1:])
        )

        db_msgs = db.get_messages(agent.session_id)
        persisted_tail = db_msgs[-1]
        assert persisted_tail["role"] == "user"
        assert persisted_tail["content"][: len(original_parts)] == original_parts
        assert any(
            isinstance(part, dict) and "inspect image" in (part.get("text") or "")
            for part in persisted_tail["content"]
        )
        assert not any(
            previous.get("role") == current.get("role") == "user"
            for previous, current in zip(db_msgs, db_msgs[1:])
        )


class TestTodoSnapshotScaffoldingTails:
    """Scaffolding tails must never absorb the todo snapshot (#69292)."""

    @staticmethod
    def _agent_with_todo(db: SessionDB, session_id: str, tail: dict):
        db.create_session(session_id, source="cli")
        agent = _build_agent_with_db(db, session_id, platform="cli")
        agent.context_compressor.compress.return_value = [
            {"role": "user", "content": "[CONTEXT COMPACTION] summary"},
            {"role": "assistant", "content": "acknowledged"},
            tail,
        ]
        agent._todo_store.write(
            [{"id": "t1", "content": "task A", "status": "pending"}]
        )
        return agent

    def test_snapshot_stays_standalone_after_continuation_marker(
        self, tmp_path: Path
    ):
        from agent.context_compressor import (
            COMPRESSION_CONTINUATION_USER_CONTENT,
            ContextCompressor,
        )

        db = SessionDB(db_path=tmp_path / "state.db")
        agent = self._agent_with_todo(
            db,
            "PARENT_TODO_MARKER_TAIL",
            {
                "role": "user",
                "content": COMPRESSION_CONTINUATION_USER_CONTENT,
            },
        )

        compressed, _ = agent._compress_context(
            _msgs(), "sys", approx_tokens=120_000
        )

        tail = compressed[-1]
        assert tail["role"] == "user"
        assert tail.get("_todo_snapshot_synthetic") is True
        assert "task A" in tail["content"]
        # The continuation marker keeps its exact text so it stays
        # recognizable as scaffolding after SessionDB projection.
        marker_rows = [
            message
            for message in compressed
            if message.get("content") == COMPRESSION_CONTINUATION_USER_CONTENT
        ]
        assert len(marker_rows) == 1
        # Zero-user provenance: neither the marker nor the snapshot may read
        # as a real user turn once SessionDB projection strips the flags
        # (#69292). The fixture's stub summary text is not a real handoff
        # prefix, so assert on the projected scaffolding rows directly.
        assert not ContextCompressor._transcript_has_real_user_turn(
            [
                {"role": "user", "content": marker_rows[0]["content"]},
                {"role": "user", "content": tail["content"]},
            ]
        )

    def test_snapshot_stays_standalone_after_summary_as_user_tail(
        self, tmp_path: Path
    ):
        from agent.context_compressor import SUMMARY_PREFIX, ContextCompressor

        summary_as_user = f"{SUMMARY_PREFIX}\nzero-user summary body"
        db = SessionDB(db_path=tmp_path / "state.db")
        agent = self._agent_with_todo(
            db,
            "PARENT_TODO_SUMMARY_TAIL",
            {"role": "user", "content": summary_as_user},
        )

        compressed, _ = agent._compress_context(
            _msgs(), "sys", approx_tokens=120_000
        )

        tail = compressed[-1]
        assert tail.get("_todo_snapshot_synthetic") is True
        assert "task A" in tail["content"]
        # The summary handoff prefix must stay at the START of its own
        # message for downstream summary detection.
        summary_rows = [
            message
            for message in compressed
            if str(message.get("content") or "").startswith(SUMMARY_PREFIX)
        ]
        assert len(summary_rows) == 1
        # Zero-user provenance (#69292): after SessionDB projection strips
        # the flags, both the summary-as-user handoff and the standalone
        # snapshot must still classify as synthetic — the merge would have
        # buried the header/prefix markers mid-content.
        assert not ContextCompressor._transcript_has_real_user_turn(
            [
                {"role": "user", "content": summary_rows[0]["content"]},
                {"role": "user", "content": tail["content"]},
            ]
        )

    def test_stale_snapshot_row_is_refreshed_not_stacked(self, tmp_path: Path):
        from tools.todo_tool import TODO_INJECTION_HEADER

        stale = f"{TODO_INJECTION_HEADER}\n- [ ] t0. old finished task (pending)"
        db = SessionDB(db_path=tmp_path / "state.db")
        agent = self._agent_with_todo(
            db,
            "PARENT_TODO_STALE_ROW",
            {"role": "user", "content": stale},
        )

        compressed, _ = agent._compress_context(
            _msgs(), "sys", approx_tokens=120_000
        )

        tail = compressed[-1]
        assert tail.get("_todo_snapshot_synthetic") is True
        assert "task A" in tail["content"]
        assert "old finished task" not in tail["content"]
        snapshot_rows = [
            message
            for message in compressed
            if str(message.get("content") or "").startswith(TODO_INJECTION_HEADER)
        ]
        assert len(snapshot_rows) == 1

    def test_previously_merged_snapshot_is_stripped_before_reinjection(
        self, tmp_path: Path
    ):
        from tools.todo_tool import TODO_INJECTION_HEADER

        previously_merged = (
            "please fix the login bug\n\n"
            f"{TODO_INJECTION_HEADER}\n- [ ] t0. old finished task (pending)"
        )
        db = SessionDB(db_path=tmp_path / "state.db")
        agent = self._agent_with_todo(
            db,
            "PARENT_TODO_RESTRIP",
            {"role": "user", "content": previously_merged},
        )

        compressed, _ = agent._compress_context(
            _msgs(), "sys", approx_tokens=120_000
        )

        tail = compressed[-1]
        assert tail["role"] == "user"
        assert "please fix the login bug" in tail["content"]
        assert "task A" in tail["content"]
        assert "old finished task" not in tail["content"]
        assert tail["content"].count(TODO_INJECTION_HEADER) == 1
        assert not any(
            previous.get("role") == current.get("role") == "user"
            for previous, current in zip(compressed, compressed[1:])
        )

    def test_empty_todo_store_injects_nothing(self, tmp_path: Path):
        from tools.todo_tool import TODO_INJECTION_HEADER

        db = SessionDB(db_path=tmp_path / "state.db")
        session_id = "PARENT_TODO_EMPTY"
        db.create_session(session_id, source="cli")
        agent = _build_agent_with_db(db, session_id, platform="cli")
        expected = [
            {"role": "user", "content": "[CONTEXT COMPACTION] summary"},
            {"role": "assistant", "content": "acknowledged"},
            {"role": "user", "content": "tail"},
        ]
        agent.context_compressor.compress.return_value = [
            dict(message) for message in expected
        ]
        agent._todo_store.write(
            [{"id": "t1", "content": "done thing", "status": "completed"}]
        )

        compressed, _ = agent._compress_context(
            _msgs(), "sys", approx_tokens=120_000
        )

        assert compressed == expected
        assert not any(
            TODO_INJECTION_HEADER in str(message.get("content") or "")
            for message in compressed
        )
