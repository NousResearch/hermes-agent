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
import sqlite3
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

import agent.conversation_compression as compression_module
from agent.conversation_compression import recover_rotated_compression_session
from agent.context_compressor import ContextCompressor
from agent.memory_manager import MemoryManager
from agent.memory_provider import MemoryProvider
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
    # Lifecycle methods must be explicit instance attributes so the fixture
    # models the declared ContextCompressor contract instead of relying on
    # MagicMock.__getattr__ to synthesize them dynamically.
    compressor.on_session_start = MagicMock()
    compressor.bind_session_state = MagicMock()
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

        # Atomic publication failure must leave the live parent and caller's
        # original list untouched even when a plugin compressor mutates in place.
        original = _msgs()

        def _mutating_compress(live_messages, **_kwargs):
            live_messages[:] = [
                {"role": "user", "content": "mutated compacted snapshot"}
            ]
            return live_messages

        agent.context_compressor.compress.side_effect = _mutating_compress

        def _boom(*a, **k):
            raise RuntimeError("simulated atomic publication failure")

        with patch.object(db, "publish_compression_child", side_effect=_boom):
            returned, _system_prompt = agent._compress_context(
                original, "sys", approx_tokens=120_000
            )

        assert agent.session_id == parent
        assert [(m["role"], m["content"]) for m in returned] == [
            (m["role"], m["content"]) for m in _msgs()
        ]
        assert returned is original
        parent_row = db.get_session(parent)
        assert parent_row is not None
        assert parent_row["ended_at"] is None
        assert db.find_live_compression_child(parent) is None


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


class TestRotatedSessionRecovery:
    def test_recovers_live_tip_across_multiple_compressions(self, tmp_path: Path):
        db = SessionDB(db_path=tmp_path / "state.db")
        parent = "STALE_MULTI_HOP_PARENT"
        db.create_session(parent, source="webui")
        agent = _build_agent_with_db(db, parent, platform="webui")

        db.end_session(parent, "compression")
        db.create_session("COMPRESSED_CHILD_1", source="webui", parent_session_id=parent)
        db.end_session("COMPRESSED_CHILD_1", "compression")
        db.create_session(
            "COMPRESSED_CHILD_2",
            source="webui",
            parent_session_id="COMPRESSED_CHILD_1",
        )
        db.end_session("COMPRESSED_CHILD_2", "compression")
        db.create_session(
            "LIVE_MULTI_HOP_TIP",
            source="webui",
            parent_session_id="COMPRESSED_CHILD_2",
        )
        db.replace_messages(
            "LIVE_MULTI_HOP_TIP",
            [{"role": "user", "content": "latest compacted history"}],
        )
        db.set_compression_fallback_streak(parent, 1)
        db.set_compression_ineffective_count(parent, 1)
        db.set_compression_fallback_streak("COMPRESSED_CHILD_2", 2)
        db.set_compression_ineffective_count("COMPRESSED_CHILD_2", 2)
        db.set_compression_fallback_streak("LIVE_MULTI_HOP_TIP", 7)
        db.set_compression_ineffective_count("LIVE_MULTI_HOP_TIP", 8)
        compressor = _bound_context_compressor(db, parent)
        setattr(agent, "context_compressor", compressor)
        class _RecordingMemoryManager:
            def __init__(self):
                self.on_session_switch = MagicMock(return_value=True)

        memory_manager = _RecordingMemoryManager()
        setattr(agent, "_memory_manager", memory_manager)

        with patch.object(
            compressor,
            "on_session_start",
            wraps=compressor.on_session_start,
        ) as on_session_start:
            recovered = recover_rotated_compression_session(agent)

        assert recovered is not None
        assert [message["content"] for message in recovered] == [
            "latest compacted history"
        ]
        assert getattr(agent, "session_id") == "LIVE_MULTI_HOP_TIP"
        assert compressor._fallback_compression_streak == 7
        assert compressor._ineffective_compression_count == 8
        assert db.get_compression_fallback_streak("LIVE_MULTI_HOP_TIP") == 7
        assert db.get_compression_ineffective_count("LIVE_MULTI_HOP_TIP") == 8
        assert on_session_start.call_args.kwargs["boundary_reason"] == "resume"
        assert on_session_start.call_args.kwargs["old_session_id"] == "COMPRESSED_CHILD_2"
        assert on_session_start.call_args.kwargs["recovered_from_compression"] is True
        memory_manager.on_session_switch.assert_called_once_with(
            "LIVE_MULTI_HOP_TIP",
            parent_session_id="COMPRESSED_CHILD_2",
            reset=False,
            reason="resume",
        )

    def test_revalidation_retries_tip_that_rotates_during_load(self, tmp_path: Path):
        db = SessionDB(db_path=tmp_path / "state.db")
        parent = "STALE_RACING_PARENT"
        db.create_session(parent, source="webui")
        agent = _build_agent_with_db(db, parent, platform="webui")
        db.end_session(parent, "compression")
        db.create_session("RACING_TIP", source="webui", parent_session_id=parent)
        db.replace_messages(
            "RACING_TIP",
            [{"role": "user", "content": "loaded before rotation"}],
        )
        original_loader = SessionDB.get_messages_as_conversation

        def _load_then_rotate(session_db: SessionDB, session_id: str):
            loaded = original_loader(session_db, session_id)
            if session_id == "RACING_TIP":
                session_db.end_session("RACING_TIP", "compression")
                session_db.create_session(
                    "NEW_RACING_TIP",
                    source="webui",
                    parent_session_id="RACING_TIP",
                )
                session_db.replace_messages(
                    "NEW_RACING_TIP",
                    [{"role": "user", "content": "new durable tip"}],
                )
            return loaded

        with patch.object(
            SessionDB,
            "get_messages_as_conversation",
            _load_then_rotate,
        ):
            recovered = recover_rotated_compression_session(agent)

        assert recovered is not None
        assert [message["content"] for message in recovered] == [
            "new durable tip"
        ]
        assert getattr(agent, "session_id") == "NEW_RACING_TIP"
        getattr(agent, "context_compressor").on_session_start.assert_called_once()
        assert (
            getattr(agent, "context_compressor").on_session_start.call_args.args[0]
            == "NEW_RACING_TIP"
        )

    def test_adoption_holds_tip_lease_through_final_validation(self, tmp_path: Path):
        db = SessionDB(db_path=tmp_path / "state.db")
        parent = "LEASE_RACING_PARENT"
        tip = "LEASE_RACING_TIP"
        successor = "LEASE_RACING_SUCCESSOR"
        db.create_session(parent, source="webui")
        agent = _build_agent_with_db(db, parent, platform="webui")
        db.end_session(parent, "compression")
        db.create_session(tip, source="webui", parent_session_id=parent)
        db.replace_messages(
            tip,
            [{"role": "user", "content": "durable tip history"}],
        )
        original_finder = SessionDB.find_live_compression_child
        finder_calls = 0
        competitor_acquired: list[bool] = []

        def _find_then_compete(session_db: SessionDB, session_id: str):
            nonlocal finder_calls
            found = original_finder(session_db, session_id)
            finder_calls += 1
            if finder_calls == 2:
                acquired = session_db.try_acquire_compression_lock(
                    tip,
                    "competing-compressor",
                    ttl_seconds=60,
                )
                competitor_acquired.append(acquired)
                if acquired:
                    try:
                        session_db.publish_compression_child(
                            parent_session_id=tip,
                            child_session_id=successor,
                            source="webui",
                            messages=[
                                {"role": "user", "content": "successor history"}
                            ],
                            compression_lock_holder="competing-compressor",
                        )
                    finally:
                        session_db.release_compression_lock(
                            tip,
                            "competing-compressor",
                        )
            return found

        with patch.object(
            SessionDB,
            "find_live_compression_child",
            _find_then_compete,
        ):
            recovered = recover_rotated_compression_session(agent)

        assert competitor_acquired == [False]
        assert recovered is not None
        assert [message["content"] for message in recovered] == [
            "durable tip history"
        ]
        assert getattr(agent, "session_id") == tip
        tip_row = db.get_session(tip)
        assert tip_row is not None
        assert tip_row["ended_at"] is None
        assert db.get_session(successor) is None
        assert db.get_compression_lock_holder(tip) is None

    def test_adoption_loads_transcript_after_acquiring_tip_lease(
        self,
        tmp_path: Path,
    ):
        db = SessionDB(db_path=tmp_path / "state.db")
        parent = "POST_LEASE_LOAD_PARENT"
        tip = "POST_LEASE_LOAD_TIP"
        db.create_session(parent, source="webui")
        agent = _build_agent_with_db(db, parent, platform="webui")
        db.end_session(parent, "compression")
        db.create_session(tip, source="webui", parent_session_id=parent)
        db.replace_messages(
            tip,
            [{"role": "user", "content": "initial"}],
        )
        original_acquire = SessionDB.try_acquire_compression_lock
        appended_before_acquire: list[bool] = []

        def _append_then_acquire(
            session_db: SessionDB,
            session_id: str,
            holder: str,
            ttl_seconds: float = 300.0,
            *,
            patience_s: float = 20.0,
            raise_on_error: bool = False,
        ) -> bool:
            if holder.endswith(":adoption"):
                session_db.append_message(
                    tip,
                    role="user",
                    content="late-before-lease",
                )
                appended_before_acquire.append(True)
            return original_acquire(
                session_db,
                session_id,
                holder,
                ttl_seconds=ttl_seconds,
                patience_s=patience_s,
                raise_on_error=raise_on_error,
            )

        with patch.object(
            SessionDB,
            "try_acquire_compression_lock",
            _append_then_acquire,
        ):
            recovered = recover_rotated_compression_session(agent)

        assert appended_before_acquire == [True]
        assert recovered is not None
        assert [message["content"] for message in recovered] == [
            "initial",
            "late-before-lease",
        ]
        assert getattr(agent, "session_id") == tip

    def test_lost_adoption_lease_fails_closed_without_replaying_hooks(
        self,
        tmp_path: Path,
    ):
        db = SessionDB(db_path=tmp_path / "state.db")
        parent = "EXPIRED_ADOPTION_PARENT"
        tip = "EXPIRED_ADOPTION_TIP"
        successor = "EXPIRED_ADOPTION_SUCCESSOR"
        db.create_session(parent, source="webui")
        agent = _build_agent_with_db(db, parent, platform="webui")
        db.end_session(parent, "compression")
        db.create_session(tip, source="webui", parent_session_id=parent)
        db.replace_messages(
            tip,
            [{"role": "user", "content": "stale tip history"}],
        )
        raced: list[bool] = []

        def _expire_during_lifecycle(session_id: str, **_kwargs) -> None:
            if session_id != tip or raced:
                return
            with db._lock:
                assert db._conn is not None
                db._conn.execute(
                    "UPDATE compression_locks SET expires_at = 0 WHERE session_id = ?",
                    (tip,),
                )
                db._conn.commit()
            acquired = db.try_acquire_compression_lock(
                tip,
                "competing-compressor",
                ttl_seconds=60,
            )
            raced.append(acquired)
            assert acquired
            try:
                db.publish_compression_child(
                    parent_session_id=tip,
                    child_session_id=successor,
                    source="webui",
                    messages=[
                        {"role": "user", "content": "successor history"}
                    ],
                    compression_lock_holder="competing-compressor",
                )
            finally:
                db.release_compression_lock(tip, "competing-compressor")

        getattr(agent, "context_compressor").on_session_start.side_effect = (
            _expire_during_lifecycle
        )

        # Failure injection: the background refresher is deliberately stalled,
        # forcing the synchronous ownership check to detect the reclaimed lease.
        with patch.object(
            compression_module._CompressionLockLeaseRefresher,
            "start",
            lambda self: self,
        ):
            with pytest.raises(
                compression_module.CompressionRecoveryUnavailableError
            ) as exc_info:
                recover_rotated_compression_session(agent)

        assert raced == [True]
        assert exc_info.value.reason == "lease_lost_after_lifecycle"
        assert exc_info.value.session_id == parent
        assert exc_info.value.retryable is True
        assert getattr(agent, "session_id") == parent
        getattr(agent, "context_compressor").on_session_start.assert_called_once()
        assert (
            getattr(agent, "context_compressor").on_session_start.call_args.args[0]
            == tip
        )
        getattr(agent, "context_compressor").bind_session_state.assert_called_once_with(
            db,
            parent,
        )
        tip_row = db.get_session(tip)
        assert tip_row is not None
        assert tip_row["ended_at"] is not None
        assert db.get_compression_lock_holder(tip) is None

    def test_expired_adoption_lease_cannot_hide_intervening_append(
        self,
        tmp_path: Path,
    ):
        db = SessionDB(db_path=tmp_path / "state.db")
        parent = "EXPIRED_GAP_PARENT"
        tip = "EXPIRED_GAP_TIP"
        db.create_session(parent, source="webui")
        agent = _build_agent_with_db(db, parent, platform="webui")
        db.end_session(parent, "compression")
        db.create_session(tip, source="webui", parent_session_id=parent)
        db.replace_messages(
            tip,
            [{"role": "user", "content": "loaded-before-gap"}],
        )

        def _expire_and_append(session_id: str, **_kwargs) -> None:
            if session_id != tip:
                return
            with db._lock:
                assert db._conn is not None
                db._conn.execute(
                    "UPDATE compression_locks SET expires_at = 0 WHERE session_id = ?",
                    (tip,),
                )
                db._conn.commit()
            db.append_message(
                tip,
                role="user",
                content="committed-during-lease-gap",
            )

        getattr(agent, "context_compressor").on_session_start.side_effect = (
            _expire_and_append
        )

        with patch.object(
            compression_module._CompressionLockLeaseRefresher,
            "start",
            lambda self: self,
        ):
            with pytest.raises(
                compression_module.CompressionRecoveryUnavailableError
            ) as exc_info:
                recover_rotated_compression_session(agent)

        assert exc_info.value.reason == "lease_lost_after_lifecycle"
        assert getattr(agent, "session_id") == parent
        assert [
            message["content"] for message in db.get_messages_as_conversation(tip)
        ] == ["loaded-before-gap", "committed-during-lease-gap"]

    def test_lifecycle_hook_false_restores_parent_and_aborts(self, tmp_path: Path):
        db = SessionDB(db_path=tmp_path / "state.db")
        parent = "HOOK_FALSE_PARENT"
        tip = "HOOK_FALSE_TIP"
        db.create_session(parent, source="webui")
        agent = _build_agent_with_db(db, parent, platform="webui")
        db.end_session(parent, "compression")
        db.create_session(tip, source="webui", parent_session_id=parent)
        db.replace_messages(tip, [{"role": "user", "content": "tip history"}])
        compressor = getattr(agent, "context_compressor")
        compressor.on_session_start.return_value = False

        with pytest.raises(
            compression_module.CompressionRecoveryUnavailableError
        ) as exc_info:
            recover_rotated_compression_session(agent)

        assert exc_info.value.reason == "lifecycle_binding_failed"
        assert getattr(agent, "session_id") == parent
        compressor.bind_session_state.assert_called_once_with(db, parent)

    def test_lifecycle_hook_failure_restores_parent_and_aborts(self, tmp_path: Path):
        db = SessionDB(db_path=tmp_path / "state.db")
        parent = "HOOK_FAILURE_PARENT"
        tip = "HOOK_FAILURE_TIP"
        db.create_session(parent, source="webui")
        agent = _build_agent_with_db(db, parent, platform="webui")
        db.end_session(parent, "compression")
        db.create_session(tip, source="webui", parent_session_id=parent)
        db.replace_messages(tip, [{"role": "user", "content": "tip history"}])
        compressor = getattr(agent, "context_compressor")
        bound_state = {"session_id": parent}

        def _partially_bind_then_fail(session_id, **_kwargs):
            bound_state["session_id"] = session_id
            raise RuntimeError("plugin hook failed")

        def _bind_state(_session_db, session_id):
            bound_state["session_id"] = session_id

        compressor.on_session_start.side_effect = _partially_bind_then_fail
        compressor.bind_session_state.side_effect = _bind_state

        with pytest.raises(
            compression_module.CompressionRecoveryUnavailableError
        ) as exc_info:
            recover_rotated_compression_session(agent)

        assert exc_info.value.reason == "lifecycle_binding_failed"
        assert getattr(agent, "session_id") == parent
        assert bound_state["session_id"] == parent
        compressor.on_session_start.assert_called_once()
        compressor.bind_session_state.assert_called_once_with(db, parent)

    def test_persistent_adoption_conflict_fails_closed(self, tmp_path: Path):
        db = SessionDB(db_path=tmp_path / "state.db")
        parent = "BUSY_ADOPTION_PARENT"
        tip = "BUSY_ADOPTION_TIP"
        db.create_session(parent, source="webui")
        agent = _build_agent_with_db(db, parent, platform="webui")
        db.end_session(parent, "compression")
        db.create_session(tip, source="webui", parent_session_id=parent)
        db.replace_messages(
            tip,
            [{"role": "user", "content": "canonical tip history"}],
        )
        assert db.try_acquire_compression_lock(
            tip,
            "competing-compressor",
            ttl_seconds=60,
        )
        with patch.object(compression_module.time, "sleep", return_value=None):
            with pytest.raises(
                compression_module.CompressionRecoveryUnavailableError
            ) as exc_info:
                recover_rotated_compression_session(agent)

        assert exc_info.value.reason == "tip_busy"
        assert exc_info.value.session_id == parent
        assert exc_info.value.retryable is True
        assert getattr(agent, "session_id") == parent
        assert db.get_compression_lock_holder(tip) == "competing-compressor"

    def test_sqlite_lock_error_is_not_retried_as_tip_contention(self, tmp_path: Path):
        db = SessionDB(db_path=tmp_path / "state.db")
        parent = "SQLITE_ERROR_PARENT"
        tip = "SQLITE_ERROR_TIP"
        db.create_session(parent, source="webui")
        agent = _build_agent_with_db(db, parent, platform="webui")
        db.end_session(parent, "compression")
        db.create_session(tip, source="webui", parent_session_id=parent)
        db.replace_messages(tip, [{"role": "user", "content": "tip history"}])

        acquire_calls = 0

        def _acquire_sqlite_error(
            _db,
            session_id,
            holder,
            ttl_seconds=300.0,
            *,
            patience_s=None,
            raise_on_error=False,
        ):
            del session_id, holder, ttl_seconds, patience_s, raise_on_error
            nonlocal acquire_calls
            acquire_calls += 1
            raise sqlite3.OperationalError("database is locked")

        with patch.object(
            SessionDB,
            "try_acquire_compression_lock",
            new=_acquire_sqlite_error,
        ):
            with patch.object(compression_module.time, "sleep", return_value=None):
                with pytest.raises(
                    compression_module.CompressionRecoveryUnavailableError
                ) as exc_info:
                    recover_rotated_compression_session(agent)

        assert acquire_calls == 1
        assert exc_info.value.reason == "sqlite_error"
        assert exc_info.value.session_id == parent
        assert exc_info.value.retryable is True
        assert isinstance(exc_info.value.__cause__, sqlite3.OperationalError)
        assert getattr(agent, "session_id") == parent

    def test_real_sqlite_contention_respects_composed_recovery_budget(
        self,
        tmp_path: Path,
    ):
        db_path = tmp_path / "state.db"
        db = SessionDB(db_path=db_path)
        parent = "REAL_BUSY_PARENT"
        tip = "REAL_BUSY_TIP"
        db.create_session(parent, source="webui")
        agent = _build_agent_with_db(db, parent, platform="webui")
        db.end_session(parent, "compression")
        db.create_session(tip, source="webui", parent_session_id=parent)
        db.replace_messages(tip, [{"role": "user", "content": "tip history"}])

        blocker = sqlite3.connect(str(db_path), timeout=0, isolation_level=None)
        blocker.execute("BEGIN IMMEDIATE")
        started = time.monotonic()
        try:
            with pytest.raises(
                compression_module.CompressionRecoveryUnavailableError
            ) as exc_info:
                recover_rotated_compression_session(agent)
        finally:
            elapsed = time.monotonic() - started
            blocker.rollback()
            blocker.close()

        assert exc_info.value.reason == "sqlite_error"
        assert exc_info.value.retryable is True
        assert elapsed < 1.5
        assert getattr(agent, "session_id") == parent

    def test_declared_instance_session_db_adapter_can_recover(self, tmp_path: Path):
        db = SessionDB(db_path=tmp_path / "state.db")
        parent = "ADAPTER_PARENT"
        tip = "ADAPTER_TIP"
        db.create_session(parent, source="webui")
        agent = _build_agent_with_db(db, parent, platform="webui")
        db.end_session(parent, "compression")
        db.create_session(tip, source="webui", parent_session_id=parent)
        db.replace_messages(tip, [{"role": "user", "content": "tip history"}])

        class _DeclaredAdapter:
            pass

        adapter = _DeclaredAdapter()
        for name in (
            "get_session",
            "find_live_compression_child",
            "get_messages_as_conversation",
            "get_compression_lock_holder",
            "try_acquire_compression_lock",
            "refresh_compression_lock",
            "release_compression_lock",
        ):
            setattr(adapter, name, getattr(db, name))
        setattr(agent, "_session_db", adapter)

        recovered = recover_rotated_compression_session(agent)

        assert recovered is not None
        assert [message["content"] for message in recovered] == ["tip history"]
        assert getattr(agent, "session_id") == tip

    def test_lock_adapter_without_bounded_error_contract_is_rejected(self, tmp_path: Path):
        db = SessionDB(db_path=tmp_path / "state.db")
        parent = "LEGACY_LOCK_ADAPTER_PARENT"
        tip = "LEGACY_LOCK_ADAPTER_TIP"
        db.create_session(parent, source="webui")
        agent = _build_agent_with_db(db, parent, platform="webui")
        db.end_session(parent, "compression")
        db.create_session(tip, source="webui", parent_session_id=parent)
        db.replace_messages(tip, [{"role": "user", "content": "tip history"}])

        class _LegacyLockAdapter:
            def get_session(self, session_id):
                return db.get_session(session_id)

            def find_live_compression_child(self, session_id):
                return db.find_live_compression_child(session_id)

            def get_messages_as_conversation(self, session_id):
                return db.get_messages_as_conversation(session_id)

            def get_compression_lock_holder(self, session_id):
                return db.get_compression_lock_holder(session_id)

            def try_acquire_compression_lock(self, session_id, holder):
                return db.try_acquire_compression_lock(session_id, holder)

            def refresh_compression_lock(self, session_id, holder):
                return db.refresh_compression_lock(session_id, holder)

            def release_compression_lock(self, session_id, holder):
                return db.release_compression_lock(session_id, holder)

        setattr(agent, "_session_db", _LegacyLockAdapter())

        with pytest.raises(
            compression_module.CompressionRecoveryUnavailableError
        ) as exc_info:
            recover_rotated_compression_session(agent)

        assert exc_info.value.reason == "unsupported_db"
        assert exc_info.value.retryable is False
        assert getattr(agent, "session_id") == parent

    def test_lock_adapter_cannot_hide_positional_only_controls_behind_kwargs(
        self, tmp_path: Path
    ):
        db = SessionDB(db_path=tmp_path / "state.db")
        parent = "POSITIONAL_LOCK_ADAPTER_PARENT"
        tip = "POSITIONAL_LOCK_ADAPTER_TIP"
        db.create_session(parent, source="webui")
        agent = _build_agent_with_db(db, parent, platform="webui")
        db.end_session(parent, "compression")
        db.create_session(tip, source="webui", parent_session_id=parent)
        db.replace_messages(tip, [{"role": "user", "content": "tip history"}])

        class _PositionalOnlyLockAdapter:
            pass

        adapter = _PositionalOnlyLockAdapter()
        for name in (
            "get_session",
            "find_live_compression_child",
            "get_messages_as_conversation",
            "get_compression_lock_holder",
            "refresh_compression_lock",
            "release_compression_lock",
        ):
            setattr(adapter, name, getattr(db, name))

        def _legacy_acquire(
            session_id,
            holder,
            ttl_seconds=300.0,
            patience_s=20.0,
            raise_on_error=False,
            /,
            **_kwargs,
        ):
            return db.try_acquire_compression_lock(
                session_id,
                holder,
                ttl_seconds,
                patience_s=patience_s,
                raise_on_error=raise_on_error,
            )

        setattr(adapter, "try_acquire_compression_lock", _legacy_acquire)
        setattr(agent, "_session_db", adapter)

        with pytest.raises(
            compression_module.CompressionRecoveryUnavailableError
        ) as exc_info:
            recover_rotated_compression_session(agent)

        assert exc_info.value.reason == "unsupported_db"

    def test_kwargs_only_lock_adapter_does_not_declare_bounded_contract(self):
        def _kwargs_only(*_args, **_kwargs):
            return True

        assert not compression_module._lock_method_accepts_bounded_recovery_contract(
            _kwargs_only,
            require_ttl=True,
        )
        assert not compression_module._lock_method_accepts_bounded_recovery_contract(
            _kwargs_only,
            require_ttl=False,
        )

    def test_undeclared_session_db_inspection_fails_closed(self, tmp_path: Path):
        db = SessionDB(db_path=tmp_path / "state.db")
        parent = "UNSUPPORTED_ADAPTER_PARENT"
        db.create_session(parent, source="webui")
        agent = _build_agent_with_db(db, parent, platform="webui")
        setattr(agent, "_session_db", object())

        with pytest.raises(
            compression_module.CompressionRecoveryUnavailableError
        ) as exc_info:
            recover_rotated_compression_session(agent)

        assert exc_info.value.reason == "unsupported_db"
        assert exc_info.value.session_id == parent
        assert exc_info.value.retryable is False

    def test_failed_hook_and_direct_binding_never_commit_agent(self, tmp_path: Path):
        db = SessionDB(db_path=tmp_path / "state.db")
        parent = "BIND_FAILURE_PARENT"
        tip = "BIND_FAILURE_TIP"
        db.create_session(parent, source="webui")
        agent = _build_agent_with_db(db, parent, platform="webui")
        db.end_session(parent, "compression")
        db.create_session(tip, source="webui", parent_session_id=parent)
        db.replace_messages(tip, [{"role": "user", "content": "tip history"}])
        compressor = getattr(agent, "context_compressor")
        compressor.on_session_start.side_effect = RuntimeError("hook failed")
        compressor.bind_session_state.side_effect = RuntimeError("bind failed")

        with pytest.raises(
            compression_module.CompressionRecoveryUnavailableError
        ) as exc_info:
            recover_rotated_compression_session(agent)

        assert exc_info.value.reason == "lifecycle_binding_failed"
        assert exc_info.value.retryable is False
        assert getattr(agent, "session_id") == parent
        compressor.on_session_start.assert_called_once()
        compressor.bind_session_state.assert_called_once_with(db, parent)

    def test_dynamic_context_lifecycle_hook_is_rejected_without_invocation(
        self,
        tmp_path: Path,
    ):
        db = SessionDB(db_path=tmp_path / "state.db")
        parent = "DYNAMIC_CONTEXT_PARENT"
        tip = "DYNAMIC_CONTEXT_TIP"
        db.create_session(parent, source="webui")
        agent = _build_agent_with_db(db, parent, platform="webui")
        db.end_session(parent, "compression")
        db.create_session(tip, source="webui", parent_session_id=parent)
        db.replace_messages(tip, [{"role": "user", "content": "tip history"}])

        class _DynamicCompressor:
            def __init__(self):
                self.lifecycle_calls = 0
                self.binding_calls = []

            def __getattr__(self, name):
                if name != "on_session_start":
                    raise AttributeError(name)

                def _dynamic_hook(*_args, **_kwargs):
                    self.lifecycle_calls += 1

                return _dynamic_hook

            def bind_session_state(self, *args, **kwargs):
                self.binding_calls.append((args, kwargs))

        compressor = _DynamicCompressor()
        setattr(agent, "context_compressor", compressor)

        with pytest.raises(
            compression_module.CompressionRecoveryUnavailableError
        ) as exc_info:
            recover_rotated_compression_session(agent)

        assert exc_info.value.reason == "lifecycle_binding_failed"
        assert compressor.lifecycle_calls == 0
        assert compressor.binding_calls == [((db, parent), {})]
        assert getattr(agent, "session_id") == parent

    def test_dynamic_restore_binder_is_rejected_without_invocation(
        self,
        tmp_path: Path,
    ):
        db = SessionDB(db_path=tmp_path / "state.db")
        parent = "DYNAMIC_RESTORE_PARENT"
        tip = "DYNAMIC_RESTORE_TIP"
        db.create_session(parent, source="webui")
        agent = _build_agent_with_db(db, parent, platform="webui")
        db.end_session(parent, "compression")
        db.create_session(tip, source="webui", parent_session_id=parent)
        db.replace_messages(tip, [{"role": "user", "content": "tip history"}])

        class _DynamicRestoreCompressor:
            def __init__(self):
                self.binding_calls = 0

            def on_session_start(self, *_args, **_kwargs):
                return False

            def __getattr__(self, name):
                if name != "bind_session_state":
                    raise AttributeError(name)

                def _dynamic_bind(*_args, **_kwargs):
                    self.binding_calls += 1

                return _dynamic_bind

        compressor = _DynamicRestoreCompressor()
        setattr(agent, "context_compressor", compressor)

        with pytest.raises(
            compression_module.CompressionRecoveryUnavailableError
        ) as exc_info:
            recover_rotated_compression_session(agent)

        assert exc_info.value.reason == "lifecycle_binding_failed"
        assert compressor.binding_calls == 0
        assert getattr(agent, "session_id") == parent

    def test_memory_switch_failure_commits_child_and_blocks_until_retry(
        self,
        tmp_path: Path,
    ):
        db = SessionDB(db_path=tmp_path / "state.db")
        parent = "MEMORY_SWITCH_FAILURE_PARENT"
        tip = "MEMORY_SWITCH_FAILURE_TIP"
        db.create_session(parent, source="webui")
        agent = _build_agent_with_db(db, parent, platform="webui")
        db.end_session(parent, "compression")
        db.create_session(tip, source="webui", parent_session_id=parent)
        db.replace_messages(tip, [{"role": "user", "content": "tip history"}])

        class _RetryingMemoryManager:
            def __init__(self):
                self.switch_calls = []
                self.retry_calls = 0

            def on_session_switch(self, session_id, **kwargs):
                self.switch_calls.append((session_id, kwargs))
                return False

            def retry_pending_session_switch(self):
                self.retry_calls += 1
                return True

        memory_manager = _RetryingMemoryManager()
        setattr(agent, "_memory_manager", memory_manager)

        with pytest.raises(
            compression_module.CompressionRecoveryUnavailableError
        ) as exc_info:
            recover_rotated_compression_session(agent)

        assert exc_info.value.reason == "memory_binding_pending"
        assert getattr(agent, "session_id") == tip
        assert getattr(agent, "_pending_compression_memory_switch") == {
            "session_id": tip,
            "parent_session_id": parent,
        }
        getattr(agent, "context_compressor").on_session_start.assert_called_once()
        assert memory_manager.switch_calls == [
            (
                tip,
                {
                    "parent_session_id": parent,
                    "reset": False,
                    "reason": "resume",
                },
            )
        ]

        resumed_history = (
            compression_module.resume_pending_compression_memory_switch(agent)
        )

        assert memory_manager.retry_calls == 1
        assert resumed_history is not None
        assert [message["content"] for message in resumed_history] == ["tip history"]
        assert getattr(agent, "_pending_compression_memory_switch") is None

    def test_memory_retry_is_not_replayed_after_transcript_reload_error(
        self,
        tmp_path: Path,
    ):
        db = SessionDB(db_path=tmp_path / "state.db")
        parent = "MEMORY_RELOAD_PARENT"
        tip = "MEMORY_RELOAD_TIP"
        db.create_session(parent, source="webui")
        db.end_session(parent, "compression")
        db.create_session(tip, source="webui", parent_session_id=parent)
        db.replace_messages(tip, [{"role": "user", "content": "durable tip"}])
        agent = _build_agent_with_db(db, tip, platform="webui")
        agent._pending_compression_memory_switch = {
            "session_id": tip,
            "parent_session_id": parent,
        }

        class _RetryingMemoryManager:
            def __init__(self):
                self.retry_calls = 0

            def retry_pending_session_switch(self):
                self.retry_calls += 1
                return True

            def on_session_switch(self, *_args, **_kwargs):
                raise AssertionError("full memory switch must not replay")

        memory_manager = _RetryingMemoryManager()
        agent._memory_manager = memory_manager
        real_loader = db.get_messages_as_conversation
        load_calls = 0

        def _load_once_then_succeed(session_id, *_args, **_kwargs):
            nonlocal load_calls
            load_calls += 1
            if load_calls == 1:
                raise sqlite3.OperationalError("database is locked")
            return real_loader(session_id)

        setattr(db, "get_messages_as_conversation", _load_once_then_succeed)

        with pytest.raises(
            compression_module.CompressionRecoveryUnavailableError
        ) as exc_info:
            compression_module.resume_pending_compression_memory_switch(agent)

        assert exc_info.value.reason == "sqlite_error"
        assert memory_manager.retry_calls == 1
        assert getattr(agent, "_pending_compression_memory_switch")["memory_bound"] is True

        resumed_history = (
            compression_module.resume_pending_compression_memory_switch(agent)
        )

        assert memory_manager.retry_calls == 1
        assert resumed_history is not None
        assert [message["content"] for message in resumed_history] == ["durable tip"]
        assert getattr(agent, "_pending_compression_memory_switch") is None

    def test_partial_memory_provider_failure_is_forward_only_and_selectively_retried(
        self,
        tmp_path: Path,
    ):
        db = SessionDB(db_path=tmp_path / "state.db")
        parent = "PARTIAL_MEMORY_PARENT"
        tip = "PARTIAL_MEMORY_TIP"
        db.create_session(parent, source="webui")
        agent = _build_agent_with_db(db, parent, platform="webui")
        db.end_session(parent, "compression")
        db.create_session(tip, source="webui", parent_session_id=parent)
        db.replace_messages(tip, [{"role": "user", "content": "tip history"}])

        class _StatefulProvider(MemoryProvider):
            def __init__(self, name: str, *, fail_once: bool = False):
                self._name = name
                self.fail_once = fail_once
                self.session_id = parent
                self.switch_calls = 0

            @property
            def name(self):
                return self._name

            def is_available(self):
                return True

            def initialize(self, session_id, **kwargs):
                self.session_id = session_id

            def get_tool_schemas(self):
                return []

            def on_session_switch(self, new_session_id, **kwargs):
                self.switch_calls += 1
                self.session_id = new_session_id
                if self.fail_once:
                    self.fail_once = False
                    raise RuntimeError("partial provider failure")
                return None

        good = _StatefulProvider("builtin")
        flaky = _StatefulProvider("external", fail_once=True)
        memory_manager = MemoryManager()
        memory_manager.add_provider(good)
        memory_manager.add_provider(flaky)
        setattr(agent, "_memory_manager", memory_manager)

        with pytest.raises(
            compression_module.CompressionRecoveryUnavailableError
        ) as exc_info:
            recover_rotated_compression_session(agent)

        assert exc_info.value.reason == "memory_binding_pending"
        assert getattr(agent, "session_id") == tip
        assert good.session_id == tip
        assert flaky.session_id == tip
        assert good.switch_calls == 1
        assert flaky.switch_calls == 1

        compression_module.resume_pending_compression_memory_switch(agent)

        assert good.switch_calls == 1
        assert flaky.switch_calls == 2
        assert getattr(agent, "_pending_compression_memory_switch") is None

    def test_final_refresh_sqlite_error_restores_parent_binding(self, tmp_path: Path):
        db = SessionDB(db_path=tmp_path / "state.db")
        parent = "FINAL_REFRESH_ERROR_PARENT"
        tip = "FINAL_REFRESH_ERROR_TIP"
        db.create_session(parent, source="webui")
        agent = _build_agent_with_db(db, parent, platform="webui")
        db.end_session(parent, "compression")
        db.create_session(tip, source="webui", parent_session_id=parent)
        db.replace_messages(tip, [{"role": "user", "content": "tip history"}])
        real_refresh = db.refresh_compression_lock
        refresh_calls = 0

        def _refresh_then_error(
            _db,
            session_id,
            holder,
            ttl_seconds=300.0,
            *,
            patience_s=None,
            raise_on_error=False,
        ):
            nonlocal refresh_calls
            refresh_calls += 1
            if refresh_calls == 1:
                return real_refresh(
                    session_id,
                    holder,
                    ttl_seconds,
                    patience_s=patience_s,
                    raise_on_error=raise_on_error,
                )
            raise sqlite3.OperationalError("database is locked")

        with patch.object(
            compression_module._CompressionLockLeaseRefresher,
            "start",
            lambda self: self,
        ):
            with patch.object(
                SessionDB,
                "refresh_compression_lock",
                new=_refresh_then_error,
            ):
                with pytest.raises(
                    compression_module.CompressionRecoveryUnavailableError
                ) as exc_info:
                    recover_rotated_compression_session(agent)

        assert exc_info.value.reason == "sqlite_error"
        assert getattr(agent, "session_id") == parent
        compressor = getattr(agent, "context_compressor")
        compressor.on_session_start.assert_called_once()
        compressor.bind_session_state.assert_called_once_with(db, parent)

    def test_failed_hook_without_direct_binding_never_commits_agent(
        self,
        tmp_path: Path,
    ):
        db = SessionDB(db_path=tmp_path / "state.db")
        parent = "NO_BINDER_PARENT"
        tip = "NO_BINDER_TIP"
        db.create_session(parent, source="webui")
        agent = _build_agent_with_db(db, parent, platform="webui")
        db.end_session(parent, "compression")
        db.create_session(tip, source="webui", parent_session_id=parent)
        db.replace_messages(tip, [{"role": "user", "content": "tip history"}])

        class _BrokenCompressor:
            @staticmethod
            def on_session_start(*_args, **_kwargs):
                raise RuntimeError("hook failed")

        setattr(agent, "context_compressor", _BrokenCompressor())

        with pytest.raises(
            compression_module.CompressionRecoveryUnavailableError
        ) as exc_info:
            recover_rotated_compression_session(agent)

        assert exc_info.value.reason == "lifecycle_binding_failed"
        assert getattr(agent, "session_id") == parent

    def test_release_failure_does_not_mask_committed_adoption(self, tmp_path: Path):
        db = SessionDB(db_path=tmp_path / "state.db")
        parent = "RELEASE_FAILURE_PARENT"
        tip = "RELEASE_FAILURE_TIP"
        db.create_session(parent, source="webui")
        agent = _build_agent_with_db(db, parent, platform="webui")
        db.end_session(parent, "compression")
        db.create_session(tip, source="webui", parent_session_id=parent)
        db.replace_messages(tip, [{"role": "user", "content": "tip history"}])

        class _DeclaredAdapter:
            pass

        adapter = _DeclaredAdapter()
        for name in (
            "get_session",
            "find_live_compression_child",
            "get_messages_as_conversation",
            "get_compression_lock_holder",
            "try_acquire_compression_lock",
            "refresh_compression_lock",
            "append_messages_batch",
        ):
            setattr(adapter, name, getattr(db, name))

        def _release_fails(
            session_id,
            holder,
            *,
            patience_s=None,
            raise_on_error=False,
        ):
            del session_id, holder, patience_s, raise_on_error
            raise RuntimeError("release failed")

        setattr(adapter, "release_compression_lock", _release_fails)
        setattr(agent, "_session_db", adapter)

        recovered = recover_rotated_compression_session(agent)

        assert recovered is not None
        assert [message["content"] for message in recovered] == ["tip history"]
        assert getattr(agent, "session_id") == tip
        active_holder = getattr(agent, "_active_compression_lock_holder", None)
        assert active_holder is not None
        assert db.get_compression_lock_holder(tip) == active_holder
        db.append_messages_batch(
            tip,
            [{"role": "assistant", "content": "owner can continue"}],
            compression_lock_holder=active_holder,
        )
        with db._lock:
            assert db._conn is not None
            db._conn.execute(
                "UPDATE compression_locks SET expires_at = 0 WHERE session_id = ?",
                (tip,),
            )
            db._conn.commit()
        assert db.try_acquire_compression_lock(tip, "successor-holder") is True
        messages = list(recovered)
        messages.append({"role": "user", "content": "must not cross reclaimed lease"})
        setattr(agent, "messages", messages)

        flush_pending = getattr(agent, "_flush_messages_to_session_db")
        assert flush_pending(messages) is False
        assert getattr(agent, "_active_compression_lock_holder", None) is None
        assert db.get_compression_lock_holder(tip) == "successor-holder"

    def test_false_release_preserves_committed_adoption_holder(self, tmp_path: Path):
        db = SessionDB(db_path=tmp_path / "state.db")
        parent = "FALSE_RELEASE_PARENT"
        tip = "FALSE_RELEASE_TIP"
        db.create_session(parent, source="webui")
        agent = _build_agent_with_db(db, parent, platform="webui")
        db.end_session(parent, "compression")
        db.create_session(tip, source="webui", parent_session_id=parent)
        db.replace_messages(tip, [{"role": "user", "content": "tip history"}])

        class _DeclaredAdapter:
            pass

        adapter = _DeclaredAdapter()
        for name in (
            "get_session",
            "find_live_compression_child",
            "get_messages_as_conversation",
            "get_compression_lock_holder",
            "try_acquire_compression_lock",
            "refresh_compression_lock",
        ):
            setattr(adapter, name, getattr(db, name))

        def _release_returns_false(
            session_id,
            holder,
            *,
            patience_s=None,
            raise_on_error=False,
        ):
            del session_id, holder, patience_s, raise_on_error
            return False

        setattr(adapter, "release_compression_lock", _release_returns_false)
        setattr(agent, "_session_db", adapter)

        recovered = recover_rotated_compression_session(agent)

        assert recovered is not None
        active_holder = getattr(agent, "_active_compression_lock_holder", None)
        assert active_holder is not None
        assert db.get_compression_lock_holder(tip) == active_holder

    def test_persist_session_propagates_batch_append_refusal(self, tmp_path: Path):
        db = SessionDB(db_path=tmp_path / "state.db")
        session_id = "PERSISTENCE_REFUSAL"
        db.create_session(session_id, source="webui")
        agent = _build_agent_with_db(db, session_id, platform="webui")
        setattr(agent, "_save_session_log", lambda _messages: None)
        setattr(agent, "_flush_messages_to_session_db", lambda *_args: False)
        setattr(db, "flush_token_counts", lambda: None)

        assert agent._persist_session([], []) is False


class TestAutomaticCompressionStateRefreshAfterLock:
    def test_prebound_agent_fails_closed_on_empty_rotated_child_before_lock(
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

        def _acquire_after_rotation(
            session_id,
            holder,
            ttl_seconds=300.0,
            *,
            patience_s=None,
            raise_on_error=False,
        ):
            db.end_session(parent_id, "compression")
            db.create_session(
                child_id,
                source="telegram",
                parent_session_id=parent_id,
            )
            return real_acquire(
                session_id,
                holder,
                ttl_seconds,
                patience_s=patience_s,
                raise_on_error=raise_on_error,
            )

        setattr(db, "try_acquire_compression_lock", _acquire_after_rotation)
        agent.context_compressor = compressor
        agent.compression_in_place = False
        agent._compression_feasibility_checked = True
        messages = _msgs()

        with patch.object(
            compressor,
            "compress",
            side_effect=AssertionError("stale parent was compressed again"),
        ) as compress:
            with pytest.raises(
                compression_module.CompressionRecoveryUnavailableError
            ) as exc_info:
                agent._compress_context(
                    messages,
                    "sys",
                    approx_tokens=120_000,
                    force=True,
                )

        children = db._conn.execute(
            "SELECT id FROM sessions WHERE parent_session_id = ?",
            (parent_id,),
        ).fetchall()
        assert exc_info.value.reason == "empty_tip"
        assert agent.session_id == parent_id
        assert [row["id"] for row in children] == [child_id]
        compress.assert_not_called()
        assert db.get_compression_lock_holder(parent_id) is None




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

    def test_ineffective_count_block_honors_durable_clear_by_another_agent(
        self,
        refresh_state_db: SessionDB,
    ):
        """The ineffective-strike counter is durable (#54923): a block owed to
        it must re-read the DB so another agent's clear (a real usage reading
        that dipped below the threshold) unblocks this compressor too."""
        db = refresh_state_db
        session_id = "INEFFECTIVE_DURABLE_BLOCK"
        db.create_session(session_id, source="telegram")
        db.set_compression_ineffective_count(session_id, 2)
        compressor = _bound_context_compressor(db, session_id)
        assert compressor._ineffective_compression_count == 2

        assert compressor._automatic_compression_blocked() is True

        # Another agent's real prompt reading dipped below the threshold and
        # zeroed the durable counter.
        db.set_compression_ineffective_count(session_id, 0)

        assert compressor._automatic_compression_blocked() is False
        assert compressor._ineffective_compression_count == 0


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


class TestArchivedParentActivityLabelsCleared:
    def test_parent_labels_cleared_after_rotation_child_lineage_intact(
        self, tmp_path: Path
    ):
        """Round-2 #4: the terminal heartbeat stamp must not stay on the parent.

        The compression activity heartbeat force-persists "context compression
        completed" against the PARENT id (agent.session_id at stamp time).
        After the out-of-place rotation the parent is archived; its activity
        labels must be cleared so it doesn't advertise a fresh
        last_activity_at + terminal label forever, while the child keeps its
        lineage.
        """
        from agent.session_activity import ActivityProvenance

        db = SessionDB(db_path=tmp_path / "state.db")
        parent = "PARENT_ACTIVITY_LABELS"
        db.create_session(parent, source="cli")
        agent = _build_agent_with_db(db, parent)

        agent._compress_context(_msgs(), "sys", approx_tokens=120_000)
        child = agent.session_id
        assert child != parent  # rotation happened

        # Child lineage intact.
        child_row = db.get_session(child)
        assert child_row is not None
        assert child_row.get("parent_session_id") == parent

        # Parent archived with cleared activity labels.
        parent_row = db.get_session(parent)
        assert parent_row is not None
        assert parent_row.get("ended_at") is not None
        assert not parent_row.get("last_activity_description"), (
            "archived compression parent kept a stale activity description "
            f"({parent_row.get('last_activity_description')!r})"
        )
        prov = parent_row.get("last_activity_provenance")
        assert not prov or prov == ActivityProvenance.UNKNOWN.value, (
            f"archived parent kept terminal provenance {prov!r}"
        )
