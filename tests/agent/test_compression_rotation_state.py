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


def _build_agent_with_db(
    db: SessionDB,
    session_id: str,
    platform: str = "telegram",
    user_id: str = None,
    chat_id: str = None,
    chat_type: str = None,
    thread_id: str = None,
    gateway_session_key: str = None,
):
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
            user_id=user_id,
            chat_id=chat_id,
            chat_type=chat_type,
            thread_id=thread_id,
            gateway_session_key=gateway_session_key,
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


class TestRoutingColumnsPersistOnRotation:
    """The compression-rotation child session row must carry the same
    chat_id/chat_type/thread_id/session_key as its parent at CREATE time.

    Previously ``create_session()`` at the rotation boundary
    (agent/conversation_compression.py) only passed session_id/source/model/
    model_config/parent_session_id — never the gateway routing columns. The
    routing peer record (chat_id/thread_id/session_key) was written in a
    SEPARATE step later, in gateway/run.py's post-turn backfill, after the
    agent result returns to the event loop. A process crash/kill landing in
    that gap left the child row permanently unroutable: NULL chat_id/thread_id
    survive forever, since nothing ever revisits an already-created row to
    backfill them, and find_latest_gateway_session_for_peer's WHERE clause
    can never match a NULL-routing row. The parent conversation becomes an
    orphan and the next inbound message on that chat/thread spawns a brand
    new, empty session instead of resuming.

    This test drives the real rotation path end-to-end (real SessionDB, real
    AIAgent, real compress_context) and asserts the child row already has
    the routing columns immediately after create_session returns — i.e.
    there is no gap for a crash to land in.
    """

    def test_child_row_has_routing_columns_immediately_after_rotation(
        self, tmp_path: Path
    ):
        db = SessionDB(db_path=tmp_path / "state.db")
        parent = "PARENT_ROUTING_ROT"
        db.create_session(
            parent,
            source="telegram",
            user_id="170829464",
            chat_id="170829464",
            chat_type="dm",
            thread_id="544520",
            session_key="agent:main:telegram:dm:170829464:544520",
        )
        agent = _build_agent_with_db(
            db,
            parent,
            platform="telegram",
            user_id="170829464",
            chat_id="170829464",
            chat_type="dm",
            thread_id="544520",
            gateway_session_key="agent:main:telegram:dm:170829464:544520",
        )

        agent._compress_context(_msgs(), "sys", approx_tokens=120_000)
        child = agent.session_id
        assert child != parent  # rotation happened

        row = db.get_session(child)
        assert row is not None, "child session row must exist immediately after rotation"
        assert row["user_id"] == "170829464"
        assert row["chat_id"] == "170829464"
        assert row["chat_type"] == "dm"
        assert row["thread_id"] == "544520"
        assert row["session_key"] == "agent:main:telegram:dm:170829464:544520"

    def test_no_routing_context_is_a_harmless_noop(self, tmp_path: Path):
        """CLI/non-gateway sessions have no user_id/chat_id/thread_id/session_key at
        all — the fix must not require them or break sessions that never had
        routing metadata in the first place."""
        db = SessionDB(db_path=tmp_path / "state.db")
        parent = "PARENT_NO_ROUTING_ROT"
        db.create_session(parent, source="cli")
        agent = _build_agent_with_db(db, parent, platform="cli")

        agent._compress_context(_msgs(), "sys", approx_tokens=120_000)
        child = agent.session_id
        assert child != parent

        row = db.get_session(child)
        assert row is not None
        assert row["user_id"] is None
        assert row["chat_id"] is None
        assert row["chat_type"] is None
        assert row["thread_id"] is None
        assert row["session_key"] is None
