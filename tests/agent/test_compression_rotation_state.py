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

import copy
import os
import time
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from agent.context_compressor import ContextCompressor, _DB_PERSISTED_MARKER
from agent.conversation_compression import (
    CompressionCommitFence,
    _is_real_user_message,
)
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


def _count_rows(rows, *, content: Any = None, role: str | None = None):
    return sum(
        1
        for row in rows
        if (content is None or row.get("content") == content)
        and (role is None or row.get("role") == role)
    )


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


class TestRotationChildFlushDedup:
    def test_summary_handoff_row_is_persisted_once_in_child(
        self, tmp_path: Path
    ):
        db = SessionDB(db_path=tmp_path / "state.db")
        parent = "PARENT_ROT_LIVE_USER"
        db.create_session(parent, source="cli")
        db.append_message(parent, "user", "persisted question")
        db.append_message(parent, "assistant", "persisted answer")

        loaded = db.get_messages_as_conversation(parent)
        messages = [*loaded, {"role": "user", "content": "live question"}]

        agent = _build_agent_with_db(db, parent)
        agent._persist_user_message_idx = len(messages) - 1
        agent.context_compressor.compress.return_value = [
            {"role": "assistant", "content": "[CONTEXT COMPACTION] summary"},
        ]

        returned, _ = agent._compress_context(messages, "sys", approx_tokens=120_000)
        assert any(
            isinstance(msg, dict)
            and msg.get("content") == "[CONTEXT COMPACTION] summary"
            and msg.get(_DB_PERSISTED_MARKER)
            for msg in returned
        )
        assert any(
            isinstance(msg, dict)
            and msg.get("content") == "live question"
            and msg.get(_DB_PERSISTED_MARKER)
            for msg in returned
        )

    def test_rotation_flush_of_original_live_list_keeps_user_once_when_handoff_already_contains_user(
        self, tmp_path: Path
    ):
        db = SessionDB(db_path=tmp_path / "state.db")
        parent = "PARENT_ROT_ORIGINAL_LIVE"
        db.create_session(parent, source="cli")
        db.append_message(parent, "user", "persisted question")
        db.append_message(parent, "assistant", "persisted answer")

        loaded = db.get_messages_as_conversation(parent)
        live_user = {
            "role": "user",
            "content": "live question",
            "timestamp": 1234.5,
        }
        messages = [*loaded, live_user]

        agent = _build_agent_with_db(db, parent)
        agent._persist_user_message_idx = len(messages) - 1
        agent.context_compressor.compress.return_value = [
            {"role": "assistant", "content": "[CONTEXT COMPACTION] summary"},
            copy.deepcopy(live_user),
        ]

        real_flush = agent._flush_messages_to_session_db
        with patch.object(
            agent,
            "_flush_messages_to_session_db",
            side_effect=RuntimeError("simulated parent flush failure"),
        ):
            returned, _ = agent._compress_context(
                messages, "sys", approx_tokens=120_000
            )

        assert agent.session_id != parent
        assert _DB_PERSISTED_MARKER in live_user
        real_flush(messages, conversation_history=loaded)

        child_rows = db.get_messages_as_conversation(
            agent.session_id, include_inactive=True
        )
        assert _count_rows(child_rows, content="live question", role="user") == 1
        assert _count_rows(
            child_rows, content="[CONTEXT COMPACTION] summary", role="assistant"
        ) == 1

    def test_failed_publish_leaves_live_user_unmarked_for_later_flush(
        self, tmp_path: Path
    ):
        db = SessionDB(db_path=tmp_path / "state.db")
        parent = "PARENT_ROT_PUBLISH_FAIL"
        db.create_session(parent, source="cli")
        db.append_message(parent, "user", "persisted question")
        db.append_message(parent, "assistant", "persisted answer")

        loaded = db.get_messages_as_conversation(parent)
        messages = [*loaded, {"role": "user", "content": "live question"}]
        live_user = messages[-1]

        agent = _build_agent_with_db(db, parent)
        agent._persist_user_message_idx = len(messages) - 1
        agent.context_compressor.compress.return_value = [
            {"role": "user", "content": "[CONTEXT COMPACTION] summary"},
            {"role": "assistant", "content": "tail"},
        ]

        real_flush = agent._flush_messages_to_session_db
        with patch.object(
            db,
            "publish_compression_child",
            side_effect=RuntimeError("simulated publish failure"),
        ):
            returned, _ = agent._compress_context(
                messages, "sys", approx_tokens=120_000
            )

        assert _DB_PERSISTED_MARKER not in live_user

        retry_session = "PARENT_ROT_PUBLISH_RETRY"
        db.create_session(retry_session, source="cli")
        agent.session_id = retry_session
        real_flush([live_user])

        retry_rows = db.get_messages_as_conversation(
            retry_session, include_inactive=True
        )
        assert _count_rows(retry_rows, content="live question", role="user") == 1

    def test_mid_tool_loop_rows_do_not_duplicate_after_failed_parent_flush(
        self, tmp_path: Path
    ):
        db = SessionDB(db_path=tmp_path / "state.db")
        parent = "PARENT_ROT_TOOL_LOOP"
        db.create_session(parent, source="cli")
        db.append_message(parent, "user", "persisted question")
        db.append_message(parent, "assistant", "persisted answer")

        loaded = db.get_messages_as_conversation(parent)
        assistant_turn = {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "call-1",
                    "type": "function",
                    "function": {"name": "lookup", "arguments": "{}"},
                }
            ],
        }
        tool_turn = {
            "role": "tool",
            "tool_call_id": "call-1",
            "content": "tool result",
        }
        messages = [
            *loaded,
            {"role": "user", "content": "live tool question"},
            assistant_turn,
            tool_turn,
        ]

        agent = _build_agent_with_db(db, parent)
        agent._persist_user_message_idx = len(loaded)
        agent.context_compressor.compress.return_value = [
            copy.deepcopy(assistant_turn),
            copy.deepcopy(tool_turn),
        ]

        with patch.object(
            agent,
            "_flush_messages_to_session_db",
            side_effect=RuntimeError("simulated parent flush failure"),
        ):
            returned, _ = agent._compress_context(
                messages, "sys", approx_tokens=120_000
            )

        agent._flush_messages_to_session_db(messages, conversation_history=loaded)

        child_rows = db.get_messages_as_conversation(
            agent.session_id, include_inactive=True
        )
        assert _count_rows(
            child_rows, content="live tool question", role="user"
        ) == 1
        assert _count_rows(child_rows, content="tool result", role="tool") == 1

    def test_mid_tool_loop_rows_do_not_duplicate_after_failed_parent_flush_direct_path(
        self, tmp_path: Path
    ):
        db = SessionDB(db_path=tmp_path / "state.db")
        parent = "PARENT_ROT_TOOL_LOOP_DIRECT"
        db.create_session(parent, source="cli")
        db.append_message(parent, "user", "persisted question")
        db.append_message(parent, "assistant", "persisted answer")

        loaded = db.get_messages_as_conversation(parent)
        assistant_turn = {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "call-1",
                    "type": "function",
                    "function": {"name": "lookup", "arguments": "{}"},
                }
            ],
        }
        tool_turn = {
            "role": "tool",
            "tool_call_id": "call-1",
            "content": "tool result",
        }
        messages = [
            *loaded,
            {"role": "user", "content": "live tool question"},
            assistant_turn,
            tool_turn,
        ]

        agent = _build_agent_with_db(db, parent)
        agent._persist_user_message_idx = len(loaded)
        agent.context_compressor.compress.return_value = [
            copy.deepcopy(assistant_turn),
            copy.deepcopy(tool_turn),
        ]

        real_flush = agent._flush_messages_to_session_db
        with patch.object(
            agent,
            "_flush_messages_to_session_db",
            side_effect=RuntimeError("simulated parent flush failure"),
        ):
            _returned, _ = agent._compress_context(
                messages,
                "sys",
                approx_tokens=120_000,
                commit_fence=CompressionCommitFence(),
            )

        assert agent.session_id != parent
        real_flush(messages, conversation_history=loaded)

        child_rows = db.get_messages_as_conversation(
            agent.session_id, include_inactive=True
        )
        assert _count_rows(
            child_rows, content="live tool question", role="user"
        ) == 1
        assert _count_rows(child_rows, content="", role="assistant") == 1
        assert _count_rows(child_rows, content="tool result", role="tool") == 1

    def test_timestampless_duplicate_content_rows_are_all_stamped(
        self, tmp_path: Path
    ):
        db = SessionDB(db_path=tmp_path / "state.db")
        parent = "PARENT_ROT_DUPLICATE_CONTENT"
        db.create_session(parent, source="cli")
        db.append_message(parent, "user", "persisted question")
        db.append_message(parent, "assistant", "persisted answer")

        loaded = db.get_messages_as_conversation(parent)
        assistant_turn = {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "call-1",
                    "type": "function",
                    "function": {"name": "lookup", "arguments": "{}"},
                }
            ],
        }
        tool_turn = {
            "role": "tool",
            "tool_call_id": "call-1",
            "content": "tool result",
        }
        messages = [
            *loaded,
            {"role": "user", "content": "live tool question"},
            assistant_turn,
            copy.deepcopy(assistant_turn),
            tool_turn,
            copy.deepcopy(tool_turn),
        ]

        agent = _build_agent_with_db(db, parent)
        agent._persist_user_message_idx = len(loaded)
        agent.context_compressor.compress.return_value = [
            copy.deepcopy(assistant_turn),
            copy.deepcopy(tool_turn),
        ]

        real_flush = agent._flush_messages_to_session_db
        with patch.object(
            agent,
            "_flush_messages_to_session_db",
            side_effect=RuntimeError("simulated parent flush failure"),
        ):
            _returned, _ = agent._compress_context(
                messages,
                "sys",
                approx_tokens=120_000,
                commit_fence=CompressionCommitFence(),
            )

        real_flush(messages, conversation_history=loaded)

        child_rows = db.get_messages_as_conversation(
            agent.session_id, include_inactive=True
        )
        assert _count_rows(
            child_rows, content="live tool question", role="user"
        ) == 1
        assert _count_rows(child_rows, content="", role="assistant") == 1
        assert _count_rows(child_rows, content="tool result", role="tool") == 1

    def test_rotation_stamps_diverged_session_messages_entry_only_when_it_matches(
        self, tmp_path: Path
    ):
        db = SessionDB(db_path=tmp_path / "state.db")
        parent = "PARENT_ROT_SESSION_MESSAGES_DIVERGE"
        db.create_session(parent, source="cli")
        db.append_message(parent, "user", "persisted question")
        db.append_message(parent, "assistant", "persisted answer")

        loaded = db.get_messages_as_conversation(parent)
        messages = [*loaded, {"role": "user", "content": "live question"}]

        agent = _build_agent_with_db(db, parent)
        agent._persist_user_message_idx = len(messages) - 1
        agent._session_messages = [
            *loaded,
            {"role": "user", "content": "different live question"},
        ]
        agent.context_compressor.compress.return_value = [
            {"role": "assistant", "content": "[CONTEXT COMPACTION] summary"},
        ]

        returned, _ = agent._compress_context(messages, "sys", approx_tokens=120_000)
        agent._flush_messages_to_session_db(messages, conversation_history=loaded)

        child_rows = db.get_messages_as_conversation(
            agent.session_id, include_inactive=True
        )
        assert _count_rows(child_rows, content="live question", role="user") == 1
        assert _DB_PERSISTED_MARKER in messages[-1]
        assert _DB_PERSISTED_MARKER not in agent._session_messages[-1]

    # ------------------------------------------------------------------
    # Item 2 review fixes — symmetric identity validation on the primary
    # stamp. The guard stamps the anchor-source row (the last real user
    # message in `messages`, the row the published child actually
    # represents), NEVER an index that may have drifted, and mirrors the
    # twin (`_session_messages`) by scoped identity against that anchor
    # source with a marker-independent exact-hit two-phase scan.
    # ------------------------------------------------------------------

    def test_rotation_never_stamps_drifted_user_role_neighbor(
        self, tmp_path: Path
    ):
        """A user-role neighbor at a drifted index must not be stamped."""
        db = SessionDB(db_path=tmp_path / "state.db")
        parent = "PARENT_ROT_DRIFTED_NEIGHBOR"
        db.create_session(parent, source="cli")
        db.append_message(parent, "user", "persisted question")
        db.append_message(parent, "assistant", "persisted answer")

        loaded = db.get_messages_as_conversation(parent)
        messages = [
            *loaded,
            {"role": "user", "content": "live question"},
            {
                "role": "user",
                "content": "drifted neighbor",
                "_todo_snapshot_synthetic": True,
            },
        ]

        agent = _build_agent_with_db(db, parent)
        # Index drifted onto the synthetic user-role neighbor (the reanchor
        # fallback / stale-index failure shape the guard must not trust).
        agent._persist_user_message_idx = len(messages) - 1
        agent.context_compressor.compress.return_value = [
            {"role": "assistant", "content": "[CONTEXT COMPACTION] summary"},
        ]

        with patch.object(
            agent,
            "_flush_messages_to_session_db",
            side_effect=RuntimeError("simulated parent flush failure"),
        ):
            _returned, _ = agent._compress_context(
                messages,
                "sys",
                approx_tokens=120_000,
                commit_fence=CompressionCommitFence(),
            )

        # The drifted neighbor is not the row the child represents.
        assert _DB_PERSISTED_MARKER not in messages[-1]
        # The anchor source (the real live question) is stamped.
        assert _DB_PERSISTED_MARKER in messages[-2]

    def test_rotation_drifted_index_does_not_duplicate_live_question_in_child(
        self, tmp_path: Path
    ):
        """Merged outcome + drifted index: real flush must not re-append the
        live question standalone (the duplicate the PR eliminates)."""
        db = SessionDB(db_path=tmp_path / "state.db")
        parent = "PARENT_ROT_DRIFTED_MERGED"
        db.create_session(parent, source="cli")
        db.append_message(parent, "user", "persisted question")
        db.append_message(parent, "assistant", "persisted answer")

        loaded = db.get_messages_as_conversation(parent)
        messages = [
            *loaded,
            {"role": "user", "content": "live question"},
            {
                "role": "user",
                "content": "drifted neighbor",
                "_todo_snapshot_synthetic": True,
            },
        ]

        agent = _build_agent_with_db(db, parent)
        agent._persist_user_message_idx = len(messages) - 1
        agent.context_compressor.compress.return_value = [
            {
                "role": "user",
                "content": "handoff scaffolding",
                "_todo_snapshot_synthetic": True,
            },
        ]

        real_flush = agent._flush_messages_to_session_db
        with patch.object(
            agent,
            "_flush_messages_to_session_db",
            side_effect=RuntimeError("simulated parent flush failure"),
        ):
            _returned, _ = agent._compress_context(
                messages,
                "sys",
                approx_tokens=120_000,
                commit_fence=CompressionCommitFence(),
            )

        real_flush(messages, conversation_history=loaded)

        child_rows = db.get_messages_as_conversation(
            agent.session_id, include_inactive=True
        )
        # No standalone "live question" row — the merged handoff already
        # represents it.
        assert _count_rows(child_rows, content="live question") == 0
        assert (
            _count_rows(
                child_rows, content="live question\n\nhandoff scaffolding"
            )
            == 1
        )

    def test_merged_outcome_still_stamps_live_question(self, tmp_path: Path):
        """Constraint: the guard must not break the legitimate merged stamp."""
        db = SessionDB(db_path=tmp_path / "state.db")
        parent = "PARENT_ROT_MERGED_LIVE"
        db.create_session(parent, source="cli")
        db.append_message(parent, "user", "persisted question")
        db.append_message(parent, "assistant", "persisted answer")

        loaded = db.get_messages_as_conversation(parent)
        messages = [*loaded, {"role": "user", "content": "live question"}]

        agent = _build_agent_with_db(db, parent)
        agent._persist_user_message_idx = len(messages) - 1
        agent.context_compressor.compress.return_value = [
            {
                "role": "user",
                "content": "scaffolding",
                "_todo_snapshot_synthetic": True,
            },
        ]

        real_flush = agent._flush_messages_to_session_db
        with patch.object(
            agent,
            "_flush_messages_to_session_db",
            side_effect=RuntimeError("simulated parent flush failure"),
        ):
            _returned, _ = agent._compress_context(
                messages,
                "sys",
                approx_tokens=120_000,
                commit_fence=CompressionCommitFence(),
            )

        assert _DB_PERSISTED_MARKER in messages[-1]
        real_flush(messages, conversation_history=loaded)

        child_rows = db.get_messages_as_conversation(
            agent.session_id, include_inactive=True
        )
        assert _count_rows(
            child_rows, content="live question", role="user"
        ) == 0
        assert (
            _count_rows(child_rows, content="live question\n\nscaffolding")
            == 1
        )

    def test_adoption_divergence_merged_stamps_both_views_and_no_duplicate(
        self, tmp_path: Path
    ):
        """REAL adoption divergence: durable parent grows under the lease,
        `messages` rebinds to the adopted snapshot while
        `agent._session_messages` stays on the old live list, and the merged
        handoff cannot be mirrored by the wrapper's scoped sync. Both views
        must carry the marker and the real post-rotation flush over the old
        live view must not re-append the live question standalone."""
        db = SessionDB(db_path=tmp_path / "state.db")
        parent = "PARENT_ROT_ADOPT_DIVERGE"
        db.create_session(parent, source="cli")
        db.append_message(parent, "user", "persisted question")
        db.append_message(parent, "assistant", "persisted answer")

        # (b) Old live list object kept alive; divergence set so the twin is
        # the SAME object the guard scans (not a fresh copy).
        loaded = db.get_messages_as_conversation(parent, include_inactive=True)
        old_live_list = [
            *loaded,
            {"role": "user", "content": "live question", "timestamp": 1234.5},
        ]

        agent = _build_agent_with_db(db, parent)
        agent._session_messages = old_live_list
        assert agent._session_messages is old_live_list

        # (c) The stale snapshot passed to _compress_context is a separate
        # object (the production frontend-snapshot shape).
        stale_snapshot = [
            {"role": "user", "content": "persisted question"},
            {"role": "assistant", "content": "persisted answer"},
        ]
        assert stale_snapshot is not agent._session_messages

        # (d) Pin the initial persist-index state: production "no known
        # un-persisted tail" shape, so the real code takes the adopt-directly
        # branch (:2994-3001) and the pre-adoption flush (:2988) is provably
        # never attempted (no fixture flush can mask the divergence).
        agent._persist_user_message_idx = None
        assert agent._persist_user_message_idx is None

        # Grow the DB AFTER the snapshot is taken so the REAL adoption
        # condition (durable parent longer than the caller snapshot) fires.
        db.append_message(parent, "user", "live question")
        durable_check = db.get_messages_as_conversation(parent)
        assert len(durable_check) == 3 > len(stale_snapshot) == 2
        # Sync the twin's timestamp to the committed row so the guard's
        # exact-timestamp twin scan matches the adopted anchor.
        old_live_list[-1]["timestamp"] = durable_check[-1]["timestamp"]

        agent.context_compressor.compress.return_value = [
            {
                "role": "user",
                "content": "handoff scaffolding",
                "_todo_snapshot_synthetic": True,
            },
        ]

        # Phase-keyed flush failure: fail ONLY the pre-publish flush (:3780);
        # a blanket failure would not distinguish the phases and a masked
        # pre-adoption flush would hide the divergence.
        flush_attempts = []

        def _fail_only_prepublish_flush(messages_arg, **kwargs):
            flush_attempts.append((messages_arg, kwargs))
            raise RuntimeError("simulated pre-publish flush failure")

        real_flush = agent._flush_messages_to_session_db
        with patch.object(
            agent,
            "_flush_messages_to_session_db",
            side_effect=_fail_only_prepublish_flush,
        ):
            _returned, _ = agent._compress_context(
                stale_snapshot,
                "sys",
                approx_tokens=120_000,
                commit_fence=CompressionCommitFence(),
            )

        # The ONLY internal flush was the single pre-publish one.
        assert len(flush_attempts) == 1

        # (e) Identity and shape asserts BEFORE markers: adoption fired, the
        # divergence is preserved, the persist index was rebound out of range.
        adopted = agent.context_compressor.compress.call_args.args[0]
        assert adopted is not stale_snapshot
        assert adopted is not agent._session_messages
        assert agent._session_messages is old_live_list
        assert agent._persist_user_message_idx == len(adopted)
        assert adopted[-1]["role"] == "user"
        assert adopted[-1]["content"] == "live question"
        assert adopted[-1].get("timestamp") is not None
        assert old_live_list[-1]["content"] == "live question"
        assert (
            old_live_list[-1].get("timestamp") == adopted[-1].get("timestamp")
        )

        # (f) Markers on BOTH views. Note: the adopted view's rows are
        # "born durable" (hermes_state stamps _DB_PERSISTED_MARKER on rows
        # materialized from the DB), so the adopted assert holds even
        # pre-fix; the DISCRIMINATING assert is the twin's — the old live
        # view is a constructed list the production code only stamps via
        # the guard's twin scan (pre-fix it stays unstamped → FAIL).
        assert _DB_PERSISTED_MARKER in adopted[-1]
        assert _DB_PERSISTED_MARKER in old_live_list[-1]
        real_flush(agent._session_messages, conversation_history=loaded)

        # (g) No standalone duplicate by EXACT SCOPED IDENTITY (content +
        # timestamp), and exactly one merged row.
        child_rows = db.get_messages_as_conversation(
            agent.session_id, include_inactive=True
        )
        assert not any(
            row.get("content") == "live question"
            and row.get("timestamp") == adopted[-1].get("timestamp")
            for row in child_rows
        )
        assert (
            _count_rows(
                child_rows, content="live question\n\nhandoff scaffolding"
            )
            == 1
        )

    def test_rotation_stamps_anchor_source_when_reanchor_fallback_rewrote_turn(
        self, tmp_path: Path
    ):
        """REAL reanchor drift: reanchor_current_turn_user_idx's last-user
        fallback lands on a trailing production-shaped todo-snapshot row
        (index 2) while the anchor-source scan selects the rewritten carrier
        (index 1). The stamp must land on the carrier, not the todo row."""
        from agent.turn_context import reanchor_current_turn_user_idx

        db = SessionDB(db_path=tmp_path / "state.db")
        parent = "PARENT_ROT_REANCHOR_DRIFT"
        db.create_session(parent, source="cli")
        db.append_message(parent, "user", "old durable")
        db.append_message(parent, "assistant", "persisted answer")

        loaded = db.get_messages_as_conversation(parent)
        # Pinned production-shaped fixture (plan §3.5): the trailing row is
        # the todo-snapshot shape compress_context appends at :3484-3489.
        # The reanchor helper's last-user-originated fallback lands on it
        # (index 2) while the anchor-source scan skips it (synthetic flag)
        # and selects the rewritten carrier (index 1) — the drift is real.
        # Deliberately a 3-row fixture (NOT prefixed with `loaded`): the
        # pinned drift values below were verified against the real
        # reanchor_current_turn_user_idx on this shape.
        messages = [
            {"role": "user", "content": "old durable"},
            {"role": "user", "content": "current ask\n\n[merged summary]"},
            {
                "role": "user",
                "content": "Current todos:\n- [ ] leftover",
                "_todo_snapshot_synthetic": True,
            },
        ]

        # Drift-first assertions: prove the reanchor index and the anchor
        # source diverge BEFORE any stamp behavior is checked.
        drifted = reanchor_current_turn_user_idx(messages, "current ask")
        anchor_source = max(
            i for i, m in enumerate(messages) if _is_real_user_message(m)
        )
        assert drifted == 2
        assert anchor_source == 1
        assert drifted != anchor_source

        agent = _build_agent_with_db(db, parent)
        agent._persist_user_message_idx = drifted
        agent.context_compressor.compress.return_value = [
            {
                "role": "user",
                "content": "handoff scaffolding",
                "_todo_snapshot_synthetic": True,
            },
        ]

        real_flush = agent._flush_messages_to_session_db
        with patch.object(
            agent,
            "_flush_messages_to_session_db",
            side_effect=RuntimeError("simulated parent flush failure"),
        ):
            _returned, _ = agent._compress_context(
                messages,
                "sys",
                approx_tokens=120_000,
                commit_fence=CompressionCommitFence(),
            )

        # The carrier (anchor source) is stamped; the todo-snapshot row the
        # drifted index points at is NOT.
        assert _DB_PERSISTED_MARKER in messages[1]
        assert _DB_PERSISTED_MARKER not in messages[2]
        real_flush(messages, conversation_history=loaded)

        child_rows = db.get_messages_as_conversation(
            agent.session_id, include_inactive=True
        )
        assert (
            _count_rows(
                child_rows,
                content="current ask\n\n[merged summary]",
                role="user",
            )
            == 0
        )
        assert (
            _count_rows(
                child_rows,
                content=(
                    "current ask\n\n[merged summary]\n\nhandoff scaffolding"
                ),
            )
            == 1
        )

    def test_no_real_user_anchor_guard_not_entered(self, tmp_path: Path):
        """Negative regression: placeholder_appended/already_present must not
        enter the anchor-source guard branch — no exception, rotation happens,
        no live row outside the handoff carries the marker."""
        db = SessionDB(db_path=tmp_path / "state.db")
        parent = "PARENT_ROT_NO_REAL_ANCHOR"
        db.create_session(parent, source="cli")

        # All-user-synthetic transcript with NO real user. The rows carry
        # enough content that compression shrinks the transcript (a single
        # short synthetic row trips the would-grow gate and aborts rotation,
        # which would make this a fixture failure, not a regression).
        messages = [
            {
                "role": "user",
                "content": f"synthetic scaffolding block {i} with enough "
                f"content to keep the compressed transcript smaller",
                "_todo_snapshot_synthetic": True,
            }
            for i in range(6)
        ]

        agent = _build_agent_with_db(db, parent)
        agent.context_compressor.compress.return_value = [
            {"role": "assistant", "content": "[CONTEXT COMPACTION] summary"},
        ]

        with patch.object(
            agent,
            "_flush_messages_to_session_db",
            side_effect=RuntimeError("simulated parent flush failure"),
        ):
            _returned, _ = agent._compress_context(
                messages,
                "sys",
                approx_tokens=120_000,
                commit_fence=CompressionCommitFence(),
            )

        # Rotation happened; no live row carries the marker.
        assert agent.session_id != parent
        assert _DB_PERSISTED_MARKER not in messages[0]

    def test_list_content_merged_outcome_still_stamps_live_question(
        self, tmp_path: Path
    ):
        """Constraint (reviewer list-content requirement): list-content anchor
        merged via the list branch (anchor_parts + target_parts) must still
        stamp the live row and not duplicate it."""
        db = SessionDB(db_path=tmp_path / "state.db")
        parent = "PARENT_ROT_LIST_MERGED"
        db.create_session(parent, source="cli")
        db.append_message(parent, "user", "persisted question")
        db.append_message(parent, "assistant", "persisted answer")

        loaded = db.get_messages_as_conversation(parent)
        messages = [
            *loaded,
            {
                "role": "user",
                "content": [{"type": "text", "text": "live question"}],
            },
        ]

        agent = _build_agent_with_db(db, parent)
        agent._persist_user_message_idx = len(messages) - 1
        agent.context_compressor.compress.return_value = [
            {
                "role": "user",
                "content": [{"type": "text", "text": "scaffolding"}],
                "_todo_snapshot_synthetic": True,
            },
        ]

        real_flush = agent._flush_messages_to_session_db
        with patch.object(
            agent,
            "_flush_messages_to_session_db",
            side_effect=RuntimeError("simulated parent flush failure"),
        ):
            _returned, _ = agent._compress_context(
                messages,
                "sys",
                approx_tokens=120_000,
                commit_fence=CompressionCommitFence(),
            )

        assert _DB_PERSISTED_MARKER in messages[-1]
        real_flush(messages, conversation_history=loaded)

        child_rows = db.get_messages_as_conversation(
            agent.session_id, include_inactive=True
        )
        # No standalone live-question row (the flush stores list content
        # flattened to its text join, so match the flattened string too).
        assert _count_rows(child_rows, content="live question") == 0
        assert (
            _count_rows(
                child_rows,
                content=[{"type": "text", "text": "live question"}],
            )
            == 0
        )
        # Exactly one merged row with the concatenated parts list.
        assert (
            _count_rows(
                child_rows,
                content=[
                    {"type": "text", "text": "live question"},
                    {"type": "text", "text": "scaffolding"},
                ],
            )
            == 1
        )

    def test_already_stamped_exact_twin_suppresses_broad_fallback(
        self, tmp_path: Path
    ):
        """Two-phase edge (reviewer scenario): an already-stamped exact twin
        must still count as an exact hit (marker-INDEPENDENT), suppressing the
        broad fallback so a timestamp-less same-content historical row is NOT
        stamped as the anchor's twin."""
        db = SessionDB(db_path=tmp_path / "state.db")
        parent = "PARENT_ROT_STAMPED_TWIN"
        db.create_session(parent, source="cli")
        db.append_message(parent, "user", "persisted question")
        db.append_message(parent, "assistant", "persisted answer")

        loaded = db.get_messages_as_conversation(parent)
        messages = [
            *loaded,
            {
                "role": "user",
                "content": "live question",
                "timestamp": "2026-01-01T00:00:00Z",
            },
        ]

        agent = _build_agent_with_db(db, parent)
        # Deliberately do NOT set _persist_user_message_idx: the guard must
        # not trust a persist index at all.
        agent._session_messages = [
            {
                "role": "user",
                "content": "live question",
                "timestamp": "2026-01-01T00:00:00Z",
                _DB_PERSISTED_MARKER: True,
            },
            {"role": "user", "content": "live question"},
        ]
        agent.context_compressor.compress.return_value = [
            {
                "role": "user",
                "content": "handoff scaffolding",
                "_todo_snapshot_synthetic": True,
            },
        ]

        with patch.object(
            agent,
            "_flush_messages_to_session_db",
            side_effect=RuntimeError("simulated parent flush failure"),
        ):
            _returned, _ = agent._compress_context(
                messages,
                "sys",
                approx_tokens=120_000,
                commit_fence=CompressionCommitFence(),
            )

        # The timestamp-less ambiguous row must NOT be stamped as a twin.
        assert _DB_PERSISTED_MARKER not in agent._session_messages[1]
        # The already-stamped exact twin keeps its marker (idempotent).
        assert _DB_PERSISTED_MARKER in agent._session_messages[0]
        # The primary anchor is still stamped.
        assert _DB_PERSISTED_MARKER in messages[-1]


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

        # Input transcript must be large enough that the fake compressor's
        # output is a genuine shrink — the no-growth commit guard refuses
        # to persist a compression that grows the transcript.
        input_msgs = [
            {
                "role": "user" if i % 2 == 0 else "assistant",
                "content": f"m{i} " + "x" * 400,
            }
            for i in range(20)
        ]
        compressed, _ = agent._compress_context(
            input_msgs, "sys", approx_tokens=120_000
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
            {
                "role": "user",
                "content": previously_merged,
                "api_content": "stale wire copy containing the old task",
            },
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
        assert "api_content" not in tail
        assert not any(
            previous.get("role") == current.get("role") == "user"
            for previous, current in zip(compressed, compressed[1:])
        )

    def test_buried_standalone_snapshot_is_replaced_not_duplicated(
        self, tmp_path: Path
    ):
        """A tool result after a snapshot must not make the next boundary stack it."""
        from tools.todo_tool import TODO_INJECTION_HEADER

        db = SessionDB(db_path=tmp_path / "state.db")
        agent = self._agent_with_todo(
            db,
            "PARENT_TODO_BURIED",
            {"role": "tool", "tool_call_id": "call-1", "content": "done"},
        )
        agent.context_compressor.compress.return_value.insert(
            -1,
            {
                "role": "user",
                "content": (
                    f"{TODO_INJECTION_HEADER}\n"
                    "- [ ] t0. stale task (pending)"
                ),
                "_todo_snapshot_synthetic": True,
            },
        )

        compressed, _ = agent._compress_context(
            _msgs(), "sys", approx_tokens=120_000
        )

        snapshot_rows = [
            message
            for message in compressed
            if TODO_INJECTION_HEADER in str(message.get("content") or "")
        ]
        assert len(snapshot_rows) == 1
        assert "task A" in str(snapshot_rows[0]["content"])
        assert "stale task" not in str(snapshot_rows[0]["content"])

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

        assert [
            {
                k: v
                for k, v in m.items()
                if k not in {"_row_id", _DB_PERSISTED_MARKER}
            }
            for m in compressed
        ] == expected
        assert not any(
            TODO_INJECTION_HEADER in str(message.get("content") or "")
            for message in compressed
        )

    def test_empty_todo_store_removes_previous_compaction_snapshot(
        self, tmp_path: Path
    ):
        """Completed items make the store authoritative and clear old work."""
        from tools.todo_tool import TODO_INJECTION_HEADER

        db = SessionDB(db_path=tmp_path / "state.db")
        session_id = "PARENT_TODO_CLEARED"
        db.create_session(session_id, source="cli")
        agent = _build_agent_with_db(db, session_id, platform="cli")
        stale_task = "- [ ] stale-task. This task was already cleared"
        agent.context_compressor.compress.return_value = [
            {"role": "user", "content": "[CONTEXT COMPACTION] summary"},
            {"role": "assistant", "content": "acknowledged"},
            {
                "role": "user",
                "content": (
                    "Keep this user context.\n\n"
                    f"{TODO_INJECTION_HEADER}\n{stale_task}"
                ),
                "api_content": "stale wire copy containing the old snapshot",
            },
        ]
        agent._todo_store.write(
            [{"id": "done", "content": "done thing", "status": "completed"}]
        )

        compressed, _ = agent._compress_context(
            _msgs(), "sys", approx_tokens=120_000
        )

        combined = "\n".join(str(m.get("content") or "") for m in compressed)
        assert "Keep this user context." in combined
        assert TODO_INJECTION_HEADER not in combined
        assert stale_task not in combined
        retained = next(
            message
            for message in compressed
            if "Keep this user context." in str(message.get("content") or "")
        )
        assert "api_content" not in retained

    def test_cancelled_only_store_is_authoritative(self, tmp_path: Path):
        """Cancelled work retires the old snapshot just like completed work."""
        from tools.todo_tool import TODO_INJECTION_HEADER

        db = SessionDB(db_path=tmp_path / "state.db")
        session_id = "PARENT_TODO_CANCELLED"
        db.create_session(session_id, source="cli")
        agent = _build_agent_with_db(db, session_id, platform="cli")
        agent.context_compressor.compress.return_value = [
            {"role": "user", "content": "Keep the request"},
            {"role": "assistant", "content": "acknowledged"},
            {
                "role": "user",
                "content": f"{TODO_INJECTION_HEADER}\n- [ ] obsolete task",
                "_todo_snapshot_synthetic": True,
            },
        ]
        agent._todo_store.write(
            [{"id": "nope", "content": "obsolete task", "status": "cancelled"}]
        )

        compressed, _ = agent._compress_context(
            _msgs(), "sys", approx_tokens=120_000
        )

        assert [message.get("role") for message in compressed] == [
            "user",
            "assistant",
        ]
        assert all(
            TODO_INJECTION_HEADER not in str(message.get("content") or "")
            for message in compressed
        )

    def test_structured_snapshot_only_tail_is_removed(self, tmp_path: Path):
        """A flagged list row is removable only when no other part remains."""
        from tools.todo_tool import TODO_INJECTION_HEADER

        db = SessionDB(db_path=tmp_path / "state.db")
        session_id = "PARENT_TODO_STRUCTURED_ONLY"
        db.create_session(session_id, source="cli")
        agent = _build_agent_with_db(db, session_id, platform="cli")
        agent.context_compressor.compress.return_value = [
            {"role": "user", "content": "Keep the request"},
            {"role": "assistant", "content": "acknowledged"},
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": f"{TODO_INJECTION_HEADER}\n- [ ] stale task",
                    }
                ],
                "_todo_snapshot_synthetic": True,
            },
        ]
        agent._todo_store.write(
            [{"id": "done", "content": "stale task", "status": "completed"}]
        )

        compressed, _ = agent._compress_context(
            _msgs(), "sys", approx_tokens=120_000
        )

        assert [message.get("role") for message in compressed] == [
            "user",
            "assistant",
        ]

    def test_non_tail_snapshot_deletion_repairs_assistant_alternation(
        self, tmp_path: Path
    ):
        from tools.todo_tool import TODO_INJECTION_HEADER

        db = SessionDB(db_path=tmp_path / "state.db")
        session_id = "PARENT_TODO_MIDDLE"
        db.create_session(session_id, source="cli")
        agent = _build_agent_with_db(db, session_id, platform="cli")
        agent.context_compressor.compress.return_value = [
            {"role": "user", "content": "Keep the request"},
            {
                "role": "assistant",
                "content": "before snapshot",
                "api_content": "stale assistant wire copy",
            },
            {
                "role": "user",
                "content": f"{TODO_INJECTION_HEADER}\n- [ ] stale task",
                "_todo_snapshot_synthetic": True,
            },
            {"role": "assistant", "content": "after snapshot"},
            {"role": "user", "content": "new request"},
        ]
        agent._todo_store.write(
            [{"id": "done", "content": "stale task", "status": "completed"}]
        )

        compressed, _ = agent._compress_context(
            _msgs(), "sys", approx_tokens=120_000
        )

        assert [message.get("role") for message in compressed] == [
            "user",
            "assistant",
            "user",
        ]
        repaired = compressed[1]
        assert "before snapshot" in repaired["content"]
        assert "after snapshot" in repaired["content"]
        assert "api_content" not in repaired
        assert not any(
            previous.get("role") == current.get("role")
            for previous, current in zip(compressed, compressed[1:])
        )

    def test_multimodal_content_survives_and_synthetic_provenance_clears(
        self, tmp_path: Path
    ):
        from tools.todo_tool import TODO_INJECTION_HEADER

        db = SessionDB(db_path=tmp_path / "state.db")
        session_id = "PARENT_TODO_MULTIMODAL_RETIRE"
        db.create_session(session_id, source="cli")
        agent = _build_agent_with_db(db, session_id, platform="cli")
        surviving_parts = [
            {"type": "text", "text": "Keep this caption"},
            {
                "type": "image_url",
                "image_url": {"url": "https://example.com/context.png"},
            },
            {"type": "input_audio", "input_audio": {"data": "audio-data"}},
        ]
        agent.context_compressor.compress.return_value = [
            {"role": "user", "content": "summary"},
            {"role": "assistant", "content": "acknowledged"},
            {
                "role": "user",
                "content": surviving_parts
                + [
                    {
                        "type": "text",
                        "text": f"{TODO_INJECTION_HEADER}\n- [ ] stale task",
                    }
                ],
                "api_content": "stale wire copy containing the snapshot",
                "_todo_snapshot_synthetic": True,
                "unrelated_metadata": "keep-me",
            },
        ]
        agent._todo_store.write(
            [{"id": "done", "content": "stale task", "status": "completed"}]
        )

        input_msgs = [
            {
                "role": "user" if index % 2 == 0 else "assistant",
                "content": f"message {index} " + "x" * 1000,
            }
            for index in range(40)
        ]
        compressed, _ = agent._compress_context(
            input_msgs, "sys", approx_tokens=120_000
        )

        retained = next(
            message for message in compressed if isinstance(message.get("content"), list)
        )
        assert retained["content"] == surviving_parts
        assert retained["unrelated_metadata"] == "keep-me"
        assert "api_content" not in retained
        assert "_todo_snapshot_synthetic" not in retained

    @pytest.mark.parametrize("authority_mode", ["missing", "raises"])
    def test_unknown_store_authority_preserves_snapshot(
        self, tmp_path: Path, authority_mode: str
    ):
        """Legacy or broken stores fail conservative, never erasing pending work."""
        from tools.todo_tool import TODO_INJECTION_HEADER

        db = SessionDB(db_path=tmp_path / "state.db")
        session_id = f"PARENT_TODO_COMPAT_{authority_mode.upper()}"
        db.create_session(session_id, source="telegram")
        agent = _build_agent_with_db(db, session_id, platform="telegram")
        pending_task = "- [ ] pending-task. Preserve conservatively"
        agent.context_compressor.compress.return_value = [
            {"role": "user", "content": "Keep the request"},
            {"role": "assistant", "content": "acknowledged"},
            {
                "role": "user",
                "content": f"{TODO_INJECTION_HEADER}\n{pending_task}",
                "_todo_snapshot_synthetic": True,
            },
        ]

        class LegacyStore:
            def format_for_injection(self):
                return None

        store = LegacyStore()
        if authority_mode == "raises":
            store.has_items = MagicMock(side_effect=RuntimeError("store unavailable"))
        agent._todo_store = store

        compressed, _ = agent._compress_context(
            _msgs(), "sys", approx_tokens=120_000
        )

        combined = "\n".join(str(message.get("content") or "") for message in compressed)
        assert TODO_INJECTION_HEADER in combined
        assert pending_task in combined

    def test_unhydrated_empty_todo_store_preserves_pending_snapshot(
        self, tmp_path: Path
    ):
        """A fresh empty store must not erase pending work retained by compression."""
        from tools.todo_tool import TODO_INJECTION_HEADER

        db = SessionDB(db_path=tmp_path / "state.db")
        session_id = "PARENT_TODO_UNHYDRATED"
        db.create_session(session_id, source="telegram")
        agent = _build_agent_with_db(db, session_id, platform="telegram")
        pending_task = "- [ ] pending-task. Continue after the next compaction"
        getattr(agent, "context_compressor").compress.return_value = [
            {"role": "user", "content": "[CONTEXT COMPACTION] summary"},
            {"role": "assistant", "content": "acknowledged"},
            {
                "role": "user",
                "content": f"{TODO_INJECTION_HEADER}\n{pending_task}",
                "_todo_snapshot_synthetic": True,
            },
        ]

        compressed, _ = agent._compress_context(
            _msgs(), "sys", approx_tokens=120_000
        )

        combined = "\n".join(str(m.get("content") or "") for m in compressed)
        assert TODO_INJECTION_HEADER in combined
        assert pending_task in combined


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


class TestAbortedRotationDoesNotGrowParent:
    """#88197 — a rotation that cannot publish must not have already written.

    The rotation flushes its un-persisted current-turn transcript to the parent
    (#47202) and only then calls ``publish_compression_child``. The abort
    handler rolls back memory but not that flush, so every failed rotation
    leaves the parent transcript longer than it found it. When the failure is
    STICKY -- a parent row stamped ``ended_at`` by something that ended the
    process rather than the conversation, e.g. the TUI gateway's
    ``_shutdown_sessions`` stamping ``end_reason='tui_shutdown'`` while the
    agent keeps running -- every subsequent auto-compaction repeats it, and the
    session grows instead of shrinking until the provider rejects the request.
    """

    @staticmethod
    def _durable_len(db: SessionDB, session_id: str) -> int:
        return len(db.get_messages_as_conversation(session_id))

    def test_automatic_stamp_no_longer_wedges_rotation(self, tmp_path: Path):
        """Flipped by the #88197 wedge fix: an AUTOMATIC stamp
        (``tui_shutdown`` — is_automatic_end_reason) is stale by construction
        for a live rotating writer, so publish now clears it in-transaction
        and the rotation COMMITS instead of aborting forever. The abort
        contract below moves to deliberate boundaries."""
        db = SessionDB(db_path=tmp_path / "state.db")
        parent = "PARENT_AUTOMATIC_STAMP_HEALS"
        db.create_session(parent, source="cli")
        agent = _build_agent_with_db(db, parent)

        # The lie at the heart of #88197: the row says ended, the agent is live.
        db.end_session(parent, "tui_shutdown")
        assert db.get_session(parent)["ended_at"] is not None

        returned, _sp = agent._compress_context(_msgs(), "sys", approx_tokens=120_000)

        # Rotation went through — no abort loop, no repeated flush growth.
        assert agent.session_id != parent
        parent_row = db.get_session(parent)
        assert parent_row["end_reason"] == "compression", (
            "parent must close with its TRUE boundary, not the stale stamp"
        )
        child_row = db.get_session(agent.session_id)
        assert child_row is not None
        assert child_row["parent_session_id"] == parent

    def test_deliberately_ended_parent_aborts_before_the_prepublish_flush(
        self, tmp_path: Path
    ):
        db = SessionDB(db_path=tmp_path / "state.db")
        parent = "PARENT_ENDED_NO_GROWTH"
        db.create_session(parent, source="cli")
        agent = _build_agent_with_db(db, parent)

        # A DELIBERATE boundary (another path owns lineage): the guard must
        # still refuse BEFORE the #47202 flush so aborted attempts cannot
        # grow the parent (#88411's contract, now scoped to non-automatic
        # reasons — automatic stamps rotate through, see the test above).
        db.end_session(parent, "session_reset")
        assert db.get_session(parent)["ended_at"] is not None

        before = self._durable_len(db, parent)

        # Three consecutive auto-compactions, as the reported incident saw.
        for attempt in range(1, 4):
            original = _msgs()
            returned, _sp = agent._compress_context(
                original, "sys", approx_tokens=120_000
            )
            assert self._durable_len(db, parent) == before, (
                f"attempt {attempt} appended to the parent it could not "
                "publish; repeated attempts grow the transcript compression "
                "exists to shrink"
            )
            # Rotation refused: the agent stays on the parent with its
            # transcript intact, same contract as any other publish failure.
            assert agent.session_id == parent
            assert returned is original
            assert [(m["role"], m["content"]) for m in returned] == [
                (m["role"], m["content"]) for m in _msgs()
            ]

        assert db.find_live_compression_child(parent) is None

    def test_live_parent_still_gets_the_prepublish_flush(self, tmp_path: Path):
        """The guard must not cost a real rotation its #47202 tail."""
        db = SessionDB(db_path=tmp_path / "state.db")
        parent = "PARENT_LIVE_FLUSH"
        db.create_session(parent, source="cli")
        agent = _build_agent_with_db(db, parent)

        agent._compress_context(_msgs(), "sys", approx_tokens=120_000)
        assert agent.session_id != parent  # rotation happened

        # The current-turn messages survive in the preserved parent transcript.
        assert self._durable_len(db, parent) >= len(_msgs())

    def test_unreadable_parent_row_fails_open(self, tmp_path: Path):
        """A guard that cannot read the row must not become a way to lose
        compression -- an unreadable parent rotates exactly as before."""
        db = SessionDB(db_path=tmp_path / "state.db")
        parent = "PARENT_UNREADABLE_ROW"
        db.create_session(parent, source="cli")
        agent = _build_agent_with_db(db, parent)

        real_get_session = db.get_session
        calls = {"n": 0}

        def _flaky_get_session(session_id: str):
            if session_id == parent and calls["n"] == 0:
                calls["n"] += 1
                raise RuntimeError("simulated read failure")
            return real_get_session(session_id)

        with patch.object(db, "get_session", side_effect=_flaky_get_session):
            agent._compress_context(_msgs(), "sys", approx_tokens=120_000)

        assert calls["n"] == 1, "the pre-flush guard never read the parent row"
        assert agent.session_id != parent  # rotation still happened
