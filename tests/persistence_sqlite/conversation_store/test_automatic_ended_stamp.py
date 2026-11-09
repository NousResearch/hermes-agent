"""Regression tests for #88197: a dirty automatic ``ended_at`` stamp on a
live compression parent must not wedge rotation.

TUI server shutdown (``_shutdown_sessions``) stamps ``ended_at`` /
``end_reason='tui_shutdown'`` on every session in its memory, including
sessions whose agent process keeps running (dist auto-reload, second TUI
instance). Nothing on the attach path clears it, so every subsequent rotation
aborted at ``publish_compression_child``'s liveness check, silently and
forever (#88197: 7 aborted attempts, 88% duplicate rows, HTTP 400; the
amplification half was fixed by #88411, the wedge half is covered here).

Contract under test: automatic-cleanup stamps (``is_automatic_end_reason``)
are stale-by-construction for a writer that holds the compression lease:
the backend's ``publish_compression_child`` clears them in its own transaction and
proceeds; deliberate boundaries (``compression``, ``session_reset``,
explicit close) still fail closed.
"""

from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from persistence.sqlite._holder import StoreHolder
from persistence.sqlite._state.common import is_automatic_end_reason
from persistence.sqlite.conversation._store.compression import publish_compression_child
from persistence.sqlite.conversation._store.messages import append_message
from persistence.sqlite.conversation._store.sessions import create_session, end_session, get_session
from persistence_assembly import open_persistence, profile_scope
from tests.fixtures.storage_holder import raw_holder


@pytest.fixture
def persistence(tmp_path: Path):
    handle = open_persistence(profile_scope(home=tmp_path), eager=True)
    try:
        yield handle
    finally:
        handle.close()


@pytest.fixture
def db(persistence) -> StoreHolder:
    return raw_holder(persistence)


def _publish(db: StoreHolder, parent: str, child: str) -> None:
    publish_compression_child(
        db,
        parent_session_id=parent,
        child_session_id=child,
        source="tui",
        messages=[{"role": "user", "content": "[CONTEXT COMPACTION] summary"}],
        require_compression_lease=False,
    )


class TestAutomaticEndReasonPredicate:
    def test_taxonomy(self) -> None:
        for reason in (
            "tui_shutdown",
            "ws_disconnect",
            "idle_timeout",
            "lru_evict",
            "ws_orphan_reap",
            "agent_close",
            "startup_orphan_reap",
            "superseded_by_resume",
        ):
            assert is_automatic_end_reason(reason), reason
        for reason in (
            "compression",
            "session_reset",
            "session_switch",
            "tui_close",
            None,
            "",
        ):
            assert not is_automatic_end_reason(reason), reason


class TestPublishHealsAutomaticStamp:
    @pytest.mark.parametrize(
        "reason", ["tui_shutdown", "ws_disconnect", "ws_orphan_reap", "idle_timeout"]
    )
    def test_rotation_publishes_through_automatic_stamp(
        self, db: StoreHolder, reason: str
    ) -> None:
        parent = f"P_{reason}"
        create_session(db, parent, source="tui")
        append_message(db, parent, "user", content="hello")
        end_session(db, parent, reason)  # the dirty stamp
        assert get_session(db, parent)["ended_at"] is not None

        _publish(db, parent, f"C_{reason}")

        parent_row = get_session(db, parent)
        # Parent closed with its TRUE boundary, not the stale stamp.
        assert parent_row["end_reason"] == "compression"
        assert parent_row["ended_at"] is not None
        child_row = get_session(db, f"C_{reason}")
        assert child_row is not None
        assert child_row["parent_session_id"] == parent

    @pytest.mark.parametrize("reason", ["compression", "session_reset", "tui_close"])
    def test_deliberate_boundary_still_fails_closed(
        self, db: StoreHolder, reason: str
    ) -> None:
        parent = f"P_{reason}"
        create_session(db, parent, source="tui")
        end_session(db, parent, reason)
        with pytest.raises(RuntimeError, match="already ended"):
            _publish(db, parent, f"C_{reason}")
        # No child row leaked from the refused publish.
        assert get_session(db, f"C_{reason}") is None

    def test_live_parent_unaffected(self, db: StoreHolder) -> None:
        create_session(db, "P_live", source="tui")
        _publish(db, "P_live", "C_live")
        assert get_session(db, "P_live")["end_reason"] == "compression"
        assert get_session(db, "C_live") is not None


class TestRotationEndToEnd:
    def _build_agent(self, persistence, session_id: str):
        with patch.dict(os.environ, {"OPENROUTER_API_KEY": "test-key"}):
            from run_agent import AIAgent

            agent = AIAgent(
                api_key="test-key",
                base_url="https://openrouter.ai/api/v1",
                model="test/model",
                platform="tui",
                quiet_mode=True,
                session_db=persistence,
                persistence=persistence,
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
        agent.compression_in_place = False  # rotation path
        return agent

    def test_dirty_tui_shutdown_stamp_does_not_wedge_rotation(
        self, persistence, db: StoreHolder
    ) -> None:
        """#88197 end-to-end: TUI shutdown stamps the live session, then
        auto-compaction rotates anyway instead of aborting forever."""
        parent = "PARENT_88197_E2E"
        create_session(db, parent, source="tui")
        agent = self._build_agent(persistence, parent)

        # Simulate the old TUI server exit stamping the still-live session.
        end_session(db, parent, "tui_shutdown")

        msgs = [{"role": "user", "content": f"m{i}"} for i in range(20)]
        agent._compress_context(list(msgs), "sys", approx_tokens=120_000)

        assert agent.session_id != parent, (
            "rotation aborted on the stale tui_shutdown stamp — "
            "the #88197 wedge is back"
        )
        parent_row = get_session(db, parent)
        assert parent_row["end_reason"] == "compression"
        child_row = get_session(db, agent.session_id)
        assert child_row is not None
        assert child_row["parent_session_id"] == parent
