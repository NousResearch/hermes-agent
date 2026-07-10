"""Regression test for /branch losing gateway routing columns (#NNNNN).

``_handle_branch_command`` (gateway/slash_commands.py) creates the branched
child session via ``create_session()`` WITHOUT chat_id/chat_type/thread_id —
identical in shape to the compression-rotation bug fixed in
agent/conversation_compression.py. The routing columns are only written
later, when ``switch_session()`` calls ``_record_gateway_session_peer()``
after the branch's session_id is live and its transcript has been copied
message-by-message.

A crash/kill landing between create_session() and switch_session() (most
plausibly mid-history-copy on a long conversation, since each
append_message call is independently try/excepted and best-effort) leaves
the branched session permanently unroutable: NULL chat_id/thread_id can
never be found by find_latest_gateway_session_for_peer, and the /resume
IDOR guard (which requires the row's chat_id/thread_id to match the
caller's) can never authorize a manual recovery either.

This test drives the REAL _handle_branch_command against a REAL SessionStore
+ SessionDB (SQLite in a tmp_path, no mocks on the DB/session-store layer)
and asserts the branched child's routing columns are present in state.db
immediately after create_session() returns — before switch_session() ever
runs — closing the gap the same way the compression fix does.
"""

from __future__ import annotations

import pytest

from gateway.config import GatewayConfig, Platform
from gateway.platforms.base import MessageEvent
from gateway.session import SessionSource, SessionStore, build_session_key
from hermes_state import AsyncSessionDB, SessionDB


@pytest.fixture()
def store(tmp_path, monkeypatch):
    """Real SessionStore backed by a real SessionDB (SQLite in tmp_path)."""
    import hermes_state

    monkeypatch.setattr(hermes_state, "DEFAULT_DB_PATH", tmp_path / "state.db")
    config = GatewayConfig()
    return SessionStore(sessions_dir=tmp_path, config=config)


def _make_source() -> SessionSource:
    return SessionSource(
        platform=Platform.TELEGRAM,
        user_id="170829464",
        chat_id="170829464",
        chat_type="dm",
        thread_id="544520",
    )


def _make_event(text: str) -> MessageEvent:
    return MessageEvent(text=text, source=_make_source(), message_id="m1")


def _make_branch_runner(store: SessionStore):
    """Minimal GatewayRunner stub wired to a REAL session_store/session_db."""
    from gateway.run import GatewayRunner

    runner = object.__new__(GatewayRunner)
    runner.adapters = {}
    runner.config = {}
    runner._background_tasks = set()
    runner._running_agents = {}
    runner._running_agents_ts = {}
    runner._busy_ack_ts = {}
    runner._pending_approvals = {}
    runner._update_prompt_pending = {}
    runner._agent_cache_lock = None
    runner.session_store = store
    runner._session_db = AsyncSessionDB(store._db)
    runner._pending_skills_reload_notes = {}
    return runner


class TestBranchRoutingColumns:
    @pytest.mark.asyncio
    async def test_branched_session_has_routing_columns_before_switch(self, store):
        """Simulates a crash/kill landing between create_session() and
        switch_session() — the exact gap where the branched child's
        routing columns are missing. switch_session() runs unconditionally
        at the end of _handle_branch_command and backfills chat_id/
        chat_type/thread_id via _record_gateway_session_peer(), so checking
        DB state AFTER the function returns successfully will always look
        fine. The real bug only bites if the process dies mid-function,
        before switch_session() gets a chance to run — so we patch
        switch_session to simulate exactly that crash point, then inspect
        the child row switch_session would otherwise have fixed up.
        """
        source = _make_source()
        parent_entry = store.get_or_create_session(source)
        store._db.append_message(parent_entry.session_id, role="user", content="hello")
        store._db.append_message(parent_entry.session_id, role="assistant", content="world")

        runner = _make_branch_runner(store)

        captured_new_session_id = {}
        real_switch_session = store.switch_session

        def _crash_before_switch(session_key, target_session_id):
            # Simulate the process dying right here — before routing gets
            # backfilled — by capturing the id and raising instead of
            # forwarding to the real switch_session().
            captured_new_session_id["id"] = target_session_id
            raise RuntimeError("simulated crash before switch_session")

        import unittest.mock as mock

        with mock.patch.object(store, "switch_session", side_effect=_crash_before_switch):
            with pytest.raises(RuntimeError, match="simulated crash"):
                await runner._handle_branch_command(_make_event("/branch"))

        new_session_id = captured_new_session_id["id"]
        assert new_session_id != parent_entry.session_id

        row = store._db.get_session(new_session_id)
        assert row is not None, "branched child row must exist in state.db"
        # THIS is the bug: without the fix, these are all None, because
        # create_session() at branch time never received them — only the
        # (now-crashed) switch_session() call would have backfilled them.
        assert row["chat_id"] == "170829464", (
            "branched session lost chat_id — unroutable if a crash lands "
            "between create_session() and switch_session()"
        )
        assert row["chat_type"] == "dm"
        assert row["thread_id"] == "544520"

        _ = real_switch_session  # silence unused

