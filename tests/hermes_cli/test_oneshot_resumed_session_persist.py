"""Regression tests for #88583 — one-shot resumed-session turns must persist.

Bot Mode's bot-to-bot send (``hermes -p <bot> chat --in ~ -c "Bot Chat"
--create-if-missing -Q -q "..."``) runs exactly one turn and exits. The
receiving agent replied, the CLI banner said it resumed the titled session,
but nothing landed in state.db when the turn's in-loop transcript flush
failed transiently (write-lock contention with a multiplex gateway sharing
state.db): the one-shot path had no end-of-run durable retry and never
finalized the session row, unlike the interactive CLI (which retries on the
next turn and ends the session with ``cli_close`` on quit).

The fix routes every one-shot exit (quiet ``-Q -q``, human ``-q``, and the
kanban SIGTERM path) through ``cli._flush_one_shot_session_store``: a final
``_persist_session`` retry (idempotent via the per-message persisted
markers), a token-count drain, and ``end_session(..., "cli_close")``.
"""

from __future__ import annotations

import os
from types import SimpleNamespace
from unittest.mock import patch

import pytest

import cli as cli_mod
from tests.fixtures.storage_read import session_row, transcript_rows
from tests.gateway.fixtures.storage_handles import close_storage_handles, storage_handle

@pytest.fixture(autouse=True)
def _reset_finalize_state(monkeypatch):
    monkeypatch.setattr(cli_mod, "_single_query_finalize_attempted_session_ids", set())
    monkeypatch.setattr(cli_mod, "_handed_off_session_ids", set())
    monkeypatch.setattr(cli_mod, "_cleanup_done", False, raising=False)

def _make_agent(store, session_id="oneshot-88583"):
    """Real AIAgent bound to a real temp store through its storage interface handle."""
    with patch.dict(os.environ, {"OPENROUTER_API_KEY": "test-key"}):
        from run_agent import AIAgent

        agent = AIAgent(
            api_key="test-key",
            base_url="https://openrouter.ai/api/v1",
            model="test/model",
            quiet_mode=True,
            persistence=store,
            session_id=session_id,
            skip_context_files=True,
            skip_memory=True,
        )
    agent._ensure_db_session()
    return agent

def _fake_cli(agent):
    return SimpleNamespace(
        agent=agent,
        session_id=agent.session_id,
        conversation_history=[],
        _session_db=agent._session_db,
        _release_active_session=lambda: None,
    )

class TestOneShotDurableFlush:
    """#88583: the one-shot exit path must retry persistence and finalize."""

    def test_finalize_single_query_persists_unflushed_turn(self, tmp_path, monkeypatch):
        """A turn whose in-loop flush failed must still reach state.db.

        Simulates the reported failure: run_conversation produced the turn's
        messages in memory (``_session_messages``) but the transcript flush
        never landed (transient write-lock loss). Without the fix,
        ``_finalize_single_query`` performs no durable write and the turn
        evaporates — this test fails.
        """
        store = storage_handle(tmp_path)
        agent = _make_agent(store)
        # The turn as run_conversation left it: in memory, un-stamped,
        # never written (the in-loop flush failed transiently).
        agent._session_messages = [
            {"role": "user", "content": "Message from 🤖 worker: hello, remember this"},
            {"role": "assistant", "content": "ack — noted."},
        ]
        assert transcript_rows(store, agent.session_id) == []

        fake = _fake_cli(agent)
        monkeypatch.setattr(cli_mod, "_run_cleanup", lambda **kw: None)
        monkeypatch.setattr(
            cli_mod, "_notify_single_query_session_finalize", lambda _c: None
        )

        cli_mod._finalize_single_query(fake)

        rows = transcript_rows(store, agent.session_id)
        assert [r["role"] for r in rows] == ["user", "assistant"], (
            "one-shot exit must durably flush the turn to state.db "
            f"(#88583); got rows: {rows}"
        )
        assert "remember this" in rows[0]["content"]

    def test_finalize_single_query_ends_session_row(self, tmp_path, monkeypatch):
        """The resumed/created one-shot session row is finalized on exit."""
        store = storage_handle(tmp_path)
        agent = _make_agent(store)
        agent._session_messages = [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hello"},
        ]
        fake = _fake_cli(agent)
        monkeypatch.setattr(cli_mod, "_run_cleanup", lambda **kw: None)
        monkeypatch.setattr(
            cli_mod, "_notify_single_query_session_finalize", lambda _c: None
        )

        cli_mod._finalize_single_query(fake)

        sess = session_row(store, agent.session_id)
        assert sess["ended_at"] is not None
        assert sess["end_reason"] == "cli_close"

    def test_flush_is_idempotent_for_already_persisted_turns(self, tmp_path, monkeypatch):
        """A turn the in-loop flush already wrote is not duplicated."""
        store = storage_handle(tmp_path)
        agent = _make_agent(store)
        messages = [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hello"},
        ]
        # Normal happy path: the in-loop flush already persisted.
        agent._flush_messages_to_session_db(messages, [])
        assert len(transcript_rows(store, agent.session_id)) == 2
        agent._session_messages = messages

        cli_mod._flush_one_shot_session_store(_fake_cli(agent))

        rows = transcript_rows(store, agent.session_id)
        assert len(rows) == 2, f"duplicate rows written: {rows}"

    def test_flush_skips_handed_off_sessions(self, tmp_path):
        """A session handed off to the gateway is owned there (#88234)."""
        store = storage_handle(tmp_path)
        agent = _make_agent(store)
        agent._session_messages = [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hello"},
        ]
        cli_mod._handed_off_session_ids.add(agent.session_id)

        cli_mod._flush_one_shot_session_store(_fake_cli(agent))

        assert transcript_rows(store, agent.session_id) == []
        assert session_row(store, agent.session_id)["ended_at"] is None

    def test_flush_skips_persist_disabled_agents(self, tmp_path):
        """Persistence-isolated forks must never write the canonical store."""
        store = storage_handle(tmp_path)
        agent = _make_agent(store)
        agent._persist_disabled = True
        agent._session_messages = [
            {"role": "user", "content": "curator harness turn"},
        ]

        cli_mod._flush_one_shot_session_store(_fake_cli(agent))

        assert transcript_rows(store, agent.session_id) == []
