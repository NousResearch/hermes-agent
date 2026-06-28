"""Regression tests for #13121 — gateway restart/shutdown must persist an
in-flight (interrupted) turn's transcript to the SQLite session store so the
immediate pre-restart context survives ``load_transcript()`` on resume.

The bug: every normal/graceful turn exit funnels through
``turn_finalizer.finalize_turn`` which calls ``_persist_session`` →
``_flush_messages_to_session_db`` (the only place a turn is written to
state.db).  During the tool loop only the *in-memory* ``_session_messages``
reference is refreshed per round — there is no incremental SQLite flush
mid-turn.

When the gateway drain times out it marks the session ``resume_pending``,
interrupts the running agents, waits a short grace window, then tears them
down via ``_finalize_shutdown_agents`` → ``_cleanup_agent_resources``.  An
agent blocked in a tool call that does not abort within the grace window
never reaches ``finalize_turn``, so its in-flight tool rounds live only in
``_session_messages`` and are never written to state.db.  On resume,
``load_transcript()`` (state.db is now the canonical store — the legacy
JSONL fallback was dropped) returns the pre-turn state, dropping the
immediate pre-restart turn.

The fix flushes ``_session_messages`` to the session DB in
``_finalize_shutdown_agents`` before teardown.  The flush is idempotent
(identity-tracked in ``_flush_messages_to_session_db``), so agents that DID
finish gracefully re-flush nothing.

These tests exercise BOTH a lightweight unit path (the flush hook is invoked
with the in-flight messages) AND a true E2E path (a real ``AIAgent`` flush
against a real ``SessionDB`` in a temp ``HERMES_HOME``, read back through the
real ``SessionStore.load_transcript``).
"""

from __future__ import annotations

import asyncio
import sys
import types
from unittest.mock import MagicMock

import pytest


@pytest.fixture(autouse=True)
def _mock_dotenv(monkeypatch):
    """gateway.run imports dotenv at module load; stub so tests run bare."""
    fake = types.ModuleType("dotenv")
    fake.load_dotenv = lambda *a, **kw: None
    monkeypatch.setitem(sys.modules, "dotenv", fake)


def _make_runner():
    from gateway.run import GatewayRunner

    runner = object.__new__(GatewayRunner)
    # _finalize_shutdown_agents offloads _cleanup_agent_resources to a worker
    # thread via _run_in_executor_with_context (#53175). Stub it to run inline
    # so the bounded-cleanup path is exercised deterministically in tests.
    async def _inline_executor(func, *args):
        return func(*args)

    runner._run_in_executor_with_context = _inline_executor
    return runner


def _finalize(runner, agents):
    """Drive the now-async _finalize_shutdown_agents from sync test bodies."""
    if not hasattr(runner, "_run_in_executor_with_context"):
        async def _inline_executor(func, *args):
            return func(*args)
        runner._run_in_executor_with_context = _inline_executor
    asyncio.run(runner._finalize_shutdown_agents(agents))


# ─────────────────────────────────────────────────────────────────────────
# Unit: _finalize_shutdown_agents calls the flush hook with the in-flight
# transcript before teardown.
# ─────────────────────────────────────────────────────────────────────────
class _FakeAgent:
    def __init__(self, session_messages=None, has_flush=True):
        if session_messages is not None:
            self._session_messages = session_messages
        if has_flush:
            self._flush_messages_to_session_db = MagicMock()
            self._drop_trailing_empty_response_scaffolding = MagicMock()
        self.shutdown_memory_provider = MagicMock()
        self.close = MagicMock()
        self.session_id = "sess-1"


class TestFinalizeShutdownFlushesInflightTranscript:
    def test_inflight_messages_flushed_before_teardown(self):
        """The mid-turn transcript (tail = pending tool result) is flushed
        to the session DB during shutdown finalization."""
        runner = _make_runner()
        inflight = [
            {"role": "user", "content": "scan the repo and summarise"},
            {"role": "assistant", "content": "", "tool_calls": [
                {"id": "c1", "function": {"name": "terminal", "arguments": "{}"}}
            ]},
            {"role": "tool", "tool_call_id": "c1", "content": "huge output..."},
        ]
        agent = _FakeAgent(session_messages=inflight)

        _finalize(runner, {"agent:main:discord:dm:42": agent})

        agent._flush_messages_to_session_db.assert_called_once_with(inflight)
        # Cleanup still happens after the flush.
        agent.close.assert_called_once()

    def test_empty_session_messages_not_flushed(self):
        """An agent that ran no turns (empty list) triggers no flush — there
        is nothing in flight to persist."""
        runner = _make_runner()
        agent = _FakeAgent(session_messages=[])

        _finalize(runner, {"k": agent})

        agent._flush_messages_to_session_db.assert_not_called()
        agent.close.assert_called_once()

    def test_missing_flush_method_is_tolerated(self):
        """A stub agent without the flush method (object.__new__ test stubs)
        must not break shutdown — teardown still runs."""
        runner = _make_runner()
        agent = _FakeAgent(session_messages=[{"role": "user", "content": "x"}],
                           has_flush=False)

        _finalize(runner, {"k": agent})

        agent.close.assert_called_once()

    def test_flush_exception_is_swallowed(self):
        """A raising flush must not prevent teardown — a transcript-flush
        failure is best-effort, losing tool resources is worse."""
        runner = _make_runner()
        agent = _FakeAgent(session_messages=[{"role": "user", "content": "x"}])
        agent._flush_messages_to_session_db.side_effect = RuntimeError("db locked")

        _finalize(runner, {"k": agent})

        agent.close.assert_called_once()


# ─────────────────────────────────────────────────────────────────────────
# E2E: real AIAgent flush → real SessionDB → real load_transcript.
# ─────────────────────────────────────────────────────────────────────────
class TestShutdownTranscriptSurvivesResumeE2E:
    def test_interrupted_turn_persisted_and_readable_on_resume(self, tmp_path, monkeypatch):
        """Drive the real flush path against a real SessionDB and confirm the
        in-flight turn is readable back through SessionStore.load_transcript —
        the exact path the resume logic reads on the next message."""
        # Isolated state.db.
        monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))

        from hermes_state import SessionDB
        from run_agent import AIAgent

        db = SessionDB(db_path=tmp_path / "state.db")
        session_id = "sess-e2e-13121"
        db.create_session(session_id=session_id, source="discord")

        # Simulate a session whose FIRST turn completed and was persisted...
        db.append_message(session_id=session_id, role="user",
                          content="hello, remember my cat is Mochi")
        db.append_message(session_id=session_id, role="assistant",
                          content="Noted — Mochi the cat.")

        # ...and a SECOND turn that was interrupted mid tool-loop. These rows
        # were NEVER flushed to the DB (only live in _session_messages).
        prior_history = [
            {"role": "user", "content": "hello, remember my cat is Mochi"},
            {"role": "assistant", "content": "Noted — Mochi the cat."},
        ]
        inflight_tail = [
            {"role": "user", "content": "now scan the whole repo for TODOs"},
            {"role": "assistant", "content": "", "tool_calls": [
                {"id": "tc1", "function": {"name": "terminal",
                                           "arguments": "{\"command\": \"grep -r TODO\"}"}}
            ]},
            {"role": "tool", "tool_call_id": "tc1", "name": "terminal",
             "content": "src/a.py: TODO fix this\nsrc/b.py: TODO and that"},
        ]
        # _session_messages is the live list: history copy + in-flight tail.
        session_messages = list(prior_history) + list(inflight_tail)

        # Build a real AIAgent shaped only with what the flush path reads.
        agent = object.__new__(AIAgent)
        agent._session_db = db
        agent._session_db_created = True
        agent.session_id = session_id
        agent.platform = "discord"
        agent._session_messages = session_messages
        # Model a real agent: turn 1 already flushed, so its message identities
        # are recorded in the dedup set. Only the in-flight turn-2 tail is new.
        agent._last_flushed_db_idx = len(prior_history)
        agent._flushed_db_message_ids = {id(m) for m in prior_history}
        agent._flushed_db_message_session_id = session_id

        # Sanity: only the 2 first-turn rows are in the DB before shutdown.
        before = db.get_messages_as_conversation(session_id)
        assert len(before) == 2, before

        # Drive the gateway shutdown finalization with this real agent.
        from gateway.run import GatewayRunner
        runner = object.__new__(GatewayRunner)
        _finalize(runner, {"agent:main:discord:dm:7": agent})

        # The in-flight turn must now be durable and readable via the SAME
        # path the resume logic uses (SessionStore.load_transcript → DB).
        after = db.get_messages_as_conversation(session_id)
        roles = [m.get("role") for m in after]
        contents = [m.get("content") for m in after]

        assert len(after) == 5, after
        # The interrupted user message survived.
        assert any("scan the whole repo for TODOs" in (c or "") for c in contents), contents
        # The pending tool result (the immediate pre-restart context) survived.
        assert any("TODO fix this" in (c or "") for c in contents), contents
        # Tail is a tool result — exactly what the _has_fresh_tool_tail resume
        # branch in _handle_message_with_agent expects to handle.
        assert roles[-1] == "tool", roles

    def test_graceful_agent_reflush_is_idempotent(self, tmp_path, monkeypatch):
        """An agent that already flushed via finalize_turn must not produce
        duplicate rows when _finalize_shutdown_agents re-flushes."""
        monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))

        from hermes_state import SessionDB
        from run_agent import AIAgent

        db = SessionDB(db_path=tmp_path / "state.db")
        session_id = "sess-e2e-idem"
        db.create_session(session_id=session_id, source="discord")

        msgs = [
            {"role": "user", "content": "what is 2+2"},
            {"role": "assistant", "content": "4"},
        ]

        agent = object.__new__(AIAgent)
        agent._session_db = db
        agent._session_db_created = True
        agent.session_id = session_id
        agent.platform = "discord"
        agent._session_messages = msgs
        agent._last_flushed_db_idx = 0
        agent._flushed_db_message_ids = set()
        agent._flushed_db_message_session_id = None

        # First flush (simulating finalize_turn).
        agent._flush_messages_to_session_db(msgs)
        assert len(db.get_messages_as_conversation(session_id)) == 2

        # Shutdown re-flush of the SAME list identity must add nothing.
        from gateway.run import GatewayRunner
        runner = object.__new__(GatewayRunner)
        _finalize(runner, {"k": agent})

        after = db.get_messages_as_conversation(session_id)
        assert len(after) == 2, after

    def test_verification_stop_scaffolding_not_persisted(self, tmp_path, monkeypatch):
        """Verification-stop retry scaffolding is model-private, not transcript content."""
        monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))

        from hermes_state import SessionDB
        from run_agent import AIAgent

        db = SessionDB(db_path=tmp_path / "state.db")
        session_id = "sess-verification-private"
        db.create_session(session_id=session_id, source="discord")

        msgs = [
            {"role": "user", "content": "please edit the script"},
            {
                "role": "assistant",
                "content": "Done without tests",
                "finish_reason": "verification_required",
            },
            {
                "role": "user",
                "content": "[System: You edited code in this turn...]",
                "_verification_stop_synthetic": True,
            },
            {"role": "assistant", "content": "", "tool_calls": [
                {"id": "tc1", "function": {"name": "terminal", "arguments": "{}"}}
            ]},
            {"role": "tool", "tool_call_id": "tc1", "content": "AD_HOC_VERIFICATION_PASSED"},
            {"role": "assistant", "content": "Verified and done."},
        ]

        agent = object.__new__(AIAgent)
        agent._session_db = db
        agent._session_db_created = True
        agent.session_id = session_id
        agent.platform = "discord"
        agent._last_flushed_db_idx = 0
        agent._flushed_db_message_ids = set()
        agent._flushed_db_message_session_id = None

        agent._flush_messages_to_session_db(msgs)

        persisted = db.get_messages_as_conversation(session_id)
        contents = [m.get("content") or "" for m in persisted]
        finish_reasons = [m.get("finish_reason") for m in persisted]

        assert "[System: You edited code in this turn...]" not in contents
        assert "Done without tests" not in contents
        assert "verification_required" not in finish_reasons
        assert "AD_HOC_VERIFICATION_PASSED" in contents
        assert "Verified and done." in contents
