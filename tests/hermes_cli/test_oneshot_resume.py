"""Tests for `hermes -z --resume` session chaining (hermes_cli.oneshot).

Oneshot historically ignored --resume entirely: every -z call built a fresh
AIAgent with a fresh session, so scripted callers (the Smith Crafts OS
gateway, cron workers) could never chain turns. These tests pin the fixed
contract:

  - --resume <existing id> loads the prior transcript as conversation_history
    and pins the agent to the SAME session id (walking compression chains).
  - --resume <unknown id> is create-on-first-use: no error, no history, the
    id is used as-is so the caller can mint stable ids up front.
  - A broken session store degrades to a stateless turn, never a failure.
"""

from unittest.mock import MagicMock, patch

import pytest

from hermes_cli.oneshot import _load_resume_history


class TestLoadResumeHistory:
    def test_no_resume_returns_none(self):
        assert _load_resume_history(MagicMock(), "") == (None, None)
        assert _load_resume_history(MagicMock(), None) == (None, None)

    def test_no_db_returns_id_stateless(self):
        sid, hist = _load_resume_history(None, "abc123")
        assert sid == "abc123"
        assert hist is None

    def test_existing_session_loads_history_and_resolves_chain(self):
        db = MagicMock()
        db.resolve_resume_session_id.return_value = "tip_id"
        db.get_messages_as_conversation.return_value = [
            {"role": "session_meta", "content": "meta"},
            {"role": "user", "content": "remember ZEBRA"},
            {"role": "assistant", "content": "OK"},
        ]
        sid, hist = _load_resume_history(db, "orig_id")
        assert sid == "tip_id"
        # session_meta rows are dropped, real turns are kept in order.
        assert hist == [
            {"role": "user", "content": "remember ZEBRA"},
            {"role": "assistant", "content": "OK"},
        ]
        db.get_messages_as_conversation.assert_called_once_with("tip_id")
        db.reopen_session.assert_called_once_with("tip_id")

    def test_unknown_id_creates_on_first_use(self):
        db = MagicMock()
        db.resolve_resume_session_id.side_effect = lambda s: s
        db.get_messages_as_conversation.return_value = []
        sid, hist = _load_resume_history(db, "brand_new_id")
        assert sid == "brand_new_id"
        assert hist is None

    def test_broken_store_degrades_to_stateless(self):
        db = MagicMock()
        db.resolve_resume_session_id.side_effect = RuntimeError("db locked")
        db.get_messages_as_conversation.side_effect = RuntimeError("db locked")
        db.reopen_session.side_effect = RuntimeError("db locked")
        sid, hist = _load_resume_history(db, "sid")
        assert sid == "sid"
        assert hist is None


class TestRunAgentResumeWiring:
    def _run(self, resume, load_result, monkeypatch):
        monkeypatch.delenv("HERMES_INFERENCE_MODEL", raising=False)
        monkeypatch.delenv("HERMES_INFERENCE_PROVIDER", raising=False)
        agent = MagicMock()
        agent.run_conversation.return_value = {"final_response": "PONG"}
        agent_cls = MagicMock(return_value=agent)
        with (
            patch("hermes_cli.oneshot._create_session_db_for_oneshot", return_value=MagicMock()),
            patch("hermes_cli.oneshot._load_resume_history", return_value=load_result),
            patch("hermes_cli.oneshot.get_fallback_chain", return_value=None),
            patch("hermes_cli.config.load_config", return_value={"model": {"default": "m1", "provider": "p1"}}),
            patch("hermes_cli.runtime_provider.resolve_runtime_provider", return_value={}),
            patch("hermes_cli.tools_config._get_platform_tools", return_value=set()),
            patch("run_agent.AIAgent", agent_cls),
        ):
            from hermes_cli.oneshot import _run_agent

            response, _result = _run_agent("hi", resume=resume)
        return agent_cls, agent, response

    def test_resume_pins_session_id_and_seeds_history(self, monkeypatch):
        history = [{"role": "user", "content": "remember ZEBRA"}]
        agent_cls, agent, response = self._run("sid1", ("sid1", history), monkeypatch)
        assert agent_cls.call_args.kwargs["session_id"] == "sid1"
        agent.run_conversation.assert_called_once_with("hi", conversation_history=history)
        assert response == "PONG"

    def test_no_resume_keeps_agent_generated_session(self, monkeypatch):
        agent_cls, agent, _ = self._run(None, (None, None), monkeypatch)
        assert agent_cls.call_args.kwargs["session_id"] is None
        agent.run_conversation.assert_called_once_with("hi", conversation_history=None)


class TestOneshotResumeIntegration:
    """Real-SQLite regression for the concern the mock-only unit tests above
    can't catch: does a genuine ``SessionDB`` on disk actually round-trip a
    resumed transcript across two separate ``_run_agent`` calls, including a
    reload through a brand-new ``SessionDB`` instance (the same shape a
    second `hermes -z` process invocation would see)?

    ``AIAgent`` is still replaced (no live LLM call — that boundary is
    correctly out of scope here) but the fake performs the SAME real
    ``session_db.create_session`` / ``append_messages_batch`` writes the
    real agent's turn-boundary flush performs, so ``_load_resume_history``
    is exercised against actual rows, not a mock's return value.
    """

    class _RecordingFakeAgent:
        """Stands in for AIAgent: records session_id/history it was given
        and performs a REAL turn-boundary flush to the real session_db,
        exactly as ``run_agent.AIAgent.run_conversation`` does on a genuine
        turn (create_session is idempotent; append_messages_batch writes the
        user+assistant pair in one transaction)."""

        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self.session_id = kwargs["session_id"]
            self.session_db = kwargs["session_db"]
            self.suppress_status_output = False
            self.stream_delta_callback = None
            self.tool_gen_callback = None

        def run_conversation(self, prompt, conversation_history=None):
            self.received_history = conversation_history
            session_id = self.session_id or f"generated_{id(self)}"
            self.session_db.create_session(session_id, source="cli")
            reply = f"ack:{prompt}"
            self.session_db.append_messages_batch(
                session_id,
                [
                    {"role": "user", "content": prompt},
                    {"role": "assistant", "content": reply},
                ],
            )
            return {"final_response": reply, "session_id": session_id}

    def _run_turn(self, prompt, resume, db_path, monkeypatch):
        from hermes_state import SessionDB

        real_db = SessionDB(db_path=db_path)
        with (
            patch("hermes_cli.oneshot._create_session_db_for_oneshot", return_value=real_db),
            patch("hermes_cli.oneshot.get_fallback_chain", return_value=None),
            patch("hermes_cli.config.load_config", return_value={"model": {"default": "m1", "provider": "p1"}}),
            patch("hermes_cli.runtime_provider.resolve_runtime_provider", return_value={}),
            patch("hermes_cli.tools_config._get_platform_tools", return_value=set()),
            patch("run_agent.AIAgent", self._RecordingFakeAgent),
        ):
            from hermes_cli.oneshot import _run_agent

            response, result = _run_agent(prompt, resume=resume)
        real_db.close()
        return response, result

    @staticmethod
    def _role_content(history):
        """Real rows carry extra bookkeeping (timestamp, etc.) that a mock
        never would — compare on the fields the caller/model actually see."""
        return [{"role": m["role"], "content": m["content"]} for m in history]

    def test_two_turns_round_trip_through_real_sqlite(self, tmp_path, monkeypatch):
        db_path = tmp_path / "state.db"
        sid = "caller_minted_abc123"

        # Turn 1: id has never been seen — create-on-first-use, no prior
        # history to seed.
        response1, _ = self._run_turn("remember ZEBRA", sid, db_path, monkeypatch)
        assert response1 == "ack:remember ZEBRA"

        # Turn 2: same id — must load turn 1's real, disk-persisted
        # transcript as conversation_history, not a fresh/empty one.
        from hermes_state import SessionDB

        probe_db = SessionDB(db_path=db_path)
        seeded_id, seeded_history = _load_resume_history(probe_db, sid)
        probe_db.close()
        assert seeded_id == sid
        assert self._role_content(seeded_history) == [
            {"role": "user", "content": "remember ZEBRA"},
            {"role": "assistant", "content": "ack:remember ZEBRA"},
        ]

        response2, _ = self._run_turn("what did I say?", sid, db_path, monkeypatch)
        assert response2 == "ack:what did I say?"

        # Reload via a BRAND NEW SessionDB instance — the same shape a
        # second `hermes -z` process invocation would see, not a cached
        # Python object from this test. Assert both turns are present, in
        # order, with no duplicate rows from the resume-and-reflush cycle.
        reload_db = SessionDB(db_path=db_path)
        final_id, final_history = _load_resume_history(reload_db, sid)
        session_row = reload_db.get_session(final_id)
        reload_db.close()

        assert final_id == sid
        assert self._role_content(final_history) == [
            {"role": "user", "content": "remember ZEBRA"},
            {"role": "assistant", "content": "ack:remember ZEBRA"},
            {"role": "user", "content": "what did I say?"},
            {"role": "assistant", "content": "ack:what did I say?"},
        ]
        # 4 real rows, not 8 — turn 1's messages were loaded as context and
        # skipped by the flush's identity-based dedup, never re-appended.
        assert session_row is not None
        assert session_row["message_count"] == 4

    def test_unknown_resume_id_is_created_fresh_on_disk(self, tmp_path, monkeypatch):
        db_path = tmp_path / "state.db"
        sid = "brand_new_never_seen"

        response, _ = self._run_turn("first ever message", sid, db_path, monkeypatch)
        assert response == "ack:first ever message"

        from hermes_state import SessionDB

        reload_db = SessionDB(db_path=db_path)
        session_row = reload_db.get_session(sid)
        reload_db.close()
        assert session_row is not None
        assert session_row["message_count"] == 2
