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
