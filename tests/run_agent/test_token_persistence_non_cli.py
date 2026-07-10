from types import ModuleType, SimpleNamespace
from unittest.mock import MagicMock, patch
import json
import sys

from run_agent import AIAgent


def _mock_response(*, usage: dict, content: str = "done"):
    msg = SimpleNamespace(content=content, tool_calls=None)
    choice = SimpleNamespace(message=msg, finish_reason="stop")
    return SimpleNamespace(
        choices=[choice],
        model="test/model",
        usage=SimpleNamespace(**usage),
    )


def _make_agent(session_db, *, platform: str):
    with (
        patch("run_agent.get_tool_definitions", return_value=[]),
        patch("run_agent.check_toolset_requirements", return_value={}),
        patch("run_agent.OpenAI"),
    ):
        agent = AIAgent(
            api_key="test-key",
            base_url="https://openrouter.ai/api/v1",
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
            session_db=session_db,
            session_id=f"{platform}-session",
            platform=platform,
        )
    agent.client = MagicMock()
    agent.client.chat.completions.create.return_value = _mock_response(
        usage={
            "prompt_tokens": 11,
            "completion_tokens": 7,
            "total_tokens": 18,
        }
    )
    return agent


def test_run_conversation_persists_tokens_for_telegram_sessions():
    session_db = MagicMock()
    agent = _make_agent(session_db, platform="telegram")

    result = agent.run_conversation("hello")

    assert result["final_response"] == "done"
    session_db.update_token_counts.assert_called_once()
    assert session_db.update_token_counts.call_args.args[0] == "telegram-session"


def test_run_conversation_persists_tokens_for_cron_sessions():
    session_db = MagicMock()
    agent = _make_agent(session_db, platform="cron")

    result = agent.run_conversation("hello")

    assert result["final_response"] == "done"
    session_db.update_token_counts.assert_called_once()
    assert session_db.update_token_counts.call_args.args[0] == "cron-session"


def test_session_search_lazily_opens_db_when_entrypoint_did_not_pass_one(monkeypatch):
    sentinel_db = object()
    captured = {}

    class FakeSessionDB:
        def __new__(cls):
            return sentinel_db

    hermes_state = ModuleType("hermes_state")
    hermes_state.SessionDB = FakeSessionDB
    monkeypatch.setitem(sys.modules, "hermes_state", hermes_state)

    session_search_mod = ModuleType("tools.session_search_tool")

    def fake_session_search(**kwargs):
        captured.update(kwargs)
        return json.dumps({"success": True, "results": []})

    session_search_mod.session_search = fake_session_search
    monkeypatch.setitem(sys.modules, "tools.session_search_tool", session_search_mod)

    agent = _make_agent(None, platform="acp")
    result = json.loads(agent._invoke_tool("session_search", {"query": "Hermes"}, "task-id"))

    assert result["success"] is True
    assert captured["db"] is sentinel_db
    assert captured["query"] == "Hermes"
    assert agent._session_db is sentinel_db


def test_session_search_forwards_profile_in_runtime_helper_path(monkeypatch):
    """Path 1: agent_runtime_helpers._execute forwards profile to session_search()."""
    captured = {}

    session_search_mod = ModuleType("tools.session_search_tool")

    def fake_session_search(**kwargs):
        captured.update(kwargs)
        return json.dumps({"success": True, "results": []})

    session_search_mod.session_search = fake_session_search
    monkeypatch.setitem(sys.modules, "tools.session_search_tool", session_search_mod)

    agent = _make_agent(MagicMock(), platform="acp")
    result = json.loads(agent._invoke_tool(
        "session_search",
        {"query": "Hermes", "profile": "work"},
        "task-id",
    ))

    assert result["success"] is True
    assert captured["profile"] == "work"
    assert captured["query"] == "Hermes"


def test_session_search_forwards_profile_in_sequential_executor_path(monkeypatch):
    """Path 2: tool_executor._execute forwards profile to session_search()."""
    from agent.tool_executor import execute_tool_calls_sequential

    captured = {}

    session_search_mod = ModuleType("tools.session_search_tool")

    def fake_session_search(**kwargs):
        captured.update(kwargs)
        return json.dumps({"success": True, "results": []})

    session_search_mod.session_search = fake_session_search
    monkeypatch.setitem(sys.modules, "tools.session_search_tool", session_search_mod)

    agent = _make_agent(MagicMock(), platform="acp")
    # Ensure _session_db is set so _get_session_db_for_recall() doesn't return None
    agent._session_db = MagicMock()

    tool_call = SimpleNamespace(
        id="call_1",
        function=SimpleNamespace(
            name="session_search",
            arguments=json.dumps({"query": "test", "profile": "staging"}),
        ),
    )
    assistant_message = SimpleNamespace(tool_calls=[tool_call])
    messages = [{"role": "user", "content": "search sessions"}]

    execute_tool_calls_sequential(agent, assistant_message, messages, "task-id")

    assert captured["profile"] == "staging"
    assert captured["query"] == "test"
