"""Runtime integration tests for evidence-backed source-first routing."""

import json
import threading
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from agent.source_routing_strategy import (
    STRATEGY_NAME,
    _uses_native_route,
    routable_domain,
    select_for_turn,
)
from hermes_state import SessionDB
from run_agent import AIAgent


def _tool_def(name: str) -> dict:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": f"{name} tool",
            "parameters": {"type": "object", "properties": {}},
        },
    }


def _tool_message(name: str = "terminal", arguments=None) -> SimpleNamespace:
    call = SimpleNamespace(
        id="call-1",
        function=SimpleNamespace(name=name, arguments=json.dumps(arguments or {})),
    )
    return SimpleNamespace(content="", tool_calls=[call])


def _tool_message_many(calls) -> SimpleNamespace:
    tool_calls = [
        SimpleNamespace(
            id=f"call-{index}",
            function=SimpleNamespace(name=name, arguments=json.dumps(arguments)),
        )
        for index, (name, arguments) in enumerate(calls)
    ]
    return SimpleNamespace(content="", tool_calls=tool_calls)


def _response(content="", finish_reason="stop", tool_calls=None) -> SimpleNamespace:
    message = SimpleNamespace(content=content, tool_calls=tool_calls)
    choice = SimpleNamespace(message=message, finish_reason=finish_reason)
    return SimpleNamespace(choices=[choice], model="test/model", usage=None)


def _make_agent(
    db: SessionDB,
    session_id: str,
    *,
    ephemeral_system_prompt=None,
) -> AIAgent:
    with (
        patch(
            "run_agent.get_tool_definitions",
            return_value=[_tool_def("terminal"), _tool_def("web_search")],
        ),
        patch("run_agent.check_toolset_requirements", return_value={}),
        patch("run_agent.OpenAI"),
    ):
        agent = AIAgent(
            model="test-model",
            api_key="test-key",
            base_url="https://example.test/v1",
            provider="custom",
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
            session_db=db,
            session_id=session_id,
            ephemeral_system_prompt=ephemeral_system_prompt,
        )
    agent.client = MagicMock()
    return agent


def test_routable_domain_is_conservative():
    assert routable_domain("Check https://github.com/NousResearch/hermes-agent") == "github.com"
    assert routable_domain("Search for Hermes on GitHub") is None
    assert routable_domain([{"type": "text", "text": "https://github.com/x/y"}]) is None


@pytest.mark.parametrize(
    ("domain", "url", "tool_name", "tool_args", "expected"),
    [
        (
            "github.com",
            "https://github.com/NousResearch/hermes-agent/pull/16122",
            "terminal",
            {"command": "gh pr view 16122 --repo NousResearch/hermes-agent"},
            True,
        ),
        (
            "github.com",
            "https://github.com/NousResearch/hermes-agent/pull/16122",
            "mcp_github_get_pull_request",
            {
                "owner": "NousResearch",
                "repo": "hermes-agent",
                "pull_number": 16122,
            },
            True,
        ),
        (
            "github.com",
            "https://github.com/NousResearch/hermes-agent/pull/16122",
            "terminal",
            {"command": "git status"},
            False,
        ),
        (
            "x.com",
            "https://x.com/example/status/12345",
            "terminal",
            {"command": "xurl read https://x.com/example/status/12345"},
            True,
        ),
        (
            "x.com",
            "https://x.com/example/status/12345",
            "terminal",
            {"command": "echo xurl read 12345"},
            False,
        ),
        (
            "youtube.com",
            "https://youtube.com/watch?v=abc123",
            "terminal",
            {"command": "python skills/media/youtube-content/scripts/fetch_transcript.py https://youtube.com/watch?v=abc123"},
            True,
        ),
        (
            "youtube.com",
            "https://youtube.com/watch?v=abc123",
            "meet_transcript",
            {"url": "https://youtube.com/watch?v=abc123"},
            False,
        ),
    ],
)
def test_native_route_requires_matching_tool_and_target(
    domain, url, tool_name, tool_args, expected
):
    assert _uses_native_route(domain, url, tool_name, tool_args) is expected


@pytest.mark.parametrize("execution_mode", ["sequential", "concurrent"])
def test_runtime_tool_event_is_strategy_tagged(tmp_path, execution_mode):
    """Exercise both production tool executors against the real SessionDB."""
    db = SessionDB(db_path=tmp_path / f"{execution_mode}.db")
    session_id = f"strategy-{execution_mode}"
    agent = _make_agent(db, session_id)
    db.create_session(session_id=session_id, source="test")
    select_for_turn(agent, "Review https://github.com/NousResearch/hermes-agent/pull/16122")

    with patch("run_agent.handle_function_call", return_value="ok"):
        executor = getattr(agent, f"_execute_tool_calls_{execution_mode}")
        executor(
            _tool_message(
                arguments={
                    "command": "gh pr view 16122 --repo NousResearch/hermes-agent"
                }
            ),
            [],
            "task-1",
        )
        # A second executed tool in the same turn must not dilute the routing
        # decision or create another promotion sample.
        executor(
            _tool_message(arguments={"command": "git status"}),
            [],
            "task-1",
        )

    events = db.get_strategy_events(strategy=STRATEGY_NAME)
    assert len(events) == 1
    assert events[0]["event_type"] == "tool_call"
    assert events[0]["tool_name"] == "terminal"
    assert events[0]["result"] == "success"
    db.close()


def test_concurrent_timeout_is_first_tool_failure_not_later_success(
    tmp_path, monkeypatch
):
    db = SessionDB(db_path=tmp_path / "concurrent-timeout.db")
    agent = _make_agent(db, "strategy-concurrent-timeout")
    db.create_session(session_id=agent.session_id, source="test")
    select_for_turn(agent, "Review https://github.com/NousResearch/hermes-agent")
    monkeypatch.setenv("HERMES_CONCURRENT_TOOL_TIMEOUT_S", "0.05")
    blocker = threading.Event()

    def _dispatch(_name, args, _task_id, **_kwargs):
        if args.get("slow"):
            blocker.wait(2)
            return "late native result"
        return "fast native result"

    message = _tool_message_many([
        (
            "terminal",
            {
                "command": "gh repo view NousResearch/hermes-agent",
                "slow": True,
            },
        ),
        (
            "terminal",
            {"command": "gh repo view NousResearch/hermes-agent"},
        ),
    ])
    try:
        with patch("run_agent.handle_function_call", side_effect=_dispatch):
            agent._execute_tool_calls_concurrent(message, [], "task-1")
    finally:
        blocker.set()

    events = db.get_strategy_events(strategy=STRATEGY_NAME)
    assert len(events) == 1
    assert events[0]["tool_name"] == "terminal"
    assert events[0]["result"] == "failure"
    db.close()


def test_real_executor_evidence_promotes_and_guides_next_session(tmp_path, monkeypatch):
    """Register -> execute -> score -> promote -> inject next-session guidance."""
    db = SessionDB(db_path=tmp_path / "promotion.db")
    agent = _make_agent(db, "strategy-source")
    strategy = db.get_strategy(STRATEGY_NAME)
    assert strategy is not None
    assert strategy["state"] == "candidate"
    assert "# Candidate strategy under evaluation" in agent.ephemeral_system_prompt

    agent._cached_system_prompt = "You are helpful."
    agent._use_prompt_caching = False
    agent.tool_delay = 0
    agent.compression_enabled = False
    agent.save_trajectories = False
    call = _tool_message(
        arguments={"command": "gh pr view 16122 --repo NousResearch/hermes-agent"}
    ).tool_calls[0]
    runtime_snapshot = {
        "model": agent.model,
        "provider": agent.provider,
        "base_url": agent.base_url,
        "client_id": id(agent.client),
        "cached_system_prompt": agent._cached_system_prompt,
        "ephemeral_system_prompt": agent.ephemeral_system_prompt,
        "tools": json.dumps(agent.tools, sort_keys=True),
    }
    responses = iter([
        _response(finish_reason="tool_calls", tool_calls=[call]),
        _response(content="Done", finish_reason="stop"),
    ])
    api_runtime_snapshots = []

    def _api_response(*_args, **_kwargs):
        api_runtime_snapshots.append({
            "model": agent.model,
            "provider": agent.provider,
            "base_url": agent.base_url,
            "client_id": id(agent.client),
            "cached_system_prompt": agent._cached_system_prompt,
            "ephemeral_system_prompt": agent.ephemeral_system_prompt,
            "tools": json.dumps(agent.tools, sort_keys=True),
        })
        return next(responses)

    agent.client.chat.completions.create.side_effect = _api_response
    with (
        patch("run_agent.handle_function_call", return_value="native GitHub result"),
        patch.object(agent, "_save_trajectory"),
        patch.object(agent, "_cleanup_task_resources"),
        patch.object(
            agent,
            "_restore_primary_runtime",
            wraps=agent._restore_primary_runtime,
        ) as restore_primary,
    ):
        result = agent.run_conversation(
            "Fix https://github.com/NousResearch/hermes-agent/pull/16122"
        )
    assert result["final_response"] == "Done"
    assert restore_primary.call_count == 1
    assert len(api_runtime_snapshots) == 2
    assert api_runtime_snapshots[0] == runtime_snapshot
    assert api_runtime_snapshots[0] == api_runtime_snapshots[1]
    assert agent.model == runtime_snapshot["model"]
    assert agent.provider == runtime_snapshot["provider"]
    assert agent.base_url == runtime_snapshot["base_url"]
    assert id(agent.client) == runtime_snapshot["client_id"]
    assert agent._cached_system_prompt == runtime_snapshot["cached_system_prompt"]
    assert agent.ephemeral_system_prompt == runtime_snapshot["ephemeral_system_prompt"]
    assert json.dumps(agent.tools, sort_keys=True) == runtime_snapshot["tools"]

    monkeypatch.setattr(SessionDB, "PROMOTE_MIN_SAMPLES", 1)
    next_agent = _make_agent(db, "strategy-destination")
    promoted = db.get_strategy(STRATEGY_NAME)
    assert promoted["state"] == "promoted"
    assert promoted["sample_count"] == 1
    assert "# Evidence-backed strategies" in next_agent.ephemeral_system_prompt
    assert STRATEGY_NAME in next_agent.ephemeral_system_prompt

    child_agent = _make_agent(
        db,
        "strategy-child",
        ephemeral_system_prompt=next_agent.ephemeral_system_prompt,
    )
    assert child_agent.ephemeral_system_prompt.count("<strategy-guidance>") == 1
    assert child_agent.ephemeral_system_prompt.count(STRATEGY_NAME) == 1
    db.close()


def test_generic_search_records_routing_failure(tmp_path):
    db = SessionDB(db_path=tmp_path / "miss.db")
    agent = _make_agent(db, "strategy-miss")
    db.create_session(session_id="strategy-miss", source="test")
    select_for_turn(agent, "Read https://github.com/NousResearch/hermes-agent")

    with patch("run_agent.handle_function_call", return_value="search result"):
        agent._execute_tool_calls_sequential(_tool_message("web_search"), [], "task-1")

    event = db.get_strategy_events(strategy=STRATEGY_NAME)[0]
    assert event["result"] == "failure"
    assert '"routing_miss": true' in event["metadata_json"]
    db.close()


def test_wrong_non_search_tool_is_not_native_success(tmp_path):
    db = SessionDB(db_path=tmp_path / "wrong-route.db")
    agent = _make_agent(db, "strategy-wrong-route")
    db.create_session(session_id="strategy-wrong-route", source="test")
    select_for_turn(agent, "Read https://github.com/NousResearch/hermes-agent")

    with patch("run_agent.handle_function_call", return_value="unrelated result"):
        agent._execute_tool_calls_sequential(
            _tool_message("terminal", {"command": "echo hello"}), [], "task-1"
        )

    event = db.get_strategy_events(strategy=STRATEGY_NAME)[0]
    assert event["result"] == "failure"
    assert '"native_route": false' in event["metadata_json"]
    db.close()
