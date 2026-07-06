"""Runtime tests for tool-call loop guardrails."""

import json
import uuid
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from run_agent import AIAgent


def _make_tool_defs(*names: str) -> list[dict]:
    return [
        {
            "type": "function",
            "function": {
                "name": name,
                "description": f"{name} tool",
                "parameters": {"type": "object", "properties": {}},
            },
        }
        for name in names
    ]


def _mock_tool_call(name="web_search", arguments="{}", call_id=None):
    return SimpleNamespace(
        id=call_id or f"call_{uuid.uuid4().hex[:8]}",
        type="function",
        function=SimpleNamespace(name=name, arguments=arguments),
    )


def _mock_response(content="Hello", finish_reason="stop", tool_calls=None):
    msg = SimpleNamespace(content=content, tool_calls=tool_calls)
    choice = SimpleNamespace(message=msg, finish_reason=finish_reason)
    return SimpleNamespace(choices=[choice], model="test/model", usage=None)


def _make_agent(*tool_names: str, max_iterations: int = 10, config: dict | None = None) -> AIAgent:
    with (
        patch("run_agent.get_tool_definitions", return_value=_make_tool_defs(*tool_names)),
        patch("run_agent.check_toolset_requirements", return_value={}),
        patch("hermes_cli.config.load_config", return_value=config or {}),
        patch("run_agent.OpenAI"),
    ):
        agent = AIAgent(
            api_key="test-key-1234567890",
            base_url="https://openrouter.ai/api/v1",
            max_iterations=max_iterations,
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
        )
    agent.client = MagicMock()
    agent._cached_system_prompt = "You are helpful."
    agent._use_prompt_caching = False
    agent.tool_delay = 0
    agent.compression_enabled = False
    agent.save_trajectories = False
    return agent


def _seed_exact_failures(agent: AIAgent, tool_name: str, args: dict, count: int = 2) -> None:
    for _ in range(count):
        agent._tool_guardrails.after_call(
            tool_name,
            args,
            json.dumps({"error": "boom"}),
            failed=True,
        )


def _hard_stop_config(**overrides) -> dict:
    cfg = {
        "tool_loop_guardrails": {
            "warnings_enabled": True,
            "hard_stop_enabled": True,
            "hard_stop_after": {
                "exact_failure": 2,
                "same_tool_failure": 8,
                "idempotent_no_progress": 5,
            },
        }
    }
    cfg["tool_loop_guardrails"].update(overrides)
    return cfg


def test_default_sequential_path_warns_repeated_exact_failure_without_blocking_execution():
    agent = _make_agent("web_search")
    args = {"query": "same"}
    _seed_exact_failures(agent, "web_search", args)
    starts = []
    progress = []
    agent.tool_start_callback = lambda *a, **k: starts.append((a, k))
    agent.tool_progress_callback = lambda *a, **k: progress.append((a, k))
    tc = _mock_tool_call("web_search", json.dumps(args), "c-soft")
    msg = SimpleNamespace(content="", tool_calls=[tc])
    messages = []

    with patch("run_agent.handle_function_call", return_value=json.dumps({"error": "boom"})) as mock_hfc:
        agent._execute_tool_calls_sequential(msg, messages, "task-1")

    mock_hfc.assert_called_once()
    assert len(starts) == 1
    assert any(event[0][0] == "tool.completed" for event in progress)
    assert len(messages) == 1
    assert messages[0]["role"] == "tool"
    assert messages[0]["tool_call_id"] == "c-soft"
    assert "repeated_exact_failure_warning" in messages[0]["content"]
    assert "repeated_exact_failure_block" not in messages[0]["content"]
    assert agent._tool_guardrail_halt_decision is None


def test_config_enabled_hard_stop_blocks_repeated_exact_failure_before_execution():
    agent = _make_agent("web_search", config=_hard_stop_config())
    args = {"query": "same"}
    _seed_exact_failures(agent, "web_search", args)
    starts = []
    progress = []
    agent.tool_start_callback = lambda *a, **k: starts.append((a, k))
    agent.tool_progress_callback = lambda *a, **k: progress.append((a, k))
    tc = _mock_tool_call("web_search", json.dumps(args), "c-block")
    msg = SimpleNamespace(content="", tool_calls=[tc])
    messages = []

    with patch("run_agent.handle_function_call", return_value="SHOULD_NOT_RUN") as mock_hfc:
        agent._execute_tool_calls_sequential(msg, messages, "task-1")

    mock_hfc.assert_not_called()
    assert starts == []
    assert progress == []
    assert len(messages) == 1
    assert messages[0]["role"] == "tool"
    assert messages[0]["tool_call_id"] == "c-block"
    assert "repeated_exact_failure_block" in messages[0]["content"]


def test_sequential_after_call_appends_guidance_to_tool_result_without_extra_messages():
    agent = _make_agent("web_search")
    args = {"query": "same"}
    _seed_exact_failures(agent, "web_search", args, count=1)
    tc = _mock_tool_call("web_search", json.dumps(args), "c-warn")
    msg = SimpleNamespace(content="", tool_calls=[tc])
    messages = []

    with patch("run_agent.handle_function_call", return_value=json.dumps({"error": "boom"})):
        agent._execute_tool_calls_sequential(msg, messages, "task-1")

    assert [m["role"] for m in messages] == ["tool"]
    assert messages[0]["tool_call_id"] == "c-warn"
    assert "Tool loop warning" in messages[0]["content"]
    assert "repeated_exact_failure_warning" in messages[0]["content"]


def test_same_tool_failure_warning_tells_model_to_recover_with_tools():
    agent = _make_agent("terminal")
    guardrails = getattr(agent, "_tool_guardrails")
    guardrails.after_call(
        "terminal",
        {"command": "bad-1"},
        json.dumps({"exit_code": 1}),
        failed=True,
    )
    guardrails.after_call(
        "terminal",
        {"command": "bad-2"},
        json.dumps({"exit_code": 1}),
        failed=True,
    )
    tc = _mock_tool_call("terminal", json.dumps({"command": "bad-3"}), "c-recover")
    msg = SimpleNamespace(content="", tool_calls=[tc])
    messages = []

    with patch("run_agent.handle_function_call", return_value=json.dumps({"exit_code": 1})):
        agent._execute_tool_calls_sequential(msg, messages, "task-1")

    content = messages[0]["content"]
    assert "same_tool_failure_warning" in content
    assert "Do not switch to text-only replies" in content
    assert "keep using tools" in content
    assert "pwd && ls -la" in content
    assert "absolute path" in content
    assert "different tool" in content


def test_terminal_usage_error_redirect_blocks_same_family_until_help_probe():
    agent = _make_agent("terminal", max_iterations=10)
    responses = [
        _mock_response(
            content="",
            finish_reason="tool_calls",
            tool_calls=[
                _mock_tool_call(
                    "terminal",
                    json.dumps({"command": "python3 .mesh/tools/meshctl.py task view foo"}),
                    "c-bad-1",
                )
            ],
        ),
        _mock_response(
            content="",
            finish_reason="tool_calls",
            tool_calls=[
                _mock_tool_call(
                    "terminal",
                    json.dumps({"command": "meshctl task status foo --bad-flag"}),
                    "c-bad-2",
                )
            ],
        ),
        _mock_response(
            content="",
            finish_reason="tool_calls",
            tool_calls=[
                _mock_tool_call(
                    "terminal",
                    json.dumps({"command": "cd ~/Workspaces && meshctl task changes foo 2>&1 | head -20"}),
                    "c-bad-3",
                )
            ],
        ),
        _mock_response(
            content="",
            finish_reason="tool_calls",
            tool_calls=[
                _mock_tool_call(
                    "terminal",
                    json.dumps({"command": "meshctl task view foo --another-guess"}),
                    "c-blocked",
                )
            ],
        ),
        _mock_response(
            content="",
            finish_reason="tool_calls",
            tool_calls=[
                _mock_tool_call(
                    "terminal",
                    json.dumps({"command": "meshctl task --help"}),
                    "c-help",
                )
            ],
        ),
        _mock_response(content="done", finish_reason="stop", tool_calls=None),
    ]
    agent.client.chat.completions.create.side_effect = responses
    executed = []
    usage_error = json.dumps(
        {
            "exit_code": 2,
            "stderr": (
                "usage: meshctl task [-h] {list,show}\n"
                "meshctl task: error: argument command: invalid choice"
            ),
        }
    )

    def fake_handle(name, args, task_id, **kwargs):
        executed.append((name, args, kwargs.get("tool_call_id")))
        if kwargs.get("tool_call_id") == "c-help":
            return json.dumps({"exit_code": 0, "stdout": "usage: meshctl task [-h] {list,show}"})
        return usage_error

    with (
        patch("run_agent.handle_function_call", side_effect=fake_handle),
        patch.object(agent, "_persist_session"),
        patch.object(agent, "_save_trajectory"),
        patch.object(agent, "_cleanup_task_resources"),
    ):
        result = agent.run_conversation("use meshctl task")

    assert result["turn_exit_reason"].startswith("text_response")
    assert result["final_response"] == "done"
    assert "guardrail" not in result
    assert [call_id for _name, _args, call_id in executed] == [
        "c-bad-1",
        "c-bad-2",
        "c-bad-3",
        "c-help",
    ]
    tool_contents = [m["content"] for m in result["messages"] if m.get("role") == "tool"]
    assert any("terminal_usage_error_warning" in content for content in tool_contents)
    assert any("terminal_usage_error_redirect" in content for content in tool_contents)
    assert any("Stop guessing new argument variants" in content for content in tool_contents)
    assert any(
        '"code": "terminal_usage_error_redirect"' in content
        or '"code":"terminal_usage_error_redirect"' in content
        for content in tool_contents
    )


def test_config_enabled_hard_stop_concurrent_path_does_not_submit_blocked_calls_and_preserves_result_order():
    agent = _make_agent("web_search", config=_hard_stop_config())
    blocked_args = {"query": "blocked"}
    allowed_args = {"query": "allowed"}
    _seed_exact_failures(agent, "web_search", blocked_args)
    starts = []
    progress_events = []
    agent.tool_start_callback = lambda tool_call_id, name, args: starts.append((tool_call_id, name, args))
    agent.tool_progress_callback = lambda event, name, preview, args, **kw: progress_events.append((event, name, args, kw))
    calls = [
        _mock_tool_call("web_search", json.dumps(blocked_args), "c-block"),
        _mock_tool_call("web_search", json.dumps(allowed_args), "c-allow"),
    ]
    msg = SimpleNamespace(content="", tool_calls=calls)
    messages = []
    executed = []

    def fake_handle(name, args, task_id, **kwargs):
        executed.append((name, args, kwargs["tool_call_id"]))
        return json.dumps({"ok": args["query"]})

    with patch("run_agent.handle_function_call", side_effect=fake_handle):
        agent._execute_tool_calls_concurrent(msg, messages, "task-1")

    assert executed == [("web_search", allowed_args, "c-allow")]
    assert [m["tool_call_id"] for m in messages] == ["c-block", "c-allow"]
    assert "repeated_exact_failure_block" in messages[0]["content"]
    assert json.loads(messages[1]["content"]) == {"ok": "allowed"}
    assert starts == [("c-allow", "web_search", allowed_args)]
    started_events = [event for event in progress_events if event[0] == "tool.started"]
    completed_events = [event for event in progress_events if event[0] == "tool.completed"]
    assert started_events == [("tool.started", "web_search", allowed_args, {})]
    assert len(completed_events) == 1
    assert completed_events[0][1] == "web_search"


def test_plugin_pre_tool_block_wins_without_counting_as_toolguard_block():
    agent = _make_agent("web_search")
    args = {"query": "same"}
    tc = _mock_tool_call("web_search", json.dumps(args), "c-plugin")
    msg = SimpleNamespace(content="", tool_calls=[tc])
    messages = []

    with (
        patch("hermes_cli.plugins.resolve_pre_tool_block", return_value="plugin policy"),
        patch("run_agent.handle_function_call", return_value="SHOULD_NOT_RUN") as mock_hfc,
    ):
        agent._execute_tool_calls_sequential(msg, messages, "task-1")

    mock_hfc.assert_not_called()
    assert "plugin policy" in messages[0]["content"]
    assert agent._tool_guardrails.before_call("web_search", args).action == "allow"


def test_default_run_conversation_warns_without_guardrail_halt():
    agent = _make_agent("web_search", max_iterations=10)
    same_args = {"query": "same"}
    responses = [
        _mock_response(
            content="",
            finish_reason="tool_calls",
            tool_calls=[_mock_tool_call("web_search", json.dumps(same_args), f"c{i}")],
        )
        for i in range(1, 4)
    ]
    responses.append(_mock_response(content="done", finish_reason="stop", tool_calls=None))
    agent.client.chat.completions.create.side_effect = responses

    with (
        patch("run_agent.handle_function_call", return_value=json.dumps({"error": "boom"})) as mock_hfc,
        patch.object(agent, "_persist_session"),
        patch.object(agent, "_save_trajectory"),
        patch.object(agent, "_cleanup_task_resources"),
    ):
        result = agent.run_conversation("search repeatedly")

    assert mock_hfc.call_count == 3
    assert result["turn_exit_reason"].startswith("text_response")
    assert "guardrail" not in result
    assert result["final_response"] == "done"
    tool_contents = [m["content"] for m in result["messages"] if m.get("role") == "tool"]
    assert any("repeated_exact_failure_warning" in content for content in tool_contents)


def test_tool_reported_loop_block_run_conversation_halts_with_default_config():
    agent = _make_agent("search_files", max_iterations=10)
    args = {"pattern": "def.*drain", "path": "/repo", "target": "content"}
    blocked_result = json.dumps(
        {
            "error": (
                "BLOCKED: You have run this exact search 4 times in a row. "
                "The results have NOT changed."
            ),
            "already_searched": 4,
        }
    )
    agent.client.chat.completions.create.return_value = _mock_response(
        content="",
        finish_reason="tool_calls",
        tool_calls=[_mock_tool_call("search_files", json.dumps(args), "c-search")],
    )

    with (
        patch("run_agent.handle_function_call", return_value=blocked_result) as mock_hfc,
        patch.object(agent, "_persist_session"),
        patch.object(agent, "_save_trajectory"),
        patch.object(agent, "_cleanup_task_resources"),
    ):
        result = agent.run_conversation("find the drain implementation")

    mock_hfc.assert_called_once()
    assert result["turn_exit_reason"] == "guardrail_halt"
    assert result["api_calls"] == 1
    assert "stopped retrying search_files" in result["final_response"].lower()
    assert result["guardrail"]["code"] == "tool_reported_loop_block"
    tool_contents = [m["content"] for m in result["messages"] if m.get("role") == "tool"]
    assert any("Tool loop hard stop" in content for content in tool_contents)


def test_low_information_search_streak_redirects_same_tool_without_halting():
    agent = _make_agent("search_files", "terminal", max_iterations=10)
    search_patterns = ["def.*drain", "drain_command", "dispatch_command", "select.*task"]
    responses = [
        _mock_response(
            content="",
            finish_reason="tool_calls",
            tool_calls=[
                _mock_tool_call(
                    "search_files",
                    json.dumps({"pattern": pattern, "path": "/repo", "target": "content"}),
                    f"c-search-{i}",
                )
            ],
        )
        for i, pattern in enumerate(search_patterns, start=1)
    ]
    responses.extend(
        [
            _mock_response(
                content="",
                finish_reason="tool_calls",
                tool_calls=[
                    _mock_tool_call(
                        "search_files",
                        json.dumps(
                            {
                                "pattern": "another search variant",
                                "path": "/repo",
                                "target": "content",
                            }
                        ),
                        "c-redirected",
                    )
                ],
            ),
            _mock_response(
                content="",
                finish_reason="tool_calls",
                tool_calls=[
                    _mock_tool_call(
                        "terminal",
                        json.dumps({"command": "pwd && rg --files | head"}),
                        "c-terminal",
                    )
                ],
            ),
            _mock_response(content="done", finish_reason="stop", tool_calls=None),
        ]
    )
    agent.client.chat.completions.create.side_effect = responses
    executed = []

    def fake_handle(name, args, task_id, **kwargs):
        executed.append((name, args, kwargs.get("tool_call_id")))
        if name == "search_files":
            return json.dumps({"total_count": 0})
        return json.dumps({"exit_code": 0, "output": "/repo\nrun_agent.py"})

    with (
        patch("run_agent.handle_function_call", side_effect=fake_handle),
        patch.object(agent, "_persist_session"),
        patch.object(agent, "_save_trajectory"),
        patch.object(agent, "_cleanup_task_resources"),
    ):
        result = agent.run_conversation("find the dispatch implementation")

    assert result["turn_exit_reason"].startswith("text_response")
    assert result["final_response"] == "done"
    assert "guardrail" not in result
    assert [name for name, _args, _call_id in executed] == [
        "search_files",
        "search_files",
        "search_files",
        "search_files",
        "terminal",
    ]
    assert all(call_id != "c-redirected" for _name, _args, call_id in executed)
    tool_contents = [m["content"] for m in result["messages"] if m.get("role") == "tool"]
    assert any("low_information_tool_redirect" in content for content in tool_contents)
    assert any("low_information_strategy_warning" in content for content in tool_contents)


def test_low_information_terminal_redirects_without_controlled_halt_by_default():
    agent = _make_agent("terminal", max_iterations=10)
    responses = [
        _mock_response(
            content="",
            finish_reason="tool_calls",
            tool_calls=[
                _mock_tool_call(
                    "terminal",
                    json.dumps(
                        {
                            "command": (
                                "cd ~/Workspaces/Projects/meshboard && "
                                f"meshctl task list 2>&1 | grep 'open-{i}' | head -20"
                            )
                        }
                    ),
                    f"c-terminal-{i}",
                )
            ],
        )
        for i in range(1, 8)
    ]
    responses.append(_mock_response(content="done", finish_reason="stop", tool_calls=None))
    agent.client.chat.completions.create.side_effect = responses

    with (
        patch(
            "run_agent.handle_function_call",
            return_value=json.dumps({"exit_code": 0, "stdout": "", "stderr": ""}),
        ) as mock_hfc,
        patch.object(agent, "_persist_session"),
        patch.object(agent, "_save_trajectory"),
        patch.object(agent, "_cleanup_task_resources"),
    ):
        result = agent.run_conversation("find a MeshBoard task")

    assert mock_hfc.call_count == 4
    assert result["turn_exit_reason"].startswith("text_response")
    assert result["final_response"] == "done"
    assert "guardrail" not in result
    tool_contents = [m["content"] for m in result["messages"] if m.get("role") == "tool"]
    assert any("low_information_tool_redirect" in content for content in tool_contents)
    assert not any("low_information_tool_halt" in content for content in tool_contents)


def test_progress_outcome_canary_nudges_after_tool_rounds_without_work_outcome():
    agent = _make_agent(
        "terminal",
        max_iterations=10,
        config={"progress_outcome_canary": {"warn_after_tool_rounds": 2}},
    )
    responses = [
        _mock_response(
            content="",
            finish_reason="tool_calls",
            tool_calls=[
                _mock_tool_call(
                    "terminal",
                    json.dumps({"command": "pwd && ls -la"}),
                    "c-probe-1",
                )
            ],
        ),
        _mock_response(
            content="",
            finish_reason="tool_calls",
            tool_calls=[
                _mock_tool_call(
                    "terminal",
                    json.dumps({"command": "git status --short"}),
                    "c-probe-2",
                )
            ],
        ),
        _mock_response(content="done", finish_reason="stop", tool_calls=None),
    ]
    agent.client.chat.completions.create.side_effect = responses
    executed = []

    def fake_handle(name, args, task_id, **kwargs):
        executed.append(kwargs.get("tool_call_id"))
        return json.dumps({"exit_code": 0, "stdout": "some diagnostic output"})

    with (
        patch("run_agent.handle_function_call", side_effect=fake_handle),
        patch.object(agent, "_persist_session"),
        patch.object(agent, "_save_trajectory"),
        patch.object(agent, "_cleanup_task_resources"),
    ):
        result = agent.run_conversation("make progress")

    assert result["final_response"] == "done"
    assert executed == ["c-probe-1", "c-probe-2"]
    assert result["progress_outcome_canary"][0]["code"] == "progress_outcome_warning"
    synthetic_users = [
        msg for msg in result["messages"]
        if msg.get("role") == "user" and msg.get("_progress_outcome_canary")
    ]
    assert len(synthetic_users) == 1
    assert "without an observable work outcome" in synthetic_users[0]["content"]
    assert "make the edit" in synthetic_users[0]["content"]


def test_progress_outcome_canary_stays_quiet_after_landed_file_mutation():
    agent = _make_agent(
        "terminal",
        "write_file",
        max_iterations=10,
        config={"progress_outcome_canary": {"warn_after_tool_rounds": 2}},
    )
    responses = [
        _mock_response(
            content="",
            finish_reason="tool_calls",
            tool_calls=[
                _mock_tool_call(
                    "terminal",
                    json.dumps({"command": "pwd && ls -la"}),
                    "c-probe",
                )
            ],
        ),
        _mock_response(
            content="",
            finish_reason="tool_calls",
            tool_calls=[
                _mock_tool_call(
                    "write_file",
                    json.dumps({"path": "/tmp/canary.txt", "content": "ok"}),
                    "c-write",
                )
            ],
        ),
        _mock_response(
            content="",
            finish_reason="tool_calls",
            tool_calls=[
                _mock_tool_call(
                    "terminal",
                    json.dumps({"command": "git status --short"}),
                    "c-status",
                )
            ],
        ),
        _mock_response(content="done", finish_reason="stop", tool_calls=None),
    ]
    agent.client.chat.completions.create.side_effect = responses

    def fake_handle(name, args, task_id, **kwargs):
        if name == "write_file":
            return json.dumps({"bytes_written": 2})
        return json.dumps({"exit_code": 0, "stdout": "some diagnostic output"})

    with (
        patch("run_agent.handle_function_call", side_effect=fake_handle),
        patch.object(agent, "_persist_session"),
        patch.object(agent, "_save_trajectory"),
        patch.object(agent, "_cleanup_task_resources"),
    ):
        result = agent.run_conversation("make progress")

    assert result["final_response"] == "done"
    assert "progress_outcome_canary" not in result
    assert not any(
        msg.get("_progress_outcome_canary")
        for msg in result["messages"]
        if msg.get("role") == "user"
    )


def test_config_enabled_hard_stop_run_conversation_returns_controlled_guardrail_halt_without_top_level_error():
    agent = _make_agent("web_search", max_iterations=10, config=_hard_stop_config())
    same_args = {"query": "same"}
    responses = [
        _mock_response(
            content="",
            finish_reason="tool_calls",
            tool_calls=[_mock_tool_call("web_search", json.dumps(same_args), f"c{i}")],
        )
        for i in range(1, 10)
    ]
    agent.client.chat.completions.create.side_effect = responses

    with (
        patch("run_agent.handle_function_call", return_value=json.dumps({"error": "boom"})) as mock_hfc,
        patch.object(agent, "_persist_session"),
        patch.object(agent, "_save_trajectory"),
        patch.object(agent, "_cleanup_task_resources"),
    ):
        result = agent.run_conversation("search repeatedly")

    assert mock_hfc.call_count == 2
    assert result["api_calls"] == 3
    assert result["api_calls"] < agent.max_iterations
    assert result["turn_exit_reason"] == "guardrail_halt"
    assert "error" not in result
    assert result["completed"] is True
    assert "stopped retrying" in result["final_response"]
    assert result["guardrail"]["code"] == "repeated_exact_failure_block"
    assert result["guardrail"]["tool_name"] == "web_search"

    assistant_tool_calls = [m for m in result["messages"] if m.get("role") == "assistant" and m.get("tool_calls")]
    for assistant_msg in assistant_tool_calls:
        call_ids = [tc["id"] for tc in assistant_msg["tool_calls"]]
        following_results = [m for m in result["messages"] if m.get("role") == "tool" and m.get("tool_call_id") in call_ids]
        assert len(following_results) == len(call_ids)


def test_guardrail_halt_emits_final_response_through_stream_delta_callback():
    """Regression for #30770: when the guardrail halts the loop, the
    synthesized halt message must be pushed through ``stream_delta_callback``
    so SSE/TUI clients see why the agent stopped instead of a silent stream
    close.  Without this the chat-completions SSE writer drains an empty
    queue and emits a finish chunk with zero content (indistinguishable
    from a crash for Open WebUI and similar clients).
    """
    agent = _make_agent("web_search", max_iterations=10, config=_hard_stop_config())
    same_args = {"query": "same"}
    responses = [
        _mock_response(
            content="",
            finish_reason="tool_calls",
            tool_calls=[_mock_tool_call("web_search", json.dumps(same_args), f"c{i}")],
        )
        for i in range(1, 10)
    ]
    agent.client.chat.completions.create.side_effect = responses

    deltas: list = []
    agent.stream_delta_callback = lambda d: deltas.append(d)
    # The mocked client returns SimpleNamespace responses which aren't
    # iterable as streaming chunks; force the non-streaming code path so
    # the guardrail-halt branch is reached without engaging the real
    # streaming machinery.
    agent._disable_streaming = True

    with (
        patch("run_agent.handle_function_call", return_value=json.dumps({"error": "boom"})),
        patch.object(agent, "_persist_session"),
        patch.object(agent, "_save_trajectory"),
        patch.object(agent, "_cleanup_task_resources"),
    ):
        result = agent.run_conversation("search repeatedly")

    assert result["turn_exit_reason"] == "guardrail_halt"
    halt_text = result["final_response"]
    assert "stopped retrying" in halt_text

    # The halt message must have been pushed through the callback at least
    # once.  Empty-queue SSE writers were the bug — clients saw no content
    # delta before the finish chunk.
    text_deltas = [d for d in deltas if isinstance(d, str)]
    assert halt_text in text_deltas, (
        f"halt message was never streamed; callback only saw {deltas!r}"
    )
