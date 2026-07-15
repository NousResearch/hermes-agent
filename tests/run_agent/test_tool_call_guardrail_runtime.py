"""Runtime tests for tool-call loop guardrails."""

import json
import logging
import uuid
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from agent.memory_manager import build_memory_context_block
from hermes_cli.plugins import get_pre_tool_call_block_message
from model_tools import _emit_post_tool_call_hook
import run_agent
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


def test_sequential_tool_callbacks_scrub_recall_blocks_but_execution_keeps_raw_args():
    agent = _make_agent("web_search")
    leaked = build_memory_context_block("operator-only peer card")
    args = {"query": leaked}
    starts = []
    completes = []
    progress = []
    executed = []
    agent.tool_start_callback = lambda tool_call_id, name, cb_args: starts.append(
        (tool_call_id, name, cb_args)
    )
    agent.tool_complete_callback = (
        lambda tool_call_id, name, cb_args, result: completes.append(
            (tool_call_id, name, cb_args, result)
        )
    )
    agent.tool_progress_callback = lambda event, name, preview, cb_args, **kw: progress.append(
        (event, name, preview, cb_args, kw)
    )
    tc = _mock_tool_call("web_search", json.dumps(args), "c-seq")
    msg = SimpleNamespace(content="", tool_calls=[tc])
    messages = []

    def fake_handle(name, raw_args, task_id, **kwargs):
        executed.append((name, raw_args, kwargs["tool_call_id"]))
        return json.dumps({"ok": True})

    with patch("run_agent.handle_function_call", side_effect=fake_handle):
        agent._execute_tool_calls_sequential(msg, messages, "task-1")

    assert executed == [("web_search", args, "c-seq")]
    assert starts == [("c-seq", "web_search", {"query": ""})]
    assert completes == [("c-seq", "web_search", {"query": ""}, '{"ok": true}')]
    started = [event for event in progress if event[0] == "tool.started"]
    assert started == [("tool.started", "web_search", None, {"query": ""}, {})]


def test_concurrent_tool_callbacks_scrub_recall_blocks_but_execution_keeps_raw_args():
    agent = _make_agent("web_search")
    leaked = build_memory_context_block("operator-only peer card")
    args = {"query": leaked}
    starts = []
    completes = []
    progress = []
    executed = []
    agent.tool_start_callback = lambda tool_call_id, name, cb_args: starts.append(
        (tool_call_id, name, cb_args)
    )
    agent.tool_complete_callback = (
        lambda tool_call_id, name, cb_args, result: completes.append(
            (tool_call_id, name, cb_args, result)
        )
    )
    agent.tool_progress_callback = lambda event, name, preview, cb_args, **kw: progress.append(
        (event, name, preview, cb_args, kw)
    )
    tc = _mock_tool_call("web_search", json.dumps(args), "c-con")
    msg = SimpleNamespace(content="", tool_calls=[tc])
    messages = []

    def fake_handle(name, raw_args, task_id, **kwargs):
        executed.append((name, raw_args, kwargs["tool_call_id"]))
        return json.dumps({"ok": True})

    with patch("run_agent.handle_function_call", side_effect=fake_handle):
        agent._execute_tool_calls_concurrent(msg, messages, "task-1")

    assert executed == [("web_search", args, "c-con")]
    assert starts == [("c-con", "web_search", {"query": ""})]
    assert completes == [("c-con", "web_search", {"query": ""}, '{"ok": true}')]
    started = [event for event in progress if event[0] == "tool.started"]
    assert started == [("tool.started", "web_search", None, {"query": ""}, {})]


def test_sequential_tool_results_scrub_recall_blocks_before_callbacks_and_messages():
    agent = _make_agent("web_search")
    leaked = build_memory_context_block("operator-only peer card")
    completes = []
    progress = []
    tc = _mock_tool_call("web_search", json.dumps({"query": "safe"}), "c-seq-result")
    msg = SimpleNamespace(content="", tool_calls=[tc])
    messages = []
    agent.tool_complete_callback = (
        lambda tool_call_id, name, cb_args, result: completes.append(
            (tool_call_id, name, cb_args, result)
        )
    )
    agent.tool_progress_callback = lambda event, name, preview, cb_args, **kw: progress.append(
        (event, name, preview, cb_args, kw)
    )

    with patch("run_agent.handle_function_call", return_value=leaked):
        agent._execute_tool_calls_sequential(msg, messages, "task-1")

    completed = [event for event in progress if event[0] == "tool.completed"]
    assert len(completed) == 1
    assert completed[0][0:4] == ("tool.completed", "web_search", None, None)
    assert completed[0][4]["is_error"] is False
    assert completed[0][4]["result"] == ""
    assert completes == [("c-seq-result", "web_search", {"query": "safe"}, "")]
    assert "operator-only peer card" not in messages[0]["content"]


def test_concurrent_tool_results_scrub_recall_blocks_before_callbacks_and_messages():
    agent = _make_agent("web_search")
    leaked = build_memory_context_block("operator-only peer card")
    completes = []
    progress = []
    tc = _mock_tool_call("web_search", json.dumps({"query": "safe"}), "c-con-result")
    msg = SimpleNamespace(content="", tool_calls=[tc])
    messages = []
    agent.tool_complete_callback = (
        lambda tool_call_id, name, cb_args, result: completes.append(
            (tool_call_id, name, cb_args, result)
        )
    )
    agent.tool_progress_callback = lambda event, name, preview, cb_args, **kw: progress.append(
        (event, name, preview, cb_args, kw)
    )

    with patch("run_agent.handle_function_call", return_value=leaked):
        agent._execute_tool_calls_concurrent(msg, messages, "task-1")

    completed = [event for event in progress if event[0] == "tool.completed"]
    assert len(completed) == 1
    assert completed[0][0:4] == ("tool.completed", "web_search", None, None)
    assert completed[0][4]["is_error"] is False
    assert completed[0][4]["result"] == ""
    assert completes == [("c-con-result", "web_search", {"query": "safe"}, "")]
    assert "operator-only peer card" not in messages[0]["content"]


def test_concurrent_tool_error_logs_scrub_recall_blocks(caplog):
    agent = _make_agent("web_search")
    leaked = build_memory_context_block("operator-only peer card")
    tc = _mock_tool_call("web_search", json.dumps({"query": "safe"}), "c-con-log")
    msg = SimpleNamespace(content="", tool_calls=[tc])
    messages = []

    with (
        caplog.at_level(logging.INFO, logger="agent.tool_executor"),
        patch("run_agent.handle_function_call", return_value=json.dumps({"error": leaked})),
    ):
        agent._execute_tool_calls_concurrent(msg, messages, "task-1")

    assert "operator-only peer card" not in caplog.text
    assert "operator-only peer card" not in messages[0]["content"]


def test_sequential_tool_error_logs_scrub_recall_blocks(caplog):
    agent = _make_agent("web_search")
    leaked = build_memory_context_block("operator-only peer card")
    tc = _mock_tool_call("web_search", json.dumps({"query": "safe"}), "c-seq-log")
    msg = SimpleNamespace(content="", tool_calls=[tc])
    messages = []

    with (
        caplog.at_level(logging.ERROR, logger="agent.tool_executor"),
        patch(
            "run_agent.handle_function_call",
            side_effect=RuntimeError(leaked),
        ),
    ):
        agent._execute_tool_calls_sequential(msg, messages, "task-1")

    assert "operator-only peer card" not in caplog.text
    assert "operator-only peer card" not in messages[0]["content"]


def test_context_engine_tool_error_logs_scrub_recall_blocks(caplog):
    agent = _make_agent("lcm_grep")
    leaked = build_memory_context_block("operator-only peer card")
    agent._context_engine_tool_names = {"lcm_grep"}
    agent.context_compressor = SimpleNamespace(
        handle_tool_call=MagicMock(side_effect=RuntimeError(leaked))
    )
    tc = _mock_tool_call("lcm_grep", json.dumps({"query": "safe"}), "c-ce-log")
    msg = SimpleNamespace(content="", tool_calls=[tc])
    messages = []

    with caplog.at_level(logging.ERROR, logger="agent.tool_executor"):
        agent._execute_tool_calls_sequential(msg, messages, "task-1")

    assert "operator-only peer card" not in caplog.text
    assert "operator-only peer card" not in messages[0]["content"]


def test_memory_provider_tool_error_logs_scrub_recall_blocks(caplog):
    agent = _make_agent("honcho_search")
    leaked = build_memory_context_block("operator-only peer card")
    memory_manager = MagicMock()
    memory_manager.has_tool.return_value = True
    memory_manager.handle_tool_call.side_effect = RuntimeError(leaked)
    agent._memory_manager = memory_manager
    tc = _mock_tool_call("honcho_search", json.dumps({"query": "safe"}), "c-mem-log")
    msg = SimpleNamespace(content="", tool_calls=[tc])
    messages = []

    with caplog.at_level(logging.ERROR, logger="agent.tool_executor"):
        agent._execute_tool_calls_sequential(msg, messages, "task-1")

    assert "operator-only peer card" not in caplog.text
    assert "operator-only peer card" not in messages[0]["content"]


def test_sequential_multimodal_tool_result_preview_does_not_crash():
    agent = _make_agent("web_search")
    agent.quiet_mode = False
    agent.verbose_logging = False
    multimodal_result = {
        "_multimodal": True,
        "content": [{"type": "text", "text": "Visible result"}],
    }
    tc = _mock_tool_call("web_search", json.dumps({"query": "safe"}), "c-seq-mm")
    msg = SimpleNamespace(content="", tool_calls=[tc])
    messages = []

    with (
        patch("run_agent.handle_function_call", return_value=multimodal_result),
        patch.object(agent, "_append_guardrail_observation", lambda *args, **kwargs: multimodal_result),
    ):
        agent._execute_tool_calls_sequential(msg, messages, "task-1")

    assert messages


def test_sequential_quiet_todo_message_scrubs_recall_blocks(monkeypatch):
    agent = _make_agent("todo")
    leaked = build_memory_context_block("operator-only peer card")
    tc = _mock_tool_call("todo", json.dumps({"todos": []}), "c-todo-quiet")
    msg = SimpleNamespace(content="", tool_calls=[tc])
    messages = []
    printed = []
    agent._vprint = printed.append
    agent._should_emit_quiet_tool_messages = lambda: True

    with patch("tools.todo_tool.todo_tool", return_value=leaked):
        agent._execute_tool_calls_sequential(msg, messages, "task-1")

    assert printed
    assert all("operator-only peer card" not in line for line in printed)


def test_sequential_quiet_generic_message_scrubs_recall_blocks():
    agent = _make_agent("web_search")
    leaked = build_memory_context_block("operator-only peer card")
    tc = _mock_tool_call("web_search", json.dumps({"query": leaked}), "c-generic-quiet")
    msg = SimpleNamespace(content="", tool_calls=[tc])
    messages = []
    printed = []
    agent._vprint = printed.append
    agent._should_emit_quiet_tool_messages = lambda: True

    with patch("run_agent.handle_function_call", return_value=leaked):
        agent._execute_tool_calls_sequential(msg, messages, "task-1")

    assert printed
    assert all("operator-only peer card" not in line for line in printed)


def test_file_mutation_verifier_footer_scrubs_recalled_error_previews():
    agent = _make_agent("write_file")
    leaked = build_memory_context_block("operator-only peer card")
    tc = _mock_tool_call(
        "write_file",
        json.dumps({"path": "test.txt", "content": "hello"}),
        "c-write-leak",
    )
    msg = SimpleNamespace(content="", tool_calls=[tc])
    messages = []
    recorded = []

    agent._record_file_mutation_result = (
        lambda function_name, function_args, function_result, is_error: recorded.append(
            (function_name, function_args, function_result, is_error)
        )
    )

    with patch("run_agent.handle_function_call", return_value=json.dumps({"error": leaked})):
        agent._execute_tool_calls_sequential(msg, messages, "task-1")

    assert recorded
    assert "operator-only peer card" not in recorded[0][2]


def test_pre_tool_hook_receives_scrubbed_args(monkeypatch):
    captured = []
    leaked = build_memory_context_block("operator-only peer card")

    monkeypatch.setattr(
        "hermes_cli.plugins.invoke_hook",
        lambda hook_name, **kwargs: captured.append((hook_name, kwargs)) or [],
    )

    block = get_pre_tool_call_block_message("web_search", {"query": leaked})

    assert block is None
    assert captured[0][0] == "pre_tool_call"
    assert captured[0][1]["args"] == {"query": ""}


def test_post_tool_hook_receives_scrubbed_args(monkeypatch):
    captured = []
    leaked = build_memory_context_block("operator-only peer card")

    monkeypatch.setattr("hermes_cli.plugins.has_hook", lambda name: name == "post_tool_call")
    monkeypatch.setattr(
        "hermes_cli.plugins.invoke_hook",
        lambda hook_name, **kwargs: captured.append((hook_name, kwargs)) or [],
    )

    _emit_post_tool_call_hook(
        function_name="web_search",
        function_args={"query": leaked},
        result=leaked,
    )

    assert captured[0][0] == "post_tool_call"
    assert captured[0][1]["args"] == {"query": ""}
    assert captured[0][1]["result"] == ""
    assert captured[0][1]["error_message"] is None


def test_transform_tool_result_hook_receives_scrubbed_args_and_result(monkeypatch):
    captured = []
    leaked = build_memory_context_block("operator-only peer card")

    monkeypatch.setattr(
        "hermes_cli.plugins.has_hook",
        lambda name: name == "transform_tool_result",
    )
    monkeypatch.setattr(
        "hermes_cli.plugins.invoke_hook",
        lambda hook_name, **kwargs: captured.append((hook_name, kwargs)) or [],
    )

    with patch("model_tools.registry.dispatch", return_value=leaked):
        result = run_agent.handle_function_call(
            "web_search",
            {"query": "safe"},
            "task-1",
            skip_pre_tool_call_hook=True,
            skip_tool_request_middleware=True,
        )

    assert result == leaked
    assert captured[0][0] == "transform_tool_result"
    assert captured[0][1]["args"] == {"query": "safe"}
    assert captured[0][1]["result"] == ""
    assert captured[0][1]["error_message"] is None


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
