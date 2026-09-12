"""Runtime tests for tool-call loop guardrails."""

import json
import uuid
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from run_agent import AIAgent
from tools import mcp_tool_handlers as mcp_handlers


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


def _make_agent(
    *tool_names: str,
    max_iterations: int = 10,
    config: dict | None = None,
    platform: str | None = None,
) -> AIAgent:
    with (
        patch("model_tools.get_tool_definitions", return_value=_make_tool_defs(*tool_names)),
        patch("model_tools.check_toolset_requirements", return_value={}),
        patch("hermes_cli.config.load_config", return_value=config or {}),
        patch("hermes_cli.config.load_config_readonly", return_value=config or {}),
        patch("agent.process_bootstrap.OpenAI"),
    ):
        agent = AIAgent(
            api_key="test-key-1234567890",
            base_url="https://openrouter.ai/api/v1",
            max_iterations=max_iterations,
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
            platform=platform or "cli",
        )
    agent.client = MagicMock()
    agent._cached_system_prompt = "You are helpful."
    agent._use_prompt_caching = False
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


def test_gateway_platform_uses_hard_stop_default_without_cli_opt_in():
    agent = _make_agent("web_search", platform="telegram")
    args = {"query": "same"}

    _seed_exact_failures(agent, "web_search", args, count=5)

    decision = getattr(agent, "_tool_guardrails").before_call("web_search", args)
    assert decision.action == "block"
    assert decision.code == "repeated_exact_failure_block"


@pytest.mark.parametrize("platform", ["desktop", "acp"])
def test_interactive_platforms_keep_warning_only_default(platform):
    agent = _make_agent("web_search", platform=platform)
    args = {"query": "same"}

    _seed_exact_failures(agent, "web_search", args, count=5)

    decision = getattr(agent, "_tool_guardrails").before_call("web_search", args)
    assert decision.action == "allow"
    assert decision.code == "allow"


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

    with patch("model_tools.handle_function_call", return_value=json.dumps({"error": "boom"})) as mock_hfc:
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

    with patch("model_tools.handle_function_call", return_value="SHOULD_NOT_RUN") as mock_hfc:
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

    with patch("model_tools.handle_function_call", return_value=json.dumps({"error": "boom"})):
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

    with patch("model_tools.handle_function_call", return_value=json.dumps({"exit_code": 1})):
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

    with patch("model_tools.handle_function_call", side_effect=fake_handle):
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


def test_relay_rewrite_precedes_sequential_policy_approval_checkpoint_and_dispatch():
    agent = _make_agent("write_file")
    original_args = {"path": "/original/path", "content": "old"}
    final_args = {"path": "/approved/path", "content": "new"}
    tc = _mock_tool_call("write_file", json.dumps(original_args), "c-rewrite")
    msg = SimpleNamespace(content="", tool_calls=[tc])
    messages = []
    observed = {
        "plugin": [],
        "guardrail": [],
        "approval": [],
        "checkpoint": [],
        "start": [],
        "dispatch": [],
    }

    original_before_call = agent._tool_guardrails.before_call

    def observe_guardrail(name, args):
        observed["guardrail"].append((name, dict(args)))
        return original_before_call(name, args)

    def relay_execute(name, args, callback, **kwargs):
        del name, args, kwargs
        return callback(dict(final_args)), dict(final_args)

    def observe_plugin(name, args, **kwargs):
        del kwargs
        observed["plugin"].append((name, dict(args)))
        return (None, None)

    def observe_approval(name, args):
        observed["approval"].append((name, dict(args)))
        return None

    def dispatch(name, args, task_id, **kwargs):
        del task_id, kwargs
        observed["dispatch"].append((name, dict(args)))
        return json.dumps({"ok": True})

    agent._checkpoint_mgr = SimpleNamespace(
        enabled=True,
        get_working_dir_for_path=lambda path: path,
        ensure_checkpoint=lambda path, reason: observed["checkpoint"].append(
            (path, reason)
        ),
    )
    agent.tool_start_callback = lambda _call_id, name, args: observed["start"].append(
        (name, dict(args))
    )

    with (
        patch("agent.relay_tools.execute", side_effect=relay_execute),
        patch(
            "hermes_cli.plugins._dispatch_pre_tool_call_hooks",
            side_effect=observe_plugin,
        ),
        patch.object(agent._tool_guardrails, "before_call", side_effect=observe_guardrail),
        patch(
            "acp_adapter.edit_approval.maybe_require_edit_approval",
            side_effect=observe_approval,
        ),
        patch("model_tools.registry.dispatch", side_effect=dispatch),
    ):
        agent._execute_tool_calls_sequential(msg, messages, "task-1")

    expected = [("write_file", final_args)]
    assert observed["plugin"] == expected
    assert observed["guardrail"] == expected
    assert observed["approval"] == expected
    assert observed["start"] == expected
    assert observed["dispatch"] == expected
    assert observed["checkpoint"] == [
        ("/approved/path", "before write_file")
    ]


def test_relay_rewrite_is_guarded_before_dispatch_in_concurrent_path():
    agent = _make_agent("web_search", config=_hard_stop_config())
    original_args = {"query": "original"}
    blocked_args = {"query": "blocked"}
    _seed_exact_failures(agent, "web_search", blocked_args)
    tc = _mock_tool_call("web_search", json.dumps(original_args), "c-rewrite-block")
    msg = SimpleNamespace(content="", tool_calls=[tc])
    messages = []
    starts = []

    def relay_execute(name, args, callback, **kwargs):
        del name, args, kwargs
        return callback(dict(blocked_args)), dict(blocked_args)

    agent.tool_start_callback = lambda *args: starts.append(args)
    with (
        patch("agent.relay_tools.execute", side_effect=relay_execute),
        patch("model_tools.handle_function_call", return_value="SHOULD_NOT_RUN") as dispatch,
    ):
        agent._execute_tool_calls_concurrent(msg, messages, "task-1")

    dispatch.assert_not_called()
    assert starts == []
    assert "repeated_exact_failure_block" in messages[0]["content"]


def test_plugin_pre_tool_block_wins_without_counting_as_toolguard_block():
    agent = _make_agent("web_search")
    args = {"query": "same"}
    tc = _mock_tool_call("web_search", json.dumps(args), "c-plugin")
    msg = SimpleNamespace(content="", tool_calls=[tc])
    messages = []

    with (
        patch(
            "hermes_cli.plugins._dispatch_pre_tool_call_hooks",
            return_value=("plugin policy", None),
        ),
        patch("model_tools.handle_function_call", return_value="SHOULD_NOT_RUN") as mock_hfc,
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
        patch("model_tools.handle_function_call", return_value=json.dumps({"error": "boom"})) as mock_hfc,
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


def test_mcp_runtime_stop_prevents_another_model_call():
    agent = _make_agent("mcp_fleet_claim", max_iterations=10)
    agent.client.chat.completions.create.side_effect = [
        _mock_response(
            content="",
            finish_reason="tool_calls",
            tool_calls=[_mock_tool_call("mcp_fleet_claim", "{}", "claim-1")],
        ),
        AssertionError("runtime stop must prevent another model call"),
    ]

    def dispatch(*_args, **_kwargs):
        mcp_handlers._mcp_runtime_stop.set({"reason": "max_items"})
        return '{"result": "done"}'

    with (
        patch("model_tools.handle_function_call", side_effect=dispatch),
        patch.object(agent, "_persist_session"),
        patch.object(agent, "_save_trajectory"),
        patch.object(agent, "_cleanup_task_resources"),
    ):
        result = agent.run_conversation("claim work")

    assert agent.client.chat.completions.create.call_count == 1
    assert result["turn_exit_reason"] == "runtime_stop(max_items)"


def test_authoritative_result_failure_prevents_another_model_call():
    agent = _make_agent("mcp_fleet_claim", max_iterations=10)
    agent.client.chat.completions.create.side_effect = [
        _mock_response(
            content="", finish_reason="tool_calls",
            tool_calls=[_mock_tool_call("mcp_fleet_claim", "{}", "claim-1")],
        ),
        AssertionError("policy failure must prevent another model call"),
    ]

    def dispatch(*_args, **_kwargs):
        mcp_handlers._mcp_runtime_stop.set({
            "reason": "policy_error", "status": "failure", "policy": "required",
        })
        return '{"error": "trusted policy failed"}'

    with (
        patch("model_tools.handle_function_call", side_effect=dispatch),
        patch.object(agent, "_persist_session"),
        patch.object(agent, "_save_trajectory"),
        patch.object(agent, "_cleanup_task_resources"),
    ):
        result = agent.run_conversation("claim work")

    assert agent.client.chat.completions.create.call_count == 1
    assert result["turn_exit_reason"] == "runtime_stop(policy_error)"
    assert result["failed"] is True
    assert result["completed"] is False


def test_authoritative_success_is_typed_without_assistant_prose():
    agent = _make_agent("mcp_fleet_claim", max_iterations=10)
    agent.client.chat.completions.create.side_effect = [_mock_response(
        content="", finish_reason="tool_calls",
        tool_calls=[_mock_tool_call("mcp_fleet_claim", "{}", "claim-1")],
    )]

    def dispatch(*_args, **_kwargs):
        mcp_handlers._mcp_runtime_stop.set({
            "reason": "max_items", "status": "success", "policy": "fleet-runtime",
        })
        return '{"result": "done"}'

    with (
        patch("model_tools.handle_function_call", side_effect=dispatch),
        patch.object(agent, "_persist_session"),
        patch.object(agent, "_save_trajectory"),
        patch.object(agent, "_cleanup_task_resources"),
    ):
        result = agent.run_conversation("claim work")

    assert result["completed"] is True
    assert result["final_response"] is None
    assert result["trusted_terminal_outcome"] == {
        "reason": "max_items", "status": "success", "policy": "fleet-runtime",
    }




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
        patch("model_tools.handle_function_call", return_value=json.dumps({"error": "boom"})),
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


def test_runtime_stop_halts_the_rest_of_the_assistant_batch():
    """A trusted stop is authoritative for the WHOLE batch, not just the loop.

    The conversation loop only suppresses the next MODEL request. Without a
    per-call gate, an assistant batch of ``claim-1, claim-2`` could receive
    ``runtime_stop(max_items)`` from the first result and still dispatch the
    second — crossing the item boundary the policy just closed.
    """
    agent = _make_agent("mcp_fleet_claim", max_iterations=10)
    agent.client.chat.completions.create.side_effect = [
        _mock_response(
            content="",
            finish_reason="tool_calls",
            tool_calls=[
                # Distinct arguments: identical calls are deduped upstream.
                _mock_tool_call("mcp_fleet_claim", '{"item": 1}', "claim-1"),
                _mock_tool_call("mcp_fleet_claim", '{"item": 2}', "claim-2"),
            ],
        ),
        AssertionError("runtime stop must prevent another model call"),
    ]

    dispatched = []

    def dispatch(*args, **kwargs):
        dispatched.append(kwargs.get("tool_call_id"))
        mcp_handlers._mcp_runtime_stop.set({
            "reason": "max_items", "status": "success", "policy": "fleet-runtime",
        })
        return '{"result": "done"}'

    with (
        patch("model_tools.handle_function_call", side_effect=dispatch),
        patch.object(agent, "_persist_session"),
        patch.object(agent, "_save_trajectory"),
        patch.object(agent, "_cleanup_task_resources"),
    ):
        result = agent.run_conversation("claim work")

    assert dispatched == ["claim-1"], "the second dispatcher must never be entered"
    assert agent.client.chat.completions.create.call_count == 1
    assert result["turn_exit_reason"] == "runtime_stop(max_items)"

    # Every call still needs a paired tool result or the next provider request
    # violates message-role alternation.
    tool_rows = [m for m in result["messages"] if m.get("role") == "tool"]
    assert [row["tool_call_id"] for row in tool_rows] == ["claim-1", "claim-2"]
    assert "was not executed" in tool_rows[1]["content"]


def test_malformed_stop_directive_under_policy_still_halts_the_batch():
    """An unreadable directive must fail closed, not vanish.

    The directive is trusted input from an MCP server, so it can arrive
    malformed. Reading ``reason`` off it raised inside a bare ``except:
    pass``, which dropped the stop entirely and let ``claim-2`` run after
    the policy had already closed the item boundary — the fail-open path
    this hook exists to prevent.
    """
    agent = _make_agent("mcp_fleet_claim", max_iterations=10)
    agent.runtime_policy = "fleet-runtime"
    agent.runtime_task_id = "fire-1"
    agent.client.chat.completions.create.side_effect = [
        _mock_response(
            content="",
            finish_reason="tool_calls",
            tool_calls=[
                _mock_tool_call("mcp_fleet_claim", '{"item": 1}', "claim-1"),
                _mock_tool_call("mcp_fleet_claim", '{"item": 2}', "claim-2"),
            ],
        ),
        AssertionError("an unreadable stop must prevent another model call"),
    ]

    dispatched = []

    def dispatch(*args, **kwargs):
        dispatched.append(kwargs.get("tool_call_id"))
        # No "reason" key: indexing it raises inside the consumer.
        mcp_handlers._mcp_runtime_stop.set({"status": "success", "policy": "fleet-runtime"})
        return '{"result": "done"}'

    with (
        patch("model_tools.handle_function_call", side_effect=dispatch),
        patch.object(agent, "_persist_session"),
        patch.object(agent, "_save_trajectory"),
        patch.object(agent, "_cleanup_task_resources"),
    ):
        result = agent.run_conversation("claim work")

    assert dispatched == ["claim-1"], "the second dispatcher must never be entered"
    assert agent.client.chat.completions.create.call_count == 1
    assert result["turn_exit_reason"] == "runtime_stop(invalid_stop_directive)"

    # A directive that could not be validated must never book as success.
    assert result["trusted_terminal_outcome"]["status"] == "failure"

    tool_rows = [m for m in result["messages"] if m.get("role") == "tool"]
    assert [row["tool_call_id"] for row in tool_rows] == ["claim-1", "claim-2"]
    assert "was not executed" in tool_rows[1]["content"]


@pytest.mark.parametrize("policy", [None, "observer"])
def test_malformed_stop_directive_without_policy_does_not_halt_the_run(policy):
    """With no policy there is no authority to enforce.

    Observer-path directives are advisory, so a malformed one is logged and
    dropped rather than terminating an ordinary interactive run.
    """
    agent = _make_agent("mcp_fleet_claim", max_iterations=10)
    agent.runtime_policy = policy
    agent.client.chat.completions.create.side_effect = [
        _mock_response(
            content="",
            finish_reason="tool_calls",
            tool_calls=[_mock_tool_call("mcp_fleet_claim", '{"item": 1}', "claim-1")],
        ),
        _mock_response(content="done", finish_reason="stop"),
    ]

    def dispatch(*args, **kwargs):
        mcp_handlers._mcp_runtime_stop.set({"status": "success"})
        return '{"result": "done"}'

    with (
        patch("model_tools.handle_function_call", side_effect=dispatch),
        patch.object(agent, "_persist_session"),
        patch.object(agent, "_save_trajectory"),
        patch.object(agent, "_cleanup_task_resources"),
    ):
        result = agent.run_conversation("claim work")

    assert agent.client.chat.completions.create.call_count == 2
    assert not str(result.get("turn_exit_reason") or "").startswith("runtime_stop")


def test_runtime_stop_halts_later_segments_of_a_mixed_batch():
    """A stop inside one segment must stop every later segment too."""
    from agent import tool_executor

    agent = _make_agent("mcp_fleet_claim", "web_search", max_iterations=10)
    dispatched = []

    def dispatch(function_name, *args, **kwargs):
        dispatched.append(function_name)
        if function_name == "mcp_fleet_claim":
            mcp_handlers._mcp_runtime_stop.set({
                "reason": "max_items", "status": "success",
                "policy": "fleet-runtime",
            })
        return '{"result": "done"}'

    assistant_message = SimpleNamespace(tool_calls=[
        _mock_tool_call("mcp_fleet_claim", "{}", "claim-1"),
        _mock_tool_call("web_search", "{}", "search-1"),
    ])
    messages: list = []

    with patch("model_tools.handle_function_call", side_effect=dispatch):
        tool_executor.execute_tool_calls_segmented(
            agent, assistant_message, messages, "task-1",
            segments=[
                ("sequential", [assistant_message.tool_calls[0]]),
                ("parallel", [assistant_message.tool_calls[1]]),
            ],
        )

    assert dispatched == ["mcp_fleet_claim"], "a later segment was still dispatched"
    assert [m["tool_call_id"] for m in messages] == ["claim-1", "search-1"]
    assert "was not executed" in messages[1]["content"]


@pytest.mark.parametrize("policy,expected", [(None, "parallel"), ("observer", "parallel"), ("fleet-runtime", "sequential")])
@pytest.mark.parametrize("entry", ["facade", "segmented"])
def test_authoritative_run_forces_mcp_calls_onto_the_barrier_path(policy, expected, entry):
    """Parallel siblings cannot be un-executed, so policy runs never fan out."""
    from agent import tool_executor

    calls = [
        _mock_tool_call("mcp__fleet__claim", "{}", "claim-1"),
        _mock_tool_call("mcp__fleet__claim", "{}", "claim-2"),
    ]
    agent = _make_agent("mcp__fleet__claim")
    agent.runtime_policy = policy
    with (
        patch("agent.tool_dispatch_helpers._is_mcp_tool_parallel_safe", return_value=True),
        patch.object(agent, "_execute_tool_calls_concurrent") as facade_parallel,
        patch.object(agent, "_execute_tool_calls_sequential") as facade_sequential,
        patch.object(tool_executor, "execute_tool_calls_concurrent") as segmented_parallel,
        patch.object(tool_executor, "execute_tool_calls_sequential") as segmented_sequential,
        patch.object(tool_executor, "_finalize_tool_batch"),
    ):
        message = SimpleNamespace(tool_calls=calls)
        if entry == "facade":
            agent._execute_tool_calls(message, [], "task-1")
            parallel, sequential = facade_parallel, facade_sequential
        else:
            tool_executor.execute_tool_calls_segmented(agent, message, [], "task-1")
            parallel, sequential = segmented_parallel, segmented_sequential
        assert parallel.call_count == (expected == "parallel")
        assert sequential.call_count == (expected == "sequential")
