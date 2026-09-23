"""Runtime tests for tool-call loop guardrails."""

import json
import time
import uuid
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

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

    decision = agent._tool_guardrails.before_call("web_search", args)
    assert decision.action == "block"
    assert decision.code == "repeated_exact_failure_block"


@pytest.mark.parametrize("platform", ["desktop", "acp"])
def test_interactive_platforms_keep_warning_only_default(platform):
    agent = _make_agent("web_search", platform=platform)
    args = {"query": "same"}

    _seed_exact_failures(agent, "web_search", args, count=5)

    decision = agent._tool_guardrails.before_call("web_search", args)
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
    assert "repeated_exact_failure_warning" in messages[0]["content"]


def test_same_tool_failure_warning_tells_model_to_recover_with_tools():
    agent = _make_agent("terminal")
    guardrails = agent._tool_guardrails
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
        (str(Path("/approved/path")), "before write_file")
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


def _compressed_args(field: str) -> dict:
    """Generate a legacy compression marker for already-contaminated session coverage."""
    from agent.compression_marker import _COMPRESSION_MARKER_TEMPLATE

    original = "z" * 2000
    marker = _COMPRESSION_MARKER_TEMPLATE.format(
        omitted=len(original) - 200,
        total=len(original),
    )
    return {field: original[:200] + marker}


def test_context_pruned_effectful_call_blocks_before_dispatch():
    agent = _make_agent("test_effectful_write")
    pruned = _compressed_args("body")
    tc = _mock_tool_call("test_effectful_write", json.dumps(pruned, ensure_ascii=False), "c-pruned-current")
    msg = SimpleNamespace(content="", tool_calls=[tc])
    messages = []

    # The hook may rewrite args; the boundary is enforced on what it returns.
    with (
        patch("hermes_cli.plugins._dispatch_pre_tool_call_hooks", return_value=(None, pruned)) as plugin,
        patch("model_tools.handle_function_call", return_value="SHOULD_NOT_RUN") as dispatch,
    ):
        agent._execute_tool_calls_sequential(msg, messages, "task-1")

    plugin.assert_called_once()
    dispatch.assert_not_called()
    payload = json.loads(messages[0]["content"])
    assert payload["error"] == "suspected_pruned_tool_arguments"
    assert payload["argument_paths"] == ["$.body"]
    assert "Recover the exact content from its durable source" in payload["message"]
    # Only the compressor's current marker is an artifact; a legacy "...[truncated]" tail is ordinary content.
    from agent.compression_marker import _COMPRESSION_MARKER_PREFIX
    from agent.tool_dispatch_helpers import _context_pruned_argument_paths

    assert _context_pruned_argument_paths("write_file", {"content": "x" * 201 + "...[truncated]"}) == []
    assert _context_pruned_argument_paths("write_file", {"content": f"see {_COMPRESSION_MARKER_PREFIX} docs"}) == []
    assert _context_pruned_argument_paths(
        "write_file",
        {"content": f"template {_COMPRESSION_MARKER_PREFIX} {{omitted:,}} of {{total:,}}"},
    ) == []
    # A marker cut before its fixed sentence is still an artifact once a count is rendered.
    for suffix in (" 1,800", " 1,800 of", " 1,800 of 2,000 chars omitted"):
        assert _context_pruned_argument_paths(
            "write_file", {"body": f"prefix {_COMPRESSION_MARKER_PREFIX}{suffix}"}
        ) == ["$.body"]


def test_read_only_tool_may_quote_current_context_prune_marker():
    agent = _make_agent("web_search")
    args = _compressed_args("query")
    tc = _mock_tool_call("web_search", json.dumps(args, ensure_ascii=False), "c-pruned-read")
    msg = SimpleNamespace(content="", tool_calls=[tc])
    messages = []

    with patch("model_tools.handle_function_call", return_value=json.dumps({"ok": True})) as dispatch:
        agent._execute_tool_calls_sequential(msg, messages, "task-1")

    dispatch.assert_called_once()


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
    assert halt_text

    # The halt message must have been pushed through the callback at least
    # once.  Empty-queue SSE writers were the bug — clients saw no content
    # delta before the finish chunk.
    text_deltas = [d for d in deltas if isinstance(d, str)]
    assert halt_text in text_deltas, (
        f"halt message was never streamed; callback only saw {deltas!r}"
    )


@pytest.mark.parametrize("executor", ["_execute_tool_calls_sequential", "_execute_tool_calls_concurrent"])
def test_policy_hook_unavailable_halts_without_dispatch(executor):
    from hermes_cli.plugins import _PRE_TOOL_CALL_TIMEOUT_BLOCK_MESSAGE

    agent = _make_agent("terminal", config=_hard_stop_config())
    args = {"command": "printf probe-1"}
    call = _mock_tool_call("terminal", json.dumps(args), "hook-unavailable-1")
    message = SimpleNamespace(content="", tool_calls=[call])
    messages = []
    with (
        patch("agent.tool_executor._pre_tool_block", return_value=(_PRE_TOOL_CALL_TIMEOUT_BLOCK_MESSAGE, args)),
        patch("model_tools.handle_function_call", return_value="MUST NOT EXECUTE") as handler,
    ):
        getattr(agent, executor)(message, messages, "hook-outage-fixture")
    handler.assert_not_called()
    decision = agent._tool_guardrail_halt_decision
    assert decision is not None
    assert decision.code == "policy_hook_unavailable"
    assert decision.should_halt
    payload = json.loads(messages[-1]["content"])
    assert payload["success"] is False
    assert payload["error_type"] == "policy_hook_unavailable"
    assert messages[-1]["tool_call_id"] == "hook-unavailable-1"


def test_policy_hook_unavailable_does_not_reclassify_normal_denial():
    agent = _make_agent("terminal", config=_hard_stop_config())
    args = {"command": "printf denied"}
    call = _mock_tool_call("terminal", json.dumps(args), "ordinary-denial")
    message = SimpleNamespace(content="", tool_calls=[call])
    messages = []
    with (
        patch("agent.tool_executor._pre_tool_block", return_value=("Explicit policy denial", args)),
        patch("model_tools.handle_function_call", return_value="MUST NOT EXECUTE") as handler,
    ):
        agent._execute_tool_calls_sequential(message, messages, "denial-fixture")
    handler.assert_not_called()
    assert agent._tool_guardrail_halt_decision is None
    assert json.loads(messages[-1]["content"])["error"] == "Explicit policy denial"


# ── C5: no-retry behavior across the full conversation loop ───────────────────


def _hang_policy_fixture():
    """Return (release_event, hung pre_tool_call callback) for outage fixtures."""
    import threading

    release = threading.Event()

    def hung_policy(**_kwargs):
        release.wait(timeout=30.0)
        return None

    return release, hung_policy


def _install_hook_manager(monkeypatch, hook_callback):
    """Install a PluginManager carrying *hook_callback* for pre_tool_call."""
    from hermes_cli.plugins import PluginManager
    import hermes_cli.plugins as plugins_mod

    mgr = PluginManager()
    mgr._discovered = True
    mgr._hooks["pre_tool_call"] = [hook_callback]
    monkeypatch.setattr(plugins_mod, "_plugin_manager", mgr)
    return mgr


def test_hook_outage_halts_turn_without_second_model_round(monkeypatch):
    """C5 boundary proof: with a hung pre_tool_call policy hook, a turn whose model
    would issue a second numbered terminal probe must (a) execute zero tool
    handlers, (b) end with guardrail.code == 'policy_hook_unavailable' before any
    further model tool round, (c) pair tool-call IDs, (d) inject no synthetic
    user message, and (e) not leave the fixture worker behind after release.
    """
    import threading

    from hermes_cli.plugins import _PRE_TOOL_CALL_TIMEOUT_BLOCK_MESSAGE

    release, hung_policy = _hang_policy_fixture()
    _install_hook_manager(monkeypatch, hung_policy)
    monkeypatch.setattr(
        "hermes_cli.plugins._resolve_hook_callback_timeout", lambda: 0.05
    )

    agent = _make_agent("terminal", max_iterations=10, config=_hard_stop_config())
    # The model would offer probe-2 if asked again: a hung hook must never get there.
    responses = [
        _mock_response(
            content="",
            finish_reason="tool_calls",
            tool_calls=[_mock_tool_call("terminal", json.dumps({"command": "printf probe-1"}), "outage-call-1")],
        ),
        _mock_response(
            content="",
            finish_reason="tool_calls",
            tool_calls=[_mock_tool_call("terminal", json.dumps({"command": "printf probe-2"}), "outage-call-2")],
        ),
    ]
    responses.append(_mock_response(content="unreachable", finish_reason="stop", tool_calls=None))
    agent.client.chat.completions.create.side_effect = responses
    agent._disable_streaming = True

    worker_threads_before = threading.active_count()
    with (
        patch("model_tools.handle_function_call", return_value="MUST NOT EXECUTE") as handler,
        patch.object(agent, "_persist_session"),
        patch.object(agent, "_save_trajectory"),
        patch.object(agent, "_cleanup_task_resources"),
    ):
        result = agent.run_conversation("run the first terminal probe")

    # (a) no tool ever dispatched
    handler.assert_not_called()

    # (b) turn ended with the controlled halt before another model tool round
    assert result["turn_exit_reason"] == "guardrail_halt"
    assert result["guardrail"]["code"] == "policy_hook_unavailable"
    assert result["guardrail"]["action"] in ("block", "halt")
    assert "policy hook" in result["final_response"] or "hook" in result["final_response"]
    model_calls = agent.client.chat.completions.create.call_count
    assert model_calls == 1, f"expected exactly one model round, got {model_calls}"

    # (c) tool-call IDs still pair; the blocked result lands on outage-call-1
    tool_rows = [m for m in result["messages"] if m.get("role") == "tool"]
    assert [row["tool_call_id"] for row in tool_rows] == ["outage-call-1"]
    payload = json.loads(tool_rows[0]["content"])
    assert payload["error_type"] == "policy_hook_unavailable"
    assert _PRE_TOOL_CALL_TIMEOUT_BLOCK_MESSAGE in payload["error"]

    # (d) no synthetic user message was injected after the original prompt:
    # exactly one user row whose content is the original prompt verbatim
    # (metadata like timestamp is fine; an injected nudge/repair row is not).
    user_rows = [m for m in result["messages"] if m.get("role") == "user"]
    assert len(user_rows) == 1, user_rows
    assert user_rows[0]["content"] == "run the first terminal probe"

    # (e) releasing the fixture event cleans up its worker
    release.set()
    deadline = time.monotonic() + 5.0
    while threading.active_count() > worker_threads_before and time.monotonic() < deadline:
        time.sleep(0.01)
    assert threading.active_count() <= worker_threads_before, "hung hook worker leaked"


def test_hook_outage_execute_code_gets_same_controlled_outcome(monkeypatch):
    """execute_code under an unavailable policy hook gets the identical controlled
    halt (not a dispatch, not a free pass)."""
    from hermes_cli.plugins import _PRE_TOOL_CALL_TIMEOUT_BLOCK_MESSAGE

    release, hung_policy = _hang_policy_fixture()
    _install_hook_manager(monkeypatch, hung_policy)
    monkeypatch.setattr(
        "hermes_cli.plugins._resolve_hook_callback_timeout", lambda: 0.05
    )

    agent = _make_agent("execute_code", max_iterations=10, config=_hard_stop_config())
    call = _mock_tool_call("execute_code", json.dumps({"code": "print('probe')"}), "outage-code-1")
    message = SimpleNamespace(content="", tool_calls=[call])
    messages = []
    with (
        patch("model_tools.handle_function_call", return_value="MUST NOT EXECUTE") as handler,
    ):
        agent._execute_tool_calls_sequential(message, messages, "outage-code-fixture")

    handler.assert_not_called()
    decision = agent._tool_guardrail_halt_decision
    assert decision is not None and decision.code == "policy_hook_unavailable"
    payload = json.loads(messages[-1]["content"])
    assert payload["error_type"] == "policy_hook_unavailable"
    assert _PRE_TOOL_CALL_TIMEOUT_BLOCK_MESSAGE in payload["error"]
    release.set()


def test_normal_policy_denial_stays_denied_and_dispatches_nothing():
    """An explicit (non-timeout) policy denial keeps the plain block path: no halt,
    no dispatch — distinct from the controlled outage halt."""
    agent = _make_agent("terminal", config=_hard_stop_config())
    args = {"command": "printf denied"}
    call = _mock_tool_call("terminal", json.dumps(args), "denial-2")
    message = SimpleNamespace(content="", tool_calls=[call])
    messages = []
    with (
        patch("agent.tool_executor._pre_tool_block", return_value=("Policy forbids this", args)),
        patch("model_tools.handle_function_call", return_value="MUST NOT EXECUTE") as handler,
    ):
        agent._execute_tool_calls_concurrent(message, messages, "denial-2-fixture")
    handler.assert_not_called()
    assert agent._tool_guardrail_halt_decision is None
    assert json.loads(messages[-1]["content"])["error"] == "Policy forbids this"


def test_fresh_agent_with_healthy_hooks_runs_tool_after_outage():
    """After the outage fixture releases, a SEPARATE fresh agent with healthy hooks
    runs a harmless tool successfully — recovery is a new process, not a retry."""
    agent = _make_agent("terminal", max_iterations=10, config=_hard_stop_config())
    responses = [
        _mock_response(
            content="",
            finish_reason="tool_calls",
            tool_calls=[_mock_tool_call("terminal", json.dumps({"command": "printf fresh-ok"}), "fresh-call-1")],
        ),
        _mock_response(content="done", finish_reason="stop", tool_calls=None),
    ]
    agent.client.chat.completions.create.side_effect = responses
    agent._disable_streaming = True

    with (
        patch("hermes_cli.lifecycle.invoke_hook", return_value=[]),
        patch("model_tools.handle_function_call", return_value=json.dumps({"output": "fresh-ok", "exit_code": 0})) as handler,
        patch.object(agent, "_persist_session"),
        patch.object(agent, "_save_trajectory"),
        patch.object(agent, "_cleanup_task_resources"),
    ):
        result = agent.run_conversation("run the harmless tool")

    assert handler.call_count == 1
    assert result["final_response"] == "done"
    assert "guardrail" not in result
    tool_rows = [m for m in result["messages"] if m.get("role") == "tool"]
    assert [row["tool_call_id"] for row in tool_rows] == ["fresh-call-1"]
