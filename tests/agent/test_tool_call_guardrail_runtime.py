"""Runtime tests for tool-call loop guardrails."""

import json
import uuid
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


def _compressed_args(field: str) -> dict:
    """Generate the current model-visible prune marker through the real compressor."""
    from agent.context_compressor import _COMPRESSION_MARKER_PREFIX, _truncate_tool_call_args_json

    raw = json.dumps({field: "z" * 2000})
    parsed = json.loads(_truncate_tool_call_args_json(raw))
    assert _COMPRESSION_MARKER_PREFIX in parsed[field]
    return parsed


def test_context_pruned_effectful_call_blocks_before_plugins_and_dispatch():
    agent = _make_agent("test_effectful_write")
    args = _compressed_args("body")
    tc = _mock_tool_call("test_effectful_write", json.dumps(args, ensure_ascii=False), "c-pruned-current")
    msg = SimpleNamespace(content="", tool_calls=[tc])
    messages = []

    with (
        patch("hermes_cli.plugins._dispatch_pre_tool_call_hooks") as plugin,
        patch("model_tools.handle_function_call", return_value="SHOULD_NOT_RUN") as dispatch,
    ):
        agent._execute_tool_calls_sequential(msg, messages, "task-1")

    plugin.assert_not_called()
    dispatch.assert_not_called()
    payload = json.loads(messages[0]["content"])
    assert payload["error"] == "suspected_pruned_tool_arguments"
    assert payload["argument_paths"] == ["$.body"]
    assert "Recover the exact content from its durable source" in payload["message"]


def test_original_pruned_args_never_reach_real_managed_relay(tmp_path, monkeypatch):
    """A Relay execution interceptor may short-circuit Hermes entirely, so it must
    never receive synthetic compressor content from the model-facing history."""
    pytest.importorskip("nemo_relay")
    from agent import relay_runtime

    session_id = "session-pruned-relay"
    consumer = "test.context-pruned-relay-guard"
    interceptor_name = "test-context-pruned-short-circuit"
    seen = []

    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "profile"))
    relay_runtime._reset_for_tests()
    lease = relay_runtime.SESSION_COORDINATOR.acquire_conversation(
        profile_key=relay_runtime.current_profile_key(),
        session_id=session_id,
        platform="cli",
    )
    turn = relay_runtime.SESSION_COORDINATOR.begin_turn(
        lease, turn_id="turn-pruned-relay", task_id="task-1",
    )
    lease.host.retain_managed_execution(consumer)
    relay = lease.host.relay

    async def short_circuit(_name, args, next_call):
        del next_call
        seen.append(dict(args))
        return relay.ToolExecutionInterceptOutcome({"intercepted": True})

    relay.intercepts.register_tool_execution(interceptor_name, 1, short_circuit)
    try:
        assert lease.host.managed_execution_enabled()

        agent = _make_agent("test_effectful_write")
        agent.session_id = session_id
        pruned = _compressed_args("body")
        tc = _mock_tool_call(
            "test_effectful_write",
            json.dumps(pruned, ensure_ascii=False),
            "c-pruned-real-relay",
        )
        msg = SimpleNamespace(content="", tool_calls=[tc])
        messages = []

        with patch("model_tools.handle_function_call", return_value="SHOULD_NOT_RUN") as dispatch:
            agent._execute_tool_calls_sequential(msg, messages, "task-1")

        # The guard is before relay_tools.execute(): a configured native interceptor
        # never gets authority to observe, forward, or short-circuit poisoned args.
        assert seen == []
        dispatch.assert_not_called()
        payload = json.loads(messages[0]["content"])
        assert payload["error"] == "suspected_pruned_tool_arguments"
        assert payload["argument_paths"] == ["$.body"]
    finally:
        relay.intercepts.deregister_tool_execution(interceptor_name)
        lease.host.release_managed_execution(consumer)
        relay_runtime.SESSION_COORDINATOR.end_turn(turn, outcome="success")
        relay_runtime.SESSION_COORDINATOR.release_conversation(lease)
        relay_runtime._reset_for_tests()


def test_relay_request_rewrite_is_blocked_before_execution_interceptor(tmp_path, monkeypatch):
    """Clean model args may become poisoned inside Relay request middleware; the
    terminal scope-local guard must reject that effective request before execution."""
    pytest.importorskip("nemo_relay")
    from agent import relay_runtime

    session_id = "session-pruned-relay-rewrite"
    consumer = "test.context-pruned-relay-rewrite-guard"
    request_name = "test-context-pruned-request-rewrite"
    execution_name = "test-context-pruned-execution-short-circuit"
    execution_seen = []

    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "profile"))
    relay_runtime._reset_for_tests()
    lease = relay_runtime.SESSION_COORDINATOR.acquire_conversation(
        profile_key=relay_runtime.current_profile_key(),
        session_id=session_id,
        platform="cli",
    )
    turn = relay_runtime.SESSION_COORDINATOR.begin_turn(
        lease, turn_id="turn-pruned-relay-rewrite", task_id="task-1",
    )
    lease.host.retain_managed_execution(consumer)
    relay = lease.host.relay
    pruned = _compressed_args("body")

    def rewrite_request(_name, args):
        assert args == {"body": "complete"}
        return dict(pruned)

    async def short_circuit(_name, args, next_call):
        del next_call
        execution_seen.append(dict(args))
        return relay.ToolExecutionInterceptOutcome({"intercepted": True})

    # Stronger than the original reviewer repro: even a breaking request rewrite
    # must not bypass the post-rewrite execution-boundary validator.
    relay.intercepts.register_tool_request(request_name, 1, True, rewrite_request)
    relay.intercepts.register_tool_execution(execution_name, 1, short_circuit)
    try:
        assert lease.host.managed_execution_enabled()

        agent = _make_agent("test_effectful_write")
        agent.session_id = session_id
        tc = _mock_tool_call(
            "test_effectful_write",
            json.dumps({"body": "complete"}),
            "c-pruned-relay-request-rewrite",
        )
        msg = SimpleNamespace(content="", tool_calls=[tc])
        messages = []

        with (
            patch("hermes_cli.middleware.apply_tool_request_middleware") as hermes_request_middleware,
            patch("model_tools.handle_function_call", return_value="SHOULD_NOT_RUN") as dispatch,
        ):
            agent._execute_tool_calls_sequential(msg, messages, "task-1")

        # Relay request middleware did rewrite the call, but the terminal Relay
        # request guard rejected it before either execution middleware or Hermes'
        # downstream request/dispatch pipeline received the poisoned payload.
        assert execution_seen == []
        hermes_request_middleware.assert_not_called()
        dispatch.assert_not_called()
        payload = json.loads(messages[0]["content"])
        assert payload["error"] == "suspected_pruned_tool_arguments"
        assert payload["argument_paths"] == ["$.body"]
    finally:
        relay.intercepts.deregister_tool_execution(execution_name)
        relay.intercepts.deregister_tool_request(request_name)
        lease.host.release_managed_execution(consumer)
        relay_runtime.SESSION_COORDINATOR.end_turn(turn, outcome="success")
        relay_runtime.SESSION_COORDINATOR.release_conversation(lease)
        relay_runtime._reset_for_tests()


def test_pruned_block_sanitizes_post_hook_and_outbound_tool_input():
    agent = _make_agent("test_effectful_write")
    args = _compressed_args("body")
    args["note"] = "safe metadata stays intact"
    original = json.loads(json.dumps(args))
    tc = _mock_tool_call(
        "test_effectful_write", json.dumps(args, ensure_ascii=False), "c-pruned-hook-redaction"
    )
    msg = SimpleNamespace(content="", tool_calls=[tc])
    messages = []
    lifecycle_events = []

    def capture_lifecycle(event_name, **kwargs):
        lifecycle_events.append((event_name, kwargs))
        return []

    with (
        patch("hermes_cli.lifecycle.has_hook", side_effect=lambda name: name == "post_tool_call"),
        patch("hermes_cli.lifecycle.invoke_hook", side_effect=capture_lifecycle),
        patch("model_tools.handle_function_call", return_value="SHOULD_NOT_RUN") as dispatch,
    ):
        agent._execute_tool_calls_sequential(msg, messages, "task-1")

    dispatch.assert_not_called()
    assert args == original, "sanitizing hook payloads must not mutate the source args"
    assert len(lifecycle_events) == 1
    event_name, event = lifecycle_events[0]
    assert event_name == "post_tool_call"
    hook_args = event["args"]
    assert hook_args["body"] == "[context-compression artifact removed]"
    assert hook_args["note"] == "safe metadata stays intact"
    assert "HERMES-CONTEXT-COMPRESSION" not in json.dumps(hook_args, ensure_ascii=False)

    from agent import outbound_webhooks

    body = outbound_webhooks._serialize_payload(
        event_name,
        event,
        "did-pruned-redaction",
    )
    payload = json.loads(body)
    assert payload["tool_input"]["body"] == "[context-compression artifact removed]"
    assert payload["tool_input"]["note"] == "safe metadata stays intact"
    assert "HERMES-CONTEXT-COMPRESSION" not in body.decode("utf-8")

    refusal = json.loads(messages[0]["content"])
    assert refusal["error"] == "suspected_pruned_tool_arguments"
    assert refusal["argument_paths"] == ["$.body"]


def test_pruned_argument_redaction_is_shape_preserving_and_leaf_scoped():
    from agent.tool_dispatch_helpers import _redact_context_pruned_arguments

    args = {
        "outer": [{"body": "x" * 20 + "...[truncated]", "keep": "literal ...[truncated] then more"}],
        "count": 3,
    }

    redacted = _redact_context_pruned_arguments("test_effectful_write", args)

    assert redacted is not args
    assert redacted["outer"] is not args["outer"]
    assert redacted["outer"][0]["body"] == "[context-compression artifact removed]"
    assert redacted["outer"][0]["keep"] == "literal ...[truncated] then more"
    assert redacted["count"] == 3
    assert args["outer"][0]["body"].endswith("...[truncated]")


def test_request_middleware_pruned_args_block_before_short_circuit_execution_middleware():
    agent = _make_agent("test_effectful_write")
    pruned = _compressed_args("body")
    tc = _mock_tool_call(
        "test_effectful_write", json.dumps({"body": "complete"}), "c-pruned-request-middleware"
    )
    msg = SimpleNamespace(content="", tool_calls=[tc])
    messages = []

    request_result = SimpleNamespace(payload=pruned, trace=[{"source": "test_request_middleware"}])
    with (
        patch("hermes_cli.middleware.apply_tool_request_middleware", return_value=request_result),
        patch("hermes_cli.middleware.run_tool_execution_middleware", return_value="SHORT_CIRCUIT") as execution_middleware,
        patch("model_tools.handle_function_call", return_value="SHOULD_NOT_RUN") as dispatch,
    ):
        agent._execute_tool_calls_sequential(msg, messages, "task-1")

    execution_middleware.assert_not_called()
    dispatch.assert_not_called()
    payload = json.loads(messages[0]["content"])
    assert payload["error"] == "suspected_pruned_tool_arguments"
    assert payload["argument_paths"] == ["$.body"]


def test_execution_middleware_rewrite_is_rechecked_before_dispatch():
    agent = _make_agent("test_effectful_write")
    pruned = _compressed_args("body")
    tc = _mock_tool_call(
        "test_effectful_write", json.dumps({"body": "complete"}), "c-pruned-execution-middleware"
    )
    msg = SimpleNamespace(content="", tool_calls=[tc])
    messages = []

    def rewrite_then_continue(tool_name, args, next_call, **_context):
        assert tool_name == "test_effectful_write"
        assert args == {"body": "complete"}
        return next_call(pruned)

    with (
        patch("hermes_cli.middleware.run_tool_execution_middleware", side_effect=rewrite_then_continue),
        patch("model_tools.handle_function_call", return_value="SHOULD_NOT_RUN") as dispatch,
    ):
        agent._execute_tool_calls_sequential(msg, messages, "task-1")

    dispatch.assert_not_called()
    payload = json.loads(messages[0]["content"])
    assert payload["error"] == "suspected_pruned_tool_arguments"
    assert payload["argument_paths"] == ["$.body"]


def test_plugin_modified_args_are_rechecked_for_context_prune_markers():
    agent = _make_agent("test_effectful_write")
    pruned = _compressed_args("body")
    tc = _mock_tool_call("test_effectful_write", json.dumps({"body": "complete"}), "c-pruned-plugin")
    msg = SimpleNamespace(content="", tool_calls=[tc])
    messages = []

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


def test_read_only_tool_may_quote_current_context_prune_marker():
    agent = _make_agent("web_search")
    args = _compressed_args("query")
    tc = _mock_tool_call("web_search", json.dumps(args, ensure_ascii=False), "c-pruned-read")
    msg = SimpleNamespace(content="", tool_calls=[tc])
    messages = []

    with patch("model_tools.handle_function_call", return_value=json.dumps({"ok": True})) as dispatch:
        agent._execute_tool_calls_sequential(msg, messages, "task-1")

    dispatch.assert_called_once()


def test_mcp_pruned_refusal_keeps_untrusted_result_framing():
    agent = _make_agent("mcp_write")
    args = _compressed_args("body")
    tc = _mock_tool_call("mcp_write", json.dumps(args, ensure_ascii=False), "c-pruned-mcp-envelope")
    msg = SimpleNamespace(content="", tool_calls=[tc])
    messages = []

    with patch("model_tools.handle_function_call", return_value="SHOULD_NOT_RUN") as dispatch:
        agent._execute_tool_calls_sequential(msg, messages, "task-1")

    dispatch.assert_not_called()
    content = messages[0]["content"]
    assert content.startswith('<untrusted_tool_result source="mcp_write">')
    assert '"error": "suspected_pruned_tool_arguments"' in content
    assert '"argument_paths": ["$.body"]' in content


def test_legacy_pruned_tail_blocks_observed_short_effectful_writes():
    from agent.tool_dispatch_helpers import _context_pruned_argument_paths

    for total_len in (182, 188):
        suffix = "...[truncated]"
        value = "x" * (total_len - len(suffix)) + suffix
        assert len(value) == total_len
        assert _context_pruned_argument_paths("test_effectful_write", {"body": value}) == ["$.body"]

    # The incident signature is a poison TAIL. Ordinary prose may discuss the marker.
    assert _context_pruned_argument_paths(
        "mcp_write", {"body": "literal ...[truncated] quote followed by complete content"}
    ) == []
    assert _context_pruned_argument_paths(
        "web_search", {"query": "x" * 170 + "...[truncated]"}
    ) == []


def test_context_pruned_effectful_call_blocks_in_concurrent_path():
    agent = _make_agent("test_effectful_write")
    args = _compressed_args("body")
    tc = _mock_tool_call("test_effectful_write", json.dumps(args, ensure_ascii=False), "c-pruned-concurrent")
    msg = SimpleNamespace(content="", tool_calls=[tc])
    messages = []

    with patch("model_tools.handle_function_call", return_value="SHOULD_NOT_RUN") as dispatch:
        agent._execute_tool_calls_concurrent(msg, messages, "task-1")

    dispatch.assert_not_called()
    payload = json.loads(messages[0]["content"])
    assert payload["error"] == "suspected_pruned_tool_arguments"
    assert payload["argument_paths"] == ["$.body"]


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
