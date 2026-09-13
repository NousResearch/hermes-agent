"""End-to-end turn containment for required plugin lifecycle hooks."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import yaml

import hermes_cli.plugins as plugins_mod
from hermes_cli.required_lifecycle import (
    REQUIRED_LIFECYCLE_FAILURE_TEXT,
    quarantine_required_provider_fields,
)
from run_agent import AIAgent


REQUIRED = {
    "pre_llm_call": ["behavioral.pre_llm.v1"],
    "pre_tool_call": ["behavioral.pre_tool.v1"],
    "post_tool_call": ["behavioral.post_tool.v1"],
    "transform_llm_output": ["behavioral.output.v1"],
}


def _response(text: str, *, finish_reason: str = "stop", tool_calls=None):
    message = SimpleNamespace(
        content=text,
        tool_calls=tool_calls,
        reasoning=None,
        reasoning_content=None,
        reasoning_details=None,
    )
    choice = SimpleNamespace(message=message, finish_reason=finish_reason)
    return SimpleNamespace(choices=[choice], model="test/model", usage=None)


def _install_required_plugin(
    home: Path,
    pre_llm_expr: str,
    output_expr: str,
    *,
    post_tool_expr: str = 'required_hook_result("behavioral.post_tool.v1", None)',
    pre_tool_expr: str = 'required_hook_result("behavioral.pre_tool.v1", {})',
) -> None:
    plugin = home / "plugins" / "atlas"
    plugin.mkdir(parents=True)
    (plugin / "plugin.yaml").write_text(
        yaml.safe_dump({"name": "atlas", "version": "1.0.0"}),
        encoding="utf-8",
    )
    (plugin / "__init__.py").write_text(
        "from hermes_cli.plugins import required_hook_result\n"
        "def register(ctx):\n"
        "    ctx.register_hook(\"pre_llm_call\", lambda **kw: "
        f"{pre_llm_expr}, registration_id=\"behavioral.pre_llm.v1\")\n"
        "    ctx.register_hook(\"pre_tool_call\", lambda **kw: "
        f"{pre_tool_expr}, "
        "registration_id=\"behavioral.pre_tool.v1\")\n"
        "    ctx.register_hook(\"post_tool_call\", lambda **kw: "
        f"{post_tool_expr}, "
        "registration_id=\"behavioral.post_tool.v1\")\n"
        "    ctx.register_hook(\"transform_llm_output\", lambda **kw: "
        f"{output_expr}, registration_id=\"behavioral.output.v1\")\n",
        encoding="utf-8",
    )
    (home / "config.yaml").write_text(
        yaml.safe_dump(
            {
                "plugins": {
                    "enabled": ["atlas"],
                    "required_lifecycle_hooks": {"atlas": REQUIRED},
                }
            }
        ),
        encoding="utf-8",
    )


def _agent() -> AIAgent:
    with (
        patch("model_tools.get_tool_definitions", return_value=[]),
        patch("model_tools.check_toolset_requirements", return_value={}),
        patch("agent.process_bootstrap.OpenAI"),
    ):
        agent = AIAgent(
            session_id="required-lifecycle-test",
            api_key="test-key-1234567890",
            base_url="https://example.invalid/v1",
            provider="openai-compat",
            model="test/model",
            max_iterations=2,
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
        )
    agent.client = MagicMock()
    agent._cached_system_prompt = "You are helpful."
    agent._use_prompt_caching = False
    agent.compression_enabled = False
    agent.save_trajectories = False
    agent._fallback_chain = []
    agent._disable_streaming = True
    return agent


def test_required_pre_llm_failure_stops_before_provider_call(
    tmp_path, monkeypatch
):
    home = tmp_path / "hermes"
    _install_required_plugin(
        home,
        '(_ for _ in ()).throw(RuntimeError("raw secret"))',
        'required_hook_result("behavioral.output.v1", None)',
    )
    monkeypatch.setenv("HERMES_HOME", str(home))
    plugins_mod._reset_plugin_managers_for_tests()
    agent = _agent()
    calls = []
    agent._interruptible_api_call = lambda kwargs: calls.append(kwargs) or _response(
        "raw answer"
    )

    result = agent.run_conversation("do work", conversation_history=[], task_id="task-1")

    assert calls == []
    assert result["failed"] is True
    assert result["final_response"] == REQUIRED_LIFECYCLE_FAILURE_TEXT
    assert "raw secret" not in str(result)


def test_required_output_is_buffered_transformed_then_persisted(
    tmp_path, monkeypatch
):
    home = tmp_path / "hermes"
    _install_required_plugin(
        home,
        'required_hook_result("behavioral.pre_llm.v1", None)',
        'required_hook_result("behavioral.output.v1", "safe answer")',
    )
    monkeypatch.setenv("HERMES_HOME", str(home))
    plugins_mod._reset_plugin_managers_for_tests()
    agent = _agent()
    stream = []
    deltas = []
    delta_callback = deltas.append
    agent.stream_delta_callback = delta_callback

    def provider(_kwargs):
        assert agent._stream_callback is None
        assert agent.stream_delta_callback is None
        return _response("raw answer")

    agent._interruptible_api_call = provider
    flushed = []
    agent._flush_messages_to_session_db = lambda messages, _history=None: (
        flushed.append(copy.deepcopy(messages))
    )
    result = agent.run_conversation(
        "do work",
        conversation_history=[],
        task_id="task-1",
        stream_callback=stream.append,
    )

    assert stream == []
    assert deltas == []
    assert agent.stream_delta_callback is delta_callback
    assert result["final_response"] == "safe answer"
    assert result["pre_transform_response"] is None
    assert result["messages"][-1]["content"] == "safe answer"
    assert all(message.get("content") != "raw answer" for message in result["messages"])
    assert all(
        message.get("content") != "raw answer"
        for snapshot in flushed
        for message in snapshot
    )


def test_required_output_failure_replaces_raw_text_before_persistence(
    tmp_path, monkeypatch
):
    home = tmp_path / "hermes"
    _install_required_plugin(
        home,
        'required_hook_result("behavioral.pre_llm.v1", None)',
        '(_ for _ in ()).throw(RuntimeError("raw transform secret"))',
    )
    monkeypatch.setenv("HERMES_HOME", str(home))
    plugins_mod._reset_plugin_managers_for_tests()
    agent = _agent()
    agent._interruptible_api_call = lambda _kwargs: _response("raw answer")

    result = agent.run_conversation(
        "do work",
        conversation_history=[],
        task_id="task-1",
    )

    assert result["failed"] is True
    assert result["final_response"] == REQUIRED_LIFECYCLE_FAILURE_TEXT
    assert result["pre_transform_response"] is None
    assert all(message.get("content") != "raw answer" for message in result["messages"])
    assert "raw transform secret" not in str(result)


@pytest.mark.parametrize(
    ("output_expr", "expected", "failed"),
    (
        (
            'required_hook_result("behavioral.output.v1", "safe partial")',
            "safe partial",
            False,
        ),
        (
            '(_ for _ in ()).throw(RuntimeError("raw ceiling secret"))',
            REQUIRED_LIFECYCLE_FAILURE_TEXT,
            True,
        ),
    ),
)
def test_required_output_authorizes_length_ceiling_before_any_persistence(
    tmp_path, monkeypatch, output_expr, expected, failed
):
    home = tmp_path / "hermes"
    _install_required_plugin(
        home,
        'required_hook_result("behavioral.pre_llm.v1", None)',
        output_expr,
    )
    monkeypatch.setenv("HERMES_HOME", str(home))
    plugins_mod._reset_plugin_managers_for_tests()
    agent = _agent()
    agent.max_iterations = 6
    fragments = [
        "raw ceiling fragment one ",
        "raw ceiling fragment two ",
        "raw ceiling fragment three ",
        "raw ceiling fragment four",
    ]
    responses = iter(_response(part, finish_reason="length") for part in fragments)
    agent._interruptible_api_call = lambda _kwargs: next(responses)
    persisted = []
    agent._flush_messages_to_session_db = lambda messages, _history=None: (
        persisted.append(copy.deepcopy(messages)) or True
    )

    result = agent.run_conversation(
        "write a long response", conversation_history=[], task_id="task-1"
    )

    assert result["final_response"] == expected
    assert bool(result.get("failed")) is failed
    assert persisted
    for raw in fragments:
        assert raw.strip() not in str(result)
        assert all(raw.strip() not in str(snapshot) for snapshot in persisted)


@pytest.mark.parametrize(
    ("output_expr", "expected"),
    (
        pytest.param(
            'required_hook_result("behavioral.output.v1", "safe incomplete")',
            "safe incomplete",
            id="transformed",
        ),
        pytest.param(
            '(_ for _ in ()).throw(RuntimeError("raw incomplete hook secret"))',
            REQUIRED_LIFECYCLE_FAILURE_TEXT,
            id="failed",
        ),
    ),
)
def test_required_output_authorizes_codex_incomplete_before_persistence(
    tmp_path, monkeypatch, output_expr, expected
):
    from agent.conversation_loop import _apply_required_direct_result
    from agent.turn_truncation import continue_codex_incomplete

    home = tmp_path / "hermes"
    _install_required_plugin(
        home,
        'required_hook_result("behavioral.pre_llm.v1", None)',
        output_expr,
    )
    monkeypatch.setenv("HERMES_HOME", str(home))
    plugins_mod._reset_plugin_managers_for_tests()
    agent = _agent()
    agent._codex_incomplete_retries = 0
    agent._current_turn_id = "turn-incomplete"
    agent._current_api_request_id = "api-incomplete"
    interim = MagicMock()
    agent._emit_interim_assistant_message = interim
    persisted = []
    agent._persist_session = lambda messages, _history=None: persisted.append(
        copy.deepcopy(messages)
    )
    messages = [{"role": "user", "content": "continue"}]
    raw_fragments = [
        "raw incomplete fragment one",
        "raw incomplete fragment two",
        "raw incomplete fragment three",
    ]
    result = None
    for raw in raw_fragments:
        result = continue_codex_incomplete(
            agent,
            _response(raw, finish_reason="incomplete").choices[0].message,
            "incomplete",
            messages=messages,
            conversation_history=[],
            api_call_count=1,
        )

    assert result is not None
    assert persisted == []
    assert interim.call_count == 0
    state = SimpleNamespace(
        effective_task_id="task-1",
        turn_id="turn-incomplete",
        conversation_history=[],
    )
    result = _apply_required_direct_result(agent, result, state)

    assert result["final_response"] == expected
    assert len(persisted) == 1
    observable = str((result, persisted, interim.call_args_list))
    assert all(raw not in observable for raw in raw_fragments)
    assert "raw incomplete hook secret" not in observable


@pytest.mark.parametrize(
    "post_tool_expr",
    [
        pytest.param(
            '(_ for _ in ()).throw(RuntimeError("raw post secret"))',
            id="raises",
        ),
        pytest.param(
            '(__import__("time").sleep(2.0), '
            'required_hook_result("behavioral.post_tool.v1", None))[1]',
            id="times-out",
        ),
    ],
)
def test_required_post_tool_failure_stops_before_next_provider_call(
    tmp_path, monkeypatch, post_tool_expr
):
    home = tmp_path / "hermes"
    _install_required_plugin(
        home,
        'required_hook_result("behavioral.pre_llm.v1", None)',
        'required_hook_result("behavioral.output.v1", None)',
        post_tool_expr=post_tool_expr,
    )
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(plugins_mod, "_resolve_hook_callback_timeout", lambda: 0.1)
    plugins_mod._reset_plugin_managers_for_tests()
    agent = _agent()
    agent.valid_tool_names.add("web_search")
    tool_calls = [
        SimpleNamespace(
            id=f"call-{index}",
            type="function",
            function=SimpleNamespace(
                name="web_search", arguments=json.dumps({"query": str(index)})
            ),
        )
        for index in (1, 2)
    ]
    responses = iter(
        (
            _response("", finish_reason="tool_calls", tool_calls=tool_calls),
            _response("provider must not be called again"),
        )
    )
    calls = []

    def provider(kwargs):
        calls.append(kwargs)
        return next(responses)

    agent._interruptible_api_call = provider
    agent.compression_enabled = True
    agent.context_compressor.should_compress = lambda _tokens: True
    compress = MagicMock(
        side_effect=AssertionError("compression must not run after settlement failure")
    )
    agent._compress_context = compress
    persisted = []
    progress = []
    completed = []
    displayed = []
    agent._flush_messages_to_session_db = lambda messages, _history=None: (
        persisted.append(copy.deepcopy(messages)) or True
    )
    agent.tool_progress_callback = lambda *parts, **kwargs: progress.append(
        (copy.deepcopy(parts), copy.deepcopy(kwargs))
    )
    agent.tool_complete_callback = lambda *parts, **kwargs: completed.append(
        (copy.deepcopy(parts), copy.deepcopy(kwargs))
    )
    agent._vprint = lambda message, **_kwargs: displayed.append(message)
    raw_tool_result = "raw tool result that settlement did not authorize"
    with patch(
        "model_tools.handle_function_call", return_value=raw_tool_result
    ) as execute:
        result = agent.run_conversation(
            "search",
            conversation_history=[],
            task_id="task-1",
        )

    assert len(calls) == 1
    assert execute.call_count == 1
    assert compress.call_count == 0
    assert result["failed"] is True
    assert result["final_response"] == REQUIRED_LIFECYCLE_FAILURE_TEXT
    assert "raw post secret" not in str(result)
    assert raw_tool_result not in str(
        (result, persisted, progress, completed, displayed)
    )
    tool_results = [
        message for message in result["messages"] if message.get("role") == "tool"
    ]
    assert len(tool_results) == 2
    assert all(
        REQUIRED_LIFECYCLE_FAILURE_TEXT in message["content"]
        for message in tool_results
    )
    assert [message["effect_disposition"] for message in tool_results] == [
        "none",
        "none",
    ]
    assert completed == []
    assert all("tool.completed" not in str(event) for event in progress)


@pytest.mark.parametrize(
    ("response_id", "call_id", "expected_pairing_id"),
    (
        pytest.param("call-batch-health", None, "call-batch-health", id="id-only"),
        pytest.param(None, "call-batch-health", "call-batch-health", id="call-id-only"),
        pytest.param(
            "call-batch-health|item-batch-health",
            None,
            "call-batch-health",
            id="composite-id",
        ),
    ),
)
def test_batch_local_required_lifecycle_failure_stops_before_compression_or_provider(
    monkeypatch, response_id, call_id, expected_pairing_id,
):
    from agent.transports.types import NormalizedResponse, ToolCall
    from hermes_cli.required_lifecycle import RequiredLifecycleError

    agent = _agent()
    agent.valid_tool_names.add("web_search")
    tool_call = ToolCall(
        id=response_id,
        name="web_search",
        arguments="{}",
        provider_data={"call_id": call_id} if call_id else None,
    )
    normalized = NormalizedResponse(
        content="",
        tool_calls=[tool_call],
        finish_reason="tool_calls",
    )
    monkeypatch.setattr(
        agent._get_transport(), "normalize_response", lambda _response: normalized
    )
    provider_calls = []

    def provider(kwargs):
        provider_calls.append(kwargs)
        return _response("", finish_reason="tool_calls", tool_calls=[tool_call])

    health_checks = 0

    def health(**_kwargs):
        nonlocal health_checks
        health_checks += 1
        if health_checks >= 2:
            raise RequiredLifecycleError("required_lifecycle_generation_changed")

    agent._interruptible_api_call = provider
    agent.compression_enabled = True
    agent.context_compressor.should_compress = lambda _tokens: True
    compress = MagicMock(
        side_effect=AssertionError("compression must not run after batch health failure")
    )
    agent._compress_context = compress
    persisted = []
    agent._flush_messages_to_session_db = lambda messages, _history=None: (
        persisted.append(copy.deepcopy(messages)) or True
    )

    with (
        patch(
            "hermes_cli.plugins.assert_required_lifecycle_turn_healthy",
            side_effect=health,
        ),
        patch("model_tools.handle_function_call") as execute,
    ):
        result = agent.run_conversation(
            "search", conversation_history=[], task_id="task-1"
        )

    assert len(provider_calls) == 1
    execute.assert_not_called()
    compress.assert_not_called()
    assert result["failed"] is True
    assert result["final_response"] == REQUIRED_LIFECYCLE_FAILURE_TEXT
    assistant_calls = [
        message
        for message in result["messages"]
        if message.get("role") == "assistant" and message.get("tool_calls")
    ]
    assert assistant_calls[-1]["tool_calls"][0]["id"] == expected_pairing_id
    tool_results = [
        message for message in result["messages"] if message.get("role") == "tool"
    ]
    assert [message["tool_call_id"] for message in tool_results] == [expected_pairing_id]
    assert tool_results[0]["effect_disposition"] == "none"
    assert persisted and persisted[-1] == result["messages"]


def test_required_post_failure_quarantines_quiet_error_output(
    tmp_path, monkeypatch
):
    home = tmp_path / "hermes"
    _install_required_plugin(
        home,
        'required_hook_result("behavioral.pre_llm.v1", None)',
        'required_hook_result("behavioral.output.v1", None)',
        post_tool_expr='(_ for _ in ()).throw(RuntimeError("raw post secret"))',
    )
    monkeypatch.setenv("HERMES_HOME", str(home))
    plugins_mod._reset_plugin_managers_for_tests()
    agent = _agent()
    agent.platform = "cli"
    agent.valid_tool_names.add("web_search")
    agent.tool_progress_callback = None
    agent._should_start_quiet_spinner = lambda: False
    displayed = []
    persisted = []
    agent._vprint = lambda message, **_kwargs: displayed.append(message)
    agent._flush_messages_to_session_db = lambda messages, _history=None: (
        persisted.append(copy.deepcopy(messages)) or True
    )
    tool_call = SimpleNamespace(
        id="call-quiet",
        type="function",
        function=SimpleNamespace(
            name="web_search", arguments=json.dumps({"query": "atlas"})
        ),
    )
    agent._interruptible_api_call = lambda _kwargs: _response(
        "", finish_reason="tool_calls", tool_calls=[tool_call]
    )
    raw_tool_result = json.dumps(
        {"error": "raw quiet result secret"}, ensure_ascii=False
    )

    with patch("model_tools.handle_function_call", return_value=raw_tool_result):
        result = agent.run_conversation(
            "search", conversation_history=[], task_id="task-1"
        )

    assert result["failed"] is True
    assert result["final_response"] == REQUIRED_LIFECYCLE_FAILURE_TEXT
    assert raw_tool_result not in str((result, persisted, displayed))
    assert "raw quiet result secret" not in str((result, persisted, displayed))
    assert displayed
    assert "Hermes blocked this turn" in str(displayed)


def test_required_post_persistence_failure_stops_deferred_spinner_safely(
    tmp_path, monkeypatch
):
    from agent.tool_executor import execute_tool_calls_sequential

    home = tmp_path / "hermes"
    _install_required_plugin(
        home,
        'required_hook_result("behavioral.pre_llm.v1", None)',
        'required_hook_result("behavioral.output.v1", None)',
    )
    monkeypatch.setenv("HERMES_HOME", str(home))
    plugins_mod._reset_plugin_managers_for_tests()
    agent = _agent()
    agent.platform = "cli"
    agent.valid_tool_names.add("web_search")
    agent._current_turn_id = "turn-persistence"
    agent._current_api_request_id = "api-persistence"
    agent._flush_messages_to_session_db = MagicMock(return_value=False)
    spinner = MagicMock()
    tool_call = SimpleNamespace(
        id="call-persistence",
        type="function",
        function=SimpleNamespace(name="web_search", arguments="{}"),
    )
    raw_tool_result = json.dumps({"error": "raw persistence secret"})

    with (
        patch(
            "agent.tool_executor._start_quiet_tool_spinner",
            return_value=spinner,
        ),
        patch(
            "model_tools.handle_function_call", return_value=raw_tool_result
        ),
    ):
        execute_tool_calls_sequential(
            agent,
            SimpleNamespace(tool_calls=[tool_call]),
            [],
            effective_task_id="task-1",
        )

    assert agent._incremental_persistence_failed is True
    spinner.stop.assert_called_once()
    assert "raw persistence secret" not in str(spinner.stop.call_args)


@pytest.mark.parametrize(
    ("scenario", "arguments", "post_tool_expr"),
    (
        pytest.param(
            "invalid_arguments",
            '{"raw invalid secret":',
            '(_ for _ in ()).throw(RuntimeError("raw invalid post secret"))',
            id="invalid-arguments-raises",
        ),
        pytest.param(
            "interrupted",
            "{}",
            '(__import__("time").sleep(2.0), '
            'required_hook_result("behavioral.post_tool.v1", None))[1]',
            id="pre-dispatch-interrupt-times-out",
        ),
    ),
)
def test_required_post_failure_settles_preexecution_terminal_paths(
    tmp_path, monkeypatch, scenario, arguments, post_tool_expr
):
    from agent.tool_executor import execute_tool_calls_sequential
    from hermes_cli.required_lifecycle import RequiredLifecycleError

    home = tmp_path / "hermes"
    _install_required_plugin(
        home,
        'required_hook_result("behavioral.pre_llm.v1", None)',
        'required_hook_result("behavioral.output.v1", None)',
        post_tool_expr=post_tool_expr,
    )
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(plugins_mod, "_resolve_hook_callback_timeout", lambda: 0.1)
    plugins_mod._reset_plugin_managers_for_tests()
    agent = _agent()
    agent.valid_tool_names.add("web_search")
    agent._current_turn_id = "turn-preexecution"
    agent._current_api_request_id = "api-preexecution"
    agent._interrupt_requested = scenario == "interrupted"
    persisted = []
    displayed = []
    agent._flush_messages_to_session_db = lambda messages, _history=None: (
        persisted.append(copy.deepcopy(messages)) or True
    )
    agent._vprint = lambda message, **_kwargs: displayed.append(message)
    tool_call = SimpleNamespace(
        id="call-preexecution",
        type="function",
        function=SimpleNamespace(name="web_search", arguments=arguments),
    )
    assistant_message = SimpleNamespace(tool_calls=[tool_call])
    messages = []

    with (
        patch("model_tools.handle_function_call") as execute,
        pytest.raises(RequiredLifecycleError),
    ):
        execute_tool_calls_sequential(
            agent,
            assistant_message,
            messages,
            effective_task_id="task-1",
        )

    assert execute.call_count == 0
    assert len(messages) == 1
    assert messages[0]["role"] == "tool"
    assert messages[0]["tool_call_id"] == "call-preexecution"
    assert REQUIRED_LIFECYCLE_FAILURE_TEXT in messages[0]["content"]
    assert messages[0]["effect_disposition"] == "none"
    assert persisted and persisted[-1] == messages
    assert "raw invalid secret" not in str((messages, persisted, displayed))
    assert "raw invalid post secret" not in str((messages, persisted, displayed))
    assert "Tool execution cancelled" not in str((messages, persisted))


def test_required_pre_tool_failure_blocks_sequential_dispatch(
    tmp_path, monkeypatch
):
    home = tmp_path / "hermes"
    _install_required_plugin(
        home,
        'required_hook_result("behavioral.pre_llm.v1", None)',
        'required_hook_result("behavioral.output.v1", None)',
        pre_tool_expr='(_ for _ in ()).throw(RuntimeError("raw pre secret"))',
    )
    monkeypatch.setenv("HERMES_HOME", str(home))
    plugins_mod._reset_plugin_managers_for_tests()
    agent = _agent()
    agent.valid_tool_names.add("web_search")
    tool_call = SimpleNamespace(
        id="call-1",
        type="function",
        function=SimpleNamespace(name="web_search", arguments="{}"),
    )
    agent._interruptible_api_call = lambda _kwargs: _response(
        "", finish_reason="tool_calls", tool_calls=[tool_call]
    )

    with patch("model_tools.handle_function_call") as execute:
        result = agent.run_conversation(
            "search", conversation_history=[], task_id="task-1"
        )

    assert execute.call_count == 0
    assert result["failed"] is True
    assert result["final_response"] == REQUIRED_LIFECYCLE_FAILURE_TEXT
    assert "raw pre secret" not in str(result)
    tool_results = [
        message for message in result["messages"] if message.get("role") == "tool"
    ]
    assert len(tool_results) == 1
    assert tool_results[0]["effect_disposition"] == "none"


@pytest.mark.parametrize("block_kind", ("scope", "guardrail"))
def test_required_post_failure_marks_policy_block_as_no_effect(
    monkeypatch, block_kind
):
    from agent.tool_executor import execute_tool_calls_sequential
    from hermes_cli.required_lifecycle import RequiredLifecycleError

    agent = _agent()
    agent.valid_tool_names.add("terminal")
    agent._current_turn_id = "turn-policy-block"
    agent._current_api_request_id = "api-policy-block"
    persisted = []
    agent._flush_messages_to_session_db = lambda messages, _history=None: (
        persisted.append(copy.deepcopy(messages)) or True
    )
    if block_kind == "scope":
        monkeypatch.setattr(
            "agent.tool_executor._unwrap_tool_search_call",
            lambda *_args, **_kwargs: ("terminal", {}, "scope denied"),
        )
    else:
        agent._tool_guardrails.before_call = MagicMock(
            return_value=SimpleNamespace(
                allows_execution=False,
                message="guardrail denied",
            )
        )
    call = SimpleNamespace(
        id=f"call-{block_kind}",
        type="function",
        function=SimpleNamespace(name="terminal", arguments="{}"),
    )
    required_error = RequiredLifecycleError(
        "required_lifecycle_delivery_failed", "post_tool_call"
    )

    with (
        patch("hermes_cli.plugins.requires_hook", return_value=True),
        patch("model_tools.handle_function_call") as execute,
        patch(
            "agent.tool_executor._emit_terminal_post_tool_call",
            side_effect=required_error,
        ),
        pytest.raises(RequiredLifecycleError),
    ):
        execute_tool_calls_sequential(
            agent,
            SimpleNamespace(tool_calls=[call]),
            [],
            effective_task_id="task-policy-block",
        )

    execute.assert_not_called()
    tool_results = [
        message for message in persisted[-1] if message.get("role") == "tool"
    ]
    assert len(tool_results) == 1
    assert tool_results[0]["effect_disposition"] == "none"
    assert REQUIRED_LIFECYCLE_FAILURE_TEXT in tool_results[0]["content"]


def test_required_post_failure_scopes_settlement_to_current_tool_frame(monkeypatch):
    from agent.tool_executor import execute_tool_calls_sequential
    from hermes_cli.required_lifecycle import RequiredLifecycleError

    agent = _agent()
    agent.valid_tool_names.add("terminal")
    agent._current_turn_id = "turn-current"
    agent._current_api_request_id = "api-current"
    persisted = []
    agent._flush_messages_to_session_db = lambda messages, _history=None: (
        persisted.append(copy.deepcopy(messages)) or True
    )
    old_result = {
        "role": "tool",
        "tool_call_id": "reused-call-id",
        "name": "terminal",
        "content": "old turn result",
    }
    messages = [old_result]
    current_call = SimpleNamespace(
        id="reused-call-id",
        type="function",
        function=SimpleNamespace(name="terminal", arguments="{}"),
    )
    required_error = RequiredLifecycleError(
        "required_lifecycle_delivery_failed", "post_tool_call"
    )

    with (
        patch("hermes_cli.plugins.requires_hook", return_value=True),
        patch("model_tools.handle_function_call", return_value="raw current result"),
        patch(
            "agent.tool_executor._emit_terminal_post_tool_call",
            side_effect=required_error,
        ),
        pytest.raises(RequiredLifecycleError) as caught,
    ):
        execute_tool_calls_sequential(
            agent,
            SimpleNamespace(tool_calls=[current_call]),
            messages,
            effective_task_id="task-current",
        )

    assert caught.value is required_error
    assert messages[0] == old_result
    assert len(messages) == 2
    assert messages[1]["tool_call_id"] == "reused-call-id"
    assert messages[1]["effect_disposition"] == "unknown"
    assert REQUIRED_LIFECYCLE_FAILURE_TEXT in messages[1]["content"]
    assert "raw current result" not in str((messages, persisted))
    assert persisted and persisted[-1] == messages


def test_required_post_failure_does_not_hide_duplicate_current_call_id(monkeypatch):
    from agent.tool_executor import execute_tool_calls_sequential
    from hermes_cli.required_lifecycle import RequiredLifecycleError

    agent = _agent()
    agent.valid_tool_names.add("terminal")
    agent._current_turn_id = "turn-duplicate"
    agent._current_api_request_id = "api-duplicate"
    persisted = []
    agent._flush_messages_to_session_db = lambda messages, _history=None: (
        persisted.append(copy.deepcopy(messages)) or True
    )
    calls = [
        SimpleNamespace(
            id="duplicate-call-id",
            type="function",
            function=SimpleNamespace(name="terminal", arguments="{}"),
        )
        for _ in range(2)
    ]
    required_error = RequiredLifecycleError(
        "required_lifecycle_delivery_failed", "post_tool_call"
    )

    with (
        patch("hermes_cli.plugins.requires_hook", return_value=True),
        patch(
            "model_tools.handle_function_call",
            side_effect=("first result", "raw second result"),
        ) as execute,
        patch(
            "agent.tool_executor._emit_terminal_post_tool_call",
            side_effect=(None, required_error),
        ),
        pytest.raises(RequiredLifecycleError) as caught,
    ):
        execute_tool_calls_sequential(
            agent,
            SimpleNamespace(tool_calls=calls),
            [],
            effective_task_id="task-duplicate",
        )

    assert caught.value is required_error
    assert execute.call_count == 2
    tool_results = [
        message for message in persisted[-1] if message.get("role") == "tool"
    ]
    assert len(tool_results) == 2
    assert tool_results[0]["content"] == "first result"
    assert tool_results[1]["effect_disposition"] == "unknown"
    assert REQUIRED_LIFECYCLE_FAILURE_TEXT in tool_results[1]["content"]
    assert "raw second result" not in str(persisted)


def test_required_pre_tool_failure_blocks_direct_invoke_tool(
    tmp_path, monkeypatch
):
    home = tmp_path / "hermes"
    _install_required_plugin(
        home,
        'required_hook_result("behavioral.pre_llm.v1", None)',
        'required_hook_result("behavioral.output.v1", None)',
        pre_tool_expr='(_ for _ in ()).throw(RuntimeError("raw direct secret"))',
    )
    monkeypatch.setenv("HERMES_HOME", str(home))
    plugins_mod._reset_plugin_managers_for_tests()
    agent = _agent()
    agent._current_turn_id = "turn-direct"
    agent._current_api_request_id = "api-direct"
    agent.valid_tool_names.add("web_search")

    with patch("model_tools.handle_function_call") as execute:
        result = agent._invoke_tool(
            "web_search", {"query": "x"}, "task-1", tool_call_id="call-1"
        )

    assert execute.call_count == 0
    assert REQUIRED_LIFECYCLE_FAILURE_TEXT in result
    assert "raw direct secret" not in result


@pytest.mark.parametrize("entrypoint", ("sequential", "invoke", "registry"))
def test_required_pre_tool_error_reaches_every_dispatch_boundary(
    monkeypatch, entrypoint
):
    from hermes_cli.required_lifecycle import RequiredLifecycleError

    required_error = RequiredLifecycleError(
        "required_lifecycle_delivery_failed", "pre_tool_call"
    )
    monkeypatch.setattr(
        plugins_mod,
        "_dispatch_pre_tool_call_hooks",
        MagicMock(side_effect=required_error),
    )
    agent = SimpleNamespace(
        session_id="required-pre-boundary",
        _current_turn_id="turn-pre-boundary",
        _current_api_request_id="api-pre-boundary",
    )
    registry_dispatch = None

    if entrypoint == "sequential":
        from agent.tool_executor import _ToolCallRef, _pre_tool_block

        invoke = lambda: _pre_tool_block(  # noqa: E731
            agent,
            _ToolCallRef(
                "web_search", {}, "task-1", "call-pre-boundary", []
            ),
        )
    elif entrypoint == "invoke":
        from agent.agent_runtime_helpers import _pre_tool_block_message

        invoke = lambda: _pre_tool_block_message(  # noqa: E731
            agent,
            "web_search",
            {},
            "task-1",
            "call-pre-boundary",
            [],
        )
    else:
        from model_tools import handle_function_call, registry

        registry_dispatch = MagicMock(
            side_effect=AssertionError("registry dispatch must not start")
        )
        monkeypatch.setattr(registry, "dispatch", registry_dispatch)
        invoke = lambda: handle_function_call(  # noqa: E731
            "web_search",
            {},
            task_id="task-1",
            session_id="required-pre-boundary",
            tool_call_id="call-pre-boundary",
            turn_id="turn-pre-boundary",
            api_request_id="api-pre-boundary",
            skip_tool_request_middleware=True,
            skip_tool_execution_middleware=True,
        )

    with pytest.raises(RequiredLifecycleError) as caught:
        invoke()

    assert caught.value is required_error
    if registry_dispatch is not None:
        registry_dispatch.assert_not_called()


def test_required_pre_tool_blocks_uncorrelated_external_registry_caller(
    tmp_path, monkeypatch
):
    from hermes_cli.required_lifecycle import RequiredLifecycleError
    from model_tools import handle_function_call, registry

    home = tmp_path / "hermes"
    _install_required_plugin(
        home,
        'required_hook_result("behavioral.pre_llm.v1", None)',
        'required_hook_result("behavioral.output.v1", None)',
    )
    monkeypatch.setenv("HERMES_HOME", str(home))
    plugins_mod._reset_plugin_managers_for_tests()
    registry_dispatch = MagicMock(
        side_effect=AssertionError("uncorrelated external dispatch must not start")
    )
    monkeypatch.setattr(registry, "dispatch", registry_dispatch)

    with pytest.raises(RequiredLifecycleError) as caught:
        handle_function_call(
            "terminal",
            {"command": "true"},
            skip_tool_request_middleware=True,
            skip_tool_execution_middleware=True,
        )

    assert caught.value.reason_code == "required_lifecycle_identity_missing"
    registry_dispatch.assert_not_called()


def test_required_output_hides_tool_narration_before_persist_and_interim_egress(
    tmp_path, monkeypatch
):
    home = tmp_path / "hermes"
    _install_required_plugin(
        home,
        'required_hook_result("behavioral.pre_llm.v1", None)',
        'required_hook_result("behavioral.output.v1", "safe answer")',
    )
    monkeypatch.setenv("HERMES_HOME", str(home))
    plugins_mod._reset_plugin_managers_for_tests()
    agent = _agent()
    agent.valid_tool_names.add("web_search")
    raw_text = "raw narration before tool"
    tool_call = SimpleNamespace(
        id="call-1",
        type="function",
        extra_content={"thought_signature": "private tool signature"},
        function=SimpleNamespace(name="web_search", arguments="{}"),
    )
    first_response = _response(
        raw_text,
        finish_reason="tool_calls",
        tool_calls=[tool_call],
    )
    first_response.choices[0].message.reasoning_content = "private continuity"
    first_response.choices[0].message.reasoning_details = [
        {"type": "reasoning", "text": "private chain of thought"}
    ]
    responses = iter((first_response, _response("raw terminal answer")))
    provider_calls = []

    def provider(kwargs):
        provider_calls.append(copy.deepcopy(kwargs))
        return next(responses)

    agent._interruptible_api_call = provider
    persisted = []
    interim = []
    progress = []
    displayed = []
    post_api = []
    manager = plugins_mod._delivery_manager()
    manager._hooks.setdefault("post_api_request", []).append(
        lambda **payload: post_api.append(copy.deepcopy(payload))
    )
    agent._flush_messages_to_session_db = lambda messages, _history=None: (
        persisted.append(copy.deepcopy(messages)) or True
    )
    agent.interim_assistant_callback = lambda message: interim.append(
        copy.deepcopy(message)
    )
    agent.tool_progress_callback = lambda *parts: progress.append(parts)
    agent._vprint = lambda message, **_kwargs: displayed.append(message)

    with patch("model_tools.handle_function_call", return_value="search result"):
        result = agent.run_conversation(
            "search",
            conversation_history=[],
            task_id="task-1",
        )

    observable = str((result, persisted, interim, progress, displayed, post_api))
    assert result["final_response"] == "safe answer"
    assert raw_text not in observable
    assert "raw terminal answer" not in observable
    assert "private continuity" not in observable
    assert "private chain of thought" not in observable
    assert "private tool signature" not in observable
    assert interim == []
    assert post_api == []
    assert any(
        message.get("reasoning_details")
        == [{"type": "reasoning", "text": "private chain of thought"}]
        for message in provider_calls[1]["messages"]
        if isinstance(message, dict)
    ), provider_calls[1]["messages"]


def test_required_output_authorizes_content_filter_direct_return(
    tmp_path, monkeypatch, caplog
):
    home = tmp_path / "hermes"
    _install_required_plugin(
        home,
        'required_hook_result("behavioral.pre_llm.v1", None)',
        'required_hook_result("behavioral.output.v1", "safe refusal")',
    )
    monkeypatch.setenv("HERMES_HOME", str(home))
    plugins_mod._reset_plugin_managers_for_tests()
    agent = _agent()
    raw = "private provider refusal"
    agent._interruptible_api_call = lambda _kwargs: _response(
        raw, finish_reason="content_filter"
    )

    result = agent.run_conversation(
        "do work", conversation_history=[], task_id="task-1"
    )

    assert result["final_response"] == "safe refusal"
    assert result["pre_transform_response"] is None
    assert raw not in str(result)
    assert raw not in caplog.text


def test_exceptional_turn_exit_scrubs_ephemeral_provider_continuity(monkeypatch):
    agent = _agent()

    def fail_after_quarantine(*_args, **_kwargs):
        message = {
            "role": "assistant",
            "timestamp": 1.0,
            "finish_reason": "tool_calls",
            "reasoning_content": "private continuity",
            "tool_calls": [{"id": "call-1", "call_id": "call-1"}],
        }
        quarantine_required_provider_fields(agent, message)
        raise RuntimeError("provider failed")

    monkeypatch.setattr("agent.conversation_loop.run_conversation", fail_after_quarantine)

    with pytest.raises(RuntimeError, match="provider failed"):
        agent.run_conversation("hello", conversation_history=[])

    assert agent._required_provider_continuity == {}


def test_outer_turn_scope_restores_required_stream_state_after_direct_return():
    """A direct inner-loop return must not poison the cached agent's next turn."""
    agent = _agent()
    original_delta_callback = object()
    agent.stream_delta_callback = None
    agent._disable_streaming = True
    agent._required_lifecycle_stream_delta_callback = original_delta_callback
    agent._required_lifecycle_disable_streaming = False
    agent._current_turn_id = "turn-direct-return"

    with patch(
        "agent.conversation_loop.run_conversation",
        return_value={
            "final_response": "bounded failure",
            "messages": [],
            "api_calls": 0,
            "completed": False,
            "failed": True,
        },
    ):
        agent.run_conversation(
            "do work",
            conversation_history=[],
            task_id="task-direct-return",
        )

    assert agent.stream_delta_callback is original_delta_callback
    assert agent._disable_streaming is False
    assert not hasattr(agent, "_required_lifecycle_stream_delta_callback")
    assert not hasattr(agent, "_required_lifecycle_disable_streaming")
