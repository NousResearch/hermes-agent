import json
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from agent.turn_result_ledger import TURN_RESULT_LEDGER_MARKER
from agent.tool_guardrails import ToolGuardrailDecision
from agent.turn_result_ledger import MIN_SUBSTANTIVE_TOOL_COMPLETIONS
from run_agent import AIAgent


def _tool_call(index: int, name: str, arguments: dict) -> SimpleNamespace:
    return SimpleNamespace(
        id=f"call-{index}",
        type="function",
        function=SimpleNamespace(name=name, arguments=json.dumps(arguments)),
    )


def _response(
    content: str,
    finish_reason: str = "stop",
    tool_calls: list | None = None,
    usage=None,
) -> SimpleNamespace:
    message = SimpleNamespace(
        content=content,
        tool_calls=tool_calls or [],
        reasoning_content=None,
        reasoning_details=None,
    )
    return SimpleNamespace(
        choices=[SimpleNamespace(message=message, finish_reason=finish_reason)],
        usage=usage,
    )


@pytest.fixture
def long_running_agent(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
    monkeypatch.setenv("HERMES_VERIFY_ON_STOP", "0")
    monkeypatch.setattr(
        "hermes_cli.plugins.invoke_hook", lambda *args, **kwargs: [], raising=False
    )

    with (
        patch("run_agent.get_tool_definitions", return_value=[]),
        patch("run_agent.check_toolset_requirements", return_value={}),
        patch("run_agent.OpenAI"),
    ):
        agent = AIAgent(
            session_id="long-turn-final-response",
            model="test/model",
            api_key="test-key",
            base_url="https://example.invalid/v1",
            provider="openai-compat",
            enabled_toolsets=[],
            max_iterations=170,
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
        )

    agent.valid_tool_names = ["patch", "read_file", "terminal"]
    agent.tools = []
    agent._cached_system_prompt = "stable test prompt"
    agent._session_db = None
    agent._session_json_enabled = False
    agent.save_trajectories = False
    agent.compression_enabled = False
    agent._cleanup_task_resources = lambda *args, **kwargs: None
    agent._save_trajectory = lambda *args, **kwargs: None
    agent._persist_session = lambda *args, **kwargs: None
    yield agent

    from agent.auxiliary_client import clear_runtime_main

    clear_runtime_main()


def _install_tool_executor(agent, *, rotate_after_index=None):
    def execute_tool_calls(assistant_message, messages, task_id, api_call_count):
        for call in assistant_message.tool_calls:
            index = int(call.id.rsplit("-", 1)[-1])
            if call.function.name == "patch":
                result = (
                    "Implemented bounded resumable reconciliation in ricochet_history.py. "
                    "EARLY_LEDGER_SENTINEL."
                )
                agent._turn_file_mutation_paths.add("ricochet_history.py")
            elif call.function.name == "terminal" and index == 148:
                result = "63 passed in 4.2s; exit_code=0"
            elif call.function.name == "terminal":
                result = "All checks passed; exit_code=0"
            else:
                result = f"middle inspection {index}"
            messages.append({
                "role": "tool",
                "name": call.function.name,
                "tool_call_id": call.id,
                "content": result,
            })
            if index == rotate_after_index:
                current_user_index = max(
                    message_index
                    for message_index, message in enumerate(messages)
                    if message.get("role") == "user"
                )
                messages[:] = [
                    *messages[: current_user_index + 1],
                    *messages[-20:],
                ]

    agent._execute_tool_calls = execute_tool_calls


def _next_tool_response(index: int) -> SimpleNamespace:
    if index == 0:
        tool_call = _tool_call(
            index,
            "patch",
            {
                "path": "ricochet_history.py",
                "patch": "Implement one-worker priority and round-robin reconciliation.",
            },
        )
    elif index < 148:
        tool_call = _tool_call(index, "read_file", {"path": f"fixture-{index}.txt"})
    else:
        command = "python -m pytest -q" if index == 148 else "ruff check ."
        tool_call = _tool_call(index, "terminal", {"command": command})
    return _response("", "tool_calls", [tool_call])


def _install_fresh_finalizer(agent, observed):
    def summary_call(**api_kwargs):
        wire_messages = json.dumps(api_kwargs["messages"], ensure_ascii=False)
        observed.append(wire_messages)
        assert [message["role"] for message in api_kwargs["messages"]] == [
            "system",
            "user",
        ]
        assert TURN_RESULT_LEDGER_MARKER in wire_messages
        assert "bounded resumable reconciliation" in wire_messages
        assert "EARLY_LEDGER_SENTINEL" in wire_messages
        assert "63 passed" in wire_messages
        assert "tools" not in api_kwargs
        return _response(
            "Implemented bounded resumable reconciliation with one-worker priority "
            "and round-robin verification. Final verification: 63 passed and Ruff passed.",
            usage=SimpleNamespace(
                prompt_tokens=42_000,
                completion_tokens=500,
                total_tokens=42_500,
                prompt_tokens_details=None,
                completion_tokens_details=None,
            ),
        )

    agent._test_finalizer_call = summary_call


def _dispatch_model_or_finalizer(agent, model_call):
    def dispatch(api_kwargs):
        request_messages = api_kwargs.get("messages") or api_kwargs.get("input") or []
        wire = json.dumps(request_messages, ensure_ascii=False)
        finalizer_call = getattr(agent, "_test_finalizer_call", None)
        if finalizer_call is not None and TURN_RESULT_LEDGER_MARKER in wire:
            return finalizer_call(**api_kwargs)
        return model_call(api_kwargs)

    return dispatch


def test_long_normal_stop_uses_fresh_bounded_finalizer(long_running_agent):
    agent = long_running_agent
    _install_tool_executor(agent)
    streamed_text: list[str] = []
    agent.stream_delta_callback = streamed_text.append
    summary_requests: list[str] = []
    _install_fresh_finalizer(agent, summary_requests)
    main_requests: list[str] = []
    call_index = 0

    def model_call(api_kwargs):
        nonlocal call_index
        main_requests.append(json.dumps(api_kwargs["messages"], ensure_ascii=False))
        if call_index < 150:
            response = _next_tool_response(call_index)
            call_index += 1
            return response
        agent._fire_stream_delta("Fresh verification completed: 63 passed.")
        return _response("Fresh verification completed: 63 passed.")

    agent._interruptible_api_call = _dispatch_model_or_finalizer(agent, model_call)
    agent._interruptible_streaming_api_call = lambda api_kwargs, on_first_delta=None: (
        model_call(api_kwargs)
    )
    result = agent.run_conversation(
        "Implement the reconciliation approach you recommended.",
        conversation_history=[
            {
                "role": "assistant",
                "content": (
                    "I recommend bounded resumable reconciliation with one worker, "
                    "new-lead priority, and a round-robin verification cursor."
                ),
            }
        ],
    )

    assert result["api_calls"] == 152
    assert "bounded resumable reconciliation" in result["final_response"]
    assert "63 passed" in result["final_response"]
    assert "Ruff passed" in result["final_response"]
    assert result["response_transformed"] is False
    assert len(summary_requests) == 1
    assert "Fresh verification completed: 63 passed." in summary_requests[0]
    assert all(TURN_RESULT_LEDGER_MARKER not in request for request in main_requests)

    canonical_wire = json.dumps(result["messages"], ensure_ascii=False)
    assert TURN_RESULT_LEDGER_MARKER not in canonical_wire
    assert canonical_wire.count("Final verification: 63 passed") == 1
    assert result["messages"][-1]["content"] == result["final_response"]
    visible_stream = "".join(item for item in streamed_text if isinstance(item, str))
    assert visible_stream == result["final_response"]
    assert "Fresh verification completed" not in visible_stream
    assert agent.session_input_tokens == 42_000
    assert agent.session_output_tokens == 500
    assert agent.session_api_calls == 1


def test_long_turn_retains_early_work_across_production_message_rotation(
    long_running_agent,
):
    agent = long_running_agent
    _install_tool_executor(agent, rotate_after_index=59)
    summary_requests: list[str] = []
    _install_fresh_finalizer(agent, summary_requests)
    call_index = 0

    def model_call(api_kwargs):
        nonlocal call_index
        if call_index < 150:
            response = _next_tool_response(call_index)
            call_index += 1
            return response
        return _response("Fresh verification completed: 63 passed.")

    agent._interruptible_api_call = _dispatch_model_or_finalizer(agent, model_call)
    result = agent.run_conversation(
        "Implement the reconciliation approach you recommended.",
        conversation_history=[
            {
                "role": "assistant",
                "content": "Use bounded resumable reconciliation with one worker.",
            }
        ],
    )

    assert "EARLY_LEDGER_SENTINEL" in summary_requests[0]
    assert "63 passed" in summary_requests[0]
    assert "Final verification: 63 passed" in result["final_response"]


def test_long_iteration_limit_uses_same_fresh_bounded_finalizer(long_running_agent):
    agent = long_running_agent
    agent.max_iterations = 150
    _install_tool_executor(agent)
    summary_requests: list[str] = []
    _install_fresh_finalizer(agent, summary_requests)
    call_index = 0

    def model_call(api_kwargs):
        nonlocal call_index
        response = _next_tool_response(call_index)
        call_index += 1
        return response

    agent._interruptible_api_call = _dispatch_model_or_finalizer(agent, model_call)
    result = agent.run_conversation(
        "Implement the reconciliation approach you recommended.",
        conversation_history=[
            {
                "role": "assistant",
                "content": (
                    "I recommend bounded resumable reconciliation with one worker, "
                    "new-lead priority, and a round-robin verification cursor."
                ),
            }
        ],
    )

    assert result["api_calls"] == 151
    assert result["completed"] is False
    assert result["turn_exit_reason"].startswith("max_iterations_reached")
    assert "bounded resumable reconciliation" in result["final_response"]
    assert "63 passed" in result["final_response"]
    assert len(summary_requests) == 1
    assert TURN_RESULT_LEDGER_MARKER not in json.dumps(result["messages"])
    assert result["messages"][-1]["content"] == result["final_response"]


def test_short_turn_does_not_call_fresh_finalizer(long_running_agent):
    agent = long_running_agent
    summary_calls = 0

    def unexpected_summary(**api_kwargs):
        nonlocal summary_calls
        summary_calls += 1
        return _response("unexpected")

    client = SimpleNamespace(
        chat=SimpleNamespace(
            completions=SimpleNamespace(create=unexpected_summary),
        )
    )
    agent._ensure_primary_openai_client = lambda *args, **kwargs: client
    agent._test_finalizer_call = unexpected_summary
    agent._interruptible_api_call = lambda api_kwargs: _response("Short answer.")

    result = agent.run_conversation("Say hello.", conversation_history=[])

    assert result["final_response"] == "Short answer."
    assert result["response_transformed"] is False
    assert summary_calls == 0


def test_long_turn_finalizer_failure_preserves_original_draft(long_running_agent):
    agent = long_running_agent
    _install_tool_executor(agent)
    streamed_text: list[str] = []
    agent.stream_delta_callback = streamed_text.append
    call_index = 0
    summary_calls = 0

    def model_call(api_kwargs):
        nonlocal call_index
        if call_index < 150:
            response = _next_tool_response(call_index)
            call_index += 1
            return response
        agent._fire_stream_delta("Fresh verification completed: 63 passed.")
        return _response("Fresh verification completed: 63 passed.")

    def failed_summary(**api_kwargs):
        nonlocal summary_calls
        summary_calls += 1
        raise RuntimeError("finalizer unavailable")

    client = SimpleNamespace(
        chat=SimpleNamespace(
            completions=SimpleNamespace(create=failed_summary),
        )
    )
    agent._ensure_primary_openai_client = lambda *args, **kwargs: client
    agent._test_finalizer_call = failed_summary
    agent._interruptible_api_call = _dispatch_model_or_finalizer(agent, model_call)
    agent._interruptible_streaming_api_call = lambda api_kwargs, on_first_delta=None: (
        model_call(api_kwargs)
    )

    result = agent.run_conversation(
        "Implement the reconciliation approach you recommended.",
        conversation_history=[],
    )

    assert result["final_response"] == "Fresh verification completed: 63 passed."
    assert result["messages"][-1]["content"] == result["final_response"]
    assert result["response_transformed"] is False
    assert summary_calls == 1
    assert (
        "".join(item for item in streamed_text if isinstance(item, str))
        == result["final_response"]
    )


def test_long_empty_exhaustion_persists_then_publishes_only_explainer(
    long_running_agent,
):
    agent = long_running_agent
    _install_tool_executor(agent)
    streamed_text: list[str] = []
    persisted: list[list[dict]] = []
    agent.stream_delta_callback = streamed_text.append
    agent._persist_session = lambda messages, _history: persisted.append(
        json.loads(json.dumps(messages))
    )
    call_index = 0

    def model_call(api_kwargs):
        nonlocal call_index
        if call_index < MIN_SUBSTANTIVE_TOOL_COMPLETIONS:
            index = call_index
            call_index += 1
            return _response(
                "",
                "tool_calls",
                [_tool_call(index, "read_file", {"path": f"fixture-{index}.txt"})],
            )
        return _response("")

    agent._interruptible_api_call = model_call
    agent._interruptible_streaming_api_call = lambda api_kwargs, on_first_delta=None: (
        model_call(api_kwargs)
    )

    result = agent.run_conversation(
        "Inspect the fixtures and report the completed result.",
        conversation_history=[],
    )

    assert result["turn_exit_reason"] == "empty_response_exhausted"
    assert result["final_response"].strip() not in {"", "(empty)"}
    canonical_writes = [
        snapshot for snapshot in persisted if snapshot[-1]["role"] == "assistant"
    ]
    assert len(canonical_writes) == 1
    assert canonical_writes[0][-1]["content"] == result["final_response"]
    assert result["messages"][-1]["content"] == result["final_response"]
    visible = "".join(item for item in streamed_text if isinstance(item, str))
    assert visible == result["final_response"]
    assert "(empty)" not in visible


def test_long_finalizer_postprocesses_before_canonical_persist_and_publish(
    long_running_agent,
    monkeypatch,
):
    agent = long_running_agent
    _install_tool_executor(agent)
    streamed_text: list[str] = []
    persisted: list[list[dict]] = []
    agent.stream_delta_callback = streamed_text.append
    agent._persist_session = lambda messages, _history: persisted.append(
        json.loads(json.dumps(messages))
    )
    agent._file_mutation_verifier_enabled = lambda: True
    agent._format_file_mutation_failure_footer = lambda _failed: "MUTATION FOOTER"

    def plugin_hook(name, **kwargs):
        if name == "transform_llm_output":
            return [kwargs["response_text"] + "\nPLUGIN TRANSFORM"]
        return []

    monkeypatch.setattr("hermes_cli.plugins.invoke_hook", plugin_hook)

    def summary_call(**api_kwargs):
        wire_messages = json.dumps(api_kwargs["messages"], ensure_ascii=False)
        assert TURN_RESULT_LEDGER_MARKER in wire_messages
        assert "EARLY_LEDGER_SENTINEL" in wire_messages
        agent._turn_failed_file_mutations = {"ricochet_history.py": "patch failed"}
        return _response("REFINED LONG-TURN RESULT")

    client = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=summary_call))
    )
    agent._ensure_primary_openai_client = lambda *args, **kwargs: client
    agent._test_finalizer_call = summary_call
    call_index = 0

    def model_call(api_kwargs):
        nonlocal call_index
        if call_index < MIN_SUBSTANTIVE_TOOL_COMPLETIONS:
            index = call_index
            call_index += 1
            name = "patch" if index == 0 else "read_file"
            return _response(
                "",
                "tool_calls",
                [_tool_call(index, name, {"path": f"fixture-{index}.txt"})],
            )
        agent._fire_stream_delta("TERSE TERMINAL DRAFT")
        return _response("TERSE TERMINAL DRAFT")

    agent._interruptible_api_call = _dispatch_model_or_finalizer(agent, model_call)
    agent._interruptible_streaming_api_call = lambda api_kwargs, on_first_delta=None: (
        model_call(api_kwargs)
    )

    result = agent.run_conversation(
        "Implement the reconciliation approach you recommended.",
        conversation_history=[],
    )

    expected = "REFINED LONG-TURN RESULT\n\nMUTATION FOOTER\nPLUGIN TRANSFORM"
    assert result["final_response"] == expected
    assert result["response_transformed"] is False
    canonical_writes = [
        snapshot for snapshot in persisted if snapshot[-1]["role"] == "assistant"
    ]
    assert len(canonical_writes) == 1
    assert canonical_writes[0][-1]["content"] == expected
    assert result["messages"][-1]["content"] == expected
    assert "".join(item for item in streamed_text if isinstance(item, str)) == expected


def test_long_tool_prose_flushes_before_direct_guardrail_halt_exactly_once(
    long_running_agent,
):
    agent = long_running_agent
    streamed: list[str | None] = []
    persisted: list[list[dict]] = []
    agent.stream_delta_callback = streamed.append
    agent._persist_session = lambda messages, _history: persisted.append(
        json.loads(json.dumps(messages))
    )
    decision = ToolGuardrailDecision(
        action="halt",
        code="test_guardrail_halt",
        message="Repeated failure stopped safely.",
        tool_name="read_file",
        count=3,
    )

    def execute_tool_calls(assistant_message, messages, task_id, api_call_count):
        for call in assistant_message.tool_calls:
            index = int(call.id.rsplit("-", 1)[-1])
            messages.append({
                "role": "tool",
                "name": call.function.name,
                "tool_call_id": call.id,
                "content": f"inspection {index}",
            })
            if index == MIN_SUBSTANTIVE_TOOL_COMPLETIONS:
                agent._tool_guardrail_halt_decision = decision

    agent._execute_tool_calls = execute_tool_calls
    call_index = 0
    interim = "Interim evidence before the guarded tool."

    def model_call(api_kwargs):
        nonlocal call_index
        index = call_index
        call_index += 1
        if index == MIN_SUBSTANTIVE_TOOL_COMPLETIONS:
            agent._fire_stream_delta(interim)
            return _response(
                interim,
                "tool_calls",
                [_tool_call(index, "read_file", {"path": "guarded.txt"})],
            )
        return _response(
            "",
            "tool_calls",
            [_tool_call(index, "read_file", {"path": f"fixture-{index}.txt"})],
        )

    agent._interruptible_api_call = model_call
    agent._interruptible_streaming_api_call = lambda api_kwargs, on_first_delta=None: (
        model_call(api_kwargs)
    )

    result = agent.run_conversation("Inspect until the guardrail stops the run.")

    assert result["turn_exit_reason"] == "guardrail_halt"
    halt = result["final_response"]
    visible = "".join(item for item in streamed if isinstance(item, str))
    assert interim in visible
    assert visible.count(interim) == 1
    assert visible.count(halt) == 1
    canonical_writes = [
        snapshot for snapshot in persisted if snapshot[-1]["role"] == "assistant"
    ]
    assert len(canonical_writes) == 1
    assert canonical_writes[0][-1]["content"] == halt


def test_stop_during_fresh_finalizer_interrupts_turn_without_publishing_draft(
    long_running_agent,
):
    agent = long_running_agent
    _install_tool_executor(agent)
    streamed: list[str | None] = []
    agent.stream_delta_callback = streamed.append
    call_index = 0

    def model_call(api_kwargs):
        nonlocal call_index
        if call_index < MIN_SUBSTANTIVE_TOOL_COMPLETIONS:
            index = call_index
            call_index += 1
            return _response(
                "",
                "tool_calls",
                [_tool_call(index, "read_file", {"path": f"fixture-{index}.txt"})],
            )
        agent._fire_stream_delta("TERMINAL DRAFT HELD FOR FINALIZATION")
        return _response("TERMINAL DRAFT HELD FOR FINALIZATION")

    def dispatch(api_kwargs):
        request_messages = api_kwargs.get("messages") or api_kwargs.get("input") or []
        if TURN_RESULT_LEDGER_MARKER in json.dumps(request_messages):
            raise InterruptedError("Agent interrupted during API call")
        return model_call(api_kwargs)

    agent._interruptible_api_call = dispatch
    agent._interruptible_streaming_api_call = lambda api_kwargs, on_first_delta=None: (
        model_call(api_kwargs)
    )

    result = agent.run_conversation("Inspect the fixtures, then summarize.")

    assert result["interrupted"] is True
    assert result["turn_exit_reason"] == "interrupted_by_user"
    assert result["final_response"] is None
    assert result["api_calls"] == MIN_SUBSTANTIVE_TOOL_COMPLETIONS + 2
    assert agent.session_api_calls == 1
    assert "".join(item for item in streamed if isinstance(item, str)) == ""


def test_toolless_finalizer_suppresses_private_codex_deltas(monkeypatch):
    from agent import chat_completion_helpers

    visible_text: list[str] = []
    visible_reasoning: list[str] = []

    class StubAgent:
        def __init__(self):
            self._discard_private_response = False

        def _reset_stream_delivery_tracking(self):
            pass

        def _start_provider_response_gate(self, *, enabled, discard=False):
            self._discard_private_response = bool(enabled and discard)

        def _finish_provider_response_gate(self, *, terminal, discard=False):
            self._discard_private_response = False

        def _fire_stream_delta(self, text):
            if not self._discard_private_response:
                visible_text.append(text)

        def _fire_reasoning_delta(self, text):
            if not self._discard_private_response:
                visible_reasoning.append(text)

    agent = StubAgent()

    def fake_handle(agent, messages, api_call_count, **kwargs):
        assert messages == []
        assert kwargs["summary_request"] == "finalize"
        assert kwargs["announce"] is False
        agent._fire_stream_delta("private finalizer text")
        agent._fire_reasoning_delta("private finalizer reasoning")
        return "refined"

    monkeypatch.setattr(
        chat_completion_helpers,
        "handle_max_iterations",
        fake_handle,
    )

    assert (
        chat_completion_helpers.request_toolless_completion(agent, "finalize", 150)
        == "refined"
    )
    assert visible_text == []
    assert visible_reasoning == []

    agent._fire_stream_delta("visible")
    agent._fire_reasoning_delta("visible reasoning")
    assert visible_text == ["visible"]
    assert visible_reasoning == ["visible reasoning"]


def test_provider_response_gate_preserves_midstream_markdown_newlines(
    long_running_agent,
):
    agent = long_running_agent
    agent._reset_stream_delivery_tracking()
    agent._start_provider_response_gate(enabled=True)

    agent._fire_stream_delta("First paragraph")
    agent._fire_stream_delta("\n\nSecond paragraph")

    assert agent._current_provider_response_text() == (
        "First paragraph\n\nSecond paragraph"
    )


def test_gated_repeated_truncation_flushes_last_partial_before_direct_return(
    long_running_agent,
):
    agent = long_running_agent
    _install_tool_executor(agent)
    visible_stream: list[str] = []
    agent.stream_delta_callback = visible_stream.append
    call_index = 0
    truncation_index = 0

    def model_call(api_kwargs):
        nonlocal call_index, truncation_index
        if call_index < MIN_SUBSTANTIVE_TOOL_COMPLETIONS:
            response = _next_tool_response(call_index)
            call_index += 1
            return response
        part = f"TRUNCATED_PART_{truncation_index}"
        truncation_index += 1
        agent._fire_stream_delta(part)
        return _response(part, finish_reason="length")

    agent._interruptible_api_call = model_call
    agent._interruptible_streaming_api_call = (
        lambda api_kwargs, on_first_delta=None: model_call(api_kwargs)
    )

    result = agent.run_conversation("Complete a long task.")

    assert result["partial"] is True
    assert result["completed"] is False
    assert "TRUNCATED_PART_3" in result["final_response"]
    streamed_text = "".join(item for item in visible_stream if isinstance(item, str))
    assert "TRUNCATED_PART_3" in streamed_text


def test_gated_content_filter_publishes_canonical_refusal_once(long_running_agent):
    agent = long_running_agent
    _install_tool_executor(agent)
    visible_stream: list[str] = []
    persisted_snapshots: list[list[dict]] = []
    agent.stream_delta_callback = visible_stream.append
    agent._persist_session = lambda messages, history: persisted_snapshots.append(
        [dict(message) for message in messages]
    )
    call_index = 0

    def model_call(api_kwargs):
        nonlocal call_index
        if call_index < MIN_SUBSTANTIVE_TOOL_COMPLETIONS:
            response = _next_tool_response(call_index)
            call_index += 1
            return response
        agent._fire_stream_delta("PROVIDER_REFUSAL_EXPLANATION")
        return _response(
            "PROVIDER_REFUSAL_EXPLANATION",
            finish_reason="content_filter",
        )

    agent._interruptible_api_call = model_call
    agent._interruptible_streaming_api_call = (
        lambda api_kwargs, on_first_delta=None: model_call(api_kwargs)
    )

    result = agent.run_conversation("Complete a long task.")

    streamed_text = "".join(item for item in visible_stream if isinstance(item, str))
    assert streamed_text == result["final_response"]
    assert streamed_text.count("PROVIDER_REFUSAL_EXPLANATION") == 1
    assert "safety refusal" in streamed_text
    assert result["messages"][-1]["role"] == "assistant"
    assert result["messages"][-1]["content"] == result["final_response"]
    assert sum(
        bool(
            snapshot
            and snapshot[-1].get("role") == "assistant"
            and snapshot[-1].get("content") == result["final_response"]
        )
        for snapshot in persisted_snapshots
    ) == 1


def test_gated_terminal_provider_exception_persists_and_publishes_last_partial(
    long_running_agent,
    monkeypatch,
):
    import agent.conversation_loop as conversation_loop

    agent = long_running_agent
    _install_tool_executor(agent)
    visible_stream: list[str] = []
    persisted_snapshots: list[list[dict]] = []
    agent.stream_delta_callback = visible_stream.append
    agent._persist_session = lambda messages, history: persisted_snapshots.append(
        [dict(message) for message in messages]
    )
    monkeypatch.setattr(conversation_loop, "jittered_backoff", lambda *args, **kwargs: 0)
    monkeypatch.setattr(
        conversation_loop,
        "adaptive_rate_limit_backoff",
        lambda *args, **kwargs: 0,
    )
    call_index = 0
    failure_index = 0

    def model_call(api_kwargs):
        nonlocal call_index, failure_index
        if call_index < MIN_SUBSTANTIVE_TOOL_COMPLETIONS:
            response = _next_tool_response(call_index)
            call_index += 1
            return response
        partial = f"EXCEPTION_PART_{failure_index}"
        failure_index += 1
        agent._fire_stream_delta(partial)
        raise RuntimeError("provider failed after a partial stream")

    agent._interruptible_api_call = model_call
    agent._interruptible_streaming_api_call = (
        lambda api_kwargs, on_first_delta=None: model_call(api_kwargs)
    )

    result = agent.run_conversation("Complete a long task.")

    last_partial = f"EXCEPTION_PART_{failure_index - 1}"
    streamed_text = "".join(item for item in visible_stream if isinstance(item, str))
    assert result["failed"] is True
    assert last_partial in result["final_response"]
    assert streamed_text.count(last_partial) == 1
    assert result["messages"][-1]["role"] == "assistant"
    assert result["messages"][-1]["content"] == result["final_response"]
    assert sum(
        bool(
            snapshot
            and snapshot[-1].get("role") == "assistant"
            and last_partial in snapshot[-1].get("content", "")
        )
        for snapshot in persisted_snapshots
    ) == 1
    assert agent._active_provider_response_gate is None


def test_router_hidden_tool_truncation_persists_and_publishes_canonical_error(
    long_running_agent,
):
    agent = long_running_agent
    _install_tool_executor(agent)
    visible_stream: list[str] = []
    persisted_snapshots: list[list[dict]] = []
    agent.stream_delta_callback = visible_stream.append
    agent._persist_session = lambda messages, history: persisted_snapshots.append(
        [dict(message) for message in messages]
    )
    call_index = 0

    def model_call(api_kwargs):
        nonlocal call_index
        if call_index < MIN_SUBSTANTIVE_TOOL_COMPLETIONS:
            response = _next_tool_response(call_index)
            call_index += 1
            return response
        agent._fire_stream_delta("ROUTER_PARTIAL")
        truncated_call = SimpleNamespace(
            id="router-truncated",
            type="function",
            function=SimpleNamespace(
                name="read_file",
                arguments='{"path":"unfinished',
            ),
        )
        return _response(
            "ROUTER_PARTIAL",
            finish_reason="tool_calls",
            tool_calls=[truncated_call],
        )

    agent._interruptible_api_call = model_call
    agent._interruptible_streaming_api_call = (
        lambda api_kwargs, on_first_delta=None: model_call(api_kwargs)
    )

    result = agent.run_conversation("Complete a long task.")

    streamed_text = "".join(item for item in visible_stream if isinstance(item, str))
    assert result["final_response"] == "Response truncated due to output length limit"
    assert streamed_text == result["final_response"]
    assert "ROUTER_PARTIAL" not in streamed_text
    assert result["messages"][-1]["content"] == result["final_response"]
    assert sum(
        bool(
            snapshot
            and snapshot[-1].get("content") == result["final_response"]
        )
        for snapshot in persisted_snapshots
    ) == 1


def test_exception_classified_policy_refusal_is_persisted_and_gate_is_closed(
    long_running_agent,
):
    agent = long_running_agent
    _install_tool_executor(agent)
    visible_stream: list[str] = []
    persisted_snapshots: list[list[dict]] = []
    agent.stream_delta_callback = visible_stream.append
    agent._persist_session = lambda messages, history: persisted_snapshots.append(
        [dict(message) for message in messages]
    )
    call_index = 0

    class PolicyError(RuntimeError):
        status_code = 400

    def model_call(api_kwargs):
        nonlocal call_index
        if call_index < MIN_SUBSTANTIVE_TOOL_COMPLETIONS:
            response = _next_tool_response(call_index)
            call_index += 1
            return response
        raise PolicyError(
            "The response was filtered: ResponsibleAIPolicyViolation "
            "(finish_reason=content_filter)."
        )

    agent._interruptible_api_call = model_call
    agent._interruptible_streaming_api_call = (
        lambda api_kwargs, on_first_delta=None: model_call(api_kwargs)
    )

    result = agent.run_conversation("Complete a long task.")

    streamed_text = "".join(item for item in visible_stream if isinstance(item, str))
    assert result["failed"] is True
    assert result["error"].startswith("content_policy_blocked:")
    assert "safety filter" in result["final_response"]
    assert streamed_text == result["final_response"]
    assert result["messages"][-1]["content"] == result["final_response"]
    assert sum(
        bool(
            snapshot
            and snapshot[-1].get("content") == result["final_response"]
        )
        for snapshot in persisted_snapshots
    ) == 1
    assert agent._active_provider_response_gate is None


def test_overflowed_abnormal_gate_keeps_streamed_draft_as_canonical_response(
    long_running_agent,
):
    from agent.provider_response_gate import MAX_BUFFERED_VISIBLE_BYTES

    agent = long_running_agent
    _install_tool_executor(agent)
    visible_stream: list[str] = []
    persisted_snapshots: list[list[dict]] = []
    agent.stream_delta_callback = visible_stream.append
    agent._persist_session = lambda messages, history: persisted_snapshots.append(
        [dict(message) for message in messages]
    )
    call_index = 0
    overflowed_draft = "X" * (MAX_BUFFERED_VISIBLE_BYTES + 1)

    def model_call(api_kwargs):
        nonlocal call_index
        if call_index < MIN_SUBSTANTIVE_TOOL_COMPLETIONS:
            response = _next_tool_response(call_index)
            call_index += 1
            return response
        agent._fire_stream_delta(overflowed_draft)
        return _response(overflowed_draft, finish_reason="content_filter")

    agent._interruptible_api_call = model_call
    agent._interruptible_streaming_api_call = (
        lambda api_kwargs, on_first_delta=None: model_call(api_kwargs)
    )

    result = agent.run_conversation("Complete a long task.")

    streamed_text = "".join(item for item in visible_stream if isinstance(item, str))
    assert result["final_response"] == streamed_text
    assert streamed_text.endswith(overflowed_draft)
    assert streamed_text.count(overflowed_draft) == 1
    assert result["messages"][-1]["content"] == streamed_text
    assert sum(
        bool(snapshot and snapshot[-1].get("content") == streamed_text)
        for snapshot in persisted_snapshots
    ) == 1
    assert agent._active_provider_response_gate is None


def test_nous_rate_guard_commits_canonical_response_after_long_tool_turn(
    long_running_agent,
    monkeypatch,
):
    from agent import nous_rate_guard

    agent = long_running_agent
    agent.provider = "nous"
    _install_tool_executor(agent)
    visible_stream: list[str] = []
    persisted_snapshots: list[list[dict]] = []
    agent.stream_delta_callback = visible_stream.append
    agent._persist_session = lambda messages, history: persisted_snapshots.append(
        [dict(message) for message in messages]
    )
    agent._try_activate_fallback = lambda: False
    call_index = 0

    monkeypatch.setattr(
        nous_rate_guard,
        "nous_rate_limit_remaining",
        lambda: 60.0 if call_index >= MIN_SUBSTANTIVE_TOOL_COMPLETIONS else None,
    )
    monkeypatch.setattr(nous_rate_guard, "format_remaining", lambda remaining: "1m")

    def model_call(api_kwargs):
        nonlocal call_index
        response = _next_tool_response(call_index)
        call_index += 1
        return response

    agent._interruptible_api_call = model_call
    agent._interruptible_streaming_api_call = (
        lambda api_kwargs, on_first_delta=None: model_call(api_kwargs)
    )

    result = agent.run_conversation("Complete a long task.")

    streamed_text = "".join(item for item in visible_stream if isinstance(item, str))
    assert "Nous Portal rate limit active" in result["final_response"]
    assert streamed_text == result["final_response"]
    assert result["messages"][-1]["content"] == result["final_response"]
    assert sum(
        bool(
            snapshot
            and snapshot[-1].get("content") == result["final_response"]
        )
        for snapshot in persisted_snapshots
    ) == 1


def test_normal_terminal_overflow_locks_streamed_text_before_postprocessors(
    long_running_agent,
):
    from agent.provider_response_gate import MAX_BUFFERED_VISIBLE_BYTES

    agent = long_running_agent
    _install_tool_executor(agent)
    visible_stream: list[str] = []
    persisted_snapshots: list[list[dict]] = []
    agent.stream_delta_callback = visible_stream.append
    agent._persist_session = lambda messages, history: persisted_snapshots.append(
        [dict(message) for message in messages]
    )
    agent._file_mutation_verifier_enabled = lambda: True
    agent._format_file_mutation_failure_footer = lambda failed: "MUTATION FOOTER"
    call_index = 0
    overflowed_draft = "Y" * (MAX_BUFFERED_VISIBLE_BYTES + 1)

    def model_call(api_kwargs):
        nonlocal call_index
        if call_index < MIN_SUBSTANTIVE_TOOL_COMPLETIONS:
            response = _next_tool_response(call_index)
            call_index += 1
            return response
        agent._turn_failed_file_mutations = {"broken.py": "patch failed"}
        agent._fire_stream_delta(overflowed_draft)
        return _response(overflowed_draft)

    agent._interruptible_api_call = model_call
    agent._interruptible_streaming_api_call = (
        lambda api_kwargs, on_first_delta=None: model_call(api_kwargs)
    )

    result = agent.run_conversation("Complete a long task.")

    streamed_text = "".join(item for item in visible_stream if isinstance(item, str))
    assert result["final_response"] == streamed_text
    assert streamed_text.endswith(overflowed_draft)
    assert "MUTATION FOOTER" not in streamed_text
    assert result["messages"][-1]["content"] == streamed_text
    assert sum(
        bool(snapshot and snapshot[-1].get("content") == streamed_text)
        for snapshot in persisted_snapshots
    ) == 1
    assert agent._long_turn_terminal_gate_overflowed is False


@pytest.mark.parametrize(
    "api_mode",
    [
        "chat_completions",
        "codex_responses",
        "anthropic_messages",
        "bedrock_converse",
    ],
)
def test_toolless_finalizer_routes_every_provider_mode_through_interruptible_dispatch(
    long_running_agent,
    monkeypatch,
    api_mode,
):
    from agent import chat_completion_helpers

    agent = long_running_agent
    agent.api_mode = api_mode
    agent.prefill_messages = [
        {"role": "user", "content": "large ordinary-call prefill"},
        {"role": "assistant", "content": "ordinary prefill acknowledgement"},
    ]
    dispatched: list[dict] = []
    agent._build_api_kwargs = lambda messages: {
        "model": agent.model,
        "messages": messages,
        "tools": [{"type": "function"}],
        "toolConfig": {"tools": [{"toolSpec": {"name": "read_file"}}]},
    }
    agent._get_transport = lambda: SimpleNamespace(
        normalize_response=lambda response, **kwargs: response
    )

    def interruptible_dispatch(api_kwargs):
        dispatched.append(api_kwargs)
        return SimpleNamespace(content="interruptible refined", usage=None)

    agent._interruptible_api_call = interruptible_dispatch
    agent._ensure_primary_openai_client = lambda *args, **kwargs: (_ for _ in ()).throw(
        AssertionError("direct OpenAI SDK dispatch is not interruptible")
    )
    agent._run_codex_stream = lambda *args, **kwargs: (_ for _ in ()).throw(
        AssertionError("direct Codex dispatch bypassed the interrupt worker")
    )
    agent._anthropic_messages_create = lambda *args, **kwargs: (_ for _ in ()).throw(
        AssertionError("direct Anthropic dispatch bypassed the interrupt worker")
    )
    monkeypatch.setattr(
        chat_completion_helpers,
        "_dispatch_nonstreaming_api_request",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("direct native dispatch bypassed the interrupt worker")
        ),
    )

    response = chat_completion_helpers.request_toolless_completion(
        agent,
        "finalize",
        150,
    )

    assert response == "interruptible refined"
    assert len(dispatched) == 1
    assert "tools" not in dispatched[0]
    assert "toolConfig" not in dispatched[0]
    assert [message["role"] for message in dispatched[0]["messages"]] == [
        "system",
        "user",
    ]
    assert dispatched[0]["messages"][0]["content"] != agent._cached_system_prompt


def test_toolless_finalizer_uses_native_bedrock_dispatch(
    long_running_agent, monkeypatch
):
    from agent import chat_completion_helpers

    agent = long_running_agent
    agent.api_mode = "bedrock_converse"
    agent._build_api_kwargs = lambda messages: {
        "messages": messages,
        "__bedrock_converse__": True,
        "__bedrock_region__": "us-east-1",
    }
    agent._get_transport = lambda: SimpleNamespace(
        normalize_response=lambda response: response
    )
    agent._ensure_primary_openai_client = lambda *args, **kwargs: (_ for _ in ()).throw(
        AssertionError("Bedrock must not build an OpenAI client")
    )
    dispatched: list[dict] = []

    def native_dispatch(agent_arg, api_kwargs, *, make_client):
        assert agent_arg is agent
        assert api_kwargs["__bedrock_converse__"] is True
        dispatched.append(api_kwargs)
        return SimpleNamespace(
            content="Bedrock refined response",
            tool_calls=[],
            finish_reason="stop",
            reasoning_details=None,
            response_id=None,
            codex_message_items=None,
        )

    monkeypatch.setattr(
        chat_completion_helpers,
        "_dispatch_nonstreaming_api_request",
        native_dispatch,
    )

    response = chat_completion_helpers.request_toolless_completion(
        agent,
        "finalize",
        150,
    )

    assert response == "Bedrock refined response"
    assert len(dispatched) == 1


def test_toolless_finalizer_usage_updates_tokens_cost_and_call_count(
    long_running_agent, monkeypatch
):
    from agent import chat_completion_helpers, usage_pricing

    agent = long_running_agent
    agent._session_db = MagicMock()
    agent._session_db_created = True
    monkeypatch.setattr(
        usage_pricing,
        "estimate_usage_cost",
        lambda *args, **kwargs: SimpleNamespace(
            amount_usd=0.25,
            status="estimated",
            source="model_pricing",
        ),
    )
    response = _response(
        "refined",
        usage=SimpleNamespace(
            prompt_tokens=42_000,
            completion_tokens=500,
            total_tokens=42_500,
            prompt_tokens_details=None,
            completion_tokens_details=None,
        ),
    )

    chat_completion_helpers._record_toolless_completion_usage(agent, response)

    assert agent.session_prompt_tokens == 42_000
    assert agent.session_completion_tokens == 500
    assert agent.session_total_tokens == 42_500
    assert agent.session_input_tokens == 42_000
    assert agent.session_output_tokens == 500
    assert agent.session_api_calls == 1
    assert agent.session_estimated_cost_usd == 0.25
    assert agent.session_cost_status == "estimated"
    assert agent.session_cost_source == "model_pricing"
    assert agent._last_toolless_api_calls == 1
    agent._session_db.update_token_counts.assert_called_once_with(
        agent.session_id,
        input_tokens=42_000,
        output_tokens=500,
        cache_read_tokens=0,
        cache_write_tokens=0,
        reasoning_tokens=0,
        estimated_cost_usd=0.25,
        cost_status="estimated",
        cost_source="model_pricing",
        billing_provider=agent.provider,
        billing_base_url=agent.base_url,
        model=agent.model,
        api_call_count=1,
    )


def test_toolless_finalizer_accounts_for_moa_reference_usage_and_cost(
    long_running_agent, monkeypatch
):
    from agent import chat_completion_helpers, usage_pricing
    from agent.usage_pricing import CanonicalUsage

    agent = long_running_agent
    agent._session_db = MagicMock()
    agent._session_db_created = True
    agent.client = SimpleNamespace(
        last_aggregator_slot={
            "model": "aggregator/model",
            "provider": "openai-compat",
            "base_url": "https://aggregator.invalid/v1",
        },
        consume_reference_usage=lambda: (
            CanonicalUsage(input_tokens=1_000, output_tokens=100),
            0.75,
        ),
    )
    monkeypatch.setattr(
        usage_pricing,
        "estimate_usage_cost",
        lambda *args, **kwargs: SimpleNamespace(
            amount_usd=0.25,
            status="estimated",
            source="model_pricing",
        ),
    )
    response = _response(
        "refined",
        usage=SimpleNamespace(
            prompt_tokens=42_000,
            completion_tokens=500,
            total_tokens=42_500,
            prompt_tokens_details=None,
            completion_tokens_details=None,
        ),
    )

    chat_completion_helpers._record_toolless_completion_usage(agent, response)

    assert agent.session_input_tokens == 43_000
    assert agent.session_output_tokens == 600
    assert agent.session_estimated_cost_usd == 1.0
    persisted = agent._session_db.update_token_counts.call_args.kwargs
    assert persisted["input_tokens"] == 43_000
    assert persisted["output_tokens"] == 600
    assert persisted["estimated_cost_usd"] == 1.0


def test_failed_toolless_moa_attempt_still_consumes_reference_usage_and_cost(
    long_running_agent,
):
    from agent import chat_completion_helpers
    from agent.usage_pricing import CanonicalUsage

    agent = long_running_agent
    agent._session_db = MagicMock()
    agent._session_db_created = True
    agent.client = SimpleNamespace(
        consume_reference_usage=lambda: (
            CanonicalUsage(input_tokens=2_000, output_tokens=200),
            0.80,
        )
    )

    chat_completion_helpers._record_toolless_completion_usage(agent, None)

    assert agent.session_input_tokens == 2_000
    assert agent.session_output_tokens == 200
    assert agent.session_estimated_cost_usd == 0.80
    assert agent.session_cost_status == "estimated"
    assert agent.session_cost_source == "moa_reference_usage"
    persisted = agent._session_db.update_token_counts.call_args.kwargs
    assert persisted["input_tokens"] == 2_000
    assert persisted["output_tokens"] == 200
    assert persisted["estimated_cost_usd"] == 0.80
    assert persisted["api_call_count"] == 1


def test_toolless_finalizer_counts_empty_and_failed_attempts_in_memory_and_db(
    long_running_agent,
):
    from agent import chat_completion_helpers

    agent = long_running_agent
    agent._session_db = MagicMock()
    agent._session_db_created = True
    attempts = iter([_response("", usage=None), RuntimeError("provider failed")])

    def provider_attempt(*args, **kwargs):
        outcome = next(attempts)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    client = SimpleNamespace(
        chat=SimpleNamespace(
            completions=SimpleNamespace(create=provider_attempt),
        )
    )
    agent._ensure_primary_openai_client = lambda *args, **kwargs: client
    agent._build_api_kwargs = lambda messages: {
        "model": agent.model,
        "messages": messages,
        "tools": [{"type": "function"}],
    }
    agent._interruptible_api_call = lambda api_kwargs: provider_attempt(**api_kwargs)

    response = chat_completion_helpers.request_toolless_completion(
        agent,
        "finalize",
        150,
    )

    assert response == ""
    assert agent._last_toolless_api_calls == 2
    assert agent.session_api_calls == 2
    assert agent._session_db.update_token_counts.call_count == 2
    assert all(
        call.kwargs["api_call_count"] == 1
        for call in agent._session_db.update_token_counts.call_args_list
    )


def test_toolless_finalizer_propagates_stop_interrupt_after_accounting_attempt(
    long_running_agent,
):
    from agent import chat_completion_helpers

    agent = long_running_agent
    agent._session_db = MagicMock()
    agent._session_db_created = True

    def interrupted_attempt(*args, **kwargs):
        raise InterruptedError("Agent interrupted during API call")

    client = SimpleNamespace(
        chat=SimpleNamespace(
            completions=SimpleNamespace(create=interrupted_attempt),
        )
    )
    agent._ensure_primary_openai_client = lambda *args, **kwargs: client
    agent._build_api_kwargs = lambda messages: {
        "model": agent.model,
        "messages": messages,
        "tools": [{"type": "function"}],
    }
    agent._interruptible_api_call = lambda api_kwargs: interrupted_attempt(**api_kwargs)

    with pytest.raises(InterruptedError, match="interrupted"):
        chat_completion_helpers.request_toolless_completion(
            agent,
            "finalize",
            150,
        )

    assert agent._last_toolless_api_calls == 1
    assert agent.session_api_calls == 1
    agent._session_db.update_token_counts.assert_called_once()
