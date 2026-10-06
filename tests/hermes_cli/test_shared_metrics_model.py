"""Per-model tool-call quality, friction and context-peak shared metrics."""

from __future__ import annotations

import json
from types import SimpleNamespace

from hermes_cli import lifecycle
from hermes_cli.observability import relay_shared_metrics
from tests.hermes_cli.test_relay_shared_metrics_runtime import (  # noqa: F401 - fixture
    _stored_values,
    direct_runtime,
)


def _flush() -> None:
    relay_shared_metrics._get_runtime(retry_failed=True).relay.subscribers.flush()


def _tool_call(call_id, name, arguments, **function_extra):
    return SimpleNamespace(
        id=call_id, type="function",
        function=SimpleNamespace(name=name, arguments=arguments, **function_extra),
    )


def _agent(provider, model):
    tools = [
        {"type": "function", "function": {"name": "read_file", "parameters": {
            "type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}}},
        {"type": "function", "function": {"name": "todo", "parameters": {"type": "object", "properties": {}}}},
    ]
    return SimpleNamespace(
        provider=provider, model=model, session_id="s1", tools=tools,
        valid_tool_names={"read_file", "todo"}, log_prefix="", _invalid_tool_retries=0, _invalid_json_retries=0,
        _uniquify_tool_call_ids=lambda calls: None,
        _repair_tool_call=lambda name: "read_file" if name == "Read_File" else None,
        _vprint=lambda *a, **k: None, _buffer_vprint=lambda *a, **k: None, _flush_status_buffer=lambda: None,
        _build_assistant_message=lambda message, finish_reason: {"role": "assistant", "content": ""},
        _persist_session=lambda *a: None, _cleanup_task_resources=lambda *a: None,
    )


def _validate(agent, tool_calls) -> None:
    from agent.turn_tool_validation import validate_tool_calls

    validate_tool_calls(
        agent, SimpleNamespace(content="", tool_calls=tool_calls), "tool_calls",
        messages=[], conversation_history=[], api_call_count=1, effective_task_id="t1",
    )


def test_every_emitted_tool_call_counts_once_with_its_issue(direct_runtime, tmp_path):
    """Clean calls count as ``none`` (the rate denominator); each defective call counts once with
    the issue the model caused; a user-named provider never leaves the machine."""
    _validate(_agent("openrouter", "anthropic/claude-sonnet"), [
        _tool_call("c1", "read_file", '{"path": "a"}'),
        _tool_call("c2", "todo", ""),                        # parameterless: "" means {}
        _tool_call("c3", "read_file", '{"path": "a"'),       # truncated JSON
        _tool_call("c4", "acme_secret_tool", "{}"),
        _tool_call("c5", "read_file", ""),                   # required param, no arguments
        _tool_call("c6", "read_file", '{"file": "a"}'),      # missing required key
        _tool_call("c7", "Read_File", '{"path": "a"}'),      # name auto-repaired
        _tool_call("c8", "read_file", '{"path": "a"}', args_repaired=True),
    ])
    _validate(_agent("custom:acme-private", "acme-internal"), [_tool_call("c9", "todo", "{}")])
    _flush()

    rows = _stored_values(tmp_path, "hermes.model_tool_quality.count")
    assert "acme" not in json.dumps(rows)
    assert sorted((d["provider"], d["model"], d["call_role"], d["issue"], v) for d, v in rows) == sorted([
        ("openrouter", "anthropic/claude-sonnet", "primary", "none", 2),
        ("openrouter", "anthropic/claude-sonnet", "primary", "invalid_json", 1),
        ("openrouter", "anthropic/claude-sonnet", "primary", "unknown_tool", 1),
        ("openrouter", "anthropic/claude-sonnet", "primary", "empty_arguments", 1),
        ("openrouter", "anthropic/claude-sonnet", "primary", "schema_mismatch", 1),
        ("openrouter", "anthropic/claude-sonnet", "primary", "repaired", 2),
        ("custom", "custom", "primary", "none", 1),
    ])


def test_tool_call_quality_records_nothing_while_disabled(direct_runtime, tmp_path, monkeypatch):
    monkeypatch.setattr("hermes_cli.config.read_raw_config_readonly", lambda: {})
    _validate(_agent("openrouter", "anthropic/claude-sonnet"), [_tool_call("c1", "nope", "{")])
    assert not (tmp_path / "hermes-home" / "telemetry").exists() or not _stored_values(
        tmp_path, "hermes.model_tool_quality.count")


def _turn(session_id, task_id, model, result, *, usage=None, context_length=None, error=None, **start):
    base = {"session_id": session_id, "task_id": task_id, "api_request_id": f"{task_id}-r",
            "provider": "openrouter", "model": model}
    lifecycle.invoke_hook("pre_llm_call", **base, platform="cli", **start)
    lifecycle.invoke_hook("pre_api_request", **base)
    if error:
        lifecycle.invoke_hook("api_request_error", **base, retryable=True, reason=error)
    if usage is not None or error is None:  # an error with no usage never recovered
        lifecycle.invoke_hook("post_api_request", **base, usage=usage, context_length=context_length)
    if result is not None:
        relay_shared_metrics.finish_task_run(session_id=session_id, task_id=task_id, platform="cli", result=result)


def test_friction_is_attributed_to_the_model_that_produced_the_turn(direct_runtime, tmp_path):
    """After a /model switch the retry still blames the model whose turn is retried; interrupts and
    a quick exit after a failure count once; a session-close abort and delegated work never do."""
    from hermes_cli.observability import shared_metrics_events as events
    from hermes_cli.observability.shared_metrics_model import record_model_friction

    _turn("s1", "t1", "model-a", {"completed": True})
    events.record_model_switch(from_provider="openrouter", to_provider="anthropic", surface="cli", from_model="model-a")
    record_model_friction("retry", session_id="s1", provider="anthropic", model="model-b")
    record_model_friction("undo", session_id="gone", provider="anthropic", model="model-b")
    _turn("s1", "t2", "model-a", {"interrupted": True, "turn_exit_reason": "interrupted_by_user"})
    _turn("s1", "t3", "model-a", {"failed": True, "turn_exit_reason": "failed"})
    lifecycle.finalize_session(session_id="s1")

    _turn("s2", "t1", "model-c", {"completed": True})
    _turn("s2", "t2", "model-c", None)  # still running when the session closes
    lifecycle.finalize_session(session_id="s2")
    _turn("child", "t1", "model-d", {"interrupted": True}, parent_session_id="s1")
    lifecycle.finalize_session(session_id="child")
    _flush()

    rows = _stored_values(tmp_path, "hermes.model_friction.count")
    assert sorted((d["provider"], d["model"], d["signal"], v) for d, v in rows) == sorted([
        ("openrouter", "model-a", "switch_away", 1),
        ("openrouter", "model-a", "retry", 1),
        ("anthropic", "model-b", "undo", 1),
        ("openrouter", "model-a", "interrupt", 1),
        ("openrouter", "model-a", "quick_abandon", 1),
    ])


def test_context_peak_is_one_bucketed_row_per_session(direct_runtime, tmp_path):
    """The fullest primary context a session reached, its window bucket and whether it overflowed;
    delegated children add no row."""
    _turn("s1", "t1", "model-a", {"completed": True}, usage={"prompt_tokens": 50_000}, context_length=200_000)
    _turn("s1", "t2", "model-a", {"completed": True}, usage={"prompt_tokens": 160_000}, context_length=200_000,
          error="context_overflow")
    _turn("s1", "t3", "model-a", {"completed": True}, usage={"prompt_tokens": 20_000}, context_length=200_000)
    lifecycle.finalize_session(session_id="s1")
    _turn("s2", "t1", "model-b", {"completed": True}, usage={"input_tokens": 10_000}, context_length=1_048_576)
    lifecycle.finalize_session(session_id="s2")
    _turn("s3", "t1", "model-c", None, error="context_overflow")  # every call overflowed, none finished
    relay_shared_metrics.finish_task_run(session_id="s3", task_id="t1", platform="cli", result={"failed": True})
    lifecycle.finalize_session(session_id="s3")
    _turn("child", "t1", "model-d", {"completed": True}, usage={"prompt_tokens": 1}, context_length=8_000,
          parent_session_id="s1")
    lifecycle.finalize_session(session_id="child")
    _flush()

    rows = _stored_values(tmp_path, "hermes.context_peak.count")
    assert sorted(
        (d["model"], d["peak_fill_bucket"], d["window_bucket"], d["limit_hit"], v) for d, v in rows
    ) == [
        ("model-a", "75_to_90", "128k_to_256k", "yes", 1),
        ("model-b", "lt_50", "gte_1m", "no", 1),
        ("model-c", "unknown", "unknown", "yes", 1),
    ]
