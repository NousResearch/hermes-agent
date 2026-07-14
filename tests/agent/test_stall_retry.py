from __future__ import annotations

import inspect
import json
from types import SimpleNamespace

from agent import conversation_loop
from agent.stall_retry import (
    get_stall_retry_max_chars,
    get_stall_retry_max_per_turn,
    get_stall_retry_model,
    get_stall_retry_nudge_enabled,
    looks_like_stall,
    record_stall_retry_event,
    retry_on_stall,
    stall_retry_summary,
)


def test_action_preamble_without_tool_call_is_a_stall() -> None:
    assert looks_like_stall(
        "Let me check what's on taro and figure out the right approach.",
        "stop",
        False,
        400,
    )


def test_followup_action_preamble_after_successful_retry_is_a_stall() -> None:
    assert looks_like_stall(
        "Let me look at open tasks with high priority that I can actually pick up.",
        "stop",
        False,
        400,
    )


def test_long_diagnostic_ending_with_action_promise_is_a_stall() -> None:
    content = (
        "The crash pattern is clear: when the context fills up and llama.cpp "
        "tries to allocate a new tensor for the KV cache, it runs out of GPU "
        "memory. The first allocation succeeds, but the subsequent data "
        "allocation fails and leaves the tensor in an inconsistent state. "
        * 4
    )
    content += "Let me look at the exact crash mechanism more carefully:"

    assert len(content) > 400
    assert looks_like_stall(content, "stop", False, 400)


def test_long_diagnostic_with_completion_text_is_not_a_stall() -> None:
    content = (
        "The crash pattern is clear: when the context fills up and llama.cpp "
        "tries to allocate a new tensor for the KV cache, it runs out of GPU "
        "memory. "
        * 5
    )
    content += "In summary, the task is complete."

    assert len(content) > 400
    assert not looks_like_stall(content, "stop", False, 400)


def test_completion_text_is_not_a_stall() -> None:
    assert not looks_like_stall(
        "Done. The task is complete and no further action is needed.",
        "stop",
        False,
        400,
    )


def test_incomplete_final_fragment_without_action_preamble_is_a_stall() -> None:
    assert looks_like_stall(
        (
            "I'm on main with a clean tree. The fix I made was on Taro "
            "(remote machine), not locally. The change was on Taro's "
            "`~/.gitconfig` and the worktree's local git config. These are "
            "machine-specific runtime"
        ),
        "stop",
        False,
        400,
    )


def test_empty_visible_response_uses_empty_response_recovery_not_stall_retry() -> None:
    assert not looks_like_stall("", "stop", False, 400)
    assert not looks_like_stall("<think>still reasoning</think>", "stop", False, 400)


def test_short_status_answer_without_punctuation_is_not_a_stall() -> None:
    assert not looks_like_stall("main", "stop", False, 400)


def test_complete_agentic_answer_without_action_preamble_is_not_a_stall() -> None:
    assert not looks_like_stall(
        (
            "I'm on main with a clean tree. The Taro git identity and SSH "
            "push configuration are machine-local runtime settings, so there "
            "is no repository diff to publish."
        ),
        "stop",
        False,
        400,
    )


def test_retry_on_stall_switches_model_and_returns_tool_calls(monkeypatch) -> None:
    captured: dict[str, object] = {}
    tool_call = SimpleNamespace(
        function=SimpleNamespace(name="terminal", arguments='{"cmd":"pwd"}')
    )
    normalized = SimpleNamespace(
        content="",
        tool_calls=[tool_call],
        finish_reason="tool_calls",
    )

    def interruptible_api_call(kwargs: dict[str, object]) -> object:
        captured["kwargs"] = dict(kwargs)
        return normalized

    agent = SimpleNamespace(
        api_mode="openai",
        _is_anthropic_oauth=False,
        log_prefix="",
        _build_api_kwargs=lambda messages: {
            "model": "dflash",
            "messages": messages,
            "stream": True,
        },
        _interruptible_api_call=interruptible_api_call,
        _get_transport=lambda: SimpleNamespace(
            normalize_response=lambda response, **_kwargs: response
        ),
        _vprint=lambda *_args, **_kwargs: None,
    )

    monkeypatch.setenv("HERMES_STALL_RETRY_MODEL", "qwen3.6-27b-256k")
    monkeypatch.setenv("HERMES_STALL_RETRY_TELEMETRY", "0")
    result = retry_on_stall(
        agent,
        [{"role": "user", "content": "go"}],
        "stop",
        stalled_content="Let me check the repo.",
        retry_index=1,
    )

    assert result is normalized
    kwargs = captured["kwargs"]
    assert kwargs["model"] == "qwen3.6-27b-256k"
    assert kwargs["stream"] is False
    assert kwargs["messages"][-2]["role"] == "assistant"
    assert kwargs["messages"][-2]["content"] == "Let me check the repo."
    assert kwargs["messages"][-1]["role"] == "user"
    assert "required tool call" in kwargs["messages"][-1]["content"]
    summary = stall_retry_summary(agent)
    assert summary is not None
    assert summary["attempted"] == 1
    assert summary["recovered"] == 1


def test_retry_on_stall_uses_agent_config_when_env_is_absent(monkeypatch) -> None:
    captured: dict[str, object] = {}
    tool_call = SimpleNamespace(
        function=SimpleNamespace(name="terminal", arguments='{"cmd":"pwd"}')
    )
    normalized = SimpleNamespace(
        content="",
        tool_calls=[tool_call],
        finish_reason="tool_calls",
    )

    def interruptible_api_call(kwargs: dict[str, object]) -> object:
        captured["kwargs"] = dict(kwargs)
        return normalized

    agent = SimpleNamespace(
        api_mode="openai",
        _is_anthropic_oauth=False,
        log_prefix="",
        _stall_retry_config={
            "model": "qwen3.6-27b-256k",
            "max_chars": 240,
            "max_per_turn": 7,
        },
        _build_api_kwargs=lambda messages: {
            "model": "dflash",
            "messages": messages,
            "stream": True,
        },
        _interruptible_api_call=interruptible_api_call,
        _get_transport=lambda: SimpleNamespace(
            normalize_response=lambda response, **_kwargs: response
        ),
        _vprint=lambda *_args, **_kwargs: None,
    )

    monkeypatch.delenv("HERMES_STALL_RETRY_MODEL", raising=False)
    monkeypatch.delenv("HERMES_STALL_RETRY_MAX_CHARS", raising=False)
    monkeypatch.delenv("HERMES_STALL_RETRY_MAX_PER_TURN", raising=False)
    monkeypatch.setenv("HERMES_STALL_RETRY_TELEMETRY", "0")

    assert get_stall_retry_model(agent) == "qwen3.6-27b-256k"
    assert get_stall_retry_max_chars(agent) == 240
    assert get_stall_retry_max_per_turn(agent) == 7
    assert get_stall_retry_nudge_enabled(agent)

    result = retry_on_stall(agent, [{"role": "user", "content": "go"}], "stop")

    assert result is normalized
    assert captured["kwargs"]["model"] == "qwen3.6-27b-256k"
    assert captured["kwargs"]["stream"] is False


def test_retry_on_stall_can_disable_retry_nudge(monkeypatch) -> None:
    captured: dict[str, object] = {}
    tool_call = SimpleNamespace(
        function=SimpleNamespace(name="terminal", arguments='{"cmd":"pwd"}')
    )
    normalized = SimpleNamespace(
        content="",
        tool_calls=[tool_call],
        finish_reason="tool_calls",
    )

    def interruptible_api_call(kwargs: dict[str, object]) -> object:
        captured["kwargs"] = dict(kwargs)
        return normalized

    agent = SimpleNamespace(
        api_mode="openai",
        _is_anthropic_oauth=False,
        log_prefix="",
        _stall_retry_config={
            "model": "qwen3.6-27b-256k",
            "nudge": False,
            "telemetry": False,
        },
        _build_api_kwargs=lambda messages: {
            "model": "dflash",
            "messages": messages,
            "stream": True,
        },
        _interruptible_api_call=interruptible_api_call,
        _get_transport=lambda: SimpleNamespace(
            normalize_response=lambda response, **_kwargs: response
        ),
        _vprint=lambda *_args, **_kwargs: None,
    )

    monkeypatch.delenv("HERMES_STALL_RETRY_MODEL", raising=False)
    monkeypatch.delenv("HERMES_STALL_RETRY_NUDGE", raising=False)
    monkeypatch.delenv("HERMES_STALL_RETRY_TELEMETRY", raising=False)

    assert not get_stall_retry_nudge_enabled(agent)

    result = retry_on_stall(
        agent,
        [{"role": "user", "content": "go"}],
        "stop",
        stalled_content="Let me check.",
    )

    assert result is normalized
    assert captured["kwargs"]["messages"] == [{"role": "user", "content": "go"}]


def test_stall_retry_telemetry_writes_bounded_local_jsonl(tmp_path) -> None:
    log_path = tmp_path / "stall-retry.ndjson"
    agent = SimpleNamespace(
        session_id="s1",
        model="dflash",
        provider="nous",
        _stall_retry_config={
            "telemetry": True,
            "telemetry_path": str(log_path),
        },
    )

    record_stall_retry_event(
        agent,
        "detected",
        retry_model="qwen3.6-27b-256k",
        content="Let me check the repo.\n" * 30,
    )

    lines = log_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    event = json.loads(lines[0])
    assert event["event"] == "detected"
    assert event["session_id"] == "s1"
    assert event["model"] == "dflash"
    assert event["provider"] == "nous"
    assert event["retry_model"] == "qwen3.6-27b-256k"
    assert event["content_chars"] > len(event["content_preview"])
    assert len(event["content_preview"]) <= 240
    assert "messages" not in event

    summary = stall_retry_summary(agent)
    assert summary is not None
    assert summary["detected"] == 1
    assert summary["events"] == 1
    assert summary["log_path"] == str(log_path)


def test_conversation_loop_adopts_retry_before_tool_call_branch() -> None:
    source = inspect.getsource(conversation_loop.run_conversation)
    retry_idx = source.index("retried = retry_on_stall")
    tool_branch_idx = source.index("# Check for tool calls")

    assert retry_idx < tool_branch_idx
    assert "continue  # re-enter loop top; tool-calls path handles it" not in source
    assert "stall_retry_failed_no_tool_call" in source


def test_conversation_loop_allows_bounded_multiple_stall_retries() -> None:
    source = inspect.getsource(conversation_loop.run_conversation)

    assert "_stall_retry_used" not in source
    assert "_stall_retry_count += 1" in source
    assert "get_stall_retry_max_per_turn" in source
    assert "stall_retry_limit_exhausted" in source
