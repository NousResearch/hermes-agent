from __future__ import annotations

import inspect
from types import SimpleNamespace

from agent import conversation_loop
from agent.stall_retry import (
    EMPTY_AFTER_TOOL_RETRY_NUDGE,
    looks_like_stall,
    retry_on_stall,
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
    result = retry_on_stall(agent, [{"role": "user", "content": "go"}], "stop")

    assert result is normalized
    kwargs = captured["kwargs"]
    assert kwargs["model"] == "qwen3.6-27b-256k"
    assert kwargs["stream"] is False


def test_retry_on_stall_can_accept_visible_content(monkeypatch) -> None:
    captured: dict[str, object] = {}
    normalized = SimpleNamespace(
        content="I processed the tool result and will continue.",
        tool_calls=None,
        finish_reason="stop",
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
    result = retry_on_stall(
        agent,
        [{"role": "user", "content": "go"}],
        "stop",
        accept_content=True,
        retry_nudge=EMPTY_AFTER_TOOL_RETRY_NUDGE,
    )

    assert result is normalized
    kwargs = captured["kwargs"]
    assert kwargs["model"] == "qwen3.6-27b-256k"
    assert kwargs["stream"] is False
    assert kwargs["messages"][-1]["role"] == "user"
    assert "after the tool results was empty" in kwargs["messages"][-1]["content"]


def test_retry_on_stall_still_rejects_content_without_accept_content(monkeypatch) -> None:
    normalized = SimpleNamespace(
        content="I processed the tool result and will continue.",
        tool_calls=None,
        finish_reason="stop",
    )

    agent = SimpleNamespace(
        api_mode="openai",
        _is_anthropic_oauth=False,
        log_prefix="",
        _build_api_kwargs=lambda messages: {
            "model": "dflash",
            "messages": messages,
            "stream": True,
        },
        _interruptible_api_call=lambda _kwargs: normalized,
        _get_transport=lambda: SimpleNamespace(
            normalize_response=lambda response, **_kwargs: response
        ),
        _vprint=lambda *_args, **_kwargs: None,
    )

    monkeypatch.setenv("HERMES_STALL_RETRY_MODEL", "qwen3.6-27b-256k")
    assert retry_on_stall(
        agent,
        [{"role": "user", "content": "go"}],
        "stop",
    ) is None


def test_conversation_loop_adopts_retry_before_tool_call_branch() -> None:
    source = inspect.getsource(conversation_loop.run_conversation)
    retry_idx = source.index("retried = retry_on_stall")
    tool_branch_idx = source.index("# Check for tool calls")

    assert retry_idx < tool_branch_idx
    assert "continue  # re-enter loop top; tool-calls path handles it" not in source
    assert "stall_retry_failed_no_tool_call" in source


def test_conversation_loop_retries_empty_post_tool_before_generic_stall() -> None:
    source = inspect.getsource(conversation_loop.run_conversation)
    empty_retry_idx = source.index("EMPTY_AFTER_TOOL_RETRY_NUDGE")
    generic_stall_idx = source.index("looks_like_stall(")
    tool_branch_idx = source.index("# Check for tool calls")

    assert empty_retry_idx < generic_stall_idx
    assert empty_retry_idx < tool_branch_idx
    assert "not _empty_after_tool_result and looks_like_stall" in source
    assert "accept_content=True" in source


def test_conversation_loop_allows_bounded_multiple_stall_retries() -> None:
    source = inspect.getsource(conversation_loop.run_conversation)

    assert "_stall_retry_used" not in source
    assert "_stall_retry_count += 1" in source
    assert "HERMES_STALL_RETRY_MAX_PER_TURN" in source
    assert "stall_retry_limit_exhausted" in source
