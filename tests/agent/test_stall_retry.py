from __future__ import annotations

import inspect
import json
from types import SimpleNamespace

from agent import conversation_loop
from agent.stall_retry import (
    EMPTY_AFTER_TOOL_RETRY_NUDGE,
    FAILED_STALL_RETRY_RECOVERY_NUDGE,
    activate_stall_retry_runtime,
    get_stall_retry_max_chars,
    get_stall_retry_max_per_turn,
    get_stall_retry_model,
    get_stall_retry_no_tool_recovery_max,
    get_stall_retry_nudge_enabled,
    get_stall_retry_promote_after,
    has_recent_tool_result,
    looks_like_incomplete_final_fragment,
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


def test_mixed_explanation_action_preamble_over_default_limit_is_a_stall() -> None:
    content = (
        "Now I have a clear picture. The task is to add a pre-dispatch "
        "executable-state gate in subagent_dispatch_register_command that "
        "checks if the linked task is in an executable state BEFORE registering "
        "a dispatch. Currently the stale guard only runs post-dispatch during "
        "drain and settlement. The goal is to prevent non-executable dispatches "
        "from being registered in the first place. Let me look at the register "
        "command's validation section and the _validate_subagent_dispatch_record "
        "function:"
    )

    assert len(content) > 400
    assert looks_like_stall(content, "stop", False, 400)


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


def test_completion_then_action_promise_is_still_a_stall() -> None:
    assert looks_like_stall(
        "Onboarding complete. Now let me read the STATUS.md to find a task I can pick up.",
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


def test_short_incomplete_connector_tail_is_a_stall() -> None:
    content = "I see a lot of discord-res tasks (digest Discord content) and some"

    assert looks_like_incomplete_final_fragment(content, "stop", False, 400)
    assert looks_like_stall(content, "stop", False, 400)


def test_empty_visible_response_uses_empty_response_recovery_not_stall_retry() -> None:
    assert not looks_like_stall("", "stop", False, 400)
    assert not looks_like_stall("<think>still reasoning</think>", "stop", False, 400)


def test_recent_tool_result_detector_spans_status_scaffolding() -> None:
    messages = [
        {"role": "user", "content": "do the onboarding"},
        {"role": "assistant", "tool_calls": [{"id": "call_1"}]},
        {"role": "tool", "content": "Onboarding complete"},
        {"role": "assistant", "content": ""},
        {"role": "assistant", "content": ""},
        {"role": "assistant", "content": ""},
        {"role": "system", "content": "status update"},
        {"role": "assistant", "content": ""},
    ]

    assert has_recent_tool_result(messages)


def test_recent_tool_result_detector_stops_at_current_user() -> None:
    messages = [
        {"role": "user", "content": "old task"},
        {"role": "assistant", "tool_calls": [{"id": "call_1"}]},
        {"role": "tool", "content": "old result"},
        {"role": "assistant", "content": "done"},
        {"role": "user", "content": "new task"},
        {"role": "assistant", "content": ""},
    ]

    assert not has_recent_tool_result(messages)


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
    monkeypatch.setenv("HERMES_STALL_RETRY_TELEMETRY", "0")
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
    monkeypatch.setenv("HERMES_STALL_RETRY_TELEMETRY", "0")
    assert retry_on_stall(
        agent,
        [{"role": "user", "content": "go"}],
        "stop",
    ) is None


def test_retry_on_stall_skips_same_model_before_runtime_promotion(monkeypatch) -> None:
    captured = {"called": False}

    agent = SimpleNamespace(
        api_mode="openai",
        _is_anthropic_oauth=False,
        log_prefix="",
        _build_api_kwargs=lambda messages: {
            "model": "qwen3.6-27b-256k",
            "messages": messages,
            "stream": True,
        },
        _interruptible_api_call=lambda _kwargs: captured.__setitem__("called", True),
        _get_transport=lambda: SimpleNamespace(
            normalize_response=lambda response, **_kwargs: response
        ),
        _vprint=lambda *_args, **_kwargs: None,
    )

    monkeypatch.setenv("HERMES_STALL_RETRY_MODEL", "qwen3.6-27b-256k")
    monkeypatch.setenv("HERMES_STALL_RETRY_TELEMETRY", "0")

    assert retry_on_stall(agent, [{"role": "user", "content": "go"}], "stop") is None
    assert captured["called"] is False
    summary = stall_retry_summary(agent)
    assert summary is not None
    assert any(
        event.get("event") == "skipped_same_model"
        for event in getattr(agent, "_stall_retry_events", [])
    )


def test_retry_on_stall_allows_same_model_after_runtime_promotion(monkeypatch) -> None:
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
        _stall_retry_runtime_promoted=True,
        _build_api_kwargs=lambda messages: {
            "model": "qwen3.6-27b-256k",
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
        retry_index=3,
    )

    assert result is normalized
    assert captured["kwargs"]["model"] == "qwen3.6-27b-256k"
    assert captured["kwargs"]["stream"] is False
    assert captured["kwargs"]["messages"][-1]["role"] == "user"
    assert "required tool call" in captured["kwargs"]["messages"][-1]["content"]
    summary = stall_retry_summary(agent)
    assert summary is not None
    assert any(
        event.get("event") == "same_model_retry_after_promotion"
        for event in getattr(agent, "_stall_retry_events", [])
    )
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


def test_stall_retry_empty_agent_config_falls_back_to_loaded_config(monkeypatch) -> None:
    import hermes_cli.config as config_mod

    monkeypatch.delenv("HERMES_STALL_RETRY_MODEL", raising=False)
    monkeypatch.delenv("HERMES_STALL_RETRY_PROVIDER", raising=False)
    monkeypatch.delenv("HERMES_STALL_RETRY_MAX_PER_TURN", raising=False)
    monkeypatch.setattr(
        config_mod,
        "load_config",
        lambda: {
            "stall_retry": {
                "max_per_turn": 3,
                "model": "qwen3.6-27b-256k",
                "provider": "taro",
            }
        },
    )

    empty_agent = SimpleNamespace(_stall_retry_config={})
    provider_only_agent = SimpleNamespace(_stall_retry_config={"provider": "ko-mac"})

    assert get_stall_retry_model(empty_agent) == "qwen3.6-27b-256k"
    assert get_stall_retry_max_per_turn(empty_agent) == 3
    assert get_stall_retry_model(provider_only_agent) == "qwen3.6-27b-256k"
    assert get_stall_retry_max_per_turn(provider_only_agent) == 3


def test_stall_retry_promote_after_uses_config_and_env(monkeypatch) -> None:
    agent = SimpleNamespace(_stall_retry_config={"promote_after": 4})

    monkeypatch.delenv("HERMES_STALL_RETRY_PROMOTE_AFTER", raising=False)
    assert get_stall_retry_promote_after(agent) == 4

    monkeypatch.setenv("HERMES_STALL_RETRY_PROMOTE_AFTER", "0")
    assert get_stall_retry_promote_after(agent) == 0


def test_stall_retry_no_tool_recovery_max_uses_config_and_env(monkeypatch) -> None:
    agent = SimpleNamespace(_stall_retry_config={"no_tool_recovery_max": 4})

    monkeypatch.delenv("HERMES_STALL_RETRY_NO_TOOL_RECOVERY_MAX", raising=False)
    assert get_stall_retry_no_tool_recovery_max(agent) == 4

    monkeypatch.setenv("HERMES_STALL_RETRY_NO_TOOL_RECOVERY_MAX", "0")
    assert get_stall_retry_no_tool_recovery_max(agent) == 0


def test_activate_stall_retry_runtime_promotes_for_current_turn(monkeypatch) -> None:
    captured: dict[str, object] = {}
    fake_client = SimpleNamespace(
        api_key="retry-key",
        base_url="http://taro:8080/v1",
        _custom_headers={"X-Test": "1"},
    )

    def fake_resolve_provider_client(**kwargs: object) -> tuple[object, str]:
        captured["resolve_kwargs"] = dict(kwargs)
        return fake_client, "qwen3.6-27b-256k"

    def update_model(**kwargs: object) -> None:
        captured["context_model"] = dict(kwargs)

    agent = SimpleNamespace(
        model="dflash",
        provider="custom",
        base_url="http://primary:8080/v1",
        api_key="primary-key",
        api_mode="chat_completions",
        _client_kwargs={"api_key": "primary-key", "base_url": "http://primary:8080/v1"},
        _stall_retry_config={
            "model": "qwen3.6-27b-256k",
            "provider": "taro",
            "base_url": "http://taro:8080/v1",
        },
        _transport_cache={"chat": object()},
        _config_context_length=262144,
        _custom_providers=[],
        context_compressor=SimpleNamespace(update_model=update_model),
        _anthropic_prompt_cache_policy=lambda **_kwargs: (False, False),
        _ensure_lmstudio_runtime_loaded=lambda: None,
        _emit_status=lambda message: captured.setdefault("status", message),
        _fallback_activated=False,
    )

    monkeypatch.setenv("HERMES_STALL_RETRY_TELEMETRY", "0")
    monkeypatch.setattr(
        "agent.auxiliary_client.resolve_provider_client",
        lambda provider, **kwargs: fake_resolve_provider_client(
            provider=provider, **kwargs
        ),
    )
    monkeypatch.setattr(
        "agent.model_metadata.get_model_context_length",
        lambda *_args, **_kwargs: 262144,
    )

    assert activate_stall_retry_runtime(
        agent,
        "qwen3.6-27b-256k",
        promote_after=2,
        successful_retries=2,
    )
    assert agent.model == "qwen3.6-27b-256k"
    assert agent.provider == "taro"
    assert str(agent.base_url) == "http://taro:8080/v1"
    assert agent._fallback_activated is True
    assert agent._stall_retry_runtime_promoted is True
    assert agent._stall_retry_promoted_from == "dflash"
    assert agent._transport_cache == {}
    assert captured["resolve_kwargs"]["provider"] == "taro"
    assert captured["context_model"]["model"] == "qwen3.6-27b-256k"
    assert "rest of this turn" in captured["status"]


def test_retry_on_stall_uses_configured_retry_provider(monkeypatch) -> None:
    captured: dict[str, object] = {}
    tool_call = SimpleNamespace(
        function=SimpleNamespace(name="terminal", arguments='{"cmd":"pwd"}')
    )
    normalized = SimpleNamespace(
        content="",
        tool_calls=[tool_call],
        finish_reason="tool_calls",
    )

    class FakeCompletions:
        def create(self, **kwargs: object) -> object:
            captured["kwargs"] = dict(kwargs)
            return normalized

    fake_client = SimpleNamespace(
        api_key="retry-key",
        base_url="http://taro:8080/v1",
        chat=SimpleNamespace(completions=FakeCompletions()),
    )

    def fake_resolve_provider_client(**kwargs: object) -> tuple[object, str]:
        captured["resolve_kwargs"] = dict(kwargs)
        return fake_client, "qwen3.6-27b-256k"

    agent = SimpleNamespace(
        api_mode="openai",
        _is_anthropic_oauth=False,
        base_url="http://taro:8080/v1",
        log_prefix="",
        _stall_retry_config={
            "model": "qwen3.6-27b-256k",
            "provider": "taro",
            "api_key_env": "HERMES_RETRY_TEST_KEY",
        },
        _build_api_kwargs=lambda messages: {
            "model": "dflash",
            "messages": messages,
            "stream": True,
        },
        _interruptible_api_call=lambda _kwargs: (_ for _ in ()).throw(
            AssertionError("same-client retry should not be used")
        ),
        _get_transport=lambda: SimpleNamespace(
            normalize_response=lambda response, **_kwargs: response
        ),
        _vprint=lambda *_args, **_kwargs: None,
    )

    monkeypatch.delenv("HERMES_STALL_RETRY_MODEL", raising=False)
    monkeypatch.setenv("HERMES_RETRY_TEST_KEY", "retry-key")
    monkeypatch.setenv("HERMES_STALL_RETRY_TELEMETRY", "0")
    monkeypatch.setattr(
        "agent.auxiliary_client.resolve_provider_client",
        lambda provider, **kwargs: fake_resolve_provider_client(provider=provider, **kwargs),
    )

    result = retry_on_stall(agent, [{"role": "user", "content": "go"}], "stop")

    assert result is normalized
    assert captured["resolve_kwargs"]["provider"] == "taro"
    assert captured["resolve_kwargs"]["model"] == "qwen3.6-27b-256k"
    assert captured["resolve_kwargs"]["explicit_api_key"] == "retry-key"
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
    assert "FAILED_STALL_RETRY_RECOVERY_NUDGE" in source
    assert "no_tool_recovery_prompt" in source
    assert "agent._session_messages = messages" in source
    assert "continue" in source


def test_conversation_loop_retries_empty_post_tool_before_tool_branch() -> None:
    source = inspect.getsource(conversation_loop.run_conversation)
    empty_retry_idx = source.index("empty_after_tool_result")
    generic_stall_idx = source.index("looks_like_stall(")
    tool_branch_idx = source.index("# Check for tool calls")

    assert empty_retry_idx < generic_stall_idx
    assert empty_retry_idx < tool_branch_idx
    assert "not _empty_after_tool_result and looks_like_stall" in source
    assert "EMPTY_AFTER_TOOL_RETRY_NUDGE" in source
    assert "accept_content=True" in source


def test_conversation_loop_uses_configured_stall_retry_model() -> None:
    source = inspect.getsource(conversation_loop.run_conversation)

    assert "get_stall_retry_model(agent)" in source
    assert 'os.environ.get("HERMES_STALL_RETRY_MODEL"' in source


def test_conversation_loop_accepts_content_for_incomplete_final_fragment() -> None:
    source = inspect.getsource(conversation_loop.run_conversation)

    assert "looks_like_incomplete_final_fragment" in source
    assert "accept_content=_retry_accepts_content" in source


def test_conversation_loop_allows_bounded_multiple_stall_retries() -> None:
    source = inspect.getsource(conversation_loop.run_conversation)

    assert "_stall_retry_used" not in source
    assert "_stall_retry_count += 1" in source
    assert "get_stall_retry_max_per_turn" in source
    assert "stall_retry_limit_exhausted" in source


def test_conversation_loop_does_not_treat_recovered_stalls_as_failures() -> None:
    source = inspect.getsource(conversation_loop.run_conversation)

    assert "_stall_retry_failed_count" in source
    assert "if _stall_retry_count >= _stall_retry_max_per_turn" not in source
    assert "A recovered tool call is work advanced" in source


def test_conversation_loop_promotes_retry_lane_after_repeated_rescues() -> None:
    source = inspect.getsource(conversation_loop.run_conversation)

    assert "get_stall_retry_promote_after" in source
    assert "_stall_retry_success_count += 1" in source
    assert "activate_stall_retry_runtime(" in source
    assert "_stall_retry_success_count >= _stall_retry_promote_after" in source


def test_conversation_loop_configures_no_tool_recovery_limit() -> None:
    source = inspect.getsource(conversation_loop.run_conversation)

    assert "get_stall_retry_no_tool_recovery_max(agent)" in source
    assert "_stall_retry_no_tool_recovery_count" in source
    assert "_stall_retry_no_tool_recovery_max" in source
    assert FAILED_STALL_RETRY_RECOVERY_NUDGE
