"""Regression test for re-arming the compression budget after tool progress."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from run_agent import AIAgent


def _tool_call():
    return SimpleNamespace(
        id="call_1",
        type="function",
        function=SimpleNamespace(name="web_search", arguments='{"query": "x"}'),
    )


def _tool_response(prompt_tokens: int):
    message = SimpleNamespace(
        content=None,
        reasoning_content=None,
        reasoning=None,
        tool_calls=[_tool_call()],
    )
    return SimpleNamespace(
        choices=[SimpleNamespace(message=message, finish_reason="tool_calls")],
        model="test/model",
        usage=SimpleNamespace(
            prompt_tokens=prompt_tokens,
            completion_tokens=1,
            total_tokens=prompt_tokens + 1,
        ),
    )


def _final_response():
    message = SimpleNamespace(
        content="done",
        reasoning_content=None,
        reasoning=None,
        tool_calls=None,
    )
    return SimpleNamespace(
        choices=[SimpleNamespace(message=message, finish_reason="stop")],
        model="test/model",
        usage=None,
    )


def _malformed_response():
    return SimpleNamespace(choices=[], model="test/model", usage=None)


def _tool_definition():
    return {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": "Search the web",
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
            },
        },
    }


@pytest.mark.parametrize(
    ("prompt_tokens", "expected_compactions", "provider_recovery"),
    [(50, 1, False), (150, 1, False), (50, 2, True)],
    ids=[
        "pressure-cleared-anchored-no-recompaction",
        "pressure-still-high-stays-capped",
        "pressure-cleared-rearms-after-provider-recovery",
    ],
)
def test_pre_api_compression_budget_rearms_only_after_pressure_clears(
    prompt_tokens: int,
    expected_compactions: int,
    provider_recovery: bool,
):
    """Only provider-confirmed headroom starts a new pressure episode.

    Usage-anchored accounting update: once the provider reports
    ``prompt_tokens=50`` for the full transcript, later pre-API checks anchor
    on that real reading plus a delta estimate of the few appended messages —
    the scripted whole-history rough estimate (200) no longer drives the
    decision, so the pressure-cleared case performs exactly ONE compaction
    (the pre-anchor one). The budget-rearm mechanics remain covered by the
    provider-recovery variant, whose first response carries no usage (no
    anchor) and therefore still compacts on the rough estimate.
    """
    with (
        patch("model_tools.get_tool_definitions", return_value=[_tool_definition()]),
        patch("model_tools.check_toolset_requirements", return_value={}),
        patch("agent.process_bootstrap.OpenAI"),
        patch("agent.model_metadata.get_model_context_length", return_value=256_000),
        patch("agent.context_compressor.get_model_context_length", return_value=256_000),
    ):
        agent = AIAgent(
            api_key="test-key-1234567890",
            base_url="https://openrouter.ai/api/v1",
            model="test/model",
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
            max_iterations=6,
        )

    agent.client = MagicMock()
    responses = [_tool_response(prompt_tokens), _final_response()]
    if provider_recovery:
        responses.insert(0, _malformed_response())
        agent._fallback_chain = [object()]
        agent._try_activate_fallback = MagicMock(return_value=True)
    agent.client.chat.completions.create.side_effect = responses
    agent._cached_system_prompt = "You are helpful."
    agent._use_prompt_caching = False
    agent._disable_streaming = True
    agent.tool_delay = 0
    agent.save_trajectories = False
    agent.max_compression_attempts = 1

    compressor = MagicMock()
    compressor.protect_first_n = 3
    compressor.protect_last_n = 20
    compressor.threshold_tokens = 100
    compressor.context_length = 1_000
    compressor.last_prompt_tokens = -1
    compressor._verify_compaction_cleared_threshold = False
    compressor.awaiting_real_usage_after_compression = False
    compressor.should_compress.side_effect = lambda tokens: tokens >= 100
    compressor.should_compress_info.return_value = (False, None)
    compressor.should_compress_preflight.return_value = False
    compressor.should_defer_preflight_to_real_usage.return_value = False
    compressor.get_active_compression_failure_cooldown.return_value = None
    compressor.select_context.return_value = None
    compressor.get_automatic_compaction_status_message.return_value = ""

    def _update_from_response(usage):
        # Mirror the real compressor: the next provider usage reading
        # consumes the completed-compaction verification latch.
        compressor.last_prompt_tokens = int(usage.get("prompt_tokens", 0) or 0)
        compressor._verify_compaction_cleared_threshold = False
        compressor.awaiting_real_usage_after_compression = False

    compressor.update_from_response.side_effect = _update_from_response
    agent.compression_enabled = True
    agent.context_compressor = compressor

    estimate_values = iter([200, 190, 200, 10])
    _last_estimate = [10]

    def _next_estimate(messages=None, *_args, **_kwargs):
        # The scripted sequence prices the WHOLE history; a usage-anchored gate
        # estimates only the few messages appended since the real reading.
        if isinstance(messages, list) and len(messages) <= 4:
            return 10
        # The provider-recovery variant re-runs the pre-API preflight after
        # fallback activation (#84733), consuming an extra estimate reading.
        # Hold the final low-pressure value once the scripted sequence is
        # exhausted instead of raising StopIteration.
        try:
            _last_estimate[0] = next(estimate_values)
        except StopIteration:
            pass
        return _last_estimate[0]

    compress_calls = []

    def _fake_compress(messages, _system_message, **_kwargs):
        compress_calls.append(messages)
        # Arm the same provider-verification boundary the real compression
        # path arms after a completed compaction.
        compressor._verify_compaction_cleared_threshold = True
        compressor._pending_history_compaction_verdict = True
        compressor.awaiting_real_usage_after_compression = True
        return list(messages), "compressed prompt"

    def _fake_execute_tool_calls(assistant_message, messages, *_args):
        tool_call = assistant_message.tool_calls[0]
        messages.append(
            {
                "role": "tool",
                "name": tool_call.function.name,
                "tool_call_id": tool_call.id,
                "content": "ok",
            }
        )

    history = [
        {"role": "user" if i % 2 == 0 else "assistant", "content": f"msg {i}"}
        for i in range(30)
    ]

    with (
        patch(
            "agent.turn_context.estimate_request_tokens_rough",
            return_value=10,
        ),
        patch(
            "agent.model_metadata.estimate_messages_tokens_rough",
            side_effect=_next_estimate,
        ),
        patch(
            "agent.conversation_loop._estimate_tools_tokens_rough",
            return_value=0,
        ),
        patch.object(agent, "_compress_context", side_effect=_fake_compress),
        patch.object(agent, "_execute_tool_calls", side_effect=_fake_execute_tool_calls),
        patch.object(agent, "_flush_messages_to_session_db", return_value=True),
        patch.object(agent, "_persist_session"),
        patch.object(agent, "_save_trajectory"),
        patch.object(agent, "_cleanup_task_resources"),
    ):
        result = agent.run_conversation("do a lot of tool work", conversation_history=history)

    assert result["completed"] is True
    assert result["final_response"] == "done"
    assert len(compress_calls) == expected_compactions, (
        "same-turn compression must re-arm only after the provider confirms "
        f"headroom; got {len(compress_calls)} compactions for "
        f"prompt_tokens={prompt_tokens}"
    )


@pytest.fixture
def recovery_agent():
    """Real loop/compressor; external clients, summary I/O and persistence are isolated."""
    from agent.context_compressor import ContextCompressor

    with (
        patch("model_tools.get_tool_definitions", return_value=[_tool_definition()]),
        patch("model_tools.check_toolset_requirements", return_value={}),
        patch("agent.process_bootstrap.OpenAI"),
        patch("agent.model_metadata.get_model_context_length", return_value=128_000),
        patch("agent.context_compressor.get_model_context_length", return_value=256_000),
    ):
        agent = AIAgent(
            api_key="test-key-1234567890", base_url="https://openrouter.ai/api/v1",
            model="test/model", provider="openrouter", quiet_mode=True,
            skip_context_files=True, skip_memory=True, max_iterations=6,
        )
        agent.client = MagicMock()
        agent._cached_system_prompt = "You are helpful."
        agent._use_prompt_caching = False
        agent._disable_streaming = True
        agent.tool_delay = 0
        agent.save_trajectories = False
        agent.max_compression_attempts = 1
        agent._session_db = None
        agent.context_compressor = ContextCompressor(
            model=agent.model, provider=agent.provider, base_url=agent.base_url,
            api_mode=agent.api_mode, quiet_mode=True, protect_first_n=1,
            protect_last_n=2, threshold_tokens_cap=100,
        )
        agent.context_compressor.tail_token_budget = 30
        with (
            patch.object(agent, "_try_refresh_env_client_credentials"),
            patch.object(agent, "_build_system_prompt", return_value="You are helpful."),
            patch.object(agent, "_flush_messages_to_session_db", return_value=True),
            patch.object(agent, "_persist_session"),
            patch.object(agent, "_save_trajectory"),
            patch.object(agent, "_cleanup_task_resources"),
            patch.object(agent.context_compressor, "_generate_summary", return_value="Earlier work completed."),
        ):
            yield agent


def _recovery_history():
    return [
        {"role": "user" if i % 2 == 0 else "assistant", "content": f"message {i} " + "detail " * 100}
        for i in range(30)
    ]


def _commit_history(agent):
    history = _recovery_history()
    compressed, _ = agent._compress_context(history, "You are helpful.", force=True)
    assert len(compressed) < len(history)
    assert agent.context_compressor._verify_compaction_cleared_threshold is True
    return compressed


def _activate_recovery_fallback(agent):
    client = MagicMock()
    client.base_url = "https://api.openai.com/v1"
    client.api_key = "test-fallback-key"
    agent._fallback_chain = [{
        "provider": "openai", "model": "test/fallback", "api_key": "test-fallback-key",
        "api_mode": "chat_completions",
    }]
    agent._fallback_index = 0
    with patch("agent.auxiliary_client.resolve_provider_client", return_value=(client, None)):
        assert agent._try_activate_fallback() is True
    compressor = agent.context_compressor
    assert compressor.model == "test/fallback"
    assert compressor.last_prompt_tokens == 0
    assert compressor.awaiting_real_usage_after_compression is False
    assert compressor._verify_compaction_cleared_threshold is False
    return client


def _usage_verdict(agent, prompt_tokens, attempts=1):
    from agent.turn_usage import record_response_usage

    response = _final_response()
    if prompt_tokens == "absent":
        del response.usage
    elif prompt_tokens is not None:
        response.usage = _tool_response(prompt_tokens).usage
    return record_response_usage(
        agent, response, messages=[{"role": "user", "content": "continue"}],
        api_call_count=1, api_duration=0.01, compression_attempts=attempts,
        max_compression_attempts=1,
    )


@pytest.mark.parametrize("attempts", [0, 1])
@pytest.mark.parametrize("first_usage", ["absent", None, 0, 100, 150, 50])
def test_committed_history_survives_fallback_noop_for_one_response(
    recovery_agent, attempts, first_usage,
):
    agent = recovery_agent
    _commit_history(agent)
    _activate_recovery_fallback(agent)
    short = [{"role": "user", "content": "continue"}]
    unchanged, _ = agent._compress_context(short, "You are helpful.", force=True)
    assert unchanged == short
    assert agent.context_compressor._last_compression_made_progress is False

    outcome = _usage_verdict(agent, first_usage, attempts)
    assert outcome.rearmed is (first_usage == 50)
    assert outcome.compression_attempts == (0 if first_usage == 50 else attempts)
    # A later positive reading cannot reuse an absent/zero/high or successful verdict.
    later = _usage_verdict(agent, 50, attempts=1)
    assert later.rearmed is False
    assert later.compression_attempts == 1


@pytest.mark.parametrize("operation", ["noop", "aborted", "model_reset"])
def test_uncommitted_compression_cannot_rearm(recovery_agent, operation):
    agent = recovery_agent
    if operation == "model_reset":
        _activate_recovery_fallback(agent)
    elif operation == "noop":
        short = [{"role": "user", "content": "continue"}]
        unchanged, _ = agent._compress_context(short, "You are helpful.", force=True)
        assert unchanged == short
    else:
        agent.context_compressor.abort_on_summary_failure = True
        history = _recovery_history()
        with patch.object(agent.context_compressor, "_generate_summary", return_value=None):
            unchanged, _ = agent._compress_context(history, "You are helpful.", force=True)
        assert unchanged == history
        assert agent.context_compressor._last_compress_aborted is True
    verdict = _usage_verdict(agent, 50)
    assert verdict.rearmed is False
    assert verdict.compression_attempts == 1


@pytest.mark.parametrize("fallback_before_return", [False, True])
def test_prologue_compaction_clears_block_with_no_loop_attempts(recovery_agent, fallback_before_return):
    from agent.conversation_loop import run_preflight_gate
    from agent.turn_usage import record_response_usage

    agent = recovery_agent
    original_compress = agent._compress_context
    compressions = []
    blocks = []
    verdicts = []

    def compress(*args, **kwargs):
        result = original_compress(*args, **kwargs)
        compressions.append(result)
        if fallback_before_return:
            _activate_recovery_fallback(agent)
        agent.client.chat.completions.create.side_effect = [_tool_response(50), _final_response()]
        return result

    def gate(*args, **kwargs):
        blocks.append(kwargs["_preflight_compression_blocked"])
        if fallback_before_return and len(blocks) == 1:
            # The no-op's structural backoff must not prevent the prologue from
            # establishing the stale block this response needs to clear.
            short = [{"role": "user", "content": "continue"}]
            assert original_compress(short, "You are helpful.", force=True)[0] == short
        return run_preflight_gate(*args, **kwargs)

    import functools
    gate = functools.wraps(run_preflight_gate)(gate)

    def usage(*args, **kwargs):
        outcome = record_response_usage(*args, **kwargs)
        verdicts.append((kwargs["compression_attempts"], outcome.rearmed))
        return outcome

    def tool_result(message, messages, *_args):
        messages.append({"role": "tool", "tool_call_id": "call_1", "name": "web_search", "content": "ok"})

    with (
        patch.object(agent, "_compress_context", side_effect=compress),
        patch.object(agent, "_execute_tool_calls", side_effect=tool_result),
        patch("agent.turn_context._preflight_request_tokens", side_effect=[200, 190]),
        patch("agent.turn_context._should_run_preflight_estimate", return_value=True),
        patch.object(agent.context_compressor, "should_defer_preflight_to_real_usage", return_value=False),
        patch("agent.conversation_loop.run_preflight_gate", new=gate),
        patch("agent.turn_response_check.record_response_usage", side_effect=usage),
    ):
        result = agent.run_conversation("continue", conversation_history=_recovery_history())
    assert result["completed"] is True
    assert len(compressions) == 1
    assert verdicts[0] == (0, True)
    assert blocks == [True, False]


def test_pending_history_does_not_cross_consecutive_turns(recovery_agent):
    from agent.turn_usage import record_response_usage

    agent = recovery_agent
    observations = []

    def usage(*args, **kwargs):
        outcome = record_response_usage(*args, **kwargs)
        observations.append(outcome.rearmed)
        return outcome

    # A turn can stop after a committed prologue before receiving any provider response.
    with patch("agent.conversation_loop.build_turn_context", side_effect=RuntimeError("stop after commit")) as build:
        def stop(*args, **kwargs):
            _commit_history(agent)
            raise RuntimeError("stop after commit")
        build.side_effect = stop
        with pytest.raises(RuntimeError, match="stop after commit"):
            agent.run_conversation("first turn")
    agent.client.chat.completions.create.return_value = _final_response()
    agent.client.chat.completions.create.return_value.usage = _tool_response(50).usage
    with patch("agent.turn_response_check.record_response_usage", side_effect=usage):
        result = agent.run_conversation("second turn")
    assert result["completed"] is True
    assert observations == [False]


def test_api_exception_does_not_consume_committed_history(recovery_agent):
    agent = recovery_agent
    _commit_history(agent)
    client = _activate_recovery_fallback(agent)
    client.chat.completions.create.side_effect = [RuntimeError("provider unavailable"), _final_response()]
    with pytest.raises(RuntimeError, match="provider unavailable"):
        client.chat.completions.create()
    # Only the successful response enters usage accounting.
    response = client.chat.completions.create()
    response.usage = _tool_response(50).usage
    from agent.turn_usage import record_response_usage
    outcome = record_response_usage(
        agent, response, messages=[], api_call_count=1, api_duration=0.01,
        compression_attempts=1, max_compression_attempts=1,
    )
    assert outcome.rearmed is True
    assert outcome.compression_attempts == 0
