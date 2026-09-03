"""Contract tests for the opt-in non-reasoning compression fast lane."""

import time
from types import SimpleNamespace
from unittest.mock import MagicMock, patch


def _resolve(config, *, provider="ollama", model="qwen3:8b", requested_model=None):
    from agent.auxiliary_client import resolve_compression_fast_lane

    with patch(
        "agent.auxiliary_client._get_auxiliary_task_config",
        return_value=config,
    ):
        return resolve_compression_fast_lane(
            provider,
            model,
            requested_model=requested_model,
        )


def test_explicit_non_reasoning_compression_route_is_certified_and_bounded():
    lane = _resolve(
        {
            "provider": "ollama",
            "model": "qwen3:8b",
            "reasoning_effort": "none",
            "max_output_tokens": 1400,
        }
    )

    assert lane.certified_non_reasoning is True
    assert lane.max_tokens == 1400
    assert lane.reasoning_config == {"enabled": False, "effort": "none"}


def test_inherited_auto_or_uncertified_compression_routes_remain_uncapped():
    inherited = _resolve({"provider": "auto", "model": "", "reasoning_effort": "none", "max_output_tokens": 1400})
    unknown = _resolve({"provider": "ollama", "model": "qwen3:8b", "max_output_tokens": 1400})
    reasoning = _resolve(
        {
            "provider": "ollama",
            "model": "qwen3:8b",
            "reasoning_effort": "low",
            "max_output_tokens": 1400,
        }
    )

    for lane in (inherited, unknown, reasoning):
        assert lane.certified_non_reasoning is False
        assert lane.max_tokens is None
        assert lane.reasoning_config is None


def test_inherited_reasoning_control_is_preserved_without_enabling_a_cap():
    from agent.auxiliary_client import _get_task_extra_body

    certified = {
        "provider": "ollama",
        "model": "qwen3:8b",
        "reasoning_effort": "none",
        "max_output_tokens": 1400,
    }
    inherited = {
        "provider": "auto",
        "model": "",
        "reasoning_effort": "none",
        "max_output_tokens": 1400,
    }

    with patch("agent.auxiliary_client._get_auxiliary_task_config", return_value=certified):
        assert _get_task_extra_body("compression")["reasoning"] == {
            "enabled": False,
        }
    with patch("agent.auxiliary_client._get_auxiliary_task_config", return_value=inherited):
        assert _get_task_extra_body("compression")["reasoning"] == {
            "enabled": False,
        }


def test_summary_model_override_is_certified_against_the_effective_model():
    config = {
        "provider": "ollama",
        "model": "qwen3:8b",
        "reasoning_effort": "none",
        "max_output_tokens": 1400,
    }

    override = _resolve(
        config,
        provider="ollama",
        model="qwen3:14b",
        requested_model="qwen3:14b",
    )
    drifted = _resolve(
        config,
        provider="ollama",
        model="server-selected-model",
        requested_model="qwen3:14b",
    )

    assert override.max_tokens == 1400
    assert drifted.max_tokens is None


def test_compression_latency_records_delayed_first_provider_chunk():
    from agent.auxiliary_client import _notify_aux_progress, call_llm

    class _DelayedSemaphore:
        def acquire(self):
            time.sleep(0.01)

        def release(self):
            pass

    timings = {}

    client = MagicMock()
    client.base_url = "http://127.0.0.1:11434/v1"

    def _chunks():
        time.sleep(0.02)
        yield SimpleNamespace(
            id="chunk-1",
            model="qwen3:8b",
            usage=None,
            choices=[
                SimpleNamespace(
                    finish_reason="stop",
                    delta=SimpleNamespace(content="summary", tool_calls=None),
                )
            ],
        )

    def _resolve_client(*_args, **_kwargs):
        # Pre-dispatch liveness must not count as provider response progress.
        _notify_aux_progress()
        return client, "qwen3:8b"

    client.chat.completions.create.side_effect = lambda **_kwargs: _chunks()

    with (
        patch("agent.auxiliary_client._acquire_sync_aux_semaphore", return_value=_DelayedSemaphore()),
        patch("agent.auxiliary_client._get_cached_client", side_effect=_resolve_client),
    ):
        response = call_llm(
            task="compression",
            messages=[{"role": "user", "content": "summary request"}],
            latency_info=timings,
        )

    assert response.choices[0].message.content == "summary"
    assert timings["queue_wait_ms"] >= 5
    assert timings["provider_dispatch_ms"] >= 0
    assert timings["time_to_first_progress_ms"] >= 15
    assert timings["time_to_first_progress_ms"] >= timings["provider_dispatch_ms"]
    assert timings["summary_generation_ms"] >= timings["time_to_first_progress_ms"]


def test_certified_fast_lane_sends_the_configured_cap_to_its_provider():
    from agent.auxiliary_client import call_llm

    config = {
        "provider": "ollama",
        "model": "qwen3:8b",
        "reasoning_effort": "none",
        "max_output_tokens": 1400,
    }
    client = MagicMock()
    client.base_url = "http://127.0.0.1:11434/v1"
    response = object()
    client.chat.completions.create.return_value = response

    with (
        patch("agent.auxiliary_client._get_auxiliary_task_config", return_value=config),
        patch("agent.auxiliary_client._get_cached_client", return_value=(client, "qwen3:8b")),
        patch("agent.auxiliary_client._validate_llm_response", return_value=response),
    ):
        assert call_llm(
            task="compression",
            messages=[{"role": "user", "content": "summary request"}],
        ) is response

    request = client.chat.completions.create.call_args.kwargs
    assert request["max_tokens"] == 1400
    assert request["extra_body"]["reasoning"] == {"enabled": False}


def test_uncertified_effective_primary_route_does_not_receive_fast_cap():
    from agent.auxiliary_client import call_llm

    config = {
        "provider": "ollama",
        "model": "qwen3:8b",
        "reasoning_effort": "none",
        "max_output_tokens": 1400,
    }
    client = MagicMock()
    client.base_url = "http://127.0.0.1:11434/v1"
    response = object()
    client.chat.completions.create.return_value = response

    with (
        patch("agent.auxiliary_client._get_auxiliary_task_config", return_value=config),
        patch(
            "agent.auxiliary_client._get_cached_client",
            return_value=(client, "server-selected-model"),
        ),
        patch("agent.auxiliary_client._validate_llm_response", return_value=response),
    ):
        assert call_llm(
            task="compression",
            messages=[{"role": "user", "content": "summary request"}],
        ) is response

    request = client.chat.completions.create.call_args.kwargs
    assert "max_tokens" not in request
    assert "max_completion_tokens" not in request
    assert "reasoning" not in request.get("extra_body", {})


def test_boolean_cap_drift_stays_uncapped_and_preserves_existing_reasoning():
    from agent.auxiliary_client import call_llm

    config = {
        "provider": "ollama",
        "model": "qwen3:8b",
        "reasoning_effort": "none",
        "max_output_tokens": True,
    }
    client = MagicMock()
    client.base_url = "http://127.0.0.1:11434/v1"
    response = object()
    client.chat.completions.create.return_value = response

    with (
        patch("agent.auxiliary_client._get_auxiliary_task_config", return_value=config),
        patch(
            "agent.auxiliary_client._get_cached_client",
            return_value=(client, "server-selected-model"),
        ),
        patch("agent.auxiliary_client._validate_llm_response", return_value=response),
    ):
        assert call_llm(
            task="compression",
            messages=[{"role": "user", "content": "summary request"}],
        ) is response

    request = client.chat.completions.create.call_args.kwargs
    assert "max_tokens" not in request
    assert "max_completion_tokens" not in request
    assert request["extra_body"]["reasoning"] == {"enabled": False}


def test_bedrock_converse_ttfp_waits_for_the_nonstreaming_response():
    from agent.auxiliary_client import BedrockAuxiliaryClient, call_llm

    config = {
        "provider": "auto",
        "model": "",
        "max_output_tokens": 0,
    }
    client = BedrockAuxiliaryClient("us-east-1", "amazon.nova-lite-v1:0")
    response = SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(content="summary"),
                finish_reason="stop",
            )
        ],
        usage=None,
    )
    timings = {}

    def _delayed_converse(**_kwargs):
        time.sleep(0.02)
        return response

    with (
        patch("agent.auxiliary_client._get_auxiliary_task_config", return_value=config),
        patch(
            "agent.auxiliary_client._get_cached_client",
            return_value=(client, "amazon.nova-lite-v1:0"),
        ),
        patch("agent.bedrock_adapter.call_converse", side_effect=_delayed_converse),
    ):
        assert call_llm(
            task="compression",
            messages=[{"role": "user", "content": "summary request"}],
            latency_info=timings,
        ) is response

    assert timings["provider_dispatch_ms"] >= 0
    assert timings["time_to_first_progress_ms"] >= 15
    assert timings["time_to_first_progress_ms"] >= timings["provider_dispatch_ms"]


def test_summary_model_override_cap_uses_the_actual_primary_request():
    from agent.auxiliary_client import call_llm

    config = {
        "provider": "ollama",
        "model": "qwen3:8b",
        "reasoning_effort": "none",
        "max_output_tokens": 1400,
    }
    client = MagicMock()
    client.base_url = "http://127.0.0.1:11434/v1"
    response = object()
    client.chat.completions.create.return_value = response

    with (
        patch("agent.auxiliary_client._get_auxiliary_task_config", return_value=config),
        patch("agent.auxiliary_client._get_cached_client", return_value=(client, "qwen3:14b")),
        patch("agent.auxiliary_client._validate_llm_response", return_value=response),
    ):
        assert call_llm(
            task="compression",
            model="qwen3:14b",
            messages=[{"role": "user", "content": "summary request"}],
        ) is response

    request = client.chat.completions.create.call_args.kwargs
    assert request["model"] == "qwen3:14b"
    assert request["max_tokens"] == 1400


def test_fallback_cap_requires_independent_route_certification():
    from agent.auxiliary_client import _call_fallback_candidate_sync

    response = object()

    def _request_for(entry):
        config = {
            "fallback_chain": [entry],
            "provider": "ollama",
            "model": "qwen3:8b",
            "reasoning_effort": "none",
            "max_output_tokens": 1400,
        }
        client = MagicMock()
        client.base_url = "http://127.0.0.1:11434/v1"
        client.chat.completions.create.return_value = response
        with (
            patch("agent.auxiliary_client._get_auxiliary_task_config", return_value=config),
            patch("agent.auxiliary_client._validate_llm_response", return_value=response),
        ):
            assert _call_fallback_candidate_sync(
                client,
                "qwen3:14b",
                "fallback_chain[0](ollama)",
                task="compression",
                messages=[{"role": "user", "content": "summary request"}],
                temperature=None,
                max_tokens=None,
                tools=None,
                effective_timeout=300,
                effective_extra_body={"reasoning": {"enabled": False, "effort": "none"}},
                reasoning_config=None,
            ) is response
        return client.chat.completions.create.call_args.kwargs

    uncertified = _request_for({"provider": "ollama", "model": "qwen3:14b"})
    certified = _request_for(
        {
            "provider": "ollama",
            "model": "qwen3:14b",
            "reasoning_effort": "none",
            "max_output_tokens": 900,
        }
    )

    assert "max_tokens" not in uncertified
    assert "max_completion_tokens" not in uncertified
    assert "reasoning" not in uncertified.get("extra_body", {})
    assert certified["max_tokens"] == 900
    assert certified["extra_body"]["reasoning"] == {
        "enabled": False,
        "effort": "none",
    }


def test_reasoning_effort_aliases_certify_like_none():
    """Every spelling parse_reasoning_effort treats as disabled must certify.

    _get_task_extra_body uses parse_reasoning_effort to disable reasoning for
    "false"/"disabled"/YAML False exactly like "none"; the certification
    predicate must agree or those users silently lose the fast lane.
    """
    base = {"provider": "ollama", "model": "qwen3:8b", "max_output_tokens": 1400}

    for alias in ("none", "false", "disabled", False):
        lane = _resolve({**base, "reasoning_effort": alias})
        assert lane.certified_non_reasoning is True, alias
        assert lane.max_tokens == 1400, alias

    # Empty/unset (provider default) and real efforts must NOT certify.
    for not_disabled in ("", None, "low", "high", True):
        lane = _resolve({**base, "reasoning_effort": not_disabled})
        assert lane.certified_non_reasoning is False, not_disabled
        assert lane.max_tokens is None, not_disabled


def test_timing_hooks_propagate_to_protected_call_worker_thread():
    """The protected daemon path must carry the timing hooks across threads.

    _run_protected_sync_provider_call runs the provider callback on a daemon
    worker. The dispatch/provider-response hooks are threading.local, so
    without explicit propagation provider_dispatch_ms and
    time_to_first_progress_ms silently vanish whenever compression takes the
    protected path (the common case: aux_interrupt_protection + hard-cancel
    source both active).
    """
    from agent.auxiliary_client import (
        _aux_timing_hook,
        _aux_dispatch,
        _aux_provider_response,
        _notify_aux_dispatch,
        _notify_aux_provider_response,
        _run_protected_sync_provider_call,
        aux_interrupt_protection,
    )

    seen = []

    def _callback(_kwargs):
        # Runs on the daemon worker thread — both notifies must reach the
        # hooks installed on the owner thread.
        _notify_aux_dispatch()
        _notify_aux_provider_response()
        return "ok"

    with (
        _aux_timing_hook(_aux_dispatch, lambda: seen.append("dispatch")),
        _aux_timing_hook(_aux_provider_response, lambda: seen.append("response")),
        aux_interrupt_protection(cancel_check=lambda: False),
    ):
        result = _run_protected_sync_provider_call(_callback, {})

    assert result == "ok"
    assert "dispatch" in seen
    assert "response" in seen


def test_explicit_caller_max_tokens_keeps_provider_quirk_handling():
    """An explicit caller cap must NOT be force-injected as a wire param.

    _build_call_kwargs deliberately omits max_tokens for most
    OpenAI-compatible providers (ZAI vision 400s on it; GPT-5/Copilot need
    max_completion_tokens). Only a cap the certified lane itself produced may
    bypass that handling. Before this guard, a caller-passed max_tokens on
    the compression task flowed through _compression_fast_lane_controls as a
    passthrough and was misread as a lane cap — forcing the param onto
    providers where the omission was intentional (pre-fast-lane behavior).
    """
    from agent.auxiliary_client import call_llm

    config = {"provider": "auto", "model": "", "max_output_tokens": 0}
    client = MagicMock()
    client.base_url = "http://127.0.0.1:11434/v1"
    response = object()
    client.chat.completions.create.return_value = response

    with (
        patch("agent.auxiliary_client._get_auxiliary_task_config", return_value=config),
        patch("agent.auxiliary_client._get_cached_client", return_value=(client, "qwen3:8b")),
        patch("agent.auxiliary_client._validate_llm_response", return_value=response),
    ):
        assert call_llm(
            task="compression",
            messages=[{"role": "user", "content": "summary request"}],
            max_tokens=1500,
        ) is response

    request = client.chat.completions.create.call_args.kwargs
    assert "max_tokens" not in request
    assert "max_completion_tokens" not in request
