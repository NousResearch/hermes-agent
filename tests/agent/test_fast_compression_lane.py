"""Contract tests for the opt-in non-reasoning compression fast lane."""

import time
from unittest.mock import MagicMock, patch


def _resolve(config):
    from agent.auxiliary_client import resolve_compression_fast_lane

    with patch(
        "agent.auxiliary_client._get_auxiliary_task_config",
        return_value=config,
    ):
        return resolve_compression_fast_lane()


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


def test_only_a_certified_lane_forwards_non_reasoning_request_controls():
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
            "effort": "none",
        }
    with patch("agent.auxiliary_client._get_auxiliary_task_config", return_value=inherited):
        assert "reasoning" not in _get_task_extra_body("compression")


def test_compression_latency_records_queue_wait_and_first_progress():
    from agent.auxiliary_client import _notify_aux_progress, call_llm

    class _DelayedSemaphore:
        def acquire(self):
            time.sleep(0.01)

        def release(self):
            pass

    timings = {}

    def _progressing_call(**_kwargs):
        _notify_aux_progress()
        return object()

    with (
        patch("agent.auxiliary_client._acquire_sync_aux_semaphore", return_value=_DelayedSemaphore()),
        patch("agent.auxiliary_client._call_llm_impl", side_effect=_progressing_call),
    ):
        call_llm(
            task="compression",
            messages=[{"role": "user", "content": "summary request"}],
            latency_info=timings,
        )

    assert timings["queue_wait_ms"] >= 5
    assert timings["time_to_first_progress_ms"] >= 0
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
            max_tokens=1400,
        ) is response

    request = client.chat.completions.create.call_args.kwargs
    assert request["max_tokens"] == 1400
    assert request["extra_body"]["reasoning"] == {"enabled": False, "effort": "none"}
