"""Transient Bedrock exception frames mid-stream are retried like a 5xx, not ended as a local bug (#138580)."""
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from agent import retry_utils
from run_agent import AIAgent


@pytest.fixture
def agent():
    with (
        patch("model_tools.get_tool_definitions", return_value=[]),
        patch("model_tools.check_toolset_requirements", return_value={}),
        patch("agent.process_bootstrap.OpenAI"),
    ):
        instance = AIAgent(
            api_key="test-key-1234567890",
            base_url="https://openrouter.ai/api/v1",
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
        )
    instance.client = MagicMock()
    instance._cached_system_prompt = "You are helpful."
    instance._use_prompt_caching = False
    instance.tool_delay = 0
    instance.compression_enabled = False
    instance.save_trajectories = False
    instance._fallback_chain = []
    instance._api_max_retries = 2
    return instance


def _response(content):
    message = SimpleNamespace(content=content, tool_calls=None)
    choice = SimpleNamespace(message=message, finish_reason="stop")
    return SimpleNamespace(choices=[choice], model="test/model", usage=None)


@pytest.mark.parametrize(
    "exception_type",
    [
        "internalServerException",
        "modelStreamErrorException",
        "throttlingException",
        "serviceUnavailableException",
        "modelTimeoutException",
    ],
)
def test_bedrock_transient_value_error_retries(agent, exception_type):
    error = ValueError(
        "Bad response code, expected 200: "
        f"{{':event-type': 'exception', ':exception-type': '{exception_type}'}}"
    )
    with (
        patch.object(
            agent,
            "_interruptible_api_call",
            side_effect=[error, _response("Recovered")],
        ) as api_call,
        patch.object(agent, "_persist_session"),
        patch.object(agent, "_save_trajectory"),
        patch.object(agent, "_cleanup_task_resources"),
        patch.object(retry_utils, "jittered_backoff", return_value=0.0),
    ):
        result = agent.run_conversation("hello")
    assert api_call.call_count == 2
    assert result["completed"] is True
    assert result["final_response"] == "Recovered"


def test_bedrock_validation_error_does_not_retry(agent):
    error = ValueError(
        "Bad response code, expected 200: "
        "{':event-type': 'exception', ':exception-type': 'validationException'}"
    )
    with (
        patch.object(agent, "_interruptible_api_call", side_effect=error) as api_call,
        patch.object(agent, "_persist_session"),
        patch.object(agent, "_save_trajectory"),
        patch.object(agent, "_cleanup_task_resources"),
        patch.object(agent, "_dump_api_request_debug"),
    ):
        result = agent.run_conversation("hello")
    assert api_call.call_count == 1
    assert result["completed"] is False
    assert result["failed"] is True
    assert "validationException" in result["error"]


def _decoder_error(exception_type, message):
    # Same text the SDK's Bedrock decoder raises: the repr of botocore's ``to_response_dict()``.
    response_dict = {
        "status_code": 400,
        "headers": {
            ":exception-type": exception_type,
            ":content-type": "application/json",
            ":message-type": "exception",
        },
        "body": ('{"message":"%s"}' % message).encode(),
    }
    return ValueError(f"Bad response code, expected 200: {response_dict}")


def test_validation_frame_naming_a_transient_type_in_its_body_does_not_retry(agent):
    error = _decoder_error(
        "validationException",
        "Malformed input. Retry only on internalServerException or throttlingException.",
    )
    with (
        patch.object(agent, "_interruptible_api_call", side_effect=error) as api_call,
        patch.object(agent, "_persist_session"),
        patch.object(agent, "_save_trajectory"),
        patch.object(agent, "_cleanup_task_resources"),
        patch.object(agent, "_dump_api_request_debug"),
    ):
        result = agent.run_conversation("hello")
    assert api_call.call_count == 1
    assert result["failed"] is True


def test_fault_frame_classifier_reads_the_exception_type_header():
    from agent.api_error_summary import is_provider_stream_fault_frame

    assert is_provider_stream_fault_frame(_decoder_error("internalServerException", "Try again."))
    assert not is_provider_stream_fault_frame(
        _decoder_error("validationException", "see internalServerException")
    )
    assert not is_provider_stream_fault_frame(ValueError("internalServerException"))
