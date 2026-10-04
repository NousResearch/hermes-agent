"""Retain remote-overflow recovery and actual Chat tool-call replay accounting."""
import httpx
import pytest
from openai import OpenAI
from tests.run_agent.test_413_compression import (
    agent,
    TestHTTP413Compression as HTTPRecoveryCases,
    TestToolResultPreflightCompression as ToolRecoveryCases,
)
from agent.final_wire_admission import (
    COVERED_MAIN, FinalAttemptIdentity, bind_attempt_identity,
    intercepted_openai_class, wrap_httpx_client_transports,
)


@pytest.mark.parametrize("case", ["vision", "assembled"])
def test_existing_remote_recovery_path_accounting_compatibility(agent, case):
    if case == "vision":
        HTTPRecoveryCases().test_413_strips_vision_payloads_when_compression_cannot_reduce_messages(agent)
    else:
        ToolRecoveryCases().test_mid_turn_retry_compares_fully_assembled_requests(agent)


def test_actual_sdk_tool_call_replay_is_counted_and_dispatchable():
    seen = []
    def receive(request):
        seen.append(request)
        return httpx.Response(200, json={"id": "inert", "object": "chat.completion", "choices": [{"index": 0, "message": {"role": "assistant", "content": "ok"}, "finish_reason": "stop"}]})
    with httpx.Client(transport=httpx.MockTransport(receive)) as http:
        wrap_httpx_client_transports(http)
        sdk = intercepted_openai_class(OpenAI)(api_key="inert", base_url="https://inert.invalid", http_client=http)
        identity = FinalAttemptIdentity(COVERED_MAIN, "chat_completions", "inert", "https://inert.invalid", 1000, "replay")
        with bind_attempt_identity(identity):
            sdk.chat.completions.create(model="inert", max_tokens=1, messages=[{"role": "user", "content": "hi"}, {"role": "assistant", "content": None, "tool_calls": [{"type": "function", "id": "inert", "function": {"name": "inert", "arguments": "{}"}}]}, {"role": "tool", "tool_call_id": "inert", "content": "done"}])
    assert len(seen) == 1
