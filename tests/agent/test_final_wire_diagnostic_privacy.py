"""Inert private-looking sentinels never enter local diagnostics or telemetry."""
import json
import logging
from contextlib import ExitStack
from unittest.mock import patch

import httpx
import pytest
from openai import OpenAI
from agent.final_wire_admission import intercepted_openai_class, wrap_httpx_client_transports
from tests.run_agent.test_413_compression import agent  # noqa: F401

PAYLOAD = "INERT_PRIVATE_MESSAGE_SENTINEL"
SECRET = "INERT_AUTH_SECRET_SENTINEL"
SCHEMA = "INERT_RAW_SCHEMA_SENTINEL"


@pytest.mark.parametrize("kind", ["pressure", "invalid", "unsupported"])
def test_actual_loop_local_refusal_diagnostics_are_private_typed_not_timeout(agent, caplog, kind):
    a = agent
    a.provider, a.api_mode, a.model, a.base_url = "openai", "chat_completions", "inert", "https://inert.invalid"
    a._base_url_lower = a.base_url
    a.context_compressor.context_length = 1000
    a.context_compressor.threshold_tokens = 500
    a._cached_system_prompt = "inert system"
    a.max_tokens = 1
    a.tools = []
    delegated = []
    def mutate(request):
        body = json.loads(request.content)
        body["messages"] = [{"role": "user", "content": PAYLOAD * (200 if kind == "pressure" else 1)}]
        body["tools"] = [{"type": "function", "function": {"name": "inert", "description": SCHEMA, "parameters": {"type": "object"}}}]
        if kind == "invalid":
            body["max_tokens"] = 0
        if kind == "unsupported":
            body["unknown_context_field"] = PAYLOAD
        request.headers["Authorization"] = "Bearer " + SECRET
        request.stream = httpx.ByteStream(json.dumps(body).encode())
        if hasattr(request, "_content"):
            del request._content
        request.read()
    with httpx.Client(transport=httpx.MockTransport(lambda request: delegated.append(request) or httpx.Response(200)), event_hooks={"request": [mutate]}) as http:
        wrap_httpx_client_transports(http)
        sdk = intercepted_openai_class(OpenAI)(api_key=SECRET, base_url=a.base_url, http_client=http, max_retries=2)
        with ExitStack() as stack:
            stack.enter_context(patch.object(a, "_create_request_openai_client", return_value=sdk))
            stack.enter_context(patch("agent.turn_context.estimate_request_tokens_rough", return_value=10))
            stack.enter_context(patch.object(a, "_supports_reasoning_extra_body", return_value=False))
            statuses = stack.enter_context(patch.object(a, "_emit_status"))
            stack.enter_context(patch.object(a, "_persist_session"))
            stack.enter_context(patch.object(a, "_save_trajectory"))
            stack.enter_context(patch.object(a, "_cleanup_task_resources"))
            fallback = stack.enter_context(patch.object(a, "_try_activate_fallback"))
            caplog.clear()
            caplog.set_level(logging.INFO)
            result = a.run_conversation("hi")
        assert delegated == []
        assert fallback.call_count == 0
        assert result["failed"] and not result["completed"]
        diagnostics = caplog.text + str(statuses.call_args_list) + result["final_response"]
        for sentinel in (PAYLOAD, SECRET, SCHEMA):
            assert sentinel not in diagnostics
        assert "ReadError" not in diagnostics
        assert "timeout" not in diagnostics.lower()
        assert "timed out" not in diagnostics.lower()
        assert f"local_accounting_{kind}" in diagnostics
        assert "not sent" in result["final_response"]
