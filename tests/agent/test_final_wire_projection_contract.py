"""Final semantic family projections, not source-shaped assertions."""
from copy import deepcopy

import httpx
import pytest

from agent.final_wire_admission import (
    COVERED_MAIN, FinalAttemptIdentity, ProviderBoundUnsupportedAccounting,
    bind_attempt_identity, wrap_httpx_client_transports, project_final_body,
)


def ident(family, window=1000):
    return FinalAttemptIdentity(COVERED_MAIN, family, "inert", "https://inert.invalid", window, "projection")


@pytest.mark.parametrize("family,body", [
    ("chat_completions", {"messages": [{"role": "assistant", "content": "", "reasoning_details": [{"text": "x" * 4400}]}]}),
    ("codex_responses", {"input": [{"type": "function_call", "name": "inert", "arguments": "x" * 4400}]}),
    ("gemini_native", {"contents": [{"role": "user", "parts": [{"text": "x" * 4400}]}], "generationConfig": {"maxOutputTokens": 1}}),
    ("chat_completions", {"messages": [{"role": "user", "content": "hi"}], "stop": ["x" * 4400]}),
    ("bedrock_converse", {"messages": [{"role": "user", "content": [{"text": "hi"}]}], "promptVariables": {"x": {"text": "x" * 4400}}}),
])
def test_transmitted_context_is_counted_at_physical_boundary(family, body):
    from agent.conversation_compression import ProviderBoundRequestOverLimit
    sent = []
    with httpx.Client(transport=httpx.MockTransport(lambda r: sent.append(r) or httpx.Response(200))) as client:
        wrap_httpx_client_transports(client)
        with bind_attempt_identity(ident(family)), pytest.raises(ProviderBoundRequestOverLimit):
            client.post("https://inert.invalid", json=body)
    assert sent == []


@pytest.mark.parametrize("family,body", [
    ("chat_completions", {"messages": [{"role": "user", "content": [{"type": "unknown_block", "payload": "hi"}]}]}),
    ("anthropic_messages", {"messages": [{"role": "user", "content": [{"type": "document", "source": {"type": "base64", "data": "abc"}}]}], "max_tokens": 1}),
    ("gemini_native", {"contents": [{"parts": [{"inlineData": {"mimeType": "audio/wav", "data": "abc"}}]}], "generationConfig": {"maxOutputTokens": 1}}),
])
def test_unknown_or_unsupported_block_is_not_zero(family, body):
    sent = []
    with httpx.Client(transport=httpx.MockTransport(lambda r: sent.append(r) or httpx.Response(200))) as client:
        wrap_httpx_client_transports(client)
        with bind_attempt_identity(ident(family)), pytest.raises(ProviderBoundUnsupportedAccounting):
            client.post("https://inert.invalid", json=body)
    assert sent == []
