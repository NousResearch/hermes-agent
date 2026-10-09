from __future__ import annotations
from agent.llm_egress_runtime import dispatch_provider_request as _dispatch_provider_request

from hashlib import sha256
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from agent.chat_completion_helpers import (
    _dispatch_nonstreaming_api_request,
)
from agent.llm_egress_firewall import EgressBlocked


def _agent(tmp_path, *, provider="nous", api_mode="chat_completions"):
    if provider == "openai-codex":
        base_url = "https://chatgpt.com/backend-api/codex"
    elif provider == "anthropic":
        base_url = "https://api.anthropic.com/v1"
    else:
        base_url = "https://inference-api.nousresearch.com/v1"
    return SimpleNamespace(
        provider=provider,
        model="test-model",
        base_url=base_url,
        api_mode=api_mode,
        session_id="session-1",
        _current_turn_id="turn-1",
        _current_api_request_id="request-1",
        _llm_egress_policy_digest=sha256(b"policy").hexdigest(),
        _llm_egress_state_dir=tmp_path,
    )


@pytest.mark.parametrize(
    "provider", ["openai-codex", "nous", "nous-portal", "nousresearch", "anthropic"]
)
def test_protected_main_provider_denies_before_callback(tmp_path, provider):
    agent = _agent(tmp_path, provider=provider)
    callback = MagicMock()

    with pytest.raises(EgressBlocked):
        _dispatch_provider_request(
            agent,
            {
                "model": "test-model",
                "messages": [{"role": "user", "content": "token=super-secret-value"}],
            },
            callback,
        )

    callback.assert_not_called()


def test_local_main_provider_keeps_zero_firewall_overhead(tmp_path):
    agent = _agent(tmp_path, provider="ollama-launch")
    agent.base_url = "http://127.0.0.1:11434/v1"
    request = {"messages": [{"role": "user", "content": "/Users/private/file.py"}]}
    callback = MagicMock(return_value="local")

    assert _dispatch_provider_request(agent, request, callback) == "local"
    callback.assert_called_once_with(request)


def test_nous_chat_completions_entrypoint_uses_firewall(tmp_path):
    agent = _agent(tmp_path)
    client = MagicMock()

    with pytest.raises(EgressBlocked):
        _dispatch_nonstreaming_api_request(
            agent,
            {
                "model": "test-model",
                "messages": [{"role": "user", "content": "/Users/private/file.py"}],
            },
            make_client=lambda *_args, **_kwargs: client,
        )

    client.chat.completions.create.assert_not_called()


def test_nous_anthropic_entrypoint_uses_firewall(tmp_path):
    agent = _agent(tmp_path, api_mode="anthropic_messages")
    agent._anthropic_messages_create = MagicMock()
    client = MagicMock()

    with pytest.raises(EgressBlocked):
        _dispatch_nonstreaming_api_request(
            agent,
            {
                "model": "test-model",
                "messages": [{"role": "user", "content": "/Users/private/file.py"}],
            },
            make_client=lambda *_args, **_kwargs: client,
        )

    agent._anthropic_messages_create.assert_not_called()

from agent.error_classifier import classify_api_error, FailoverReason

def test_egress_policy_denial_falls_back_without_retry():
    from agent.llm_egress_firewall import (
        DestinationClass,
        EgressBlocked,
        EgressDecision,
    )

    error = EgressBlocked(
        EgressDecision(
            allowed=False,
            destination_class=DestinationClass.REMOTE,
            provider="nous",
            model="test-model",
            payload_sha256="",
            serialized_bytes=0,
            estimated_tokens=0,
            source_grant_count=0,
            source_segment_count=0,
            session_id="session",
            turn_id="turn",
            request_id="request",
            policy_digest="policy",
            reason_codes=("secret_detected",),
        )
    )

    result = classify_api_error(error, provider="nous", model="test-model")

    assert result.reason is FailoverReason.egress_policy_blocked
    assert result.retryable is False
    assert result.should_fallback is True
