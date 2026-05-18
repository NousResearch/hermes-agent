"""Tests for the AWS Bedrock auxiliary-client dual-path routing.

Bedrock supports two auth modes:

  1. AWS_BEARER_TOKEN_BEDROCK (Bedrock-specific bearer token).  The
     ``anthropic.AnthropicBedrock`` SDK does NOT support bearer tokens —
     its auth helper consults the boto3 credential chain for IAM keys
     only and raises ``RuntimeError: could not resolve credentials from
     session`` when only a bearer token is present.  Bearer-token users
     must therefore go through the boto3 Converse API instead.

  2. IAM credentials (env vars / SSO / instance profile).  The
     AnthropicBedrock SDK works fine here and gives feature parity with
     the main agent loop's anthropic_messages path (prompt caching,
     thinking budgets, adaptive thinking).

These tests verify the auxiliary-client correctly picks each path —
mirroring the dual-path routing in
``hermes_cli.runtime_provider._build_runtime_provider``.

Regression for: aux client raised ``RuntimeError: could not resolve
credentials from session`` for vision tasks when the user authed with
a Bedrock bearer token, even though the main model loop worked fine.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch, MagicMock

import pytest


@pytest.fixture(autouse=True)
def _clean_aws_env(monkeypatch):
    """Strip every AWS auth env var so each test sets exactly what it needs."""
    for key in (
        "AWS_BEARER_TOKEN_BEDROCK",
        "AWS_ACCESS_KEY_ID",
        "AWS_SECRET_ACCESS_KEY",
        "AWS_SESSION_TOKEN",
        "AWS_PROFILE",
        "AWS_DEFAULT_REGION",
        "AWS_REGION",
    ):
        monkeypatch.delenv(key, raising=False)


def test_bedrock_bearer_token_routes_to_converse_client(monkeypatch):
    """AWS_BEARER_TOKEN_BEDROCK should pick BedrockConverseAuxiliaryClient,
    not the AnthropicBedrock-SDK-backed AnthropicAuxiliaryClient (which
    can't use bearer tokens)."""
    from agent.auxiliary_client import (
        resolve_provider_client,
        BedrockConverseAuxiliaryClient,
        AnthropicAuxiliaryClient,
    )

    monkeypatch.setenv("AWS_BEARER_TOKEN_BEDROCK", "bedrock-api-key-test123")
    monkeypatch.setenv("AWS_REGION", "us-east-1")

    client, model = resolve_provider_client(
        "bedrock", "us.anthropic.claude-opus-4-7"
    )

    assert client is not None, "bearer-token auth should resolve to a client"
    assert isinstance(client, BedrockConverseAuxiliaryClient), (
        f"bearer-token auth should pick BedrockConverseAuxiliaryClient, got "
        f"{type(client).__name__}"
    )
    assert not isinstance(client, AnthropicAuxiliaryClient), (
        "bearer-token auth must NOT pick AnthropicAuxiliaryClient — that "
        "wraps anthropic.AnthropicBedrock which raises 'could not resolve "
        "credentials from session' on bearer tokens"
    )
    assert model == "us.anthropic.claude-opus-4-7"
    assert client.api_key == "aws-sdk"
    assert client.base_url == "https://bedrock-runtime.us-east-1.amazonaws.com"


def test_bedrock_iam_credentials_route_to_anthropic_client(monkeypatch):
    """IAM credentials should pick AnthropicAuxiliaryClient (AnthropicBedrock
    SDK), preserving the legacy path for SSO / instance-profile / IAM-key
    users."""
    from agent.auxiliary_client import (
        resolve_provider_client,
        AnthropicAuxiliaryClient,
        BedrockConverseAuxiliaryClient,
    )

    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "AKIAIOSFODNN7EXAMPLE")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY")
    monkeypatch.setenv("AWS_REGION", "us-east-1")

    # Stub the anthropic-bedrock client builder so the test doesn't need the
    # real SDK or real AWS credentials at import time.
    fake_anthropic_bedrock = MagicMock()
    with patch(
        "agent.anthropic_adapter.build_anthropic_bedrock_client",
        return_value=fake_anthropic_bedrock,
    ):
        client, model = resolve_provider_client(
            "bedrock", "us.anthropic.claude-opus-4-7"
        )

    assert client is not None
    assert isinstance(client, AnthropicAuxiliaryClient), (
        f"IAM auth should pick AnthropicAuxiliaryClient, got "
        f"{type(client).__name__}"
    )
    assert not isinstance(client, BedrockConverseAuxiliaryClient)
    assert model == "us.anthropic.claude-opus-4-7"


def test_bedrock_converse_adapter_calls_call_converse(monkeypatch):
    """BedrockConverseAuxiliaryClient.chat.completions.create should
    delegate to bedrock_adapter.call_converse and return its
    OpenAI-compatible SimpleNamespace unchanged."""
    from agent.auxiliary_client import BedrockConverseAuxiliaryClient

    fake_response = SimpleNamespace(
        choices=[
            SimpleNamespace(
                index=0,
                message=SimpleNamespace(
                    role="assistant",
                    content="hello world",
                    tool_calls=None,
                    reasoning_content=None,
                ),
                finish_reason="stop",
            )
        ],
        usage=SimpleNamespace(prompt_tokens=10, completion_tokens=5, total_tokens=15),
        model="us.anthropic.claude-opus-4-7",
    )

    client = BedrockConverseAuxiliaryClient(
        region="us-east-1",
        model="us.anthropic.claude-opus-4-7",
        base_url="https://bedrock-runtime.us-east-1.amazonaws.com",
    )

    with patch(
        "agent.bedrock_adapter.call_converse", return_value=fake_response
    ) as mock_converse:
        resp = client.chat.completions.create(
            model="us.anthropic.claude-opus-4-7",
            messages=[{"role": "user", "content": "hi"}],
            max_tokens=128,
            temperature=0.5,
        )

    assert resp is fake_response, "adapter must return the converse response unchanged"
    mock_converse.assert_called_once()
    call_kwargs = mock_converse.call_args.kwargs
    assert call_kwargs["region"] == "us-east-1"
    assert call_kwargs["model"] == "us.anthropic.claude-opus-4-7"
    assert call_kwargs["messages"] == [{"role": "user", "content": "hi"}]
    assert call_kwargs["max_tokens"] == 128
    assert call_kwargs["temperature"] == 0.5


def test_bedrock_converse_adapter_falls_back_max_completion_tokens(monkeypatch):
    """The Anthropic OpenAI shim sometimes sends max_completion_tokens
    instead of max_tokens — the adapter must accept both."""
    from agent.auxiliary_client import BedrockConverseAuxiliaryClient

    client = BedrockConverseAuxiliaryClient(
        region="us-east-1",
        model="us.anthropic.claude-opus-4-7",
        base_url="https://bedrock-runtime.us-east-1.amazonaws.com",
    )
    with patch(
        "agent.bedrock_adapter.call_converse",
        return_value=SimpleNamespace(choices=[], usage=None, model=""),
    ) as mock_converse:
        client.chat.completions.create(
            model="us.anthropic.claude-opus-4-7",
            messages=[{"role": "user", "content": "hi"}],
            max_completion_tokens=512,
        )
    assert mock_converse.call_args.kwargs["max_tokens"] == 512


def test_bedrock_converse_adapter_normalizes_stop_string():
    """OpenAI-style ``stop="\\n"`` strings must be wrapped in a list before
    going to Bedrock Converse (which expects ``stop_sequences: List[str]``)."""
    from agent.auxiliary_client import BedrockConverseAuxiliaryClient

    client = BedrockConverseAuxiliaryClient(
        region="us-east-1",
        model="us.anthropic.claude-opus-4-7",
        base_url="https://bedrock-runtime.us-east-1.amazonaws.com",
    )
    with patch(
        "agent.bedrock_adapter.call_converse",
        return_value=SimpleNamespace(choices=[], usage=None, model=""),
    ) as mock_converse:
        client.chat.completions.create(
            model="us.anthropic.claude-opus-4-7",
            messages=[{"role": "user", "content": "hi"}],
            stop="\n",
        )
    assert mock_converse.call_args.kwargs["stop_sequences"] == ["\n"]


def test_async_bedrock_converse_client_wraps_sync(monkeypatch):
    """AsyncBedrockConverseAuxiliaryClient.chat.completions.create should
    delegate to the sync client via asyncio.to_thread."""
    import asyncio

    from agent.auxiliary_client import (
        BedrockConverseAuxiliaryClient,
        AsyncBedrockConverseAuxiliaryClient,
    )

    sync_client = BedrockConverseAuxiliaryClient(
        region="us-east-1",
        model="us.anthropic.claude-opus-4-7",
        base_url="https://bedrock-runtime.us-east-1.amazonaws.com",
    )
    async_client = AsyncBedrockConverseAuxiliaryClient(sync_client)

    fake_response = SimpleNamespace(choices=[], usage=None, model="")
    with patch(
        "agent.bedrock_adapter.call_converse", return_value=fake_response
    ) as mock_converse:
        result = asyncio.run(
            async_client.chat.completions.create(
                model="us.anthropic.claude-opus-4-7",
                messages=[{"role": "user", "content": "hi"}],
            )
        )

    assert result is fake_response
    mock_converse.assert_called_once()
    # Identity check ensures cache eviction in resolve_provider_client
    # works correctly for async clients (matches existing aux client pattern).
    assert async_client._real_client is sync_client._real_client
