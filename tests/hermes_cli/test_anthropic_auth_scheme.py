"""Explicit authentication schemes for custom Anthropic Messages providers."""

from unittest.mock import MagicMock, patch

import pytest

from hermes_cli import runtime_provider
from hermes_cli.config import (
    _normalize_custom_provider_entry,
    validate_config_structure,
)


def _provider_entry(auth_scheme=None):
    entry = {
        "name": "example-relay",
        "base_url": "https://relay.example/anthropic",
        "api_mode": "anthropic_messages",
        "api_key": "test-key-not-secret",
    }
    if auth_scheme is not None:
        entry["auth_scheme"] = auth_scheme
    return entry


@pytest.mark.parametrize(
    ("configured", "expected"),
    [(None, None), ("BEARER", "bearer"), ("X-API-KEY", "x-api-key")],
)
def test_custom_provider_auth_scheme_normalization(configured, expected):
    normalized = _normalize_custom_provider_entry(_provider_entry(configured))

    assert normalized is not None
    assert normalized.get("auth_scheme") == expected


def test_invalid_custom_provider_auth_scheme_is_rejected_by_validation():
    issues = validate_config_structure(
        {"custom_providers": [_provider_entry("basic")]}
    )

    assert any(
        issue.severity == "error" and "auth_scheme" in issue.message
        for issue in issues
    )


def test_invalid_named_custom_runtime_auth_scheme_raises(monkeypatch):
    monkeypatch.setattr(
        runtime_provider,
        "load_config",
        lambda: {"custom_providers": [_provider_entry("basic")]},
    )

    with pytest.raises(ValueError, match="auth_scheme"):
        runtime_provider.resolve_runtime_provider(requested="custom:example-relay")

@pytest.mark.parametrize("auth_scheme", ["bearer", "x-api-key"])
def test_named_custom_runtime_preserves_explicit_auth_scheme(
    monkeypatch, auth_scheme
):
    monkeypatch.setattr(
        runtime_provider,
        "load_config",
        lambda: {"custom_providers": [_provider_entry(auth_scheme)]},
    )

    runtime = runtime_provider.resolve_runtime_provider(
        requested="custom:example-relay"
    )

    assert runtime["provider"] == "custom"
    assert runtime["api_mode"] == "anthropic_messages"
    assert runtime["auth_scheme"] == auth_scheme


def test_primary_agent_forwards_explicit_auth_scheme(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))

    with (
        patch("run_agent.get_tool_definitions", return_value=[]),
        patch("run_agent.check_toolset_requirements", return_value={}),
        patch("run_agent.OpenAI"),
        patch("hermes_logging.setup_logging"),
        patch(
            "agent.anthropic_adapter.build_anthropic_client",
            return_value=MagicMock(),
        ) as build_client,
    ):
        from run_agent import AIAgent

        agent = AIAgent(
            provider="custom",
            model="example-model",
            base_url="https://relay.example/anthropic",
            api_key="test-key-not-secret",
            api_mode="anthropic_messages",
            auth_scheme="bearer",
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
        )

    assert agent._anthropic_auth_scheme == "bearer"
    assert build_client.call_args.kwargs["auth_scheme"] == "bearer"


def test_auxiliary_wrapper_uses_custom_provider_auth_scheme():
    from agent.auxiliary_client import AnthropicAuxiliaryClient, _maybe_wrap_anthropic

    with (
        patch(
            "agent.anthropic_adapter.build_anthropic_client",
            return_value=MagicMock(),
        ) as build_client,
        patch(
            "hermes_cli.config.get_custom_provider_auth_scheme",
            return_value="x-api-key",
        ),
    ):
        wrapped = _maybe_wrap_anthropic(
            MagicMock(),
            "example-model",
            "test-key-not-secret",
            "https://relay.example/anthropic",
            "anthropic_messages",
        )

    assert isinstance(wrapped, AnthropicAuxiliaryClient)
    assert wrapped.auth_scheme == "x-api-key"
    assert build_client.call_args.kwargs["auth_scheme"] == "x-api-key"
