"""Regression tests for provider request metadata crossing Gateway routing."""

from unittest.mock import patch

import gateway.run as gateway_run


_REQUEST_OVERRIDES = {
    "extra_body": {
        "enable_thinking": True,
        "tools": [{"type": "web_search"}, {"type": "web_extractor"}],
    }
}


def _resolved_runtime() -> dict:
    return {
        "api_key": "test-key",
        "base_url": "https://example.invalid/v1",
        "provider": "custom",
        "requested_provider": "custom:research-provider",
        "api_mode": "codex_responses",
        "command": None,
        "args": [],
        "credential_pool": None,
        "request_overrides": _REQUEST_OVERRIDES,
    }


def test_primary_runtime_preserves_provider_request_overrides():
    with (
        patch(
            "hermes_cli.runtime_provider.resolve_runtime_provider",
            return_value=_resolved_runtime(),
        ),
        patch("hermes_cli.runtime_provider._get_model_config", return_value={}),
    ):
        result = gateway_run._resolve_runtime_agent_kwargs()

    assert result["request_overrides"] == _REQUEST_OVERRIDES
    assert result["request_overrides"] is not _REQUEST_OVERRIDES


def test_channel_runtime_preserves_provider_request_overrides():
    with patch(
        "hermes_cli.runtime_provider.resolve_runtime_provider",
        return_value=_resolved_runtime(),
    ) as resolve:
        result = gateway_run._resolve_runtime_agent_kwargs_for_provider(
            "custom:research-provider"
        )

    resolve.assert_called_once_with(requested="custom:research-provider")
    assert result["request_overrides"] == _REQUEST_OVERRIDES
    assert result["request_overrides"] is not _REQUEST_OVERRIDES


def test_fallback_runtime_preserves_provider_request_overrides():
    fallback = {
        "provider": "custom:research-provider",
        "model": "research-model",
    }
    with (
        patch("gateway.run._load_gateway_runtime_config", return_value={}),
        patch("gateway.run.get_fallback_chain", return_value=[fallback]),
        patch(
            "hermes_cli.fallback_config.resolve_entry_api_key",
            return_value="test-key",
        ),
        patch(
            "hermes_cli.runtime_provider.resolve_runtime_provider",
            return_value=_resolved_runtime(),
        ),
    ):
        result = gateway_run._try_resolve_fallback_provider()

    assert result is not None
    assert result["request_overrides"] == _REQUEST_OVERRIDES
    assert result["request_overrides"] is not _REQUEST_OVERRIDES
