"""CLI's interactive fallback resolver must keep a named custom provider's configured
identity, same as the gateway/cron/tui_gateway fallback resolvers (#98739)."""

from unittest.mock import MagicMock

import pytest

from hermes_cli.auth import AuthError
from hermes_cli.cli_agent_setup_mixin import CLIAgentSetupMixin


class _FallbackCLI(CLIAgentSetupMixin):
    """Only what _resolve_fallback_runtime touches."""

    def __init__(self, fallback_model):
        self._fallback_model = fallback_model
        self.requested_provider = None
        self.model = None


@pytest.fixture(autouse=True)
def _stub_runtime_provider(monkeypatch):
    """resolve_runtime_provider returns the bare "custom" billing class for a named
    providers:/custom_providers: entry — the entry's configured id only survives in
    requested_provider (mirrors hermes_cli.runtime_provider's real contract)."""
    def _fake_resolve(*, requested, **_kwargs):
        return {
            "api_key": "sk-fallback",
            "base_url": "https://my-custom-llm.example/v1",
            "provider": "custom",
            "requested_provider": requested,
        }

    monkeypatch.setattr("hermes_cli.runtime_provider.resolve_runtime_provider", _fake_resolve)


def test_named_custom_fallback_keeps_configured_identity():
    cli = _FallbackCLI([{"model": "some-model", "provider": "my-custom-llm"}])

    runtime = cli._resolve_fallback_runtime(AuthError("primary auth failed"))

    assert runtime is not None
    assert runtime["provider"] == "my-custom-llm"
    assert cli.requested_provider == "my-custom-llm"
    assert cli.model == "some-model"


def test_builtin_fallback_provider_is_untouched(monkeypatch):
    def _fake_resolve(*, requested, **_kwargs):
        return {"api_key": "sk-fallback", "base_url": "https://openrouter.ai/api/v1",
                "provider": requested, "requested_provider": requested}

    monkeypatch.setattr("hermes_cli.runtime_provider.resolve_runtime_provider", _fake_resolve)
    cli = _FallbackCLI([{"model": "glm-4.6", "provider": "openrouter"}])

    runtime = cli._resolve_fallback_runtime(AuthError("primary auth failed"))

    assert runtime["provider"] == "openrouter"


def test_non_auth_error_returns_none():
    cli = _FallbackCLI([{"model": "some-model", "provider": "my-custom-llm"}])

    assert cli._resolve_fallback_runtime(RuntimeError("not an auth error")) is None
