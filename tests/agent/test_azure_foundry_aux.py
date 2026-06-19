"""Tests for azure-foundry as an auxiliary provider.

Covers:
- azure-foundry + api_mode=anthropic_messages routes through
  AnthropicAuxiliaryClient with the configured ``/anthropic`` base_url
  preserved (no /v1 rewrite).
- azure-foundry + chat_completions falls back to a plain OpenAI client on
  the /openai/v1 surface.
- Explicit per-task overrides (explicit_base_url, explicit_api_key,
  api_mode) win over model.* config from config.yaml.
- DeploymentNotFound 404 translation: when an Azure endpoint returns the
  characteristic "DeploymentNotFound" body, a clear WARNING is logged that
  names the missing deployment and the configured base_url.
"""
from __future__ import annotations

import logging
from typing import Any
from unittest.mock import MagicMock, patch

import pytest


@pytest.fixture(autouse=True)
def _isolate(tmp_path, monkeypatch):
    """Redirect HERMES_HOME and install a minimal config.yaml."""
    hermes_home = tmp_path / ".hermes"
    hermes_home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    (hermes_home / "config.yaml").write_text("model:\n  default: test-model\n")
    # Ensure no stale Azure env vars leak in.
    monkeypatch.delenv("AZURE_FOUNDRY_API_KEY", raising=False)
    monkeypatch.delenv("AZURE_FOUNDRY_BASE_URL", raising=False)


def _write_azure_config(tmp_path, **model_overrides: Any) -> None:
    """Persist an azure-foundry model.* block to the isolated HERMES_HOME."""
    import yaml
    cfg = {
        "model": {
            "default": "claude-opus-4-7",
            "provider": "azure-foundry",
            **model_overrides,
        },
    }
    (tmp_path / ".hermes" / "config.yaml").write_text(yaml.dump(cfg))


# ── Bug A: routing ────────────────────────────────────────────────────────


class TestAzureFoundryAnthropicMessages:
    """anthropic_messages should wrap in AnthropicAuxiliaryClient."""

    def test_anthropic_messages_wraps_in_anthropic_client(self, tmp_path):
        _write_azure_config(
            tmp_path,
            base_url="https://example.openai.azure.com/anthropic",
            api_mode="anthropic_messages",
        )
        from agent import auxiliary_client

        fake_anthropic = MagicMock(name="anthropic.Anthropic")
        fake_anthropic.base_url = "https://example.openai.azure.com/anthropic"

        with patch(
            "agent.anthropic_adapter.build_anthropic_client",
            return_value=fake_anthropic,
        ) as build_mock, patch.object(
            auxiliary_client,
            "resolve_api_key_provider_credentials",
            create=True,
            return_value={"api_key": "azure-key", "base_url": ""},
        ), patch(
            "hermes_cli.auth.resolve_api_key_provider_credentials",
            return_value={"api_key": "azure-key", "base_url": ""},
        ):
            client, model = auxiliary_client.resolve_provider_client(
                "azure-foundry", model="claude-opus-4-7",
            )

        assert isinstance(client, auxiliary_client.AnthropicAuxiliaryClient)
        # base_url preserved as-is (no /v1 rewrite)
        assert client.base_url == "https://example.openai.azure.com/anthropic"
        assert model == "claude-opus-4-7"
        # The adapter must have been invoked with the Anthropic base URL.
        build_mock.assert_called_once()
        called_args, called_kwargs = build_mock.call_args
        assert "azure-key" in called_args or called_kwargs.get("api_key") == "azure-key"
        url_arg = called_args[1] if len(called_args) > 1 else called_kwargs.get("base_url")
        assert url_arg == "https://example.openai.azure.com/anthropic"

    def test_anthropic_messages_strips_trailing_v1(self, tmp_path):
        _write_azure_config(
            tmp_path,
            base_url="https://example.openai.azure.com/anthropic/v1",
            api_mode="anthropic_messages",
        )
        from agent import auxiliary_client

        fake_anthropic = MagicMock(name="anthropic.Anthropic")
        with patch(
            "agent.anthropic_adapter.build_anthropic_client",
            return_value=fake_anthropic,
        ) as build_mock, patch(
            "hermes_cli.auth.resolve_api_key_provider_credentials",
            return_value={"api_key": "azure-key", "base_url": ""},
        ):
            client, _ = auxiliary_client.resolve_provider_client(
                "azure-foundry", model="claude-opus-4-7",
            )

        assert isinstance(client, auxiliary_client.AnthropicAuxiliaryClient)
        assert client.base_url == "https://example.openai.azure.com/anthropic"
        url_arg = build_mock.call_args.args[1] if len(build_mock.call_args.args) > 1 \
            else build_mock.call_args.kwargs.get("base_url")
        assert url_arg == "https://example.openai.azure.com/anthropic"


class TestAzureFoundryChatCompletions:
    """chat_completions Azure must remain on plain OpenAI client + /openai/v1."""

    def test_chat_completions_returns_plain_openai(self, tmp_path):
        _write_azure_config(
            tmp_path,
            base_url="https://example.openai.azure.com/openai/v1",
            api_mode="chat_completions",
        )
        from agent import auxiliary_client

        fake_openai = MagicMock(name="OpenAI")
        with patch.object(auxiliary_client, "OpenAI", return_value=fake_openai) as openai_mock, \
                patch("hermes_cli.auth.resolve_api_key_provider_credentials",
                      return_value={"api_key": "azure-key", "base_url": ""}):
            client, _ = auxiliary_client.resolve_provider_client(
                "azure-foundry", model="gpt-4o-mini",
            )

        # NOT wrapped in Anthropic client
        assert not isinstance(client, auxiliary_client.AnthropicAuxiliaryClient)
        openai_mock.assert_called_once()
        kwargs = openai_mock.call_args.kwargs
        # /openai/v1 surface preserved (it already ends in /v1 — no rewrite needed)
        assert kwargs["base_url"].rstrip("/").endswith("/openai/v1")
        assert kwargs["api_key"] == "azure-key"


class TestAzureFoundryExplicitOverrides:
    """Explicit per-task overrides win over model.* config."""

    def test_explicit_base_url_and_api_mode_override_config(self, tmp_path):
        # Config says chat_completions on /openai/v1; explicit override says
        # anthropic_messages on a different /anthropic URL.
        _write_azure_config(
            tmp_path,
            base_url="https://stale.openai.azure.com/openai/v1",
            api_mode="chat_completions",
        )
        from agent import auxiliary_client

        fake_anthropic = MagicMock(name="anthropic.Anthropic")
        with patch(
            "agent.anthropic_adapter.build_anthropic_client",
            return_value=fake_anthropic,
        ) as build_mock, patch(
            "hermes_cli.auth.resolve_api_key_provider_credentials",
            return_value={"api_key": "stale-key", "base_url": ""},
        ):
            client, _ = auxiliary_client.resolve_provider_client(
                "azure-foundry",
                model="claude-opus-4-7",
                explicit_base_url="https://override.openai.azure.com/anthropic",
                explicit_api_key="override-key",
                api_mode="anthropic_messages",
            )

        assert isinstance(client, auxiliary_client.AnthropicAuxiliaryClient)
        assert client.base_url == "https://override.openai.azure.com/anthropic"
        assert client.api_key == "override-key"
        # And the build_anthropic_client got the override key
        args = build_mock.call_args.args
        assert "override-key" in args


# ── Bug B: DeploymentNotFound translation ─────────────────────────────────


class _FakeAzureError(Exception):
    """Approximation of the anthropic / openai NotFoundError shape."""

    def __init__(self, status_code: int, body: dict):
        super().__init__(body.get("error", {}).get("message", "error"))
        self.status_code = status_code
        self.body = body
        self.message = body.get("error", {}).get("message", "")


class TestAzureDeploymentNotFoundTranslation:
    def test_404_with_deployment_not_found_logs_warning(self, caplog):
        from agent.auxiliary_client import _translate_azure_deployment_error

        body = {
            "error": {
                "code": "DeploymentNotFound",
                "message": (
                    "The API deployment for this resource does not exist. "
                    "If you created the deployment within the last 5 minutes, "
                    "please wait a moment and try again."
                ),
            }
        }
        exc = _FakeAzureError(404, body)

        def boom():
            raise exc

        caplog.set_level(logging.WARNING, logger="agent.auxiliary_client")
        with pytest.raises(_FakeAzureError):
            _translate_azure_deployment_error(
                boom,
                base_url="https://example.openai.azure.com/anthropic",
                model="claude-haiku-4-5",
            )

        assert any(
            "claude-haiku-4-5" in rec.getMessage()
            and "example.openai.azure.com" in rec.getMessage()
            for rec in caplog.records
        ), [r.getMessage() for r in caplog.records]

    def test_non_azure_endpoint_does_not_trigger(self, caplog):
        from agent.auxiliary_client import _translate_azure_deployment_error

        exc = _FakeAzureError(404, {"error": {"code": "DeploymentNotFound",
                                              "message": "DeploymentNotFound"}})

        def boom():
            raise exc

        caplog.set_level(logging.WARNING, logger="agent.auxiliary_client")
        with pytest.raises(_FakeAzureError):
            _translate_azure_deployment_error(
                boom,
                base_url="https://api.anthropic.com",
                model="claude-opus-4-7",
            )
        # No Azure-translation warning should have fired.
        assert not any("Azure Foundry: deployment" in r.getMessage()
                       for r in caplog.records)

    def test_non_404_passes_through(self, caplog):
        from agent.auxiliary_client import _translate_azure_deployment_error

        exc = _FakeAzureError(500, {"error": {"code": "ServerError",
                                              "message": "boom"}})

        def boom():
            raise exc

        caplog.set_level(logging.WARNING, logger="agent.auxiliary_client")
        with pytest.raises(_FakeAzureError):
            _translate_azure_deployment_error(
                boom,
                base_url="https://example.openai.azure.com/anthropic",
                model="claude-haiku-4-5",
            )
        assert not any("Azure Foundry: deployment" in r.getMessage()
                       for r in caplog.records)

    def test_400_with_deployment_not_found_text_passes_through(self, caplog):
        from agent.auxiliary_client import _translate_azure_deployment_error

        exc = _FakeAzureError(400, {
            "error": {
                "code": "BadRequest",
                "message": (
                    "Invalid request body while checking DeploymentNotFound: "
                    "the api deployment for this resource does not exist"
                ),
            }
        })

        def boom():
            raise exc

        caplog.set_level(logging.WARNING, logger="agent.auxiliary_client")
        with pytest.raises(_FakeAzureError):
            _translate_azure_deployment_error(
                boom,
                base_url="https://example.openai.azure.com/anthropic",
                model="claude-haiku-4-5",
            )
        assert not any("Azure Foundry: deployment" in r.getMessage()
                       for r in caplog.records)

    def test_success_returns_value(self):
        from agent.auxiliary_client import _translate_azure_deployment_error

        sentinel = object()
        result = _translate_azure_deployment_error(
            lambda: sentinel,
            base_url="https://example.openai.azure.com/anthropic",
            model="claude-opus-4-7",
        )
        assert result is sentinel
