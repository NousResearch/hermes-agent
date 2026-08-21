"""Provider + alias resolution for cursor-acp / acp://cursor."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from agent.auxiliary_client import _normalize_aux_provider
from hermes_cli.auth import (
    PROVIDER_REGISTRY,
    resolve_external_process_provider_credentials,
    resolve_provider,
)


class TestCursorACPAliases:
    def test_normalize_aux_provider_aliases(self) -> None:
        assert _normalize_aux_provider("cursor-acp") == "cursor-acp"
        assert _normalize_aux_provider("cursor-acp-agent") == "cursor-acp"
        assert _normalize_aux_provider("cursor-agent-acp") == "cursor-acp"

    def test_resolve_provider_aliases(self) -> None:
        assert resolve_provider("cursor-acp") == "cursor-acp"
        assert resolve_provider("cursor-acp-agent") == "cursor-acp"
        assert resolve_provider("cursor-agent-acp") == "cursor-acp"

    def test_registry_entry(self) -> None:
        assert "cursor-acp" in PROVIDER_REGISTRY
        pconfig = PROVIDER_REGISTRY["cursor-acp"]
        assert pconfig.auth_type == "external_process"
        assert pconfig.inference_base_url == "acp://cursor"
        assert pconfig.id == "cursor-acp"


class TestCursorACPRuntimeResolution:
    def test_resolve_runtime_provider(self, monkeypatch) -> None:
        monkeypatch.setattr(
            "hermes_cli.auth.shutil.which",
            lambda command: f"/usr/local/bin/{command}",
        )
        from hermes_cli.runtime_provider import resolve_runtime_provider

        result = resolve_runtime_provider(requested="cursor-acp")
        assert result["provider"] == "cursor-acp"
        assert result["api_mode"] == "chat_completions"
        assert result["api_key"] == "cursor-acp"
        assert result["base_url"] == "acp://cursor"
        assert result["command"] == "/usr/local/bin/cursor-agent"
        assert result["args"] == ["acp"]

    def test_missing_cli_is_actionable(self, monkeypatch) -> None:
        monkeypatch.setattr("hermes_cli.auth.shutil.which", lambda command: None)
        with pytest.raises(Exception, match="cursor-agent"):
            resolve_external_process_provider_credentials("cursor-acp")


class TestCursorACPClientRouting:
    def test_runtime_helper_builds_cursor_client(self) -> None:
        from agent.agent_runtime_helpers import create_openai_client

        agent = SimpleNamespace(
            provider="cursor-acp",
            _client_log_context=lambda: "",
        )
        with (
            patch("agent.cursor_acp_client.CursorACPClient") as mock_cls,
            patch("agent.agent_runtime_helpers._ra") as mock_ra,
            patch("agent.auxiliary_client._validate_proxy_env_urls"),
            patch("agent.auxiliary_client._validate_base_url"),
            patch("agent.ssl_verify.resolve_httpx_verify", return_value=True),
        ):
            mock_ra.return_value = SimpleNamespace(
                logger=SimpleNamespace(info=lambda *a, **k: None)
            )
            fake = MagicMock()
            mock_cls.return_value = fake
            client = create_openai_client(
                agent,
                {"api_key": "cursor-acp", "base_url": "acp://cursor"},
                reason="test",
                shared=False,
            )
        assert client is fake
        mock_cls.assert_called_once()

    def test_base_url_sentinel_routes_without_provider_name(self) -> None:
        from agent.agent_runtime_helpers import create_openai_client

        agent = SimpleNamespace(
            provider="custom",
            _client_log_context=lambda: "",
        )
        with (
            patch("agent.cursor_acp_client.CursorACPClient") as mock_cls,
            patch("agent.agent_runtime_helpers._ra") as mock_ra,
            patch("agent.auxiliary_client._validate_proxy_env_urls"),
            patch("agent.auxiliary_client._validate_base_url"),
            patch("agent.ssl_verify.resolve_httpx_verify", return_value=True),
            patch("openai.OpenAI") as mock_openai,
        ):
            mock_ra.return_value = SimpleNamespace(
                logger=SimpleNamespace(info=lambda *a, **k: None)
            )
            fake = MagicMock()
            mock_cls.return_value = fake
            client = create_openai_client(
                agent,
                {"api_key": "cursor-acp", "base_url": "acp://cursor"},
                reason="test",
                shared=False,
            )
        assert client is fake
        mock_openai.assert_not_called()
