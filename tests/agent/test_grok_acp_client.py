"""Tests for Grok CLI ACP provider wiring."""

from __future__ import annotations

import unittest
from unittest.mock import patch

from agent.grok_acp_client import (
    ACP_MARKER_BASE_URL,
    GrokACPClient,
    _resolve_args,
    _resolve_command,
    resolve_grok_acp_model_value,
)
from agent.acp_client_factory import is_acp_provider, create_acp_client
from hermes_cli.auth import (
    PROVIDER_REGISTRY,
    get_external_process_provider_status,
    resolve_external_process_provider_credentials,
    resolve_provider,
)


class TestGrokAcpProviderRegistry(unittest.TestCase):
    def test_registry_entry(self):
        p = PROVIDER_REGISTRY["grok-acp"]
        assert p.auth_type == "external_process"
        assert p.inference_base_url == "acp://grok"

    def test_aliases(self):
        assert resolve_provider("grok") == "xai"
        assert resolve_provider("grok-cli") == "grok-acp"
        assert resolve_provider("grok-build") == "grok-acp"
        assert resolve_provider("xai-grok-cli") == "grok-acp"

    def test_is_acp_provider(self):
        assert is_acp_provider("grok-acp")
        assert is_acp_provider(base_url="acp://grok")


class TestGrokAcpResolve(unittest.TestCase):
    def test_status_and_creds(self):
        with patch("hermes_cli.auth.shutil.which", return_value="/usr/local/bin/grok"):
            with patch(
                "hermes_cli.auth._grok_local_credentials_present",
                return_value=True,
            ):
                with patch.dict(
                    "os.environ",
                    {"HERMES_GROK_ACP_ARGS": "--no-auto-update agent stdio"},
                    clear=False,
                ):
                    status = get_external_process_provider_status("grok-acp")
                    assert status["configured"] is True
                    assert status["cli_installed"] is True
                    assert status["auth_present"] is True
                    assert status["logged_in"] is True
                    assert status["command"] == "grok"
                    assert status["resolved_command"] == "/usr/local/bin/grok"
                    assert status["args"] == ["--no-auto-update", "agent", "stdio"]
                    assert status["base_url"] == "acp://grok"

                    creds = resolve_external_process_provider_credentials("grok-acp")
                    assert creds["provider"] == "grok-acp"
                    assert creds["api_key"] == "grok-acp"
                    assert creds["base_url"] == "acp://grok"
                    assert creds["command"] == "/usr/local/bin/grok"
                    assert creds["args"] == ["--no-auto-update", "agent", "stdio"]

    def test_status_cli_without_credentials_is_not_logged_in(self):
        with patch("hermes_cli.auth.shutil.which", return_value="/usr/local/bin/grok"):
            with patch(
                "hermes_cli.auth._grok_local_credentials_present",
                return_value=False,
            ):
                status = get_external_process_provider_status("grok-acp")
        assert status["configured"] is True
        assert status["cli_installed"] is True
        assert status["auth_present"] is False
        assert status["logged_in"] is False
        assert status.get("hint")
        assert "grok login" in status["hint"]


class TestGrokAcpClientDefaults(unittest.TestCase):
    def test_marker_and_defaults(self):
        assert ACP_MARKER_BASE_URL == "acp://grok"
        with patch.dict("os.environ", {"HERMES_GROK_ACP_ARGS": ""}, clear=False):
            assert _resolve_args() == ["--no-auto-update", "agent", "stdio"]
        client = GrokACPClient(
            acp_cwd="/tmp",
            command="grok",
            args=["--no-auto-update", "agent", "stdio"],
        )
        assert client.api_key == "grok-acp"
        assert client.base_url == "acp://grok"
        assert client._acp_command == "grok"
        assert client._acp_args == ["--no-auto-update", "agent", "stdio"]

    def test_backend_model_id_maps_placeholders_to_none(self):
        from agent.grok_acp_client import _backend_model_id

        assert _backend_model_id(None) is None
        assert _backend_model_id("") is None
        assert _backend_model_id("grok-acp") is None
        assert _backend_model_id("grok-cli") is None
        assert _backend_model_id("grok-build") is None
        assert _backend_model_id("grok-4.5") == "grok-4.5"

    def test_resolve_grok_acp_model_value_maps_cli_to_acp_ids(self):
        available = [
            {"modelId": "grok-4.5", "name": "Grok 4.5"},
            {"modelId": "grok-composer-2.5-fast", "name": "Grok Composer 2.5 Fast"},
            {"modelId": "grok-build-0.1", "name": "Grok Build 0.1"},
        ]
        assert resolve_grok_acp_model_value("grok-4.5", available) == "grok-4.5"
        assert (
            resolve_grok_acp_model_value("grok-composer-2.5-fast", available)
            == "grok-composer-2.5-fast"
        )
        assert resolve_grok_acp_model_value("grok-4-5", available) == "grok-4.5"
        assert resolve_grok_acp_model_value("grok-build-0.1", available) == "grok-build-0.1"
        assert resolve_grok_acp_model_value("grok-acp", available) is None

    def test_create_acp_client_factory_returns_grok_client(self):
        client = create_acp_client(provider="grok-acp")
        assert isinstance(client, GrokACPClient)


if __name__ == "__main__":
    unittest.main()
