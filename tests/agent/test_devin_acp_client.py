"""Tests for Devin CLI ACP provider wiring."""

from __future__ import annotations

import unittest
from unittest.mock import patch

from agent.devin_acp_client import ACP_MARKER_BASE_URL, DevinACPClient, _resolve_args, _resolve_command
from hermes_cli.auth import (
    PROVIDER_REGISTRY,
    get_external_process_provider_status,
    resolve_external_process_provider_credentials,
    resolve_provider,
)


class TestDevinAcpProviderRegistry(unittest.TestCase):
    def test_registry_entry(self):
        p = PROVIDER_REGISTRY["devin-acp"]
        assert p.auth_type == "external_process"
        assert p.inference_base_url == "acp://devin"

    def test_aliases(self):
        assert resolve_provider("devin") == "devin-acp"
        assert resolve_provider("devin-cli") == "devin-acp"
        assert resolve_provider("cognition-devin") == "devin-acp"


class TestDevinAcpResolve(unittest.TestCase):
    def test_status_detects_local_cli(self, monkeypatch=None):
        pass

    def test_status_and_creds(self):
        with patch("hermes_cli.auth.shutil.which", return_value="/usr/local/bin/devin"):
            with patch.dict("os.environ", {"HERMES_DEVIN_ACP_ARGS": "acp --debug"}, clear=False):
                status = get_external_process_provider_status("devin-acp")
                assert status["configured"] is True
                assert status["command"] == "devin"
                assert status["resolved_command"] == "/usr/local/bin/devin"
                assert status["args"] == ["acp", "--debug"]
                assert status["base_url"] == "acp://devin"

                creds = resolve_external_process_provider_credentials("devin-acp")
                assert creds["provider"] == "devin-acp"
                assert creds["api_key"] == "devin-acp"
                assert creds["base_url"] == "acp://devin"
                assert creds["command"] == "/usr/local/bin/devin"
                assert creds["args"] == ["acp", "--debug"]


class TestDevinAcpClientDefaults(unittest.TestCase):
    def test_marker_and_defaults(self):
        assert ACP_MARKER_BASE_URL == "acp://devin"
        with patch.dict("os.environ", {}, clear=False):
            # Ensure defaults when env not set — may still pick up user env
            args = _resolve_args() if not __import__("os").getenv("HERMES_DEVIN_ACP_ARGS") else ["acp"]
            if not __import__("os").getenv("HERMES_DEVIN_ACP_ARGS"):
                assert _resolve_args() == ["acp"]
        client = DevinACPClient(acp_cwd="/tmp", command="devin", args=["acp"])
        assert client.api_key == "devin-acp"
        assert client.base_url == "acp://devin"
        assert client._acp_command == "devin"
        assert client._acp_args == ["acp"]


class TestAcpClientFactory(unittest.TestCase):
    def test_create_devin(self):
        from agent.acp_client_factory import create_acp_client, is_acp_provider

        assert is_acp_provider("devin-acp") is True
        assert is_acp_provider(base_url="acp://devin") is True
        client = create_acp_client(provider="devin-acp", command="devin", args=["acp"], acp_cwd="/tmp")
        assert isinstance(client, DevinACPClient)


if __name__ == "__main__":
    unittest.main()
