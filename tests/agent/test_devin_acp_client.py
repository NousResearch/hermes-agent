"""Tests for Devin CLI ACP provider wiring."""

from __future__ import annotations

import unittest
from unittest.mock import patch

from agent.devin_acp_client import ACP_MARKER_BASE_URL, DevinACPClient, _resolve_args
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
        with patch.dict("os.environ", {"HERMES_DEVIN_ACP_ARGS": ""}, clear=False):
            # Force empty env so default path is exercised even if the host
            # shell exported HERMES_DEVIN_ACP_ARGS.
            assert _resolve_args() == ["acp"]
        client = DevinACPClient(acp_cwd="/tmp", command="devin", args=["acp"])
        assert client.api_key == "devin-acp"
        assert client.base_url == "acp://devin"
        assert client._acp_command == "devin"
        assert client._acp_args == ["acp"]

    def test_empty_args_does_not_fall_through_to_copilot_defaults(self):
        """Regression: args=[] used to become ['--acp', '--stdio'] via parent."""
        with patch.dict(
            "os.environ",
            {
                "HERMES_DEVIN_ACP_ARGS": "",
                "HERMES_COPILOT_ACP_ARGS": "",
            },
            clear=False,
        ):
            via_args = DevinACPClient(acp_cwd="/tmp", command="devin", args=[])
            via_acp_args = DevinACPClient(acp_cwd="/tmp", command="devin", acp_args=[])
            via_none = DevinACPClient(acp_cwd="/tmp", command="devin")

        for client in (via_args, via_acp_args, via_none):
            assert client._acp_args == ["acp"], client._acp_args
            assert "--acp" not in client._acp_args
            assert "--stdio" not in client._acp_args

    def test_display_name_and_install_hint_are_devin(self):
        client = DevinACPClient(acp_cwd="/tmp", command="devin", args=["acp"])
        assert client._acp_display_name == "Devin ACP"
        assert "Devin CLI" in client._install_hint
        assert "Copilot" not in client._install_hint


class TestAcpClientFactory(unittest.TestCase):
    def test_create_devin(self):
        from agent.acp_client_factory import create_acp_client, is_acp_provider

        assert is_acp_provider("devin-acp") is True
        assert is_acp_provider(base_url="acp://devin") is True
        client = create_acp_client(
            provider="devin-acp", command="devin", args=["acp"], acp_cwd="/tmp"
        )
        assert isinstance(client, DevinACPClient)

    def test_create_devin_empty_args_uses_devin_defaults(self):
        from agent.acp_client_factory import create_acp_client

        with patch.dict("os.environ", {"HERMES_DEVIN_ACP_ARGS": ""}, clear=False):
            client = create_acp_client(
                provider="devin-acp", command="devin", args=[], acp_cwd="/tmp"
            )
        assert isinstance(client, DevinACPClient)
        assert client._acp_args == ["acp"]


class TestOneshotAcpWiring(unittest.TestCase):
    def test_oneshot_forwards_runtime_command_and_args(self):
        """hermes -z must pass ACP command/args into AIAgent (P0 regression)."""
        import hermes_cli.oneshot as oneshot_mod

        captured: dict = {}

        class _FakeAgent:
            def __init__(self, **kwargs):
                captured.update(kwargs)
                self.suppress_status_output = False
                self.stream_delta_callback = None
                self.tool_gen_callback = None

            def run_conversation(self, prompt):
                return {
                    "final_response": "pong",
                    "messages": [{"role": "user", "content": prompt}],
                }

        with (
            patch.object(
                oneshot_mod,
                "resolve_runtime_provider",
                create=True,
            ),
            patch("hermes_cli.oneshot.load_config", create=True),
            patch("hermes_cli.config.load_config", return_value={"model": {}}),
            patch(
                "hermes_cli.runtime_provider.resolve_runtime_provider",
                return_value={
                    "provider": "devin-acp",
                    "api_mode": "chat_completions",
                    "base_url": "acp://devin",
                    "api_key": "devin-acp",
                    "command": "/usr/bin/devin",
                    "args": ["acp"],
                    "credential_pool": None,
                },
            ),
            patch("hermes_cli.oneshot.get_fallback_chain", return_value=None),
            patch("hermes_cli.oneshot._create_session_db_for_oneshot", return_value=None),
            patch("hermes_cli.tools_config._get_platform_tools", return_value=set()),
            patch("run_agent.AIAgent", _FakeAgent),
        ):
            response, result = oneshot_mod._run_agent(
                "ping",
                model="devin-acp",
                provider="devin-acp",
                use_config_toolsets=False,
            )

        assert response == "pong"
        assert captured.get("acp_command") == "/usr/bin/devin"
        assert captured.get("acp_args") == ["acp"]
        assert captured.get("provider") == "devin-acp"
        assert captured.get("base_url") == "acp://devin"


if __name__ == "__main__":
    unittest.main()
