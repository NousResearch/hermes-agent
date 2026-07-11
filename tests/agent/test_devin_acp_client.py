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
            with patch(
                "hermes_cli.auth._devin_local_credentials_present",
                return_value=True,
            ):
                with patch.dict("os.environ", {"HERMES_DEVIN_ACP_ARGS": "acp --debug"}, clear=False):
                    status = get_external_process_provider_status("devin-acp")
                    assert status["configured"] is True
                    assert status["cli_installed"] is True
                    assert status["auth_present"] is True
                    assert status["logged_in"] is True
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

    def test_status_cli_without_credentials_is_not_logged_in(self):
        with patch("hermes_cli.auth.shutil.which", return_value="/usr/local/bin/devin"):
            with patch(
                "hermes_cli.auth._devin_local_credentials_present",
                return_value=False,
            ):
                status = get_external_process_provider_status("devin-acp")
        assert status["configured"] is True
        assert status["cli_installed"] is True
        assert status["auth_present"] is False
        assert status["logged_in"] is False
        assert status.get("hint")
        assert "devin auth login" in status["hint"]


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
        from agent.acp_client_factory import ACP_PROVIDERS, create_acp_client, is_acp_provider

        assert "devin-acp" in ACP_PROVIDERS
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

    def test_create_devin_fills_missing_command_from_resolver(self):
        from agent.acp_client_factory import create_acp_client

        with patch(
            "hermes_cli.auth.resolve_external_process_provider_credentials",
            return_value={
                "command": "/resolved/devin",
                "args": ["acp", "--from-resolver"],
            },
        ):
            client = create_acp_client(provider="devin-acp", acp_cwd="/tmp")
        assert isinstance(client, DevinACPClient)
        assert client._acp_command == "/resolved/devin"
        assert client._acp_args == ["acp", "--from-resolver"]

    def test_devin_credentials_probe_reads_marker_not_secret(self, tmp_path=None):
        import tempfile
        from pathlib import Path

        from hermes_cli.auth import _devin_local_credentials_present

        with tempfile.TemporaryDirectory() as td:
            cred_dir = Path(td) / "devin"
            cred_dir.mkdir()
            cred_file = cred_dir / "credentials.toml"
            cred_file.write_text(
                'windsurf_api_key = "devin-session-token$secret-value"\n',
                encoding="utf-8",
            )
            with patch.dict("os.environ", {"APPDATA": td, "XDG_CONFIG_HOME": td}, clear=False):
                # Point home-based candidates away from the real user home by
                # still using APPDATA/XDG which our probe checks first.
                assert _devin_local_credentials_present() is True

            missing = Path(td) / "empty"
            missing.mkdir()
            with patch.dict(
                "os.environ",
                {"APPDATA": str(missing), "XDG_CONFIG_HOME": str(missing)},
                clear=False,
            ):
                with patch("pathlib.Path.home", return_value=missing):
                    assert _devin_local_credentials_present() is False


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
