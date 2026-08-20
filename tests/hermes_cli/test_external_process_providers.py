"""Tests for the generic external_process provider path.

copilot-acp was the only external_process provider for a while and its command
resolution was hardcoded in auth.py. These tests pin the generic behaviour: any
provider declaring auth_type="external_process" resolves its command, args and
base URL from its own ProviderConfig.
"""

import pytest

from hermes_cli.auth import (
    PROVIDER_REGISTRY,
    AuthError,
    ProviderConfig,
    get_external_process_provider_status,
    is_external_process_provider,
    resolve_external_process_provider_credentials,
)


@pytest.fixture
def acme_provider(monkeypatch):
    """Register a throwaway external-process provider for the duration of a test."""
    pconfig = ProviderConfig(
        id="acme-cli",
        name="Acme CLI",
        auth_type="external_process",
        inference_base_url="acme://local",
        base_url_env_var="ACME_BASE_URL",
        command_env_vars=("HERMES_ACME_COMMAND", "ACME_CLI_PATH"),
        default_command="acme",
        default_args=("--serve", "--stdio"),
        args_env_var="HERMES_ACME_ARGS",
        remote_base_url_prefixes=("acme+tcp://",),
    )
    monkeypatch.setitem(PROVIDER_REGISTRY, "acme-cli", pconfig)
    for env_var in (
        "HERMES_ACME_COMMAND",
        "ACME_CLI_PATH",
        "HERMES_ACME_ARGS",
        "ACME_BASE_URL",
    ):
        monkeypatch.delenv(env_var, raising=False)
    return pconfig


class TestIsExternalProcessProvider:
    def test_copilot_acp_is_external_process(self):
        assert is_external_process_provider("copilot-acp") is True

    def test_api_key_provider_is_not(self):
        assert is_external_process_provider("copilot") is False

    def test_unknown_provider_is_not(self):
        assert is_external_process_provider("does-not-exist") is False


class TestGenericCommandResolution:
    def test_defaults_come_from_provider_config(self, acme_provider, monkeypatch):
        monkeypatch.setattr("hermes_cli.auth.shutil.which", lambda command: f"/usr/bin/{command}")

        creds = resolve_external_process_provider_credentials("acme-cli")

        assert creds["command"] == "/usr/bin/acme"
        assert creds["args"] == ["--serve", "--stdio"]
        assert creds["base_url"] == "acme://local"
        assert creds["api_key"] == "acme-cli"
        assert creds["source"] == "process"

    def test_command_env_vars_win_in_priority_order(self, acme_provider, monkeypatch):
        monkeypatch.setattr("hermes_cli.auth.shutil.which", lambda command: f"/usr/bin/{command}")
        monkeypatch.setenv("ACME_CLI_PATH", "acme-fallback")
        monkeypatch.setenv("HERMES_ACME_COMMAND", "acme-preferred")

        creds = resolve_external_process_provider_credentials("acme-cli")

        assert creds["command"] == "/usr/bin/acme-preferred"

    def test_args_env_var_overrides_defaults(self, acme_provider, monkeypatch):
        monkeypatch.setattr("hermes_cli.auth.shutil.which", lambda command: f"/usr/bin/{command}")
        monkeypatch.setenv("HERMES_ACME_ARGS", "--rpc --port 9000")

        creds = resolve_external_process_provider_credentials("acme-cli")

        assert creds["args"] == ["--rpc", "--port", "9000"]

    def test_missing_command_raises_with_provider_specific_hint(self, acme_provider, monkeypatch):
        monkeypatch.setattr("hermes_cli.auth.shutil.which", lambda command: None)

        with pytest.raises(AuthError) as excinfo:
            resolve_external_process_provider_credentials("acme-cli")

        message = str(excinfo.value)
        assert "Acme CLI" in message
        assert "HERMES_ACME_COMMAND" in message
        assert "copilot" not in message.lower()

    def test_remote_base_url_does_not_require_local_command(self, acme_provider, monkeypatch):
        monkeypatch.setattr("hermes_cli.auth.shutil.which", lambda command: None)
        monkeypatch.setenv("ACME_BASE_URL", "acme+tcp://127.0.0.1:9000")

        creds = resolve_external_process_provider_credentials("acme-cli")

        assert creds["base_url"] == "acme+tcp://127.0.0.1:9000"
        assert creds["command"] == "acme"

    def test_status_reports_configured_when_command_found(self, acme_provider, monkeypatch):
        monkeypatch.setattr("hermes_cli.auth.shutil.which", lambda command: f"/usr/bin/{command}")

        status = get_external_process_provider_status("acme-cli")

        assert status["configured"] is True
        assert status["logged_in"] is True
        assert status["name"] == "Acme CLI"
        assert status["args"] == ["--serve", "--stdio"]

    def test_status_reports_unconfigured_when_command_missing(self, acme_provider, monkeypatch):
        monkeypatch.setattr("hermes_cli.auth.shutil.which", lambda command: None)

        status = get_external_process_provider_status("acme-cli")

        assert status["configured"] is False
        assert status["logged_in"] is False


class TestCopilotAcpBehaviourUnchanged:
    """copilot-acp must resolve exactly as it did before the generalisation."""

    @pytest.fixture(autouse=True)
    def _clean_env(self, monkeypatch):
        for env_var in (
            "HERMES_COPILOT_ACP_COMMAND",
            "COPILOT_CLI_PATH",
            "HERMES_COPILOT_ACP_ARGS",
            "COPILOT_ACP_BASE_URL",
        ):
            monkeypatch.delenv(env_var, raising=False)

    def test_defaults(self, monkeypatch):
        monkeypatch.setattr("hermes_cli.auth.shutil.which", lambda command: f"/usr/bin/{command}")

        creds = resolve_external_process_provider_credentials("copilot-acp")

        assert creds["command"] == "/usr/bin/copilot"
        assert creds["args"] == ["--acp", "--stdio"]
        assert creds["base_url"] == "acp://copilot"
        assert creds["api_key"] == "copilot-acp"

    def test_legacy_env_vars_still_honoured(self, monkeypatch):
        monkeypatch.setattr("hermes_cli.auth.shutil.which", lambda command: f"/usr/bin/{command}")
        monkeypatch.setenv("COPILOT_CLI_PATH", "copilot-nightly")

        creds = resolve_external_process_provider_credentials("copilot-acp")

        assert creds["command"] == "/usr/bin/copilot-nightly"

    def test_acp_tcp_base_url_skips_command_lookup(self, monkeypatch):
        monkeypatch.setattr("hermes_cli.auth.shutil.which", lambda command: None)
        monkeypatch.setenv("COPILOT_ACP_BASE_URL", "acp+tcp://127.0.0.1:4000")

        creds = resolve_external_process_provider_credentials("copilot-acp")

        assert creds["base_url"] == "acp+tcp://127.0.0.1:4000"

    def test_non_external_process_provider_rejected(self):
        with pytest.raises(AuthError):
            resolve_external_process_provider_credentials("copilot")
