"""Regression tests for hermes secrets bitwarden setup non-TTY guard.

Issue #40274: cmd_setup() crashes with EOFError when stdin is not a TTY
because getpass.getpass() and console.input() require an interactive terminal.
"""
from __future__ import annotations

import argparse
from unittest.mock import patch

import pytest

from hermes_cli.secret_prompt import (
    capture_pre_dotenv_rotation_inputs,
    reset_pre_dotenv_rotation_inputs,
)


@pytest.fixture(autouse=True)
def _reset_pre_dotenv_rotation_inputs():
    reset_pre_dotenv_rotation_inputs()
    yield
    reset_pre_dotenv_rotation_inputs()


class TestCmdSetupNonTtyGuard:
    """cmd_setup should fail early with a clear error in non-TTY environments."""

    @staticmethod
    def _make_args(**overrides):
        ns = argparse.Namespace(
            access_token=overrides.get("access_token", ""),
            server_url=overrides.get("server_url", ""),
            project_id=overrides.get("project_id", ""),
        )
        return ns


    def test_missing_access_token_only(self, monkeypatch, capsys):
        """Non-TTY with server-url and project-id but no token → reports --access-token."""
        monkeypatch.setattr("sys.stdin.isatty", lambda: False)
        monkeypatch.setattr(
            "hermes_cli.secrets_cli.bw.find_bws", lambda install_if_missing=False: "/usr/bin/bws"
        )
        monkeypatch.setattr(
            "hermes_cli.secrets_cli._bws_version", lambda _: "2.0.0"
        )

        from hermes_cli.secrets_cli import cmd_setup

        result = cmd_setup(self._make_args(
            server_url="https://vault.bitwarden.com",
            project_id="aaaa-bbbb",
        ))
        assert result == 1
        captured = capsys.readouterr()
        # The "Missing:" line should list --access-token only
        assert "Missing:" in captured.out
        assert "--access-token" in captured.out
        # The usage example contains --server-url and --project-id, so check
        # the missing line specifically: it should NOT list them as missing
        missing_line = [l for l in captured.out.split("\n") if "Missing:" in l][0]
        assert "--access-token" in missing_line
        assert "--server-url" not in missing_line
        assert "--project-id" not in missing_line

    def test_missing_server_url_with_env_var_passes(self, monkeypatch, capsys):
        """Non-TTY with BWS_SERVER_URL env set → server-url not required."""
        monkeypatch.setattr("sys.stdin.isatty", lambda: False)
        monkeypatch.setenv("BWS_SERVER_URL", "https://vault.bitwarden.com")
        monkeypatch.setattr(
            "hermes_cli.secrets_cli.bw.find_bws", lambda install_if_missing=False: "/usr/bin/bws"
        )
        monkeypatch.setattr(
            "hermes_cli.secrets_cli._bws_version", lambda _: "2.0.0"
        )
        monkeypatch.setattr("hermes_cli.secrets_cli.load_config", lambda: {})
        monkeypatch.setattr("hermes_cli.secrets_cli.save_env_value", lambda *a: None)
        monkeypatch.setattr("hermes_cli.secrets_cli.get_env_path", lambda: "/tmp/.env")
        monkeypatch.setattr(
            "hermes_cli.secrets_cli.bw.fetch_bitwarden_secrets",
            lambda **kw: ({"KEY": "val"}, []),
        )

        from hermes_cli.secrets_cli import cmd_setup

        result = cmd_setup(self._make_args(
            access_token="0.valid-token",
            project_id="aaaa-bbbb",
        ))
        assert result == 0
        output = capsys.readouterr().out
        assert "process listings" in output
        assert "0.valid-token" not in output

    def test_malformed_secrets_config_is_normalized(self, monkeypatch):
        monkeypatch.setattr("sys.stdin.isatty", lambda: False)
        monkeypatch.setenv("BWS_SERVER_URL", "https://vault.bitwarden.com")
        monkeypatch.setattr(
            "hermes_cli.secrets_cli.bw.find_bws",
            lambda install_if_missing=False: "/usr/bin/bws",
        )
        monkeypatch.setattr(
            "hermes_cli.secrets_cli._bws_version", lambda _: "2.0.0"
        )
        monkeypatch.setattr(
            "hermes_cli.secrets_cli.load_config",
            lambda: {"secrets": None},
        )
        monkeypatch.setattr("hermes_cli.secrets_cli.save_config", lambda _: None)
        monkeypatch.setattr("hermes_cli.secrets_cli.save_env_value", lambda *a: None)
        monkeypatch.setattr("hermes_cli.secrets_cli.get_env_path", lambda: "/tmp/.env")
        monkeypatch.setattr(
            "hermes_cli.secrets_cli.bw.fetch_bitwarden_secrets",
            lambda **kwargs: ({"KEY": "value"}, []),
        )

        from hermes_cli.secrets_cli import cmd_setup

        assert cmd_setup(
            self._make_args(
                access_token="0.valid-token",
                project_id="aaaa-bbbb",
            )
        ) == 0

    @pytest.mark.parametrize("bad_name", ["BAD NAME", "1BAD", "BAD=INJECT"])
    def test_malformed_token_env_uses_default(self, monkeypatch, bad_name):
        monkeypatch.setattr("sys.stdin.isatty", lambda: False)
        monkeypatch.setenv("BWS_SERVER_URL", "https://vault.bitwarden.com")
        monkeypatch.setattr(
            "hermes_cli.secrets_cli.bw.find_bws",
            lambda install_if_missing=False: "/usr/bin/bws",
        )
        monkeypatch.setattr("hermes_cli.secrets_cli._bws_version", lambda _: "2.0.0")
        monkeypatch.setattr(
            "hermes_cli.secrets_cli.load_config",
            lambda: {
                "secrets": {"bitwarden": {"access_token_env": bad_name}}
            },
        )
        monkeypatch.setattr("hermes_cli.secrets_cli.save_config", lambda _: None)
        monkeypatch.setattr("hermes_cli.secrets_cli.save_env_value", lambda *a: None)
        monkeypatch.setattr("hermes_cli.secrets_cli.get_env_path", lambda: "/tmp/.env")
        monkeypatch.setattr(
            "hermes_cli.secrets_cli.bw.fetch_bitwarden_secrets",
            lambda **kwargs: ({"KEY": "value"}, []),
        )

        from hermes_cli.secrets_cli import cmd_setup

        assert cmd_setup(
            self._make_args(access_token="0.valid-token", project_id="aaaa-bbbb")
        ) == 0

    def test_non_tty_accepts_secret_provider_token_env(self, monkeypatch, capsys):
        monkeypatch.setattr("sys.stdin.isatty", lambda: False)
        monkeypatch.setenv("BWS_ACCESS_TOKEN", "0.from-secret-provider")
        monkeypatch.setattr(
            "hermes_cli.secrets_cli.bw.find_bws", lambda install_if_missing=False: "/usr/bin/bws"
        )
        monkeypatch.setattr(
            "hermes_cli.secrets_cli._bws_version", lambda _: "2.0.0"
        )
        monkeypatch.setattr("hermes_cli.secrets_cli.load_config", lambda: {})
        monkeypatch.setattr("hermes_cli.secrets_cli.save_config", lambda _: None)
        monkeypatch.setattr(
            "hermes_cli.secrets_cli.save_env_value", lambda *a: None
        )
        monkeypatch.setattr("hermes_cli.secrets_cli.get_env_path", lambda: "/tmp/.env")
        captured = {}

        def fake_fetch(**kwargs):
            captured.update(kwargs)
            return {"KEY": "val"}, []

        monkeypatch.setattr(
            "hermes_cli.secrets_cli.bw.fetch_bitwarden_secrets", fake_fetch
        )

        from hermes_cli.secrets_cli import cmd_setup

        result = cmd_setup(self._make_args(
            server_url="https://vault.bitwarden.com",
            project_id="aaaa-bbbb",
        ))

        assert result == 0
        assert captured["access_token"] == "0.from-secret-provider"
        output = capsys.readouterr().out
        assert "0.from-secret-provider" not in output

    def test_non_tty_setup_prefers_injected_token_over_stale_dotenv(
        self, monkeypatch
    ):
        monkeypatch.setattr("sys.stdin.isatty", lambda: False)
        monkeypatch.setenv("BWS_ACCESS_TOKEN", "0.injected-before-dotenv")
        capture_pre_dotenv_rotation_inputs(
            ["hermes", "secrets", "bitwarden", "setup"],
            config={"secrets": {"bitwarden": {}}},
        )
        monkeypatch.setenv("BWS_ACCESS_TOKEN", "0.stale-dotenv")
        monkeypatch.setattr(
            "hermes_cli.secrets_cli.bw.find_bws",
            lambda install_if_missing=False: "/usr/bin/bws",
        )
        monkeypatch.setattr(
            "hermes_cli.secrets_cli._bws_version", lambda _: "2.0.0"
        )
        monkeypatch.setattr("hermes_cli.secrets_cli.load_config", lambda: {})
        monkeypatch.setattr("hermes_cli.secrets_cli.save_config", lambda _: None)
        saved = {}
        monkeypatch.setattr(
            "hermes_cli.secrets_cli.save_env_value",
            lambda name, value: saved.__setitem__(name, value),
        )
        monkeypatch.setattr(
            "hermes_cli.secrets_cli.get_env_path", lambda: "/tmp/.env"
        )
        captured = {}

        def fake_fetch(**kwargs):
            captured.update(kwargs)
            return {"KEY": "val"}, []

        monkeypatch.setattr(
            "hermes_cli.secrets_cli.bw.fetch_bitwarden_secrets", fake_fetch
        )

        from hermes_cli.secrets_cli import cmd_setup

        result = cmd_setup(self._make_args(
            server_url="https://vault.bitwarden.com",
            project_id="aaaa-bbbb",
        ))

        assert result == 0
        assert captured["access_token"] == "0.injected-before-dotenv"
        assert saved == {"BWS_ACCESS_TOKEN": "0.injected-before-dotenv"}
