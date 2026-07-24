"""Regression tests for hermes secrets bitwarden setup non-TTY guard.

Issue #40274: cmd_setup() crashes with EOFError when stdin is not a TTY
because getpass.getpass() and console.input() require an interactive terminal.
"""
from __future__ import annotations

import argparse
from unittest.mock import MagicMock

import pytest


def _mock_bw(**overrides):
    """Build a MagicMock standing in for the bitwarden module.

    #70697: secrets_cli now lazy-imports bitwarden via ``_bw()`` instead of
    a module-level ``bw`` binding, so tests patch ``_bw`` rather than ``bw``.
    """
    mock = MagicMock()
    mock.find_bws.return_value = overrides.get("find_bws", "/usr/bin/bws")
    mock.install_bws.return_value = overrides.get("install_bws", "/usr/bin/bws")
    mock.fetch_bitwarden_secrets.return_value = overrides.get(
        "fetch_bitwarden_secrets", ({"KEY": "val"}, [])
    )
    mock.clear_caches.return_value = None
    mock._BWS_VERSION = overrides.get("_BWS_VERSION", "2.0.0")
    return mock


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

    def test_missing_all_flags_returns_1(self, monkeypatch, capsys):
        """Non-TTY with no flags → exit 1 with missing flags listed."""
        monkeypatch.setattr("sys.stdin.isatty", lambda: False)
        monkeypatch.setattr("hermes_cli.secrets_cli._bw", lambda: _mock_bw())
        monkeypatch.setattr(
            "hermes_cli.secrets_cli._bws_version", lambda _: "2.0.0"
        )

        from hermes_cli.secrets_cli import cmd_setup

        result = cmd_setup(self._make_args())
        assert result == 1
        captured = capsys.readouterr()
        assert "Non-interactive mode" in captured.out
        assert "--access-token" in captured.out
        assert "--server-url" in captured.out
        assert "--project-id" in captured.out

    def test_missing_access_token_only(self, monkeypatch, capsys):
        """Non-TTY with server-url and project-id but no token → reports --access-token."""
        monkeypatch.setattr("sys.stdin.isatty", lambda: False)
        monkeypatch.setattr("hermes_cli.secrets_cli._bw", lambda: _mock_bw())
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

    def test_missing_server_url_with_env_var_passes(self, monkeypatch):
        """Non-TTY with BWS_SERVER_URL env set → server-url not required."""
        monkeypatch.setattr("sys.stdin.isatty", lambda: False)
        monkeypatch.setenv("BWS_SERVER_URL", "https://vault.bitwarden.com")
        monkeypatch.setattr("hermes_cli.secrets_cli._bw", lambda: _mock_bw())
        monkeypatch.setattr(
            "hermes_cli.secrets_cli._bws_version", lambda _: "2.0.0"
        )
        monkeypatch.setattr("hermes_cli.secrets_cli.load_config", lambda: {})
        monkeypatch.setattr("hermes_cli.secrets_cli.save_env_value", lambda *a: None)
        monkeypatch.setattr("hermes_cli.secrets_cli.get_env_path", lambda: "/tmp/.env")

        from hermes_cli.secrets_cli import cmd_setup

        result = cmd_setup(self._make_args(
            access_token="0.valid-token",
            project_id="aaaa-bbbb",
        ))
        assert result == 0

    def test_all_flags_provided_passes_guard(self, monkeypatch):
        """Non-TTY with all three flags → guard passes, proceeds to setup."""
        monkeypatch.setattr("sys.stdin.isatty", lambda: False)
        monkeypatch.setattr("hermes_cli.secrets_cli._bw", lambda: _mock_bw())
        monkeypatch.setattr(
            "hermes_cli.secrets_cli._bws_version", lambda _: "2.0.0"
        )
        monkeypatch.setattr("hermes_cli.secrets_cli.load_config", lambda: {})
        monkeypatch.setattr("hermes_cli.secrets_cli.save_env_value", lambda *a: None)
        monkeypatch.setattr("hermes_cli.secrets_cli.get_env_path", lambda: "/tmp/.env")

        from hermes_cli.secrets_cli import cmd_setup

        result = cmd_setup(self._make_args(
            access_token="0.valid-token",
            server_url="https://vault.bitwarden.com",
            project_id="aaaa-bbbb",
        ))
        assert result == 0

    def test_tty_does_not_trigger_guard(self, monkeypatch):
        """With TTY, the guard should not trigger (interactive mode allowed)."""
        monkeypatch.setattr("sys.stdin.isatty", lambda: True)
        monkeypatch.setattr("hermes_cli.secrets_cli._bw", lambda: _mock_bw())
        monkeypatch.setattr(
            "hermes_cli.secrets_cli._bws_version", lambda _: "2.0.0"
        )
        monkeypatch.setattr(
            "hermes_cli.secrets_cli.masked_secret_prompt", lambda prompt: "0.valid-token"
        )
        monkeypatch.setattr("hermes_cli.secrets_cli.load_config", lambda: {})
        monkeypatch.setattr("hermes_cli.secrets_cli.save_env_value", lambda *a: None)
        monkeypatch.setattr("hermes_cli.secrets_cli.get_env_path", lambda: "/tmp/.env")
        monkeypatch.setattr(
            "hermes_cli.secrets_cli._resolve_server_url",
            lambda *a: "https://vault.bitwarden.com",
        )

        from hermes_cli.secrets_cli import cmd_setup

        # With TTY + all flags → should complete without hitting guard
        result = cmd_setup(self._make_args(
            access_token="0.valid-token",
            server_url="https://vault.bitwarden.com",
            project_id="aaaa-bbbb",
        ))
        assert result == 0
