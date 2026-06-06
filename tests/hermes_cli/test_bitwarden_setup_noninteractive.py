from __future__ import annotations

from argparse import Namespace
from pathlib import Path
import sys
import types
from unittest.mock import patch


def _import_secrets_cli():
    if "rich" not in sys.modules:
        rich_mod = types.ModuleType("rich")
        rich_console = types.ModuleType("rich.console")
        rich_panel = types.ModuleType("rich.panel")
        rich_table = types.ModuleType("rich.table")

        class _Console:
            def print(self, *args, **kwargs):
                return None

            def input(self, prompt: str) -> str:
                return ""

        class _Panel:
            @staticmethod
            def fit(*args, **kwargs):
                return object()

        class _Table:
            def __init__(self, *args, **kwargs):
                pass

            def add_column(self, *args, **kwargs):
                pass

            def add_row(self, *args, **kwargs):
                pass

        rich_console.Console = _Console
        rich_panel.Panel = _Panel
        rich_table.Table = _Table
        sys.modules["rich"] = rich_mod
        sys.modules["rich.console"] = rich_console
        sys.modules["rich.panel"] = rich_panel
        sys.modules["rich.table"] = rich_table

    from hermes_cli import secrets_cli

    return secrets_cli


secrets_cli = _import_secrets_cli()


def _args(**overrides) -> Namespace:
    return Namespace(
        access_token=overrides.get("access_token"),
        project_id=overrides.get("project_id"),
        server_url=overrides.get("server_url"),
    )


class _FakeConsole:
    messages: list[str] = []

    def print(self, *args, **kwargs) -> None:
        for arg in args:
            if isinstance(arg, str):
                self.messages.append(arg)

    def input(self, prompt: str) -> str:
        raise AssertionError(f"console.input should not be called: {prompt}")


def test_bitwarden_setup_non_tty_with_all_flags_skips_prompts(monkeypatch):
    cfg = {"secrets": {"bitwarden": {}}}
    _FakeConsole.messages = []

    monkeypatch.setattr(secrets_cli, "Console", _FakeConsole)
    monkeypatch.setattr(secrets_cli.sys.stdin, "isatty", lambda: False)

    with (
        patch.object(secrets_cli.bw, "find_bws", return_value=Path("/tmp/bws")) as mock_find,
        patch.object(secrets_cli, "_bws_version", return_value="1.0.0"),
        patch.object(secrets_cli, "load_config", return_value=cfg),
        patch.object(secrets_cli, "save_config") as mock_save_config,
        patch.object(secrets_cli, "save_env_value") as mock_save_env,
        patch.object(secrets_cli.bw, "fetch_bitwarden_secrets", return_value=({}, [])) as mock_fetch,
        patch.object(secrets_cli, "masked_secret_prompt", side_effect=AssertionError("masked_secret_prompt should not be called")),
    ):
        rc = secrets_cli.cmd_setup(
            _args(
                access_token="0.test-token",
                project_id="project-123",
                server_url="https://vault.bitwarden.com",
            )
        )

    assert rc == 0
    mock_find.assert_called_once_with(install_if_missing=False)
    mock_save_env.assert_called_once_with("BWS_ACCESS_TOKEN", "0.test-token")
    mock_fetch.assert_called_once_with(
        access_token="0.test-token",
        project_id="project-123",
        binary=Path("/tmp/bws"),
        use_cache=False,
        server_url="https://vault.bitwarden.com",
    )
    assert cfg["secrets"]["bitwarden"]["server_url"] == "https://vault.bitwarden.com"
    mock_save_config.assert_called_once_with(cfg)


def test_bitwarden_setup_non_tty_requires_access_token_flag(monkeypatch):
    _FakeConsole.messages = []

    monkeypatch.setattr(secrets_cli, "Console", _FakeConsole)
    monkeypatch.setattr(secrets_cli.sys.stdin, "isatty", lambda: False)

    with (
        patch.object(secrets_cli.bw, "find_bws", return_value=Path("/tmp/bws")),
        patch.object(secrets_cli, "_bws_version", return_value="1.0.0"),
        patch.object(secrets_cli, "load_config", return_value={"secrets": {"bitwarden": {}}}),
        patch.object(secrets_cli, "masked_secret_prompt", side_effect=AssertionError("masked_secret_prompt should not be called")),
    ):
        rc = secrets_cli.cmd_setup(_args(access_token=None, project_id="project-123", server_url=None))

    assert rc == 1
    assert any("--access-token" in message for message in _FakeConsole.messages)


def test_bitwarden_setup_non_tty_requires_project_id_flag(monkeypatch):
    _FakeConsole.messages = []

    monkeypatch.setattr(secrets_cli, "Console", _FakeConsole)
    monkeypatch.setattr(secrets_cli.sys.stdin, "isatty", lambda: False)

    with (
        patch.object(secrets_cli.bw, "find_bws", return_value=Path("/tmp/bws")),
        patch.object(secrets_cli, "_bws_version", return_value="1.0.0"),
        patch.object(secrets_cli, "load_config", return_value={"secrets": {"bitwarden": {}}}),
        patch.object(secrets_cli, "save_env_value"),
        patch.object(secrets_cli, "_list_projects", side_effect=AssertionError("_list_projects should not be called")),
    ):
        rc = secrets_cli.cmd_setup(
            _args(
                access_token="0.test-token",
                project_id=None,
                server_url="https://vault.bitwarden.com",
            )
        )

    assert rc == 1
    assert any("--project-id" in message for message in _FakeConsole.messages)


def test_bitwarden_setup_non_tty_requires_server_url_flag(monkeypatch):
    _FakeConsole.messages = []

    monkeypatch.setattr(secrets_cli, "Console", _FakeConsole)
    monkeypatch.setattr(secrets_cli.sys.stdin, "isatty", lambda: False)

    with (
        patch.object(secrets_cli.bw, "find_bws", return_value=Path("/tmp/bws")),
        patch.object(secrets_cli, "_bws_version", return_value="1.0.0"),
        patch.object(secrets_cli, "load_config", return_value={"secrets": {"bitwarden": {}}}),
        patch.object(secrets_cli, "save_env_value"),
        patch.object(secrets_cli, "masked_secret_prompt", side_effect=AssertionError("masked_secret_prompt should not be called")),
        patch.object(secrets_cli.bw, "fetch_bitwarden_secrets", side_effect=AssertionError("fetch_bitwarden_secrets should not be called")),
    ):
        rc = secrets_cli.cmd_setup(
            _args(access_token="0.test-token", project_id="project-123", server_url=None)
        )

    assert rc == 1
    assert any("--server-url" in message for message in _FakeConsole.messages)
