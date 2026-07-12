"""Regression tests for CLI/TUI CWD resolution in ``load_cli_config``.

These tests exercise the real config loader against a temporary HERMES_HOME.
An explicit local ``terminal.cwd`` is configuration, not inherited state: it
must be honored while CLI still overwrites a stale ``TERMINAL_CWD`` environment
variable. Placeholder values continue to resolve to the launch directory.
"""

import os
from pathlib import Path

import pytest

import cli


def _load_cli_config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, config: str) -> dict:
    hermes_home = tmp_path / ".hermes"
    hermes_home.mkdir()
    (hermes_home / "config.yaml").write_text(config)
    monkeypatch.setattr(cli, "_hermes_home", hermes_home)
    return cli.load_cli_config()


def test_explicit_local_cwd_honors_config_and_overwrites_stale_env(tmp_path, monkeypatch):
    launch_dir = tmp_path / "launch"
    profile_dir = tmp_path / "profile"
    launch_dir.mkdir()
    profile_dir.mkdir()
    monkeypatch.chdir(launch_dir)
    monkeypatch.setenv("TERMINAL_CWD", str(tmp_path / "stale-parent-cwd"))

    config = _load_cli_config(
        tmp_path,
        monkeypatch,
        f"terminal:\n  backend: local\n  cwd: {profile_dir}\n",
    )

    assert config["terminal"]["cwd"] == str(profile_dir)
    assert config["terminal"]["env_type"] == "local"
    assert config["terminal"]["cwd"] != str(launch_dir)
    assert os.environ["TERMINAL_CWD"] == str(profile_dir)


@pytest.mark.parametrize("placeholder", [".", "auto", "cwd"])
def test_local_cwd_placeholder_uses_launch_directory(tmp_path, monkeypatch, placeholder):
    launch_dir = tmp_path / "launch"
    launch_dir.mkdir()
    monkeypatch.chdir(launch_dir)
    monkeypatch.setenv("TERMINAL_CWD", str(tmp_path / "stale-parent-cwd"))

    config = _load_cli_config(
        tmp_path,
        monkeypatch,
        f"terminal:\n  backend: local\n  cwd: {placeholder}\n",
    )

    assert config["terminal"]["cwd"] == str(launch_dir)
    assert os.environ["TERMINAL_CWD"] == str(launch_dir)


def test_local_default_cwd_overwrites_inherited_env(tmp_path, monkeypatch):
    launch_dir = tmp_path / "launch"
    launch_dir.mkdir()
    monkeypatch.chdir(launch_dir)
    monkeypatch.setenv("TERMINAL_CWD", str(tmp_path / "stale-parent-cwd"))

    config = _load_cli_config(tmp_path, monkeypatch, "model: test-model\n")

    assert config["terminal"]["cwd"] == str(launch_dir)
    assert os.environ["TERMINAL_CWD"] == str(launch_dir)


def test_nonlocal_explicit_cwd_is_preserved(tmp_path, monkeypatch):
    workspace_dir = tmp_path / "workspace"
    workspace_dir.mkdir()
    monkeypatch.delenv("TERMINAL_CWD", raising=False)

    config = _load_cli_config(
        tmp_path,
        monkeypatch,
        f"terminal:\n  backend: ssh\n  cwd: {workspace_dir}\n",
    )

    assert config["terminal"]["cwd"] == str(workspace_dir)
    assert os.environ["TERMINAL_CWD"] == str(workspace_dir)


def test_nonlocal_placeholder_defers_to_backend_default(tmp_path, monkeypatch):
    monkeypatch.delenv("TERMINAL_CWD", raising=False)

    config = _load_cli_config(
        tmp_path,
        monkeypatch,
        "terminal:\n  backend: docker\n  cwd: .\n",
    )

    assert "cwd" not in config["terminal"]
    assert "TERMINAL_CWD" not in os.environ


def test_gateway_lazy_import_does_not_clobber_terminal_cwd(tmp_path, monkeypatch):
    gateway_cwd = tmp_path / "gateway-workspace"
    gateway_cwd.mkdir()
    monkeypatch.setenv("_HERMES_GATEWAY", "1")
    monkeypatch.setenv("TERMINAL_CWD", str(gateway_cwd))

    _load_cli_config(
        tmp_path,
        monkeypatch,
        "terminal:\n  backend: local\n  cwd: /configured-for-gateway\n",
    )

    assert os.environ["TERMINAL_CWD"] == str(gateway_cwd)
