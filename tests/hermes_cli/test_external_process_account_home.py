"""Regression for #151733: discover account-local CLIs from a nested profile HOME."""

import os
from pathlib import Path

import pytest

from hermes_cli.auth_external_process import resolve_external_process_command


@pytest.mark.platforms("posix")
def test_nested_home_finds_account_install_without_changing_environment(tmp_path, monkeypatch):
    account = tmp_path / "account"
    profile = account / "profile"
    nested = profile / "home"
    nested.mkdir(parents=True)
    installed = account / ".local" / "bin" / "hermes-lookup-fixture"
    installed.parent.mkdir(parents=True)
    installed.write_text("#!/bin/sh\nexit 0\n")
    installed.chmod(0o700)
    monkeypatch.setenv("HOME", str(nested))
    monkeypatch.setenv("HERMES_HOME", str(profile))
    monkeypatch.setenv("HERMES_REAL_HOME", str(account))
    monkeypatch.setenv("PATH", str(tmp_path / "empty-path"))
    before = dict(os.environ)

    assert resolve_external_process_command(installed.name) == str(installed)
    assert dict(os.environ) == before
    assert not (nested / ".local").exists()


@pytest.mark.platforms("posix")
def test_nested_home_keeps_path_and_explicit_command_precedence(tmp_path, monkeypatch):
    account = tmp_path / "account"
    nested = tmp_path / "profile" / "home"
    nested.mkdir(parents=True)
    candidates = []
    for directory in (account / ".local" / "bin", tmp_path / "path-bin", nested / "bin"):
        directory.mkdir(parents=True)
        command = directory / "hermes-lookup-fixture"
        command.write_text("#!/bin/sh\nexit 0\n")
        command.chmod(0o700)
        candidates.append(str(command))
    monkeypatch.setenv("HOME", str(nested))
    monkeypatch.setenv("HERMES_HOME", str(nested.parent))
    monkeypatch.setenv("HERMES_REAL_HOME", str(account))
    monkeypatch.setenv("PATH", str(Path(candidates[1]).parent))
    monkeypatch.chdir(nested)

    assert resolve_external_process_command("hermes-lookup-fixture") == candidates[1]
    assert resolve_external_process_command("./bin/hermes-lookup-fixture") == candidates[2]
    assert resolve_external_process_command("~/bin/hermes-lookup-fixture") == candidates[2]
    assert resolve_external_process_command(candidates[0]) == candidates[0]
