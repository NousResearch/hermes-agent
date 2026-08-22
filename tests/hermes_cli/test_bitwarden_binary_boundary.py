"""Regression coverage for the Bitwarden CLI use-time trust boundary."""

from __future__ import annotations

import argparse
import hashlib
import io
from pathlib import Path
from unittest import mock

import pytest
from rich.console import Console

from agent.secret_sources import bitwarden as bw
from hermes_cli import secrets_cli


@pytest.fixture(autouse=True)
def _clear_hermes_home_cache():
    import hermes_constants

    if hasattr(hermes_constants, "_HERMES_HOME_CACHE"):
        hermes_constants._HERMES_HOME_CACHE = None  # type: ignore[attr-defined]
    yield
    if hasattr(hermes_constants, "_HERMES_HOME_CACHE"):
        hermes_constants._HERMES_HOME_CACHE = None  # type: ignore[attr-defined]


def _managed_bws(tmp_path: Path, monkeypatch, *, tampered: bool) -> Path:
    home = tmp_path / "hermes"
    bin_dir = home / "bin"
    bin_dir.mkdir(parents=True)
    bin_dir.chmod(0o700)
    binary = bin_dir / "bws"
    payload = b"#!/bin/sh\nprintf 'bws 2.0.0\\n'\n"
    binary.write_bytes(payload)
    binary.chmod(0o700)
    checksum = bin_dir / ".bws.sha256"
    checksum.write_text(hashlib.sha256(payload).hexdigest() + "\n")
    checksum.chmod(0o600)
    if tampered:
        binary.write_bytes(b"#!/bin/sh\nprintf 'tampered\\n'\n")

    monkeypatch.setenv("HERMES_HOME", str(home))
    import hermes_constants

    if hasattr(hermes_constants, "_HERMES_HOME_CACHE"):
        monkeypatch.setattr(
            hermes_constants, "_HERMES_HOME_CACHE", None, raising=False
        )
    return binary


def test_version_refuses_tampered_managed_binary_before_spawn(tmp_path, monkeypatch):
    binary = _managed_bws(tmp_path, monkeypatch, tampered=True)
    monkeypatch.setenv("BWS_ACCESS_TOKEN", "synthetic-token")
    run = mock.Mock()
    monkeypatch.setattr(secrets_cli.subprocess, "run", run)

    assert secrets_cli._bws_version(binary) == "version unknown"
    run.assert_not_called()


def test_projects_refuse_tampered_managed_binary_before_token_child(
    tmp_path, monkeypatch
):
    binary = _managed_bws(tmp_path, monkeypatch, tampered=True)
    run = mock.Mock()
    monkeypatch.setattr(secrets_cli.subprocess, "run", run)
    output = io.StringIO()

    projects = secrets_cli._list_projects(
        binary, "synthetic-token", Console(file=output), server_url=""
    )

    assert projects is None
    assert "Refusing unverified bws binary" in output.getvalue()
    run.assert_not_called()


def test_version_uses_credential_free_environment_for_valid_managed_binary(
    tmp_path, monkeypatch
):
    binary = _managed_bws(tmp_path, monkeypatch, tampered=False)
    monkeypatch.setenv("BWS_ACCESS_TOKEN", "synthetic-token")
    monkeypatch.setenv("LD_PRELOAD", "must-not-pass")
    result = mock.Mock(returncode=0, stdout="bws 2.0.0\n", stderr="")
    run = mock.Mock(return_value=result)
    monkeypatch.setattr(secrets_cli.subprocess, "run", run)

    assert secrets_cli._bws_version(binary) == "bws 2.0.0"

    args, kwargs = run.call_args
    assert args[0] == [str(binary), "--version"]
    assert kwargs["stdin"] is secrets_cli.subprocess.DEVNULL
    assert kwargs["env"].get("BWS_ACCESS_TOKEN") is None
    assert kwargs["env"].get("LD_PRELOAD") is None


def test_projects_accept_valid_managed_binary_and_pass_only_project_token(
    tmp_path, monkeypatch
):
    binary = _managed_bws(tmp_path, monkeypatch, tampered=False)
    result = mock.Mock(
        returncode=0,
        stdout='[{"id": "project-1", "name": "Project"}]',
        stderr="",
    )
    run = mock.Mock(return_value=result)
    monkeypatch.setattr(secrets_cli.subprocess, "run", run)

    projects = secrets_cli._list_projects(
        binary, "synthetic-token", Console(), server_url="https://vault.example"
    )

    assert projects == [{"id": "project-1", "name": "Project"}]
    args, kwargs = run.call_args
    assert args[0] == [str(binary), "project", "list", "--output", "json"]
    assert kwargs["env"]["BWS_ACCESS_TOKEN"] == "synthetic-token"
    assert kwargs["env"]["BWS_SERVER_URL"] == "https://vault.example"


def test_projects_accept_valid_path_candidate_through_policy_mocks(monkeypatch):
    binary = Path("/trusted/bin/bws")
    monkeypatch.setattr(bw, "_is_managed_bws", lambda _path: False)
    monkeypatch.setattr(bw, "_resolve_executable", lambda path, **_kw: Path(path))
    monkeypatch.setattr(bw, "_probe_binary_version", lambda *_args, **_kwargs: True)
    result = mock.Mock(returncode=0, stdout="[]", stderr="")
    run = mock.Mock(return_value=result)
    monkeypatch.setattr(secrets_cli.subprocess, "run", run)

    assert secrets_cli._list_projects(binary, "synthetic-token", Console()) == []
    args, kwargs = run.call_args
    assert args[0] == [str(binary), "project", "list", "--output", "json"]
    assert kwargs["env"]["BWS_ACCESS_TOKEN"] == "synthetic-token"


def test_status_does_not_spawn_for_tampered_managed_binary(tmp_path, monkeypatch):
    binary = _managed_bws(tmp_path, monkeypatch, tampered=True)
    monkeypatch.setattr(secrets_cli, "load_config", lambda: {
        "secrets": {"bitwarden": {
            "enabled": True,
            "access_token_env": "BWS_ACCESS_TOKEN",
            "project_id": "project-1",
        }},
    })
    monkeypatch.setenv("BWS_ACCESS_TOKEN", "synthetic-token")
    monkeypatch.setattr(secrets_cli.bw, "find_bws", lambda **_kw: binary)
    run = mock.Mock()
    monkeypatch.setattr(secrets_cli.subprocess, "run", run)

    assert secrets_cli.cmd_status(argparse.Namespace()) == 0
    run.assert_not_called()


def test_token_rotation_does_not_spawn_or_store_for_tampered_managed_binary(
    tmp_path, monkeypatch
):
    binary = _managed_bws(tmp_path, monkeypatch, tampered=True)
    monkeypatch.setattr(secrets_cli, "load_config", lambda: {
        "secrets": {"bitwarden": {
            "enabled": True,
            "access_token_env": "BWS_ACCESS_TOKEN",
            "project_id": "project-1",
        }},
    })
    monkeypatch.setattr(secrets_cli.bw, "find_bws", lambda **_kw: binary)
    saved = mock.Mock()
    monkeypatch.setattr(secrets_cli, "save_env_value", saved)
    monkeypatch.setattr(secrets_cli.bw, "clear_caches", lambda: None)
    run = mock.Mock()
    monkeypatch.setattr(secrets_cli.subprocess, "run", run)

    args = argparse.Namespace(access_token="0.synthetic-token", no_verify=False)
    assert secrets_cli.cmd_token(args) == 1
    run.assert_not_called()
    saved.assert_not_called()


def test_setup_does_not_spawn_for_tampered_managed_binary(tmp_path, monkeypatch):
    binary = _managed_bws(tmp_path, monkeypatch, tampered=True)
    monkeypatch.setattr(secrets_cli.bw, "find_bws", lambda **_kw: binary)
    monkeypatch.setattr(secrets_cli, "load_config", lambda: {})
    monkeypatch.setattr(secrets_cli, "save_config", lambda _cfg: None)
    monkeypatch.setattr(secrets_cli, "save_env_value", lambda *_args: None)
    monkeypatch.setattr(secrets_cli, "get_env_path", lambda: tmp_path / ".env")
    monkeypatch.setattr("sys.stdin.isatty", lambda: False)
    run = mock.Mock()
    monkeypatch.setattr(secrets_cli.subprocess, "run", run)

    args = argparse.Namespace(
        access_token="0.synthetic-token",
        server_url="https://vault.example",
        project_id="project-1",
    )
    assert secrets_cli.cmd_setup(args) == 1
    run.assert_not_called()
