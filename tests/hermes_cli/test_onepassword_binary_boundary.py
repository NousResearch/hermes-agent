"""The CLI must revalidate op before version and authenticated probes."""
from __future__ import annotations

import os
import stat
from unittest import mock

import pytest

from agent.secret_sources import _binary_security as security
from hermes_cli import onepassword_secrets_cli as cli


@pytest.fixture
def explicit_op(tmp_path, monkeypatch):
    path = tmp_path / "op.exe"
    path.write_text("inert executable")
    path.chmod(0o755)
    if os.name == "nt":
        monkeypatch.setenv("USERPROFILE", str(tmp_path))
    else:
        original_stat = security.Path.stat

        def private_stat(path, *args, **kwargs):
            result = original_stat(path, *args, **kwargs)
            fields = list(result)
            fields[4] = os.geteuid()
            if stat.S_ISDIR(result.st_mode):
                fields[0] &= ~(stat.S_IWGRP | stat.S_IWOTH)
            return os.stat_result(fields)

        monkeypatch.setattr(security.Path, "stat", private_stat)
    return path


@pytest.mark.skipif(os.name == "nt", reason="POSIX mode boundary")
@pytest.mark.parametrize("operation", ["version", "whoami"])
def test_cli_rejects_changed_binary_before_spawn(explicit_op, monkeypatch, operation):
    explicit_op.chmod(0o777)
    run = mock.Mock(return_value=mock.Mock(returncode=0, stdout="2.0.0\n", stderr=""))
    monkeypatch.setattr(cli.subprocess, "run", run)
    if operation == "version":
        assert cli._op_version(explicit_op) == "version unknown"
    else:
        assert cli._op_whoami(explicit_op, "", token_value="inert-token") is None
    run.assert_not_called()


def test_cli_version_probe_keeps_credentials_out_of_environment(explicit_op, monkeypatch):
    monkeypatch.setenv("OP_SERVICE_ACCOUNT_TOKEN", "inert-token")
    monkeypatch.setenv("BWS_ACCESS_TOKEN", "inert-other-token")
    run = mock.Mock(return_value=mock.Mock(returncode=0, stdout="2.0.0\n", stderr=""))
    monkeypatch.setattr(cli.subprocess, "run", run)

    assert cli._op_version(explicit_op) == "2.0.0"
    args, kwargs = run.call_args
    assert args[0] == [str(explicit_op.resolve()), "--version"]
    assert "OP_SERVICE_ACCOUNT_TOKEN" not in kwargs["env"]
    assert "BWS_ACCESS_TOKEN" not in kwargs["env"]
    assert kwargs["stdin"] is cli.subprocess.DEVNULL


def test_cli_authenticated_probe_keeps_intended_token(explicit_op, monkeypatch):
    run = mock.Mock(return_value=mock.Mock(returncode=0, stdout="identity\n", stderr=""))
    monkeypatch.setattr(cli.subprocess, "run", run)

    assert cli._op_whoami(explicit_op, "account", token_value="inert-token") == "identity"
    args, kwargs = run.call_args
    assert args[0] == [str(explicit_op.resolve()), "whoami", "--account", "account"]
    assert kwargs["env"]["OP_SERVICE_ACCOUNT_TOKEN"] == "inert-token"
