"""Credential and display boundaries for authenticated vault CLI subprocesses."""
from pathlib import Path
from unittest import mock
import pytest
from hermes_cli import secrets_cli as bw_cli
from hermes_cli import onepassword_secrets_cli as op_cli


def test_bw_cli_child_env_allows_only_provider_values(monkeypatch):
    captured = {}
    for key in ("OPENAI_API_KEY", "GH_TOKEN", "AUXILIARY_WEB_API_KEY"):
        monkeypatch.setenv(key, f"sentinel-{key}")
    monkeypatch.setenv("BWS_SERVER_URL", "https://vault.example")

    def fake_run(cmd, **kwargs):
        captured.update(kwargs["env"])
        return mock.Mock(returncode=0, stdout="[]", stderr="")

    monkeypatch.setattr(bw_cli.subprocess, "run", fake_run)

    projects = bw_cli._list_projects(
        Path("/fake/bws"), "0.synthetic-bw-cli-env", bw_cli.Console()
    )

    assert projects == []
    assert captured["BWS_ACCESS_TOKEN"] == "0.synthetic-bw-cli-env"
    assert captured["BWS_SERVER_URL"] == "https://vault.example"
    for key in ("OPENAI_API_KEY", "GH_TOKEN", "AUXILIARY_WEB_API_KEY"):
        assert key not in captured


def test_op_whoami_redacts_current_environment_token(monkeypatch):
    token = "ops.synthetic-op-current-77468"
    split_token = f"{token[:8]}\x9b31m{token[8:]}\x9b0m"
    monkeypatch.setenv("OP_SERVICE_ACCOUNT_TOKEN", token)
    for key in ("OPENAI_API_KEY", "GH_TOKEN", "AUXILIARY_WEB_API_KEY"):
        monkeypatch.setenv(key, f"sentinel-{key}")
    captured = {}

    def fake_run(*a, **kwargs):
        captured.update(kwargs["env"])
        return mock.Mock(returncode=0, stdout=f"session {split_token}", stderr="")

    monkeypatch.setattr(
        op_cli.subprocess,
        "run", fake_run,
    )

    result = op_cli._op_whoami(
        Path("/fake/op"), "", token_value=token
    )

    assert result == "session <redacted>"
    assert captured["OP_SERVICE_ACCOUNT_TOKEN"] == token
    for key in ("OPENAI_API_KEY", "GH_TOKEN", "AUXILIARY_WEB_API_KEY"):
        assert key not in captured


@pytest.mark.parametrize("auth_env", ["OP_SESSION_demo", "OP_CONNECT_TOKEN"])
@pytest.mark.parametrize("control", ["\x00", "\x09", "\x0d", "\x1b", "\x1b["])
def test_op_whoami_redacts_session_and_connect_auth(
    monkeypatch, auth_env, control
):
    auth = f"ops.synthetic-{auth_env.lower()}-whoami-77468"
    monkeypatch.delenv("OP_SERVICE_ACCOUNT_TOKEN", raising=False)
    monkeypatch.setenv(auth_env, auth)
    if auth_env == "OP_CONNECT_TOKEN":
        monkeypatch.setenv("OP_CONNECT_HOST", "https://connect.example")
    split_auth = f"{auth[:8]}{control}{auth[8:]}"
    captured = {}

    def fake_run(*a, **kwargs):
        captured.update(kwargs["env"])
        return mock.Mock(
            returncode=0,
            stdout=f"identity {split_auth} host=https://connect.example",
            stderr="",
        )

    monkeypatch.setattr(op_cli.subprocess, "run", fake_run)

    result = op_cli._op_whoami(Path("/fake/op"), "")

    assert result == "identity <redacted> host=https://connect.example"
    assert auth_env in captured
    assert auth not in result


def test_provider_version_probes_use_minimal_env_and_safe_output(monkeypatch):
    for key in (
        "OPENAI_API_KEY",
        "GH_TOKEN",
        "AUXILIARY_WEB_API_KEY",
        "BWS_ACCESS_TOKEN",
        "OP_SERVICE_ACCOUNT_TOKEN",
    ):
        monkeypatch.setenv(key, f"sentinel-{key}")
    captured = {}

    def fake_bws_run(cmd, **kwargs):
        captured["bws"] = dict(kwargs["env"])
        return mock.Mock(returncode=0, stdout="bws v2.0.0; sentinel", stderr="")

    monkeypatch.setattr(bw_cli.subprocess, "run", fake_bws_run)
    assert bw_cli._bws_version(Path("/fake/bws")) == "v2.0.0"

    def fake_op_run(cmd, **kwargs):
        captured["op"] = dict(kwargs["env"])
        return mock.Mock(
            returncode=0,
            stdout="1Password CLI 2.32.1 sentinel-token",
            stderr="",
        )

    monkeypatch.setattr(op_cli.subprocess, "run", fake_op_run)
    assert op_cli._op_version(Path("/fake/op")) == "2.32.1"

    for env in (captured["bws"], captured["op"]):
        for key in (
            "OPENAI_API_KEY",
            "GH_TOKEN",
            "AUXILIARY_WEB_API_KEY",
            "BWS_ACCESS_TOKEN",
            "OP_SERVICE_ACCOUNT_TOKEN",
        ):
            assert key not in env
