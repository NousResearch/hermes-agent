"""Tests for `hermes secrets bitwarden token` / `hermes secrets onepassword token`.

The rotation command must: verify the candidate token BEFORE persisting,
never touch .env on a rejected token, store + clear caches on success,
and fail cleanly without a TTY.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from unittest import mock

import pytest

from agent.secret_sources.base import scrub_ansi
from hermes_cli import onepassword_secrets_cli as op_cli
from hermes_cli import secrets_cli as bw_cli
from hermes_cli.secret_prompt import (
    capture_pre_dotenv_rotation_inputs,
    get_pre_dotenv_rotation_input,
    reset_pre_dotenv_rotation_inputs,
)


@pytest.fixture(autouse=True)
def _reset_pre_dotenv_rotation_inputs():
    reset_pre_dotenv_rotation_inputs()
    yield
    reset_pre_dotenv_rotation_inputs()


# ---------------------------------------------------------------------------
# Bitwarden
# ---------------------------------------------------------------------------


def _bw_args(**overrides):
    return argparse.Namespace(
        access_token=overrides.get("access_token", ""),
        no_verify=overrides.get("no_verify", False),
    )


@pytest.fixture
def bw_env(monkeypatch, tmp_path):
    saved = {}
    monkeypatch.setattr(bw_cli, "load_config", lambda: {
        "secrets": {"bitwarden": {
            "enabled": True,
            "access_token_env": "BWS_ACCESS_TOKEN",
            "project_id": "proj-1",
            "server_url": "",
        }},
    })
    monkeypatch.setattr(
        bw_cli, "save_env_value",
        lambda name, value: saved.__setitem__(name, value),
    )
    monkeypatch.setattr(bw_cli, "get_env_path", lambda: tmp_path / ".env")
    monkeypatch.setattr(
        bw_cli.bw, "find_bws",
        lambda install_if_missing=True: Path("/fake/bws"),
    )
    return saved




def test_bw_token_no_verify_skips_probe(bw_env, monkeypatch, capsys):
    probe = mock.Mock()
    monkeypatch.setattr(bw_cli, "_list_projects", probe)
    monkeypatch.setattr(bw_cli.bw, "clear_caches", lambda *a, **kw: None)
    rc = bw_cli.cmd_token(_bw_args(access_token="0.x", no_verify=True))
    assert rc == 0
    probe.assert_not_called()
    assert bw_env == {"BWS_ACCESS_TOKEN": "0.x"}
    output = capsys.readouterr().out
    assert "process listings" in output
    assert "0.x" not in output


def test_bw_token_verify_failure_redacts_provider_diagnostic(
    bw_env, monkeypatch, capsys
):
    token = "0.synthetic-bw-cli-diagnostic-77468"
    split_token = f"{token[:8]}\x1b[31m{token[8:]}\x1b[0m"
    monkeypatch.setattr(
        bw_cli.subprocess,
        "run",
        lambda *a, **kw: mock.Mock(
            returncode=1,
            stdout="",
            stderr=f"provider rejected {split_token}; invalid_client",
        ),
    )

    rc = bw_cli.cmd_token(_bw_args(access_token=token))

    assert rc == 1
    assert bw_env == {}
    output = capsys.readouterr().out
    assert token not in scrub_ansi(output)
    assert "provider rejected <redacted>" in output
    assert "invalid_client" in output


@pytest.mark.parametrize("control", ["\x00", "\x09", "\x0d", "\x1b", "\x1b["])
def test_bw_token_verify_failure_redacts_controls(
    bw_env, monkeypatch, capsys, control
):
    token = "0.synthetic-bw-cli-control-77468"
    split_token = f"{token[:8]}{control}{token[8:]}"
    monkeypatch.setattr(
        bw_cli.subprocess,
        "run",
        lambda *a, **kw: mock.Mock(
            returncode=1,
            stdout="",
            stderr=f"provider rejected {split_token}; invalid_client",
        ),
    )

    rc = bw_cli.cmd_token(_bw_args(access_token=token))

    assert rc == 1
    output = capsys.readouterr().out
    assert token not in output
    assert "provider rejected <redacted>" in output


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


def test_bw_setup_success_redacts_provider_project_echo(monkeypatch, capsys):
    token = "0.synthetic-bw-project-77468"
    split_token = f"{token[:8]}\x1b[31m{token[8:]}\x1b[0m"
    monkeypatch.setattr("sys.stdin.isatty", lambda: True)
    monkeypatch.setattr(bw_cli, "load_config", lambda: {})
    monkeypatch.setattr(
        bw_cli.bw, "find_bws", lambda install_if_missing=False: Path("/fake/bws")
    )
    monkeypatch.setattr(bw_cli, "_bws_version", lambda _: "2.0.0")
    monkeypatch.setattr(bw_cli, "save_env_value", lambda *args: None)
    monkeypatch.setattr(bw_cli, "get_env_path", lambda: Path("/tmp/.env"))
    monkeypatch.setattr(bw_cli, "save_config", lambda _: None)
    monkeypatch.setattr(
        bw_cli,
        "_list_projects",
        lambda *args, **kwargs: [{"id": split_token, "name": split_token}],
    )
    monkeypatch.setattr(
        bw_cli.bw,
        "fetch_bitwarden_secrets",
        lambda **kwargs: (
            {"KEY": "value"},
            [f"provider warning: {split_token}; retained context"],
        ),
    )
    monkeypatch.setattr(bw_cli.Console, "input", lambda self, prompt: "1")

    args = argparse.Namespace(
        access_token=token,
        server_url="https://vault.bitwarden.com",
        project_id="",
    )
    rc = bw_cli.cmd_setup(args)

    assert rc == 0
    output = capsys.readouterr().out
    assert token not in scrub_ansi(output)
    assert "<redacted>" in output
    assert "provider warning: <redacted>; retained context" in output
    assert "KEY" in output


def test_bw_token_non_tty_uses_token_env(bw_env, monkeypatch, capsys):
    monkeypatch.setattr("sys.stdin.isatty", lambda: False)
    monkeypatch.setenv("BWS_ACCESS_TOKEN", "0.from-environment")
    monkeypatch.setattr(bw_cli.bw, "clear_caches", lambda *a, **kw: None)

    rc = bw_cli.cmd_token(_bw_args(no_verify=True))

    assert rc == 0
    assert bw_env == {"BWS_ACCESS_TOKEN": "0.from-environment"}
    output = capsys.readouterr().out
    assert "No TTY" not in output
    assert "0.from-environment" not in output


def test_bw_token_non_tty_prefers_injected_value_over_dotenv(
    bw_env, monkeypatch, capsys
):
    monkeypatch.setattr("sys.stdin.isatty", lambda: False)
    monkeypatch.setenv("BWS_ACCESS_TOKEN", "0.injected-before-dotenv")
    capture_pre_dotenv_rotation_inputs(
        ["hermes", "secrets", "bitwarden", "token"],
        config={"secrets": {"bitwarden": {}}},
    )
    # Simulate load_hermes_dotenv() replacing the injected value with the
    # persisted value before the command handler runs.
    monkeypatch.setenv("BWS_ACCESS_TOKEN", "0.stale-dotenv")
    monkeypatch.setattr(bw_cli.bw, "clear_caches", lambda *a, **kw: None)

    rc = bw_cli.cmd_token(_bw_args(no_verify=True))

    assert rc == 0
    assert bw_env == {"BWS_ACCESS_TOKEN": "0.injected-before-dotenv"}
    assert "0.stale-dotenv" not in capsys.readouterr().out


def test_bw_token_non_tty_prefers_injected_custom_env_over_dotenv(
    bw_env, monkeypatch
):
    monkeypatch.setattr(
        bw_cli,
        "load_config",
        lambda: {
            "secrets": {
                "bitwarden": {
                    "enabled": True,
                    "access_token_env": "CUSTOM_BW_TOKEN",
                    "project_id": "proj-1",
                },
            },
        },
    )
    monkeypatch.setattr("sys.stdin.isatty", lambda: False)
    monkeypatch.setenv("CUSTOM_BW_TOKEN", "0.injected-custom-before-dotenv")
    capture_pre_dotenv_rotation_inputs(
        ["hermes", "secrets", "bitwarden", "token"],
        config={
            "secrets": {
                "bitwarden": {"access_token_env": "CUSTOM_BW_TOKEN"},
            },
        },
    )
    monkeypatch.setenv("CUSTOM_BW_TOKEN", "0.stale-custom-dotenv")
    monkeypatch.setattr(bw_cli.bw, "clear_caches", lambda *a, **kw: None)

    rc = bw_cli.cmd_token(_bw_args(no_verify=True))

    assert rc == 0
    assert bw_env == {"CUSTOM_BW_TOKEN": "0.injected-custom-before-dotenv"}


def test_bw_setup_resolves_user_dotenv_token_name_before_stale_value(
    bw_env, monkeypatch, tmp_path
):
    env_file = tmp_path / ".env"
    env_file.write_text(
        "TOKEN_ENV_NAME=CUSTOM_BW_TOKEN\n"
        "CUSTOM_BW_TOKEN=0.stale-dotenv\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        bw_cli,
        "load_config",
        lambda: {
            "secrets": {
                "bitwarden": {
                    "enabled": True,
                    "access_token_env": "CUSTOM_BW_TOKEN",
                    "project_id": "proj-1",
                },
            },
        },
    )
    monkeypatch.setattr("sys.stdin.isatty", lambda: False)
    monkeypatch.delenv("TOKEN_ENV_NAME", raising=False)
    monkeypatch.setenv("CUSTOM_BW_TOKEN", "0.injected-before-dotenv")
    capture_pre_dotenv_rotation_inputs(
        ["hermes", "secrets", "bitwarden", "setup"],
        config={
            "secrets": {
                "bitwarden": {
                    "access_token_env": "${env:TOKEN_ENV_NAME}",
                },
            },
        },
        dotenv_sources=[(env_file, True)],
    )
    monkeypatch.setenv("CUSTOM_BW_TOKEN", "0.stale-dotenv")
    monkeypatch.setattr(bw_cli, "_bws_version", lambda _: "2.0.0")
    monkeypatch.setattr(bw_cli, "save_config", lambda _: None)
    monkeypatch.setattr(bw_cli, "get_env_path", lambda: "/tmp/.env")
    monkeypatch.setattr(
        bw_cli.bw,
        "fetch_bitwarden_secrets",
        lambda **kwargs: ({"SAFE_KEY": "inert-value"}, []),
    )

    args = argparse.Namespace(
        access_token="",
        server_url="https://vault.bitwarden.com",
        project_id="proj-1",
    )
    rc = bw_cli.cmd_setup(args)

    assert rc == 0
    assert bw_env == {"CUSTOM_BW_TOKEN": "0.injected-before-dotenv"}


def test_bw_sync_redacts_invalid_name_warning(monkeypatch, capsys):
    token = "0.synthetic-bw-sync-warning-77468"
    monkeypatch.setattr(
        bw_cli,
        "load_config",
        lambda: {
            "secrets": {
                "bitwarden": {
                    "enabled": True,
                    "access_token_env": "BWS_ACCESS_TOKEN",
                    "project_id": "proj-1",
                },
            },
        },
    )
    monkeypatch.setenv("BWS_ACCESS_TOKEN", token)
    monkeypatch.setattr(
        bw_cli.bw,
        "fetch_bitwarden_secrets",
        lambda **kwargs: (
            {"SAFE_KEY": "inert-value"},
            [f"Skipping secret {token!r}: not a valid env-var name"],
        ),
    )

    rc = bw_cli.cmd_sync(argparse.Namespace(apply=False))

    assert rc == 0
    output = capsys.readouterr().out
    assert token not in output
    assert "Skipping secret '<redacted>'" in output


def test_bw_token_non_tty_prefers_managed_custom_env_over_dotenv(
    bw_env, monkeypatch, tmp_path
):
    managed = tmp_path / "managed"
    managed.mkdir()
    (managed / "config.yaml").write_text(
        "secrets:\n  bitwarden:\n    access_token_env: CUSTOM_BW_TOKEN\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("HERMES_MANAGED_DIR", str(managed))
    from hermes_cli import managed_scope

    managed_scope.invalidate_managed_cache()
    monkeypatch.setattr(
        bw_cli,
        "load_config",
        lambda: {
            "secrets": {
                "bitwarden": {
                    "enabled": True,
                    "access_token_env": "CUSTOM_BW_TOKEN",
                    "project_id": "proj-1",
                },
            },
        },
    )
    monkeypatch.setattr("sys.stdin.isatty", lambda: False)
    monkeypatch.setenv("CUSTOM_BW_TOKEN", "0.injected-managed-before-dotenv")
    capture_pre_dotenv_rotation_inputs(
        ["hermes", "secrets", "bitwarden", "token"], config={}
    )
    assert (
        get_pre_dotenv_rotation_input("CUSTOM_BW_TOKEN")
        == "0.injected-managed-before-dotenv"
    )
    monkeypatch.setenv("CUSTOM_BW_TOKEN", "0.stale-managed-dotenv")
    monkeypatch.setattr(bw_cli.bw, "clear_caches", lambda *a, **kw: None)

    rc = bw_cli.cmd_token(_bw_args(no_verify=True))

    assert rc == 0
    assert bw_env == {"CUSTOM_BW_TOKEN": "0.injected-managed-before-dotenv"}


# ---------------------------------------------------------------------------
# 1Password
# ---------------------------------------------------------------------------


def _op_args(**overrides):
    return argparse.Namespace(
        token=overrides.get("token", ""),
        no_verify=overrides.get("no_verify", False),
    )


@pytest.fixture
def op_env(monkeypatch, tmp_path):
    saved = {}
    monkeypatch.setattr(op_cli, "load_config", lambda: {
        "secrets": {"onepassword": {
            "enabled": True,
            "service_account_token_env": "OP_SERVICE_ACCOUNT_TOKEN",
        }},
    })
    monkeypatch.setattr(
        op_cli, "save_env_value",
        lambda name, value: saved.__setitem__(name, value),
    )
    monkeypatch.setattr(op_cli, "get_env_path", lambda: tmp_path / ".env")
    monkeypatch.setattr(
        op_cli.op_src, "find_op", lambda binary_path="": Path("/fake/op")
    )
    return saved




def test_op_token_non_tty_requires_flag(op_env, monkeypatch):
    monkeypatch.setattr("sys.stdin.isatty", lambda: False)
    rc = op_cli.cmd_token(_op_args())
    assert rc == 1
    assert op_env == {}


def test_op_setup_warns_when_token_is_in_argv(op_env, monkeypatch, capsys):
    monkeypatch.setattr(op_cli, "_op_version", lambda _: "2.0.0")
    monkeypatch.setattr(op_cli, "save_config", lambda _: None)
    monkeypatch.setattr(op_cli, "get_env_path", lambda: "/tmp/.env")
    args = argparse.Namespace(
        account="",
        token_env="",
        token="ops-from-argv",
        binary_path="",
    )

    rc = op_cli.cmd_setup(args)

    assert rc == 0
    assert op_env == {"OP_SERVICE_ACCOUNT_TOKEN": "ops-from-argv"}
    output = capsys.readouterr().out
    assert "process listings" in output
    assert "ops-from-argv" not in output


@pytest.mark.parametrize("bad_name", ["BAD NAME", "1BAD", "BAD=INJECT"])
def test_op_setup_malformed_token_env_uses_default(op_env, monkeypatch, bad_name):
    monkeypatch.setattr(
        op_cli,
        "load_config",
        lambda: {
            "secrets": {
                "onepassword": {"service_account_token_env": bad_name}
            }
        },
    )
    monkeypatch.setattr(op_cli, "_op_version", lambda _: "2.0.0")
    monkeypatch.setattr(op_cli, "save_config", lambda _: None)
    monkeypatch.setattr(op_cli, "get_env_path", lambda: "/tmp/.env")

    args = argparse.Namespace(
        account="",
        token_env="",
        token="ops-default-env",
        binary_path="",
    )

    assert op_cli.cmd_setup(args) == 0
    assert op_env == {"OP_SERVICE_ACCOUNT_TOKEN": "ops-default-env"}


def test_op_setup_interactive_prefers_captured_token_over_stale_dotenv(
    op_env, monkeypatch
):
    monkeypatch.setattr("sys.stdin.isatty", lambda: True)
    monkeypatch.setenv("OP_SERVICE_ACCOUNT_TOKEN", "ops-injected-interactive")
    capture_pre_dotenv_rotation_inputs(
        ["hermes", "secrets", "onepassword", "setup"],
        config={"secrets": {"onepassword": {}}},
    )
    monkeypatch.setenv("OP_SERVICE_ACCOUNT_TOKEN", "ops-stale-dotenv")
    monkeypatch.setattr(op_cli, "_op_version", lambda _: "2.0.0")
    monkeypatch.setattr(op_cli, "save_config", lambda _: None)

    args = argparse.Namespace(
        account="",
        token_env="",
        token="",
        binary_path="",
    )

    rc = op_cli.cmd_setup(args)

    assert rc == 0
    assert op_env == {"OP_SERVICE_ACCOUNT_TOKEN": "ops-injected-interactive"}


def test_op_setup_non_tty_prefers_injected_token_over_stale_dotenv(
    op_env, monkeypatch
):
    monkeypatch.setattr("sys.stdin.isatty", lambda: False)
    monkeypatch.setenv("OP_SERVICE_ACCOUNT_TOKEN", "ops-injected-before-dotenv")
    capture_pre_dotenv_rotation_inputs(
        ["hermes", "secrets", "onepassword", "setup"],
        config={"secrets": {"onepassword": {}}},
    )
    monkeypatch.setenv("OP_SERVICE_ACCOUNT_TOKEN", "ops-stale-dotenv")
    monkeypatch.setattr(op_cli, "_op_version", lambda _: "2.0.0")
    monkeypatch.setattr(op_cli, "save_config", lambda _: None)
    monkeypatch.setattr(op_cli, "get_env_path", lambda: "/tmp/.env")

    args = argparse.Namespace(
        account="",
        token_env="",
        token="",
        binary_path="",
    )

    rc = op_cli.cmd_setup(args)

    assert rc == 0
    assert op_env == {"OP_SERVICE_ACCOUNT_TOKEN": "ops-injected-before-dotenv"}


@pytest.mark.parametrize("provider", ["onepassword", "op", "1password"])
@pytest.mark.parametrize(
    "token_env_args",
    [
        ["--token-env", "CUSTOM_OP_TOKEN"],
        ["--token-env=CUSTOM_OP_TOKEN"],
    ],
)
def test_op_setup_non_tty_prefers_cli_custom_token_over_stale_dotenv(
    op_env, monkeypatch, provider, token_env_args
):
    monkeypatch.setattr("sys.stdin.isatty", lambda: False)
    monkeypatch.setenv("CUSTOM_OP_TOKEN", "ops-injected-cli-before-dotenv")
    capture_pre_dotenv_rotation_inputs(
        ["hermes", "secrets", provider, "setup", *token_env_args],
        config={"secrets": {"onepassword": {}}},
    )
    monkeypatch.setenv("CUSTOM_OP_TOKEN", "ops-stale-cli-dotenv")
    monkeypatch.setattr(op_cli, "_op_version", lambda _: "2.0.0")
    monkeypatch.setattr(op_cli, "save_config", lambda _: None)
    monkeypatch.setattr(op_cli, "get_env_path", lambda: "/tmp/.env")

    args = argparse.Namespace(
        account="",
        token_env="CUSTOM_OP_TOKEN",
        token="",
        binary_path="",
    )

    rc = op_cli.cmd_setup(args)

    assert rc == 0
    assert op_env == {"CUSTOM_OP_TOKEN": "ops-injected-cli-before-dotenv"}


@pytest.mark.parametrize("template", ["${TOKEN_ENV_NAME}", "${env:TOKEN_ENV_NAME}"])
@pytest.mark.parametrize(
    "op_assignments",
    [
        'TOKEN_ENV_NAME="CUSTOM_OP_TOKEN" # trailing comment\n',
        "TOKEN_ENV_NAME='CUSTOM_OP_TOKEN' # trailing comment\n",
        "SECOND_NAME=CUSTOM_OP_TOKEN\n"
        "TOKEN_ENV_NAME=${SECOND_NAME}\n",
    ],
)
def test_op_setup_non_tty_resolves_op_env_name_binding_before_stale_dotenv(
    op_env, monkeypatch, tmp_path, template, op_assignments
):
    user_env = tmp_path / ".env"
    user_env.write_text(
        "CUSTOM_OP_TOKEN=ops-stale-user-dotenv\n", encoding="utf-8"
    )
    op_dotenv = tmp_path / ".op.env"
    op_dotenv.write_text(
        op_assignments + "CUSTOM_OP_TOKEN=ops-stale-op-dotenv\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        op_cli,
        "load_config",
        lambda: {
            "secrets": {
                "onepassword": {
                    "enabled": True,
                    "service_account_token_env": "CUSTOM_OP_TOKEN",
                },
            },
        },
    )
    monkeypatch.setattr("sys.stdin.isatty", lambda: False)
    monkeypatch.delenv("TOKEN_ENV_NAME", raising=False)
    monkeypatch.delenv("SECOND_NAME", raising=False)
    monkeypatch.setenv("CUSTOM_OP_TOKEN", "ops-injected-before-dotenv")
    capture_pre_dotenv_rotation_inputs(
        ["hermes", "secrets", "onepassword", "setup"],
        config={
            "secrets": {
                "onepassword": {
                    "service_account_token_env": template,
                },
            },
        },
        dotenv_sources=[(user_env, True), (op_dotenv, False)],
    )
    monkeypatch.setenv("CUSTOM_OP_TOKEN", "ops-stale-after-dotenv")
    monkeypatch.setattr(op_cli, "_op_version", lambda _: "2.0.0")
    monkeypatch.setattr(op_cli, "save_config", lambda _: None)
    monkeypatch.setattr(op_cli, "get_env_path", lambda: "/tmp/.env")

    rc = op_cli.cmd_setup(
        argparse.Namespace(
            account="",
            token_env="",
            token="",
            binary_path="",
        )
    )

    assert rc == 0
    assert op_env == {"CUSTOM_OP_TOKEN": "ops-injected-before-dotenv"}


def test_op_token_non_tty_uses_token_env(op_env, monkeypatch, capsys):
    monkeypatch.setattr("sys.stdin.isatty", lambda: False)
    monkeypatch.setenv("OP_SERVICE_ACCOUNT_TOKEN", "ops-from-environment")
    monkeypatch.setattr(op_cli.op_src, "clear_caches", lambda *a, **kw: None)

    rc = op_cli.cmd_token(_op_args(no_verify=True))

    assert rc == 0
    assert op_env == {"OP_SERVICE_ACCOUNT_TOKEN": "ops-from-environment"}
    output = capsys.readouterr().out
    assert "No TTY" not in output
    assert "ops-from-environment" not in output


def test_op_token_non_tty_prefers_injected_value_over_dotenv(
    op_env, monkeypatch, capsys
):
    monkeypatch.setattr("sys.stdin.isatty", lambda: False)
    monkeypatch.setenv("OP_SERVICE_ACCOUNT_TOKEN", "ops-injected-before-dotenv")
    capture_pre_dotenv_rotation_inputs(
        ["hermes", "secrets", "onepassword", "token"],
        config={"secrets": {"onepassword": {}}},
    )
    monkeypatch.setenv("OP_SERVICE_ACCOUNT_TOKEN", "ops-stale-dotenv")
    monkeypatch.setattr(op_cli.op_src, "clear_caches", lambda *a, **kw: None)

    rc = op_cli.cmd_token(_op_args(no_verify=True))

    assert rc == 0
    assert op_env == {"OP_SERVICE_ACCOUNT_TOKEN": "ops-injected-before-dotenv"}
    assert "ops-stale-dotenv" not in capsys.readouterr().out


def test_op_token_non_tty_prefers_injected_custom_env_over_dotenv(
    op_env, monkeypatch
):
    monkeypatch.setattr(
        op_cli,
        "load_config",
        lambda: {
            "secrets": {
                "onepassword": {
                    "enabled": True,
                    "service_account_token_env": "CUSTOM_OP_TOKEN",
                },
            },
        },
    )
    monkeypatch.setattr("sys.stdin.isatty", lambda: False)
    monkeypatch.setenv("CUSTOM_OP_TOKEN", "ops-injected-custom-before-dotenv")
    capture_pre_dotenv_rotation_inputs(
        ["hermes", "secrets", "onepassword", "token"],
        config={
            "secrets": {
                "onepassword": {
                    "service_account_token_env": "CUSTOM_OP_TOKEN",
                },
            },
        },
    )
    monkeypatch.setenv("CUSTOM_OP_TOKEN", "ops-stale-custom-dotenv")
    monkeypatch.setattr(op_cli.op_src, "clear_caches", lambda *a, **kw: None)

    rc = op_cli.cmd_token(_op_args(no_verify=True))

    assert rc == 0
    assert op_env == {"CUSTOM_OP_TOKEN": "ops-injected-custom-before-dotenv"}


@pytest.mark.parametrize("template", ["${TOKEN_ENV_NAME}", "${env:TOKEN_ENV_NAME}"])
@pytest.mark.parametrize(
    "op_assignments",
    [
        'TOKEN_ENV_NAME="CUSTOM_OP_TOKEN" # trailing comment\n',
        "TOKEN_ENV_NAME='CUSTOM_OP_TOKEN' # trailing comment\n",
        "SECOND_NAME=CUSTOM_OP_TOKEN\n"
        "TOKEN_ENV_NAME=${SECOND_NAME}\n",
    ],
)
def test_op_token_non_tty_resolves_op_env_name_binding_before_stale_dotenv(
    op_env, monkeypatch, tmp_path, template, op_assignments
):
    user_env = tmp_path / ".env"
    user_env.write_text(
        "CUSTOM_OP_TOKEN=ops-stale-user-dotenv\n", encoding="utf-8"
    )
    op_dotenv = tmp_path / ".op.env"
    op_dotenv.write_text(
        op_assignments + "CUSTOM_OP_TOKEN=ops-stale-op-dotenv\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        op_cli,
        "load_config",
        lambda: {
            "secrets": {
                "onepassword": {
                    "enabled": True,
                    "service_account_token_env": "CUSTOM_OP_TOKEN",
                },
            },
        },
    )
    monkeypatch.setattr("sys.stdin.isatty", lambda: False)
    monkeypatch.delenv("TOKEN_ENV_NAME", raising=False)
    monkeypatch.delenv("SECOND_NAME", raising=False)
    monkeypatch.setenv("CUSTOM_OP_TOKEN", "ops-injected-before-dotenv")
    capture_pre_dotenv_rotation_inputs(
        ["hermes", "secrets", "onepassword", "token"],
        config={
            "secrets": {
                "onepassword": {
                    "service_account_token_env": template,
                },
            },
        },
        dotenv_sources=[(user_env, True), (op_dotenv, False)],
    )
    monkeypatch.setenv("CUSTOM_OP_TOKEN", "ops-stale-after-dotenv")
    monkeypatch.setattr(op_cli.op_src, "clear_caches", lambda *a, **kw: None)

    rc = op_cli.cmd_token(_op_args(no_verify=True))

    assert rc == 0
    assert op_env == {"CUSTOM_OP_TOKEN": "ops-injected-before-dotenv"}


def test_op_token_resolves_managed_dotenv_token_name_before_stale_value(
    op_env, monkeypatch, tmp_path
):
    managed = tmp_path / "managed"
    managed.mkdir()
    (managed / "config.yaml").write_text(
        "secrets:\n"
        "  onepassword:\n"
        "    service_account_token_env: '${TOKEN_ENV_NAME}'\n",
        encoding="utf-8",
    )
    managed_env = managed / ".env"
    managed_env.write_text(
        "TOKEN_ENV_NAME=CUSTOM_OP_TOKEN\n"
        "CUSTOM_OP_TOKEN=ops-stale-managed-dotenv\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("HERMES_MANAGED_DIR", str(managed))
    from hermes_cli import managed_scope

    managed_scope.invalidate_managed_cache()
    monkeypatch.setattr(
        op_cli,
        "load_config",
        lambda: {
            "secrets": {
                "onepassword": {
                    "enabled": True,
                    "service_account_token_env": "CUSTOM_OP_TOKEN",
                },
            },
        },
    )
    monkeypatch.setattr("sys.stdin.isatty", lambda: False)
    monkeypatch.delenv("TOKEN_ENV_NAME", raising=False)
    monkeypatch.setenv("CUSTOM_OP_TOKEN", "ops-injected-before-dotenv")
    capture_pre_dotenv_rotation_inputs(
        ["hermes", "secrets", "onepassword", "token"],
        config={},
        dotenv_sources=[(managed_env, True)],
    )
    monkeypatch.setenv("CUSTOM_OP_TOKEN", "ops-stale-managed-dotenv")
    monkeypatch.setattr(op_cli.op_src, "clear_caches", lambda *a, **kw: None)

    rc = op_cli.cmd_token(_op_args(no_verify=True))

    assert rc == 0
    assert op_env == {"CUSTOM_OP_TOKEN": "ops-injected-before-dotenv"}


def test_op_token_non_tty_prefers_managed_custom_env_over_dotenv(
    op_env, monkeypatch, tmp_path
):
    managed = tmp_path / "managed"
    managed.mkdir()
    (managed / "config.yaml").write_text(
        "secrets:\n  onepassword:\n    service_account_token_env: CUSTOM_OP_TOKEN\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("HERMES_MANAGED_DIR", str(managed))
    from hermes_cli import managed_scope

    managed_scope.invalidate_managed_cache()
    monkeypatch.setattr(
        op_cli,
        "load_config",
        lambda: {
            "secrets": {
                "onepassword": {
                    "enabled": True,
                    "service_account_token_env": "CUSTOM_OP_TOKEN",
                },
            },
        },
    )
    monkeypatch.setattr("sys.stdin.isatty", lambda: False)
    monkeypatch.setenv("CUSTOM_OP_TOKEN", "ops-injected-managed-before-dotenv")
    capture_pre_dotenv_rotation_inputs(
        ["hermes", "secrets", "onepassword", "token"], config={}
    )
    assert (
        get_pre_dotenv_rotation_input("CUSTOM_OP_TOKEN")
        == "ops-injected-managed-before-dotenv"
    )
    monkeypatch.setenv("CUSTOM_OP_TOKEN", "ops-stale-managed-dotenv")
    monkeypatch.setattr(op_cli.op_src, "clear_caches", lambda *a, **kw: None)

    rc = op_cli.cmd_token(_op_args(no_verify=True))

    assert rc == 0
    assert op_env == {"CUSTOM_OP_TOKEN": "ops-injected-managed-before-dotenv"}


def test_op_token_verify_success_redacts_provider_identity(
    op_env, monkeypatch, capsys
):
    token = "ops.synthetic-op-identity-77468"
    monkeypatch.setattr(
        op_cli.subprocess,
        "run",
        lambda *a, **kw: mock.Mock(
            returncode=0,
            stdout=f"service-account {token}",
            stderr="",
        ),
    )

    rc = op_cli.cmd_token(_op_args(token=token))

    assert rc == 0
    assert op_env == {"OP_SERVICE_ACCOUNT_TOKEN": token}
    output = capsys.readouterr().out
    assert token not in output
    assert "service-account <redacted>" in output


def test_op_token_verify_failure_does_not_echo_provider_output(
    op_env, monkeypatch, capsys
):
    token = "ops.synthetic-op-failure-77468"
    monkeypatch.setattr(
        op_cli.subprocess,
        "run",
        lambda *a, **kw: mock.Mock(
            returncode=1,
            stdout=f"provider rejected {token}",
            stderr="invalid token",
        ),
    )

    rc = op_cli.cmd_token(_op_args(token=token))

    assert rc == 1
    assert op_env == {}
    output = capsys.readouterr().out
    assert token not in output
    assert "New token was rejected by op" in output


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


def test_op_token_warns_when_token_is_in_argv(op_env, monkeypatch, capsys):
    monkeypatch.setattr(op_cli.op_src, "clear_caches", lambda *a, **kw: None)

    rc = op_cli.cmd_token(_op_args(token="ops-from-argv", no_verify=True))

    assert rc == 0
    assert op_env == {"OP_SERVICE_ACCOUNT_TOKEN": "ops-from-argv"}
    output = capsys.readouterr().out
    assert "process listings" in output
    assert "ops-from-argv" not in output
