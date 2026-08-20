"""Setup wizards must validate replacement credentials before persisting them."""

from __future__ import annotations

import argparse
from pathlib import Path

from hermes_cli import onepassword_secrets_cli as op_cli
from hermes_cli import secrets_cli as bw_cli


def _bw_args() -> argparse.Namespace:
    return argparse.Namespace(
        access_token="0.candidate",
        server_url="https://vault.bitwarden.com",
        project_id="project-1",
    )


def _prepare_bitwarden(monkeypatch, tmp_path, events: list[str]) -> None:
    monkeypatch.setattr(
        bw_cli.bw,
        "find_bws",
        lambda install_if_missing=False: Path("/fake/bws"),
    )
    monkeypatch.setattr(bw_cli, "_bws_version", lambda binary: "2.0.0")
    monkeypatch.setattr(
        bw_cli,
        "load_config",
        lambda: {"secrets": {"bitwarden": {"access_token_env": "BWS_ACCESS_TOKEN"}}},
    )
    monkeypatch.setattr(bw_cli, "get_env_path", lambda: tmp_path / ".env")
    monkeypatch.setattr(
        bw_cli,
        "save_env_value",
        lambda name, value: events.append("save_env"),
    )
    monkeypatch.setattr(
        bw_cli,
        "save_config",
        lambda config: events.append("save_config"),
    )


def test_bitwarden_setup_rejected_token_does_not_replace_working_token(
    monkeypatch, tmp_path
):
    events: list[str] = []
    _prepare_bitwarden(monkeypatch, tmp_path, events)
    monkeypatch.setenv("BWS_ACCESS_TOKEN", "0.working")

    def reject_candidate(**kwargs):
        events.append("fetch")
        raise RuntimeError("candidate rejected")

    monkeypatch.setattr(bw_cli.bw, "fetch_bitwarden_secrets", reject_candidate)

    assert bw_cli.cmd_setup(_bw_args()) == 1
    assert events == ["fetch"]
    assert bw_cli.os.environ["BWS_ACCESS_TOKEN"] == "0.working"


def test_bitwarden_setup_persists_only_after_successful_fetch(monkeypatch, tmp_path):
    events: list[str] = []
    _prepare_bitwarden(monkeypatch, tmp_path, events)

    def accept_candidate(**kwargs):
        assert kwargs["access_token"] == "0.candidate"
        events.append("fetch")
        return {"EXAMPLE_SECRET": "value"}, []

    monkeypatch.setattr(bw_cli.bw, "fetch_bitwarden_secrets", accept_candidate)

    assert bw_cli.cmd_setup(_bw_args()) == 0
    assert events == ["fetch", "save_env", "save_config"]
    assert bw_cli.os.environ["BWS_ACCESS_TOKEN"] == "0.candidate"


def _op_args() -> argparse.Namespace:
    return argparse.Namespace(
        account="team.1password.com",
        binary_path="",
        token="candidate-token",
        token_env="OP_SERVICE_ACCOUNT_TOKEN",
    )


def _prepare_onepassword(monkeypatch, tmp_path, events: list[str]) -> None:
    monkeypatch.setattr(
        op_cli.op_src, "find_op", lambda binary_path="": Path("/fake/op")
    )
    monkeypatch.setattr(op_cli, "_op_version", lambda binary: "2.30.0")
    monkeypatch.setattr(op_cli, "load_config", lambda: {})
    monkeypatch.setattr(op_cli, "get_env_path", lambda: tmp_path / ".env")
    monkeypatch.setattr(
        op_cli,
        "save_env_value",
        lambda name, value: events.append("save_env"),
    )
    monkeypatch.setattr(
        op_cli,
        "save_config",
        lambda config: events.append("save_config"),
    )


def test_onepassword_setup_rejected_token_does_not_replace_working_token(
    monkeypatch, tmp_path
):
    events: list[str] = []
    _prepare_onepassword(monkeypatch, tmp_path, events)
    monkeypatch.setenv("OP_SERVICE_ACCOUNT_TOKEN", "working-token")

    def reject_candidate(binary, account, *, token_value=""):
        assert token_value == "candidate-token"
        events.append("whoami")
        return None

    monkeypatch.setattr(op_cli, "_op_whoami", reject_candidate)

    assert op_cli.cmd_setup(_op_args()) == 1
    assert events == ["whoami"]
    assert op_cli.os.environ["OP_SERVICE_ACCOUNT_TOKEN"] == "working-token"


def test_onepassword_setup_persists_only_after_successful_whoami(monkeypatch, tmp_path):
    events: list[str] = []
    _prepare_onepassword(monkeypatch, tmp_path, events)

    def accept_candidate(binary, account, *, token_value=""):
        assert token_value == "candidate-token"
        events.append("whoami")
        return "service-account@example.com"

    monkeypatch.setattr(op_cli, "_op_whoami", accept_candidate)

    assert op_cli.cmd_setup(_op_args()) == 0
    assert events == ["whoami", "save_env", "save_config"]
    assert op_cli.os.environ["OP_SERVICE_ACCOUNT_TOKEN"] == "candidate-token"
