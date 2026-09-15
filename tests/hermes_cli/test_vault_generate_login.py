from __future__ import annotations

import argparse
import re

import pytest

from agent.vault_store import VaultStore
from hermes_cli import vault as vault_cli


def _parse(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="hermes vault")
    vault_cli.register_cli(parser)
    return parser.parse_args(argv)


def test_generate_login_parser_accepts_only_metadata() -> None:
    args = _parse(
        [
            "generate-login",
            "--origin",
            "https://example.com/login",
            "--identifier",
            "qa@example.com",
            "--label",
            "Disposable QA",
        ]
    )

    assert args.vault_action == "generate-login"
    assert args.origin == "https://example.com/login"
    assert args.identifier == "qa@example.com"
    assert args.identifier_type == "email"
    assert args.label == "Disposable QA"
    assert args.length == 24
    assert not hasattr(args, "password")


def test_generate_login_creates_model_blind_vault_item(
    tmp_path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    store = VaultStore(base_dir=tmp_path / "vault")
    monkeypatch.setattr("agent.vault_store.get_vault_store", lambda: store)
    monkeypatch.setattr(
        "builtins.input",
        lambda *_args, **_kwargs: pytest.fail("generate-login must be non-interactive"),
    )
    monkeypatch.setattr(
        "getpass.getpass",
        lambda *_args, **_kwargs: pytest.fail("generate-login must not ask for a password"),
    )
    args = _parse(
        [
            "generate-login",
            "--origin",
            "https://EXAMPLE.com:443/register",
            "--identifier",
            "qa@example.com",
            "--label",
            "Disposable QA",
        ]
    )

    vault_cli.vault_command(args)

    items = store.list_items()
    assert len(items) == 1
    item = items[0]
    assert item.kind == "login"
    assert item.label == "Disposable QA"
    assert item.origin == "https://example.com"
    assert item.identifier_type == "email"
    assert item.identifier == "qa@example.com"
    assert item.generated is True

    password = store.resolve_secret(item.id)["password"]
    assert len(password) == 24
    assert re.search(r"[a-z]", password)
    assert re.search(r"[A-Z]", password)
    assert re.search(r"[0-9]", password)
    assert re.search(r"[^A-Za-z0-9]", password)

    output = capsys.readouterr().out
    assert item.id in output
    assert item.origin in output
    assert item.identifier in output
    assert password not in output


def test_generate_login_rejects_unsafe_password_lengths() -> None:
    parser = argparse.ArgumentParser(prog="hermes vault")
    vault_cli.register_cli(parser)

    with pytest.raises(SystemExit):
        parser.parse_args(
            [
                "generate-login",
                "--origin",
                "https://example.com",
                "--identifier",
                "qa@example.com",
                "--length",
                "8",
            ]
        )


def test_generate_login_returns_nonzero_when_vault_rejects_metadata(
    tmp_path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    store = VaultStore(base_dir=tmp_path / "vault")
    monkeypatch.setattr("agent.vault_store.get_vault_store", lambda: store)
    args = _parse(
        [
            "generate-login",
            "--origin",
            "not-an-origin",
            "--identifier",
            "qa@example.com",
        ]
    )

    with pytest.raises(SystemExit) as raised:
        vault_cli.vault_command(args)

    assert raised.value.code == 1
    assert store.list_items() == []
    assert "origin must include a scheme" in capsys.readouterr().out
