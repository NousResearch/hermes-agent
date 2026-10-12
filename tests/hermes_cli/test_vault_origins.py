"""Human CLI authorization of exact checkout origins stays metadata-only."""

import argparse

import pytest
from rich.console import Console

from agent.vault_store import VaultStore
from hermes_cli.subcommands.vault import build_vault_parser


CARD = {"card_number": "4111111111111111", "exp_month": "7", "exp_year": "2029", "cvc": "cvc-cli-canary"}


@pytest.fixture
def local_vault(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    store = VaultStore(tmp_path / "vault")
    monkeypatch.setattr("hermes_cli.vault._console", lambda: Console(width=240, highlight=False))
    monkeypatch.setattr(VaultStore, "resolve_secret", lambda *_: pytest.fail("CLI must not resolve secrets"))
    return store


def _run(*argv):
    parser = argparse.ArgumentParser()
    build_vault_parser(parser.add_subparsers())
    args = parser.parse_args(["vault", *argv])
    args.func(args)


def test_cli_add_list_remove_history_and_duplicate_are_metadata_only(local_vault, capsys, monkeypatch):
    from agent.vault_backends.local import LocalLoginBackend

    meta = local_vault.add_item("payment", "Card", CARD, origin="https://primary.test")
    _run("origins", meta.id)
    initial = capsys.readouterr().out
    assert meta.id in initial and "payment" in initial and "https://primary.test" in initial
    _run("origins", meta.id, "--add", "HTTPS://Second.TEST:443/")
    added = capsys.readouterr().out
    assert "https://second.test" in added
    after_add = local_vault._vault_path.read_bytes()
    _run("origins", meta.id, "--add", "https://second.test")
    assert local_vault._vault_path.read_bytes() == after_add
    _run("origins", meta.id)
    listed = capsys.readouterr().out
    assert "https://primary.test" in listed and "https://second.test" in listed
    assert "add" in listed and local_vault.origin_events(meta.id)[0]["timestamp"] in listed
    monkeypatch.setattr("agent.vault_backends.enabled_backends", lambda: [LocalLoginBackend()])
    _run("list")
    all_items = capsys.readouterr().out
    assert "https://primary.test" in all_items and "https://second.test" in all_items
    _run("origins", meta.id, "--remove", "https://second.test")
    removed = capsys.readouterr().out
    assert "remove" in removed
    assert local_vault.get_meta(meta.id).allowed_origins == ("https://primary.test",)
    output = initial + added + listed + all_items + removed
    assert CARD["card_number"] not in output and CARD["cvc"] not in output
    assert "card_number" not in output


@pytest.mark.parametrize("kind,origin,flag,url", [
    (None, None, "--remove", "https://second.test"),
    ("login", "https://primary.test", "--remove", "https://second.test"),
    ("payment", None, "--remove", "https://second.test"),
    ("payment", "https://primary.test", "--remove", "https://primary.test"),
    ("payment", "https://primary.test", "--remove", "https://second.test"),
    ("payment", "https://primary.test", "--add", "https://second.test/checkout"),
    ("payment", "https://primary.test", "--add", "https://u:never-echo-this@second.test"),
])
def test_cli_errors_are_clean_and_leave_encrypted_store_unchanged(local_vault, capsys, kind, origin, flag, url):
    local_vault.add_item("payment", "Card", CARD, origin="https://primary.test")
    handle = "vault_missing"
    if kind is not None:
        secrets = {"payment": CARD, "login": {
            "identifier_type": "username", "identifier": "u", "password": "never-echo-this"}}
        handle = local_vault.add_item(kind, "Item", secrets[kind], origin=origin).id
    before = local_vault._vault_path.read_bytes()
    _run("origins", handle, flag, url)
    output = capsys.readouterr().out
    assert "Error" in output and "Traceback" not in output
    assert "never-echo-this" not in output and CARD["card_number"] not in output
    assert local_vault._vault_path.read_bytes() == before


def test_origins_flags_are_mutually_exclusive(local_vault):
    with pytest.raises(SystemExit) as error:
        _run("origins", "vault_any", "--add", "https://a.test", "--remove", "https://b.test")
    assert error.value.code == 2
