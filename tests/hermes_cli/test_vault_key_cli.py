"""`hermes vault key migrate --to keychain|file` through the real argparse tree: the key moves, the
vault still opens, and ``vault.key_storage`` follows so a recreated vault lands in the same place."""

from __future__ import annotations

import argparse

import keyring
import pytest
from keyring.backend import KeyringBackend
from keyring.backends import fail

from agent.vault_store import VaultStore


class _MemoryKeyring(KeyringBackend):
    priority = 1

    def __init__(self):
        super().__init__()
        self.entries: dict[tuple[str, str], str] = {}

    def get_password(self, service, username):
        return self.entries.get((service, username))

    def set_password(self, service, username, password):
        self.entries[(service, username)] = password

    def delete_password(self, service, username):
        self.entries.pop((service, username))


@pytest.fixture
def hermes(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    backend = _MemoryKeyring()
    keyring.set_keyring(backend)
    from hermes_cli.subcommands.vault import build_vault_parser

    parser = argparse.ArgumentParser(prog="hermes")
    build_vault_parser(parser.add_subparsers())

    def run(*argv):
        args = parser.parse_args(["vault", *argv])
        args.func(args)

    yield run, backend, tmp_path
    keyring.set_keyring(fail.Keyring())


def test_vault_key_migrate_moves_the_key_and_records_the_choice(hermes, capsys):
    run, backend, home = hermes
    item = VaultStore().add_item("login", "Example", {"identifier_type": "email", "identifier": "jane@example.com",
                                                      "password": "dummy-pw-cli-9"}, origin="https://example.com").id
    from hermes_cli.config import load_config

    run("key", "migrate", "--to", "keychain")
    assert not (home / "vault" / "vault.key").exists() and len(backend.entries) == 1
    assert load_config()["vault"]["key_storage"] == "keychain"
    run("key")
    assert "OS keychain" in capsys.readouterr().out

    run("key", "migrate", "--to", "file")
    assert (home / "vault" / "vault.key").exists() and backend.entries == {}
    assert load_config()["vault"]["key_storage"] == "file"
    assert VaultStore().resolve_secret(item)["password"] == "dummy-pw-cli-9"


@pytest.mark.parametrize("target", ["keychain", "file"])
def test_migrating_an_empty_vault_says_no_key_exists_yet(hermes, capsys, target):
    run, backend, home = hermes
    run("key", "migrate", "--to", target)
    out = capsys.readouterr().out
    assert "No vault key exists yet" in out
    assert "decrypted" not in out and "moved" not in out, "nothing was moved or verified"
    assert backend.entries == {} and not (home / "vault" / "vault.key").exists()
