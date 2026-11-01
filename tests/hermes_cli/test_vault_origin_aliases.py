"""The owner can authorize another site without supplying credentials again."""
import argparse
from unittest.mock import patch

from agent.vault_store import VaultStore


def test_cli_add_origin_uses_existing_item(tmp_path, capsys):
    from hermes_cli.vault import register_cli, vault_command

    store = VaultStore(tmp_path / "vault")
    meta = store.add_item("login", "Example", {
        "identifier_type": "email", "identifier": "person@example.com",
        "password": "synthetic-password",
    }, origin="https://example.com")
    parser = argparse.ArgumentParser()
    register_cli(parser)
    args = parser.parse_args(["add-origin", meta.id, "https://www.example.com"])
    with patch("agent.vault_store.get_vault_store", return_value=store):
        vault_command(args)
    updated = store.get_meta(meta.id)
    assert updated is not None
    assert "https://www.example.com" in updated.allowed_origins
    assert "synthetic-password" not in capsys.readouterr().out
