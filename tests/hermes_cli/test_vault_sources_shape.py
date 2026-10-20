"""`hermes vault sources --disable` must not crash on a hand-edited non-dict
``vault:`` section in config.yaml (the same YAML-shape hazard class
``_voice_cfg_dict`` guards for voice.*, #19835): the malformed section is
coerced to a dict before the opt-out is written."""

from types import SimpleNamespace

import hermes_yaml as yaml

from agent.vault_store import VaultStore
from hermes_cli.vault import _cmd_sources, vault_command


def _corrupt_vault(home):
    store = VaultStore(home / "vault")
    store.add_item(
        "login",
        "site",
        {"identifier_type": "email", "identifier": "u", "password": "p"},
        origin="https://x.example",
    )
    store._vault_path.write_bytes(store._fernet().encrypt(b"{not json"))


def test_sources_disable_tolerates_scalar_vault_section(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    (home / "config.yaml").write_text("vault: true\n")

    _cmd_sources(SimpleNamespace(enable=None, disable="bitwarden"))

    saved = yaml.safe_load((home / "config.yaml").read_text())
    assert saved["vault"]["bitwarden"]["enabled"] is False


def test_vault_command_surfaces_corrupt_store_as_clean_error(tmp_path, monkeypatch, capsys):
    """A decryptable-but-malformed vault file surfaces through the VaultError
    contract as a clean error line on every subcommand, not a traceback."""
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    _corrupt_vault(home)

    vault_command(SimpleNamespace())  # bare `hermes vault` -> _cmd_list

    out = capsys.readouterr().out
    assert "Error" in out
    assert "corrupted" in out
