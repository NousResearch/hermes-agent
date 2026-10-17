"""`hermes vault add` rejects a bad origin / authenticator key right after it
is typed and re-prompts, instead of failing only after the whole wizard."""

from types import SimpleNamespace

from agent.vault_store import VaultStore
from hermes_cli import vault as vault_cli


def _run_add(monkeypatch, tmp_path, inputs, hidden):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    prompts = []

    def fake_input(prompt=""):
        prompts.append(prompt)
        return inputs.pop(0)

    def fake_getpass(prompt=""):
        prompts.append(prompt)
        return hidden.pop(0)

    monkeypatch.setattr("builtins.input", fake_input)
    monkeypatch.setattr(vault_cli.getpass, "getpass", fake_getpass)
    vault_cli._cmd_add(SimpleNamespace(kind="login"))
    return home, prompts


def test_bad_origin_reprompts_before_next_field(monkeypatch, tmp_path, capsys):
    home, prompts = _run_add(
        monkeypatch,
        tmp_path,
        ["label", "example.com", "https://example.com", "email", "u@example.com"],
        ["pw", ""],
    )
    origin_prompts = [i for i, p in enumerate(prompts) if p.startswith("Site origin")]
    assert origin_prompts == [1, 2]  # label, bad origin, origin again, then identifier type
    assert "must include a scheme" in capsys.readouterr().out
    (item,) = VaultStore(home / "vault").list_items()
    assert item.origin == "https://example.com"


def test_bad_otp_reprompts(monkeypatch, tmp_path, capsys):
    home, prompts = _run_add(
        monkeypatch,
        tmp_path,
        ["label", "https://example.com", "email", "u@example.com"],
        ["pw", "not base32!", "JBSWY3DPEHPK3PXP"],
    )
    assert sum(p.startswith("Authenticator key") for p in prompts) == 2
    out = capsys.readouterr().out
    assert "base32" in out and "Stored." in out
    (item,) = VaultStore(home / "vault").list_items()
    assert VaultStore(home / "vault").resolve_secret(item.id)["otp_secret"] == "JBSWY3DPEHPK3PXP"
