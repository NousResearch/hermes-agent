"""Docs + CLI help: startup apply hydrates os.environ; vault holds identity.

Issue #107698 — custody warning and Passwords & Logins pointer must be
present in the secrets user-guide and in Bitwarden/1Password --apply help.
No SecretSource behavior change. Reads files from repo root via Path().
"""
from pathlib import Path


def _read(rel: str) -> str:
    return Path(rel).read_text(encoding="utf-8")


def _vault_pointer(text: str) -> bool:
    return (
        "hermes vault" in text
        or "Passwords & Logins" in text
        or "credential-vault" in text
    )


def test_secrets_index_warns_apply_hydrates_os_environ() -> None:
    doc = _read("website/docs/user-guide/secrets/index.md")
    assert "os.environ" in doc
    assert "printenv" in doc
    assert "display-layer" in doc
    # Process env is observable to the model/tools/children — not a boundary.
    assert "children" in doc or "model" in doc or "tools" in doc


def test_secrets_index_provider_keys_not_identity_vault_pointer() -> None:
    doc = _read("website/docs/user-guide/secrets/index.md")
    assert "provider keys" in doc
    assert "site passwords" in doc
    assert _vault_pointer(doc)


def test_bitwarden_sync_apply_hydrates_this_shell_and_vault() -> None:
    doc = _read("website/docs/user-guide/secrets/bitwarden.md")
    cli = doc.split("## CLI", 1)[1]
    assert "sync --apply" in cli
    assert "this shell" in cli
    assert "not only" in cli
    assert "site passwords" in cli or _vault_pointer(cli)


def test_onepassword_sync_apply_hydrates_this_shell_and_vault() -> None:
    doc = _read("website/docs/user-guide/secrets/onepassword.md")
    cli = doc.split("## CLI", 1)[1]
    assert "sync --apply" in cli
    assert "this shell" in cli
    assert "not only" in cli
    assert "site passwords" in cli or _vault_pointer(cli)


def test_command_startup_apply_custody_and_vault() -> None:
    doc = _read("website/docs/user-guide/secrets/command.md")
    assert "sync --apply" not in doc  # command source has no apply CLI
    assert "os.environ" in doc
    assert "display-layer" in doc or "printenv" in doc
    assert "site passwords" in doc
    assert _vault_pointer(doc)


def test_bitwarden_apply_flag_help_custody() -> None:
    src = _read("hermes_cli/secrets_cli.py")
    assert "visible to this process and children" in src
    assert "password-blind vault" in src


def test_onepassword_apply_flag_help_custody() -> None:
    src = _read("hermes_cli/onepassword_secrets_cli.py")
    assert "visible to this process and children" in src
    assert "password-blind vault" in src
