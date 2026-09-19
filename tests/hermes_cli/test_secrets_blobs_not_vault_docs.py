"""Docs: statement / passport / tax PDFs are blobs, not vault items.

Issue #107705 — docs plus the VAULT_KINDS boundary.
Reads documentation from repo root via Path().
"""
from pathlib import Path

from agent.vault_store import VAULT_KINDS


def _read(rel: str) -> str:
    return Path(rel).read_text(encoding="utf-8")


def test_secrets_index_blobs_not_vault_or_apply() -> None:
    doc = _read("website/docs/user-guide/secrets/index.md")
    low = doc.lower()
    assert "passport" in low
    assert "statement" in low
    assert "hermes vault add" in doc
    assert "read_file" in doc
    assert "Skyflow" in doc or "VGS" in doc
    assert "plugins-skills-and-skins" in doc
    assert "Plaid" in doc or "plaid" in low


def test_credential_vault_excludes_file_blobs() -> None:
    doc = _read("website/docs/user-guide/features/credential-vault.md")
    low = doc.lower()
    assert "passport" in low or "statement" in low
    assert "read_file" in doc
    assert "get_secret" in doc or "not a vault" in low or "not vault" in low


def test_vault_kinds_exclude_file_blobs() -> None:
    assert {"login", "payment", "address"} <= set(VAULT_KINDS)
    assert {"document", "file", "blob"}.isdisjoint(VAULT_KINDS)
