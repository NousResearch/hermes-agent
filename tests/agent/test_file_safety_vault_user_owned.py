"""~/.hermes/vault without the runtime's credential pair is user space (#131781).

The whole-dir deny exists because vault.key + vault.json.enc side by side mean
key + ciphertext. When neither file exists the dir is the user's own (e.g. an
Obsidian vault): the guards then deny only the two credential filenames, so
note-writing agents can touch the user's notes. With the pair present the
whole-dir deny must hold — including deletes, so the agent cannot manufacture
the downgrade by removing vault.key first.
"""

from __future__ import annotations

from pathlib import Path

import pytest

import agent.file_safety as fs


@pytest.fixture()
def hermes_layout(tmp_path, monkeypatch):
    """Single HERMES_HOME (home == root), patched like the sibling suites."""
    home = tmp_path / "hermes_home"
    home.mkdir(parents=True)
    monkeypatch.setattr(fs, "_hermes_home_path", lambda: home)
    monkeypatch.setattr(fs, "_hermes_root_path", lambda: home)
    return home


def _touch(base: Path, rel: str) -> Path:
    p = base / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("dummy", encoding="utf-8")
    return p


def test_user_owned_vault_is_writable_and_readable(hermes_layout):
    home = hermes_layout
    for rel in ("vault/Notes/inbox.md", "vault/Projects/hermes.md"):
        note = _touch(home, rel)
        assert fs.is_write_denied(str(note)) is False, rel
        assert fs.get_read_block_error(str(note)) is None, rel


def test_credential_filenames_stay_denied_in_user_owned_vault(hermes_layout):
    home = hermes_layout
    (home / "vault").mkdir()
    for name in ("vault.key", "vault.json.enc", "VAULT.KEY"):
        assert fs.is_write_denied(str(home / "vault" / name)), name
        assert fs.get_read_block_error(str(home / "vault" / name)), name


def test_credential_pair_present_keeps_whole_dir_denied(hermes_layout):
    home = hermes_layout
    _touch(home, "vault/vault.json.enc")
    note = _touch(home, "vault/Notes/inbox.md")
    assert fs.is_write_denied(str(note))
    assert fs.get_read_block_error(str(note))
    # The downgrade must not be reachable by deleting the pair: while either
    # file exists, deleting it is itself whole-dir write-denied.
    assert fs.is_write_denied(str(home / "vault" / "vault.json.enc"))


def test_never_initialized_vault_dir_is_user_space_too(hermes_layout):
    note = hermes_layout / "vault" / "note.md"
    assert fs.is_write_denied(str(note)) is False
    assert fs.get_read_block_error(str(note)) is None
