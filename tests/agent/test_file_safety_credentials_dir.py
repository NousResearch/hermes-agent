"""``~/.hermes/credentials/`` is denied to the agent's file tools.

master.key + store.enc side by side IS the plaintext — the exact hazard
``vault/`` is already denied for — so the whole directory is both read- and
write-denied. Values reach execution only through the credentials tool and
execute_code's ``credentials=`` parameter.
"""

from __future__ import annotations

import agent.file_safety as fs


def _layout(tmp_path, monkeypatch):
    home = tmp_path / "hermes_home"
    (home / "credentials").mkdir(parents=True)
    monkeypatch.setattr(fs, "_hermes_home_path", lambda: home)
    monkeypatch.setattr(fs, "_hermes_root_path", lambda: tmp_path / "hermes_root")
    return home


def test_credentials_dir_and_files_are_read_denied(tmp_path, monkeypatch):
    home = _layout(tmp_path, monkeypatch)
    key = home / "credentials" / "master.key"
    store = home / "credentials" / "store.enc"
    key.write_text("fkey", encoding="utf-8")
    store.write_bytes(b"ciphertext")

    for path in (home / "credentials", key, store):
        error = fs.get_read_block_error(str(path))
        assert error, f"not read-denied: {path}"
        assert "credential store" in error


def test_credentials_dir_is_write_denied(tmp_path, monkeypatch):
    home = _layout(tmp_path, monkeypatch)
    store = home / "credentials" / "store.enc"
    store.write_bytes(b"ciphertext")

    assert fs.is_write_denied(str(store))
    assert fs.is_write_denied(str(home / "credentials" / "master.key"))
    assert fs.get_write_denied_error(str(store))


def test_other_home_files_stay_readable(tmp_path, monkeypatch):
    home = _layout(tmp_path, monkeypatch)
    note = home / "notes.txt"
    note.write_text("plain", encoding="utf-8")
    assert fs.get_read_block_error(str(note)) is None