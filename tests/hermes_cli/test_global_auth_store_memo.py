"""Measured-work pins for the _load_global_auth_store() memo.

read_credential_pool() -> load_pool() runs _load_global_auth_store() once per
provider row in the /model picker, and the global-store JSON read + parse
cost ~60-100us+ per call even when nothing changed. The memo keyed on the
global auth file's path+mtime makes repeat reads a dict lookup. The store
only changes when the user authenticates at global scope (writes always go
through _save_auth_store, which touches the file), so the mtime key keeps
the memo freshness-correct.
"""

from __future__ import annotations

import json
import os

import pytest

import hermes_cli.auth as auth_mod


@pytest.fixture(autouse=True)
def _reset_cache():
    auth_mod._global_auth_store_cache = None
    yield
    auth_mod._global_auth_store_cache = None


def _make_global_store(tmp_path) -> "os.PathLike[str]":
    """Write a realistic global auth.json and return its path."""
    path = tmp_path / "global-hermes" / "auth.json"
    path.parent.mkdir(parents=True)
    path.write_text(
        json.dumps(
            {
                "version": 1,
                "providers": {
                    "openai": {"api_key": "sk-x"},
                    "anthropic": {"api_key": "an-x"},
                },
                "credential_pool": {
                    "openai": [{"id": "1", "access_token": "t"}],
                    "anthropic": [{"id": "2", "access_token": "u"}],
                },
            }
        ),
        encoding="utf-8",
    )
    return path


class TestLoadGlobalAuthStoreMemo:
    def test_repeated_calls_read_store_once(self, tmp_path, monkeypatch):
        """Repeated calls must not re-read/re-parse the global store."""
        global_path = _make_global_store(tmp_path)
        monkeypatch.setattr(
            auth_mod, "_global_auth_file_path", lambda: global_path
        )
        reads = {"n": 0}
        orig = auth_mod._load_auth_store

        def counting_load(store_path=None):
            reads["n"] += 1
            return orig(store_path)

        monkeypatch.setattr(auth_mod, "_load_auth_store", counting_load)

        first = auth_mod._load_global_auth_store()
        for _ in range(10):
            auth_mod._load_global_auth_store()
        assert reads["n"] == 1, (
            "repeated calls must be memo hits (store read once), "
            f"got {reads['n']}"
        )
        assert first.get("providers", {}).get("openai") == {"api_key": "sk-x"}

    def test_mtime_change_re_reads_once(self, tmp_path, monkeypatch):
        """A store file change on disk invalidates the memo."""
        global_path = _make_global_store(tmp_path)
        monkeypatch.setattr(
            auth_mod, "_global_auth_file_path", lambda: global_path
        )
        reads = {"n": 0}
        orig = auth_mod._load_auth_store

        def counting_load(store_path=None):
            reads["n"] += 1
            return orig(store_path)

        monkeypatch.setattr(auth_mod, "_load_auth_store", counting_load)

        auth_mod._load_global_auth_store()
        assert reads["n"] == 1

        # Bump the file mtime -> memo invalidates -> re-read once.
        os.utime(global_path, (1_700_000_000, 1_700_000_000))
        auth_mod._load_global_auth_store()
        assert reads["n"] == 2, "mtime change must force exactly one re-read"

    def test_absent_global_store_returns_empty_without_error(self, tmp_path, monkeypatch):
        """No global fallback (classic mode) returns {} and stays cheap."""
        missing = tmp_path / "no-such" / "auth.json"
        monkeypatch.setattr(
            auth_mod, "_global_auth_file_path", lambda: missing
        )
        assert auth_mod._load_global_auth_store() == {}
        assert auth_mod._global_auth_store_cache is None
