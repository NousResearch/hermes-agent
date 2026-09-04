"""Measured-work pins for the _load_global_auth_store() memo.

read_credential_pool() -> load_pool() runs _load_global_auth_store() once per
provider row in the /model picker, and the global-store JSON read + parse
cost ~60-100us+ per call even when nothing changed. The memo keyed on the
global auth file's path+content-digest makes repeat reads a dict lookup.

Keyed on content digest rather than mtime: a (path, mtime_ns) signature is
the right idiom for a config cache, but this one guards credential
identity, and an external metadata-preserving copy of auth.json (dotfile
sync, backup/restore) can replace the file's bytes while restoring its
original mtime — a stat-only key would serve stale (or entirely different)
global credentials indefinitely. See the fix that applied the same
stat-vs-digest correction to the Vertex SA credential cache
(agent/vertex_adapter.py._read_sa_file) for the reasoning this mirrors.
"""

from __future__ import annotations

import json
import os

import pytest

import hermes_cli.auth as auth_mod


@pytest.fixture(autouse=True)
def _reset_cache(monkeypatch):
    # raising=False: on pre-fix code the memo attribute doesn't exist (that
    # IS the fix); the reset is a no-op there so the measured-work assertions
    # fail genuinely instead of erroring.
    monkeypatch.setattr(
        auth_mod, "_global_auth_store_cache", None, raising=False
    )
    yield
    monkeypatch.setattr(
        auth_mod, "_global_auth_store_cache", None, raising=False
    )


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

    def test_content_change_re_reads_once(self, tmp_path, monkeypatch):
        """A store file content change on disk invalidates the memo."""
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

        # Real content change (a fresh auth write) -> memo invalidates -> one re-read.
        global_path.write_text(
            json.dumps({"version": 1, "providers": {"openai": {"api_key": "sk-NEW"}}}),
            encoding="utf-8",
        )
        second = auth_mod._load_global_auth_store()
        assert reads["n"] == 2, "content change must force exactly one re-read"
        assert second.get("providers", {}).get("openai") == {"api_key": "sk-NEW"}

    def test_metadata_preserving_replace_is_not_served_stale(self, tmp_path, monkeypatch):
        """A content change that PRESERVES the original mtime (e.g. rsync -a,
        a dotfile-sync tool, or a backup/restore that restores timestamps)
        must still invalidate the memo — a (path, mtime_ns) key would miss
        this indefinitely and keep serving the old global credentials."""
        global_path = _make_global_store(tmp_path)
        monkeypatch.setattr(
            auth_mod, "_global_auth_file_path", lambda: global_path
        )

        first = auth_mod._load_global_auth_store()
        assert first.get("providers", {}).get("openai") == {"api_key": "sk-x"}
        original_stat = global_path.stat()

        # Replace the content, then restore the ORIGINAL mtime/atime exactly
        # — simulates a metadata-preserving external copy landing different
        # bytes at the same path with the same stat signature.
        global_path.write_text(
            json.dumps({"version": 1, "providers": {"openai": {"api_key": "sk-DIFFERENT"}}}),
            encoding="utf-8",
        )
        os.utime(
            global_path,
            ns=(original_stat.st_atime_ns, original_stat.st_mtime_ns),
        )
        assert global_path.stat().st_mtime_ns == original_stat.st_mtime_ns

        second = auth_mod._load_global_auth_store()
        assert second.get("providers", {}).get("openai") == {"api_key": "sk-DIFFERENT"}, (
            "memo served stale credentials after a metadata-preserving "
            "content replace — the cache key did not detect the change"
        )

    def test_absent_global_store_returns_empty_without_error(self, tmp_path, monkeypatch):
        """No global fallback (classic mode) returns {} and stays cheap."""
        missing = tmp_path / "no-such" / "auth.json"
        monkeypatch.setattr(
            auth_mod, "_global_auth_file_path", lambda: missing
        )
        assert auth_mod._load_global_auth_store() == {}
        assert auth_mod._global_auth_store_cache is None
