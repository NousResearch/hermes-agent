"""Tests for the credential store (proxy/store.py)."""

from __future__ import annotations

import threading

import pytest

from proxy.store import CredentialStore


class TestCredentialStore:
    """Basic CRUD operations on the credential store."""

    def test_store_and_resolve(self):
        s = CredentialStore()
        s.store("token_a", "secret123")
        assert s._resolve("token_a") == "secret123"

    def test_resolve_unknown_returns_none(self):
        s = CredentialStore()
        assert s._resolve("nonexistent") is None

    def test_store_overwrites(self):
        s = CredentialStore()
        s.store("key", "v1")
        s.store("key", "v2")
        assert s._resolve("key") == "v2"

    def test_rotate_existing(self):
        s = CredentialStore()
        s.store("key", "old")
        s.rotate("key", "new")
        assert s._resolve("key") == "new"

    def test_rotate_new_name(self):
        s = CredentialStore()
        s.rotate("fresh", "value")
        assert s._resolve("fresh") == "value"

    def test_delete_existing(self):
        s = CredentialStore()
        s.store("key", "val")
        assert s.delete("key") is True
        assert s._resolve("key") is None

    def test_delete_nonexistent(self):
        s = CredentialStore()
        assert s.delete("nope") is False

    def test_list_names_sorted(self):
        s = CredentialStore()
        s.store("zeta", "z")
        s.store("alpha", "a")
        s.store("mid", "m")
        assert s.list_names() == ["alpha", "mid", "zeta"]

    def test_list_names_empty(self):
        s = CredentialStore()
        assert s.list_names() == []

    def test_len(self):
        s = CredentialStore()
        assert len(s) == 0
        s.store("a", "1")
        s.store("b", "2")
        assert len(s) == 2
        s.delete("a")
        assert len(s) == 1


class TestCredentialStoreThreadSafety:
    """Concurrent access must not corrupt the store."""

    def test_concurrent_store_and_list(self):
        s = CredentialStore()
        errors: list[Exception] = []

        def writer(prefix: str, count: int):
            try:
                for i in range(count):
                    s.store(f"{prefix}_{i}", f"val_{i}")
            except Exception as e:
                errors.append(e)

        def reader(count: int):
            try:
                for _ in range(count):
                    s.list_names()
            except Exception as e:
                errors.append(e)

        threads = [
            threading.Thread(target=writer, args=("a", 100)),
            threading.Thread(target=writer, args=("b", 100)),
            threading.Thread(target=reader, args=(100,)),
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors
        assert len(s) == 200
