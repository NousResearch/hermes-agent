"""Tests for cred_proxy.store (keyring-backed credential store)."""

import pytest

from cred_proxy.store import CredStore


@pytest.fixture
def store() -> CredStore:
    return CredStore()


def test_set_and_list_shows_name(store: CredStore) -> None:
    store.set("mytoken", "secret-value")
    assert "mytoken" in store.list()


def test_list_empty_by_default(store: CredStore) -> None:
    assert store.list() == []


def test_list_multiple_names_sorted(store: CredStore) -> None:
    store.set("zebra", "v1")
    store.set("alpha", "v2")
    store.set("middle", "v3")
    assert store.list() == ["alpha", "middle", "zebra"]


def test_get_returns_correct_value(store: CredStore) -> None:
    store.set("api_key", "top-secret-123")
    assert store._get("api_key") == "top-secret-123"


def test_delete_removes_name(store: CredStore) -> None:
    store.set("tok", "val")
    store.delete("tok")
    assert "tok" not in store.list()


def test_delete_raises_key_error_for_missing(store: CredStore) -> None:
    with pytest.raises(KeyError):
        store.delete("nonexistent")


def test_get_raises_key_error_for_missing(store: CredStore) -> None:
    with pytest.raises(KeyError):
        store._get("nonexistent")


def test_overwrite_updates_value(store: CredStore) -> None:
    store.set("key", "old-value")
    store.set("key", "new-value")
    assert store._get("key") == "new-value"
    assert store.list().count("key") == 1  # no duplicates
