"""Contract cases for the backend door the interface declares: every registered backend is a ``PersistenceBackend`` whose ``open(scope)`` returns a ``PersistenceStore`` bound to that one profile."""

from __future__ import annotations

import typing

from persistence.conversation.messages import MessageRepository
from persistence.core.backend import PersistenceBackend, PersistenceStore
from persistence.core.health import StorageHealth


def test_a_registered_backend_presents_the_declared_door(backend):
    # CONTRACT: the assembly reaches every backend through the declared protocol, not a test-only shape.
    assert isinstance(backend, PersistenceBackend)


def test_a_store_handle_carries_every_declared_repository(backend, scope_a):
    store = backend.open(scope_a)
    repositories = {name: hint for name, hint in typing.get_type_hints(PersistenceStore).items() if getattr(hint, "_is_protocol", False)}
    assert repositories["messages"] is MessageRepository and repositories["health"] is StorageHealth
    # CONTRACT: each repository attribute the store handle declares is present and satisfies its protocol.
    assert [name for name, protocol in repositories.items() if not isinstance(getattr(store, name, None), protocol)] == []


def test_open_binds_a_store_handle_to_one_profile(backend, scope_a, scope_b):
    store_a = backend.open(scope_a)
    store_b = backend.open(scope_b)
    # CONTRACT: the handle is a declared store handle carrying its scope and the backend's declared capabilities.
    assert isinstance(store_a, PersistenceStore)
    assert store_a.scope == scope_a
    assert store_b.scope == scope_b
    assert store_a.capabilities == backend.capabilities
    # CONTRACT: every repository of a handle acts on its own profile only.
    store_a.state_meta.set("door-key", "a")
    assert store_a.state_meta.get("door-key") == "a"
    assert store_b.state_meta.get("door-key") is None
    # CONTRACT: a second open of one profile reaches the same store.
    assert backend.open(scope_a).state_meta.get("door-key") == "a"
