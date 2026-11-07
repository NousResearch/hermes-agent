"""Contract cases for the backend door the interface declares: every registered backend is a ``PersistenceBackend`` whose ``open(scope)`` returns a ``PersistenceStore`` bound to that one profile."""

from __future__ import annotations

from persistence.core.backend import PersistenceBackend, PersistenceStore


def test_a_registered_backend_presents_the_declared_door(backend):
    # CONTRACT: the assembly reaches every backend through the declared protocol, not a test-only shape.
    assert isinstance(backend, PersistenceBackend)


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
