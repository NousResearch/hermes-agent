"""Contract case for a store handle's mode: ``read_only`` answers without a verb, and a handle from the backend's
``open`` is writable."""

from __future__ import annotations

from persistence import StorageProbe


def test_a_handle_from_open_answers_writable_without_touching_the_store(backend, scope_a):
    store = backend.open(scope_a)
    # CONTRACT: the mode is known at open; reading it neither creates nor opens the store.
    assert store.read_only is False
    assert store.health.probe() == StorageProbe(status="ok", detail="not initialized")
    store.state_meta.set("mode-key", "value")
    # CONTRACT: the mode stays the handle's for its life.
    assert store.read_only is False
    assert store.state_meta.get("mode-key") == "value"
