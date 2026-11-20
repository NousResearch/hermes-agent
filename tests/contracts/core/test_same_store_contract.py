"""Contract case for ``serves_same_store``: two handles on one profile serve one store; another profile's do not."""

from __future__ import annotations


def test_handles_on_one_profile_serve_the_same_store(backend, scope_a, scope_b):
    first, second, other = backend.open(scope_a), backend.open(scope_a), backend.open(scope_b)
    first.state_meta.set("same-store", "written")
    # CONTRACT: two handles on one profile serve the same store, either way round, and what one writes the other reads.
    assert first.serves_same_store(second) and second.serves_same_store(first)
    assert second.state_meta.get("same-store") == "written"
    # CONTRACT: a handle on another profile serves another store.
    assert not first.serves_same_store(other) and not other.serves_same_store(first)
