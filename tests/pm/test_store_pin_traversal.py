"""A lockfile digest names a store slot, so it has to *be* a digest.

`Store.fetch_many` builds its cache entry name as ``fetch-<sha256>`` straight
from the artifact rows, and removes that entry before re-downloading when the
cached copy cannot be reused. `pm.install._remove_downloads` removes the same
entry after publication. Both act on the path *before* anything verifies the
bytes or checks the value's shape, so a digest that carries separators
(``slot/../../victim``) turns a lockfile row into a recursive removal of the
lockfile's choosing — the store root only has to contain one ``fetch-<...>``
slot for the traversal to resolve and escape it.

The refusal belongs at the name construction, not only in the downloader:
`pm.artifact_mirror.object_key` validates the digest, but `fetch_many` reaches
its `shutil.rmtree` first, and `_remove_downloads` never calls it at all.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from pm.store import Store, artifact_entry_name


def _victim(tmp_path: Path) -> Path:
    """A directory outside the store root that a traversal would reach. It
    holds two files on purpose: `fetch_many` only skips the removal of a
    cache entry that is a directory with exactly one file."""
    victim = tmp_path / "victim"
    victim.mkdir()
    (victim / "keep-1").write_text("mine", encoding="utf-8")
    (victim / "keep-2").write_text("mine", encoding="utf-8")
    return victim


def _store_with_a_slot(tmp_path: Path, slot: str) -> Store:
    """A store whose root holds one cached archive slot: the path component a
    traversal in the digest has to walk through before it can climb out."""
    root = tmp_path / "store"
    (root / f"fetch-{slot}").mkdir(parents=True)
    return Store(root)


def test_fetch_many_refuses_a_traversing_digest(tmp_path):
    victim = _victim(tmp_path)
    store = _store_with_a_slot(tmp_path, "slot")
    artifacts = [{"url": "https://example.test/tool.tar.gz", "sha256": "slot/../../victim"}]

    with pytest.raises(ValueError):
        store.fetch_many(artifacts, tmp_path / "scratch")

    assert victim.is_dir(), "the traversing digest removed a directory outside the store root"
    assert sorted(p.name for p in victim.iterdir()) == ["keep-1", "keep-2"]


def test_remove_downloads_refuses_a_traversing_digest(tmp_path):
    from pm.install import _remove_downloads

    victim = _victim(tmp_path)
    store = _store_with_a_slot(tmp_path, "slot")

    with pytest.raises(ValueError):
        _remove_downloads(store, [{"sha256": "slot/../../victim"}])

    assert victim.is_dir(), "the traversing digest removed a directory outside the store root"


def test_artifact_entry_name_rejects_everything_that_is_not_a_digest():
    assert artifact_entry_name("a" * 64) == "fetch-" + "a" * 64

    for value in ("slot/../../victim", "..", "", None, 64, "A" * 64, "a" * 63, "a" * 65):
        with pytest.raises(ValueError):
            artifact_entry_name(value)
