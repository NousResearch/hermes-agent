"""A lock pin's sha256 becomes a `fetch-` path component under the store
root; off-shape rows must fail closed before any rmtree/publish (#138359)."""

import json

import pytest

from pm.lock import Lockfile
from pm.store import Store, fetch_entry_name

GOOD = "a" * 64


def _write_pin(path, any_artifacts):
    path.write_text(json.dumps({"schema": 1, "packages": {
        "faketool": {"version": "1", "artifacts": {"any": any_artifacts}}}}))


def _lock_with_pin(tmp_path, sha256):
    path = tmp_path / "lock.json"
    _write_pin(path, {"url": "https://example.com/a.zip", "sha256": sha256})
    return Lockfile(path)


def test_off_shape_sha256_fails_closed_at_the_lock(tmp_path):
    lock = _lock_with_pin(tmp_path, "../../../x")
    with pytest.raises(ValueError, match="malformed artifact pin"):
        lock.artifacts("faketool", "any")


def test_non_string_pin_rows_fail_closed(tmp_path):
    path = tmp_path / "lock.json"
    _write_pin(path, ["https://example.com/a.zip", GOOD])
    with pytest.raises(ValueError, match="malformed artifact pin"):
        Lockfile(path).artifacts("faketool", "any")


def test_well_formed_pin_still_reads_back(tmp_path):
    lock = _lock_with_pin(tmp_path, GOOD)
    assert lock.artifacts("faketool", "any") == [
        {"url": "https://example.com/a.zip", "sha256": GOOD}]


@pytest.mark.parametrize("digest", [
    "../../x", GOOD.upper(), GOOD[:63], GOOD + "a", "", "z" * 64, None, 5,
])
def test_fetch_entry_name_rejects_off_shape_digests(digest):
    with pytest.raises(ValueError, match="malformed artifact sha256"):
        fetch_entry_name(digest)


def test_fetch_many_refuses_traversal_digest_before_touching_disk(tmp_path):
    store_root = tmp_path / "store"
    store_root.mkdir()
    decoy = tmp_path / "precious"
    decoy.mkdir()
    (decoy / "keep.txt").write_text("data")
    with pytest.raises(ValueError, match="malformed artifact sha256"):
        Store(store_root).fetch_many(
            [{"url": "https://example.com/a.zip", "sha256": f"../../{decoy.name}"}],
            tmp_path / "scratch",
        )
    assert (decoy / "keep.txt").read_text() == "data"


def test_remove_downloads_refuses_traversal_sha(tmp_path):
    from pm.install import _remove_downloads

    store_root = tmp_path / "store"
    store_root.mkdir()
    decoy = tmp_path / "precious"
    decoy.mkdir()
    with pytest.raises(ValueError, match="malformed artifact sha256"):
        _remove_downloads(Store(store_root), [{"sha256": f"../../{decoy.name}"}])
    assert decoy.is_dir()
