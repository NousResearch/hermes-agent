"""Desktop metadata must not hide package payloads or obstruct extraction (#128588)."""

import hashlib
import io
import tarfile
from pathlib import Path

import pytest

from pm.store import extract, flatten_single_dir, tree_digest


@pytest.mark.platforms("any")
def test_metadata_does_not_change_extracted_layout_or_payload_digest(tmp_path: Path):
    archive = tmp_path / "package.tar.gz"
    with tarfile.open(archive, "w:gz") as tf:
        for name, data in (
            (".DS_Store", b"outer"),
            ("node/.DS_Store", b"inner"),
            ("node/bin/node", b"runtime"),
        ):
            info = tarfile.TarInfo(name)
            info.size = len(data)
            tf.addfile(info, io.BytesIO(data))
    tree = tmp_path / "tree"
    extract(archive, tree)
    flatten_single_dir(tree)
    assert (tree / "bin/node").read_bytes() == b"runtime"
    assert not (tree / "node").exists()
    baseline = tree_digest(tree)
    for name in (".DS_Store", "._node", "Thumbs.db", "desktop.ini", ".localized"):
        (tree / name).write_bytes(b"desktop metadata")
    assert tree_digest(tree) == baseline
    (tree / "bin/node").write_bytes(b"changed runtime")
    assert tree_digest(tree) != baseline


@pytest.mark.platforms("posix")
def test_metadata_named_directories_links_and_executables_are_hashed(tmp_path: Path):
    tree = tmp_path / "tree"
    tree.mkdir()
    directory = tree / ".DS_Store"
    directory.mkdir()
    payload = directory / "payload"
    payload.write_bytes(b"first")
    first = tree_digest(tree)
    payload.write_bytes(b"second")
    assert tree_digest(tree) != first
    link = tree / "._link"
    link.symlink_to(".DS_Store/payload")
    first = tree_digest(tree)
    link.unlink()
    link.symlink_to("missing-target")
    assert tree_digest(tree) != first
    executable = tree / "Thumbs.db"
    executable.write_bytes(b"first")
    executable.chmod(0o755)
    first = tree_digest(tree)
    executable.write_bytes(b"second")
    assert tree_digest(tree) != first
    # A metadata-looking directory is a real second top-level entry.
    wrapper = tree / "node"
    wrapper.mkdir()
    (wrapper / "bin").mkdir()
    flatten_single_dir(tree)
    assert (wrapper / "bin").is_dir()


@pytest.mark.platforms("any")
def test_facts_recorded_with_metadata_in_the_digest_still_verify(tmp_path: Path):
    from pm import build_operations
    from pm.lock import Facts, Lockfile
    from pm.package import InstallError
    from pm.registry import get_package
    from pm.store import current_target

    target, store = current_target(), tmp_path / "tools"
    lock = Lockfile(tmp_path / "lock.json")
    lock.set_pin("python", "1.0", {target: {"url": "https://invalid.test/python", "sha256": "a" * 64}})
    package = get_package("python")
    entry = store / package.store_entry("1.0", target)
    binary = package.binary(entry, target)
    assert binary is not None
    binary.parent.mkdir(parents=True)
    binary.write_bytes(b"runtime")
    (entry / ".DS_Store").write_bytes(b"desktop metadata")
    # Earlier releases hashed every regular file, metadata included.
    legacy = hashlib.sha256()
    for path in sorted((p for p in entry.rglob("*") if p.is_file()), key=lambda p: p.relative_to(entry).as_posix()):
        legacy.update(path.relative_to(entry).as_posix().encode() + b"\0" + path.read_bytes())
    Facts(store / "facts.json").record("python", "1.0", entry.name, package.env(entry, target), store,
                                       target=target, artifacts=["a" * 64], digest=legacy.hexdigest())

    selected = build_operations.verified_tools(["python"], source_store=store, target=target, lock=lock)
    assert selected.entries["python"].binary == binary
    binary.write_bytes(b"changed runtime")
    with pytest.raises(InstallError, match="source failed verification"):
        build_operations.verified_tools(["python"], source_store=store, target=target, lock=lock)


@pytest.mark.platforms("any")
def test_metadata_named_payload_collision_refuses_before_any_move(tmp_path: Path):
    tree = tmp_path / "tree"
    (tree / "pkg/.DS_Store").mkdir(parents=True)
    for name in ("a", "bin", "m", "z"):
        (tree / "pkg" / name).write_bytes(b"payload")
    (tree / ".DS_Store").write_bytes(b"desktop metadata")
    flatten_single_dir(tree)
    assert sorted(entry.name for entry in tree.iterdir()) == [".DS_Store", "pkg"]
    assert len(list((tree / "pkg").iterdir())) == 5
