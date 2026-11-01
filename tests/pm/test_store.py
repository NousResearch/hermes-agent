"""Desktop metadata must not hide package payloads or obstruct extraction (#128588)."""
import io
import tarfile
from pathlib import Path

import pytest

from pm.store import extract, flatten_single_dir, tree_digest


def test_metadata_does_not_change_extracted_layout_or_payload_digest(tmp_path: Path):
    archive = tmp_path / "package.tar.gz"
    with tarfile.open(archive, "w:gz") as tf:
        for name, data in ((".DS_Store", b"outer"), ("node/.DS_Store", b"inner"),
                           ("node/bin/node", b"runtime")):
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
