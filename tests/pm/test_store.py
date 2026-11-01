from pathlib import Path

from pm.store import flatten_single_dir, tree_digest


def test_flatten_single_dir_ignores_os_metadata(tmp_path: Path):
    tree = tmp_path / "tree"
    nested = tree / "node-v1"
    nested.mkdir(parents=True)
    (nested / "bin").mkdir()
    (nested / "bin" / "node").write_bytes(b"node")
    (tree / ".DS_Store").write_bytes(b"metadata")

    flatten_single_dir(tree)

    assert (tree / "bin" / "node").read_bytes() == b"node"
    assert not (tree / "node-v1").exists()


def test_tree_digest_ignores_os_metadata(tmp_path: Path):
    tree = tmp_path / "tree"
    tree.mkdir()
    payload = tree / "payload"
    payload.write_bytes(b"payload")
    baseline = tree_digest(tree)

    (tree / ".DS_Store").write_bytes(b"metadata")
    (tree / "._payload").write_bytes(b"resource fork")
    (tree / "Thumbs.db").write_bytes(b"metadata")

    assert tree_digest(tree) == baseline
