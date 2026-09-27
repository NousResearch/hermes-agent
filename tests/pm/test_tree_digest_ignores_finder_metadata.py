"""pm store integrity must ignore Finder browse metadata (.DS_Store).

macOS Finder writes .DS_Store whenever a human browses the pm store or the
staging tree. Package bytes, digests, and flatten decisions must not change
because a window was open. Regression tests for the Sep 2026 failed-update
class (digest mismatch → doctor ✗ → update abort).
"""
from pathlib import Path

from pm.store import flatten_single_dir, tree_digest


def _mk(root: Path):
    root.mkdir(parents=True)
    (root / "bin").mkdir()
    (root / "bin" / "tool").write_text("payload")
    return root


def test_tree_digest_ignores_ds_store(tmp_path):
    root = _mk(tmp_path / "entry")
    before = tree_digest(root)
    (root / ".DS_Store").write_bytes(b"\x00finder")
    (root / "bin" / ".DS_Store").write_bytes(b"\x00finder")
    assert tree_digest(root) == before


def test_flatten_single_dir_ignores_ds_store(tmp_path):
    # archive layout: lone wrapper dir + Finder junk beside it
    staged = tmp_path / "staged"
    staged.mkdir()
    (staged / "node-v1-darwin-arm64").mkdir()
    (staged / "node-v1-darwin-arm64" / "bin").mkdir()
    (staged / "node-v1-darwin-arm64" / "bin" / "node").write_text("x")
    (staged / ".DS_Store").write_bytes(b"\x00finder")
    flatten_single_dir(staged)
    assert (staged / "bin" / "node").exists()
    assert not (staged / "node-v1-darwin-arm64").exists()
