"""Post-exec diff hook."""

from pathlib import Path

from sandbox.hooks import DiffReport, post_exec_diff


def test_post_exec_diff_detects_new_file(tmp_path: Path):
    left = tmp_path / "a"
    right = tmp_path / "b"
    left.mkdir()
    right.mkdir()
    (left / "keep.txt").write_text("x", encoding="utf-8")
    (right / "keep.txt").write_text("x", encoding="utf-8")
    (right / "new.txt").write_text("y", encoding="utf-8")
    rep = post_exec_diff(left, right)
    assert "new.txt" in rep.only_right
    assert isinstance(rep, DiffReport)
