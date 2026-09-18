"""Permission repair stays within the directory being removed."""

import os
from pathlib import Path

import pytest

from hermes_cli.fs_remove import _make_writable


def test_permission_repair_does_not_change_external_paths(tmp_path, monkeypatch):
    root = tmp_path / "tree"
    root.mkdir()
    outside = tmp_path / "external"
    outside.write_text("keep", encoding="utf-8")
    original = os.chmod
    touched = []

    def chmod(path, mode):
        touched.append(Path(path).resolve())
        original(path, mode)

    monkeypatch.setattr(os, "chmod", chmod)
    retried = []
    _make_writable(root, retried.append, root, PermissionError("readonly"))
    _make_writable(root, retried.append, outside, PermissionError("denied"))
    assert touched == [root]
    assert retried == [root, outside]
    with pytest.raises(IsADirectoryError):
        _make_writable(root, retried.append, root, IsADirectoryError("wrong operation"))
    assert touched == [root]
    assert outside.read_text(encoding="utf-8") == "keep"
