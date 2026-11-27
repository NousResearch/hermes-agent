"""Regression for #130242: deep core workspace snapshots must use long-path roots.

The file operations are real on each host.  The wrapper observes which paths
reach the outer copytree call; it does not fake Windows or lower MAX_PATH.
"""

from pathlib import Path
import shutil

import pytest

from pm import filesystem, workspace

pytestmark = pytest.mark.platforms("windows", "posix")


def test_core_snapshot_copies_deep_package_through_long_roots(tmp_path, monkeypatch):
    source = tmp_path / "core"
    destination = tmp_path / "installs" / "workspace"
    source.mkdir()
    destination.mkdir(parents=True)
    (source / "pyproject.toml").write_text(
        '[project]\nname = "deep-copy-probe"\nversion = "0.1.0"\n'
        '[tool.setuptools.packages.find]\ninclude = ["plugins"]\n',
        encoding="utf-8",
    )

    relative = (Path("plugins") / "config_backends" / "remote" / "vendor" /
                "conformance" / "config-resolution" / ("a" * 80) /
                ("b" * 80) / "probe.json")
    original = source / relative
    copied = destination / relative
    assert len(str(copied)) > 260  # Windows' real default MAX_PATH, not a reduced limit
    filesystem.long_root(original.parent).mkdir(parents=True)
    filesystem.long_root(original).write_bytes(b'{"copied": true}\n')

    returned_roots = []

    def record_long_root(path):
        rooted = Path(filesystem.long_root(path))
        returned_roots.append(rooted)
        return rooted

    first_copy = []
    real_copytree = shutil.copytree

    def observe_copytree(src, dst, *args, **kwargs):
        if not first_copy:
            first_copy.append((src, dst))
        return real_copytree(src, dst, *args, **kwargs)

    monkeypatch.setattr(workspace, "long_root", record_long_root, raising=False)
    monkeypatch.setattr(workspace.shutil, "copytree", observe_copytree)
    try:
        workspace._copy_core_inputs(source, destination)
        assert filesystem.long_root(copied).read_bytes() == b'{"copied": true}\n'
        assert first_copy
        assert all(any(argument is rooted for rooted in returned_roots)
                   for argument in first_copy[0]), (
            "copytree must receive both extended-length roots, not plain paths"
        )
    finally:
        shutil.rmtree(filesystem.long_root(source / "plugins"), ignore_errors=True)
        shutil.rmtree(filesystem.long_root(destination / "plugins"), ignore_errors=True)
