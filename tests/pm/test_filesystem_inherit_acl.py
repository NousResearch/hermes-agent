"""pm.filesystem.inherit_parent_acl on real Windows ACLs (the publication-level case is in test_worker_publication)."""
import os
import subprocess
import tempfile
from pathlib import Path

import pytest

from pm.filesystem import inherit_parent_acl


def _entries(path: Path) -> list[str]:
    out = subprocess.run(["icacls", str(path)], capture_output=True, text=True, encoding="oem",
                         errors="replace", check=True).stdout
    return [line.replace(str(path), "").strip() for line in out.splitlines() if ":(" in line]


@pytest.mark.platforms("windows")
def test_a_tree_moved_out_of_a_temporary_directory_inherits_from_its_new_parent(tmp_path):
    parent = tmp_path / "plugins"
    parent.mkdir()
    subprocess.run(["icacls", str(parent), "/grant", "*S-1-5-32-545:(OI)(CI)(RX)"], check=True, capture_output=True)
    with tempfile.TemporaryDirectory(dir=parent) as tmp:
        staged = Path(tmp) / "plugin"
        (staged / "pkg").mkdir(parents=True)
        (staged / "pkg" / "mod.py").write_text("x = 1\n", encoding="utf-8")
        # An explicit entry inside the tree is the owner's choice and must survive.
        subprocess.run(["icacls", str(staged / "pkg" / "mod.py"), "/grant", "*S-1-5-32-546:(R)"],
                       check=True, capture_output=True)
        target = parent / "example"
        os.replace(staged, target)
    assert not any("(I)" in line for line in _entries(target))  # the staging folder's protected DACL moved

    assert inherit_parent_acl(target) is True

    assert all("(I)" in line for line in _entries(target)), _entries(target)
    for path in (target, target / "pkg", target / "pkg" / "mod.py"):
        assert any(line.startswith("BUILTIN\\Users:(I)") for line in _entries(path)), (path, _entries(path))
    assert any(line.startswith("BUILTIN\\Guests:(R)") for line in _entries(target / "pkg" / "mod.py"))


def test_never_raises(tmp_path):
    assert inherit_parent_acl(tmp_path / "missing") is False
