"""A clone must not copy a bind-mounted directory when its path is Unicode."""

import builtins
import os
import shutil

import pytest

from hermes_cli.profiles import _clone_all_copytree_ignore


@pytest.mark.platforms("linux")
def test_clone_all_skips_unicode_child_mount_with_escaped_space(tmp_path, monkeypatch):
    source = tmp_path / "profile"
    mountpoint = source / "共享 data"
    mountpoint.mkdir(parents=True)
    (mountpoint / "host-only.txt").write_text("do not copy", encoding="utf-8")
    skills = source / "skills"
    skills.mkdir()
    (skills / "keep.txt").write_text("keep", encoding="utf-8")

    # Linux mountinfo leaves Unicode unescaped, but encodes spaces as \040.
    mountinfo = tmp_path / "mountinfo"
    escaped_mountpoint = str(mountpoint).replace(" ", r"\040")
    mountinfo.write_text(
        f"36 35 0:22 / {escaped_mountpoint} rw,relatime - ext4 /dev/root rw\n",
        encoding="utf-8",
    )

    real_open = builtins.open

    def open_with_mountinfo(path, *args, **kwargs):
        if os.fspath(path) == "/proc/self/mountinfo":
            path = mountinfo
        return real_open(path, *args, **kwargs)

    monkeypatch.setattr(builtins, "open", open_with_mountinfo)
    dest = tmp_path / "clone"
    shutil.copytree(source, dest, ignore=_clone_all_copytree_ignore(source))

    assert not (dest / "共享 data").exists()
    assert (dest / "skills" / "keep.txt").read_text(encoding="utf-8") == "keep"
