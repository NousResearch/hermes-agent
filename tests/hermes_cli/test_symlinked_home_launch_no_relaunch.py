"""A permanent HERMES_HOME reached through a symlink must boot without relaunch.

Publication bakes a symlink-resolved store root into the launcher, so the
runtime leg of ``resolve_store_python`` must spell the store the same
resolved way: ``prepare_launch`` compares it lexically against
``sys.executable``, and a mismatch republishes launchers and re-execs once on
every start when HERMES_HOME itself, or its ``tools`` dir, is a symlink.
"""
import os
import shutil
import sys

import pm
import pytest
from hermes_cli import _launchers, venv_sync

from tests.hermes_cli.test_hermes_home_shared_launchers import (
    _isolate,
    _make_home,
    _make_repo,
    _publish,
)

pytestmark = pytest.mark.skipif(os.name == "nt", reason="POSIX symlinks")


def _launch_as_baked(tmp_path, monkeypatch, repo, home):
    """Run prepare_launch as the published launcher started this interpreter."""
    _isolate(tmp_path, monkeypatch, home)
    _publish(repo)
    baked = _launchers._launcher_python(repo / ".hermes" / "bin" / "hermes")
    assert baked is not None and baked.is_file()

    publish_calls = []
    real_publish = venv_sync.publish_launchers

    def counting_publish(*args, **kwargs):
        publish_calls.append(args)
        return real_publish(*args, **kwargs)

    monkeypatch.setattr(venv_sync, "publish_launchers", counting_publish)
    monkeypatch.setattr(pm, "venv_is_current", lambda **kwargs: True)
    # A real interpreter reports the path it was started by: the baked one.
    monkeypatch.setattr(sys, "executable", str(baked))
    return venv_sync.prepare_launch(repo, []), publish_calls


@pytest.mark.platforms("posix")
def test_symlinked_tools_dir_boots_without_relaunch(tmp_path, monkeypatch):
    # S2: the store lives on another disk, reached through home/tools.
    ext, _ = _make_home(tmp_path, "external-disk")
    home = tmp_path / "home"
    home.mkdir()
    (home / "tools").symlink_to(ext / "tools")
    target, publish_calls = _launch_as_baked(tmp_path, monkeypatch, _make_repo(tmp_path), home)
    assert target is None, f"relaunched into {target}"
    assert publish_calls == []


@pytest.mark.platforms("posix")
def test_symlinked_hermes_home_boots_without_relaunch(tmp_path, monkeypatch):
    # S3: HERMES_HOME itself is a symlink (e.g. ~/.hermes -> /Volumes/x/hermes).
    real, _ = _make_home(tmp_path, "real-home")
    link = tmp_path / "home-link"
    link.symlink_to(real)
    target, publish_calls = _launch_as_baked(tmp_path, monkeypatch, _make_repo(tmp_path), link)
    assert target is None, f"relaunched into {target}"
    assert publish_calls == []


@pytest.mark.platforms("posix")
def test_repointed_then_deleted_store_heals_on_next_publish(tmp_path, monkeypatch):
    # Repoint home/tools at a new disk and delete the old one: the launcher
    # names a dead interpreter until the next publish from the live root
    # rebinds it (same heal path as a deleted real store, #123238).
    old_ext, _ = _make_home(tmp_path, "old-disk")
    home = tmp_path / "home"
    home.mkdir()
    (home / "tools").symlink_to(old_ext / "tools")
    repo = _make_repo(tmp_path)
    _isolate(tmp_path, monkeypatch, home)
    _publish(repo)
    baked = _launchers._launcher_python(repo / ".hermes" / "bin" / "hermes")

    new_ext, new_python = _make_home(tmp_path, "new-disk")
    (home / "tools").unlink()
    (home / "tools").symlink_to(new_ext / "tools")
    shutil.rmtree(old_ext)
    assert baked is not None and not baked.is_file()

    healed = _publish(repo)
    for name in healed:
        assert _launchers._launcher_python(repo / ".hermes" / "bin" / name) == new_python
