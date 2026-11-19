"""A provider plugin's frozen copy outlives the age-based temp cleaners of a long-running gateway (#156493).

The copy used to live in the process temp dir. ``copytree`` keeps the plugin's install-time mtimes, so the
scratch pruner (24 h idle) or macOS's ``/var/folders`` cleaner (3 days) deleted it under a live gateway, and the
plugin's first lazy import afterwards (``from .directsdk import Client`` in ``create_client``) failed with
``No module named ...``.
"""

from __future__ import annotations

import atexit
import os
import sys
import tempfile
import time

import pytest

from tests.providers.test_provider_plugin_update import _write

pytest_plugins = ["tests.providers.test_provider_plugin_update"]


@pytest.fixture
def fresh_root(monkeypatch):
    """The module with no root yet, and the list of exit hooks the test's roots register: they are captured
    rather than registered, and run at teardown, so no owner lock or root outlives the test."""
    import providers.plugin_reload as reload

    hooks = []
    monkeypatch.setattr(reload, "_snapshot_root", None)
    monkeypatch.setattr(atexit, "register", lambda fn, *args: hooks.append(lambda: fn(*args)))
    yield reload, hooks
    for hook in hooks:
        hook()


def _delete_files_older_than(root, seconds: float) -> int:
    """What an age-based temp cleaner does (macOS dirhelper, systemd-tmpfiles); returns the count deleted."""
    cutoff, deleted = time.time() - seconds, 0
    for dirpath, _dirs, files in os.walk(root):
        for name in files:
            path = os.path.join(dirpath, name)
            if os.lstat(path).st_mtime < cutoff:
                os.unlink(path)
                deleted += 1
    return deleted


def test_a_lazy_import_after_an_os_temp_cleaner_ran_still_finds_the_plugin(plugin, fresh_root, tmp_path, monkeypatch):
    from providers import get_provider_profile

    os_temp = tmp_path / "os-temp"
    os_temp.mkdir()
    (os_temp / "stale").write_text("")
    os.utime(os_temp / "stale", (0, 0))
    monkeypatch.setattr(tempfile, "tempdir", str(os_temp))
    _write(plugin, 1)  # files stamped decades back, as copytree carries an install's mtimes into the copy
    profile = get_provider_profile("demo-provider")  # loaded; its lazy ``client`` not imported yet

    assert _delete_files_older_than(os_temp, 3 * 86400) >= 1  # the cleaner did run over the temp dir

    assert profile is not None and profile.create_client().version == "v1"


def test_a_lazy_import_after_the_scratch_pruner_ran_still_finds_the_plugin(plugin, fresh_root, monkeypatch):
    from hermes_constants import get_scratch_dir, prune_scratch_dir
    from providers import get_provider_profile

    scratch = get_scratch_dir(prune=False)
    monkeypatch.setattr(tempfile, "tempdir", str(scratch))  # the boot hook points tempfile here
    _write(plugin, 1)
    profile = get_provider_profile("demo-provider")
    # A day later: nothing under scratch has been written since, so every entry there is idle.
    for dirpath, dirs, files in os.walk(scratch):
        for name in dirs + files:
            os.utime(os.path.join(dirpath, name), (0, 0), follow_symlinks=False)

    prune_scratch_dir(scratch, max_idle_hours=24)

    assert profile is not None and profile.create_client().version == "v1"


def test_roots_of_exited_processes_are_removed_and_held_ones_kept(fresh_root):
    from pm.filesystem import lock_path

    reload, _hooks = fresh_root
    parent = reload._snapshots_parent()
    dead, never_locked, live = parent / "1-dead", parent / "2-never-locked", parent / "3-live"
    for root in (dead, never_locked, live):
        (root / "pkg").mkdir(parents=True)
    (dead / reload._OWNER_LOCK).write_text("")
    holder = lock_path(live / reload._OWNER_LOCK, timeout=0)
    assert holder is not None
    try:
        mine = reload._new_root()
        assert mine.parent == parent and live.is_dir()
        assert not dead.exists() and not never_locked.exists()
        # A second sweep (another process starting) leaves this process's root alone.
        reload._remove_dead_roots(parent)
        assert mine.is_dir()
    finally:
        os.close(holder)


@pytest.mark.skipif(sys.platform == "win32", reason="needs os.fork")
def test_a_forked_child_exiting_keeps_its_parents_copy(fresh_root):
    reload, hooks = fresh_root
    root = reload._new_root()
    (release,) = hooks
    pid = os.fork()
    if pid == 0:  # pragma: no cover - runs in the child
        release()  # the hook a normal interpreter exit in the child runs
        os._exit(0)
    os.waitpid(pid, 0)

    assert root.is_dir()
    reload._remove_dead_roots(root.parent)
    assert root.is_dir()  # this process still holds its owner lock
