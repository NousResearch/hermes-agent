"""Frozen provider-plugin copies of exited processes are reclaimed, and live ones never are (#156493).

Copies live under ``<home>/runtime/provider-plugins`` for the life of their process, so a crashed or killed
process would leave its copy forever. Each process holds a lock on ``owner.lock`` in its root; the next process
to create a root reclaims those nobody holds.
"""

from __future__ import annotations

import os
import sys
import types

import pytest


@pytest.fixture
def loader(tmp_path, monkeypatch):
    """The loader with no root yet. Its exit hooks are captured rather than registered (only this module's),
    and run at teardown in atexit's order, so no owner lock or root outlives the test."""
    import providers.plugin_reload as module

    hooks = []
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "home"))
    monkeypatch.setattr(module, "_snapshot_root", None)
    monkeypatch.setattr(module, "atexit", types.SimpleNamespace(register=lambda fn, *args: hooks.append((fn, args))))
    yield module, hooks
    for fn, args in reversed(hooks):
        fn(*args)


def _parent():
    from hermes_constants import get_process_hermes_home

    return get_process_hermes_home() / "runtime" / "provider-plugins"


def test_roots_of_exited_processes_are_reclaimed_and_everything_else_is_kept(loader):
    reload, _hooks = loader
    from pm.filesystem import lock_path

    parent = _parent()
    dead, never_locked, live = (parent / f"{reload._ROOT_PREFIX}{n}" for n in ("dead", "never-locked", "live"))
    unrelated = parent / "not-a-snapshot"
    for root in (dead, never_locked, live, unrelated):
        (root / "pkg").mkdir(parents=True)
    (dead / reload._OWNER_LOCK).write_text("")
    (unrelated / reload._OWNER_LOCK).write_text("")
    holder = lock_path(live / reload._OWNER_LOCK, timeout=0)
    assert holder is not None
    try:
        mine = reload._new_root()
        assert mine.parent == parent
        assert live.is_dir() and unrelated.is_dir()
        assert not dead.exists() and not never_locked.exists()
        assert not [e for e in parent.iterdir() if e.name.startswith(reload._REAPING_PREFIX)]
        # The next process to start leaves this one's root alone.
        reload._snapshot_root = None
        reload._new_root()
        assert mine.is_dir()
    finally:
        os.close(holder)


def test_a_sweeper_that_died_mid_delete_has_its_claims_finished(loader):
    reload, _hooks = loader
    parent = _parent()
    leftover = parent / f"{reload._REAPING_PREFIX}{reload._ROOT_PREFIX}gone"
    (leftover / "pkg").mkdir(parents=True)

    reload._new_root()

    assert not leftover.exists()


def test_a_lock_failure_keeps_its_cause_and_leaves_no_root(loader, monkeypatch):
    reload, _hooks = loader
    import pm.filesystem

    real = pm.filesystem.lock_path

    def refuse_owner(path, **kw):
        if path.name == reload._OWNER_LOCK:
            raise PermissionError(13, "denied", str(path))
        return real(path, **kw)

    monkeypatch.setattr(pm.filesystem, "lock_path", refuse_owner)
    with pytest.raises(PermissionError):
        reload._new_root()
    assert not [e for e in _parent().iterdir() if e.name.startswith(reload._ROOT_PREFIX)]


@pytest.mark.skipif(sys.platform == "win32", reason="needs os.fork")
def test_a_forked_child_exiting_keeps_its_parents_copy(loader):
    reload, hooks = loader
    root = reload._new_root()
    ((release, args),) = hooks
    pid = os.fork()
    if pid == 0:  # pragma: no cover - runs in the child
        release(*args)  # what a normal interpreter exit in the child runs
        os._exit(0)
    os.waitpid(pid, 0)

    assert root.is_dir()
    reload._snapshot_root = None
    reload._new_root()  # another process starting: this process still holds its owner lock
    assert root.is_dir()


def test_backups_and_profile_clones_leave_the_snapshots_out(tmp_path):
    from pathlib import Path

    from hermes_cli.backup import _should_exclude
    from hermes_cli.home_data_layout import provider_plugin_snapshot_dir

    for rel in ("runtime/provider-plugins/x/owner.lock", "profiles/work/runtime/provider-plugins/x/a.py"):
        assert _should_exclude(Path(rel))
    assert not _should_exclude(Path("runtime/active_sessions.json"))

    home = tmp_path / "home"
    (home / "runtime" / "provider-plugins").mkdir(parents=True)
    names = ["provider-plugins", "active_sessions.json"]
    assert provider_plugin_snapshot_dir(str(home / "runtime"), names, home) == {"provider-plugins"}
    assert provider_plugin_snapshot_dir(str(home / "skills" / "runtime"), names, home) == set()
