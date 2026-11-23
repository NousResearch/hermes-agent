"""Frozen provider-plugin copies of exited processes are reclaimed, and nothing else ever is (#156493).

Copies live under ``<root>/runtime/provider-plugins`` for the life of their process, so a crashed or killed
process would leave its copy forever. Each process holds a lock on ``owner.lock`` in its root; the next process
to create a root reclaims the owned roots nobody holds.
"""

from __future__ import annotations

import os
import shutil
import types
from pathlib import Path

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


def _parent() -> Path:
    from hermes_constants import get_default_hermes_root

    return get_default_hermes_root() / "runtime" / "provider-plugins"


def _seed(*roots: Path) -> None:
    for root in roots:
        (root / "pkg").mkdir(parents=True)


def test_dead_owned_roots_are_reclaimed_and_everything_else_is_kept(loader):
    from pm.filesystem import lock_path

    reload, _hooks = loader
    parent = _parent()
    owned = reload._OWNED_PREFIX
    dead, live = parent / f"{owned}dead", parent / f"{owned}live"
    # A root from a release without owner locks (its process may still be running), a root made without the setup
    # lock, a killed sweeper's claim lookalike that is not one, user data, and an owned-looking directory with no
    # owner.lock (not provably ours): never deleted.
    kept = [parent / n for n in ("hermes-provider-plugins-older", f"{reload._UNSWEPT_PREFIX}x", ".reaping-mine", "notes",
                                 f"{owned}important-user-data")]
    _seed(dead, live, *kept)
    for root in (dead, *kept[:-1]):  # a dead lock file is no licence outside owned-* roots
        (root / reload._OWNER_LOCK).write_text("")
    holder = lock_path(live / reload._OWNER_LOCK, timeout=0)
    assert holder is not None
    try:
        mine = reload._new_root()
        assert mine.parent == parent and mine.name.startswith(owned)
        assert live.is_dir() and all(k.is_dir() for k in kept)
        assert not dead.exists()
        assert not [e for e in parent.iterdir() if e.name.startswith(reload._REAPING_PREFIX)]
        # The next process to start leaves this one's root alone.
        reload._snapshot_root = None
        reload._new_root()
        assert mine.is_dir()
    finally:
        os.close(holder)


def test_a_sweeper_that_died_mid_delete_has_its_claims_finished(loader):
    reload, _hooks = loader
    leftover = _parent() / f"{reload._REAPING_PREFIX}gone"
    _seed(leftover)

    reload._new_root()

    assert not leftover.exists()


@pytest.mark.parametrize("failure", ["timeout", "foreign lock file"])
def test_without_the_setup_lock_the_root_is_one_no_sweep_matches(loader, monkeypatch, failure):
    import pm.filesystem

    reload, hooks = loader
    real = pm.filesystem.lock_path
    stray = _parent() / f"{reload._OWNED_PREFIX}dead"
    _seed(stray)
    (stray / reload._OWNER_LOCK).write_text("")

    def setup_unavailable(path, **kw):
        if path.name == reload._SETUP_LOCK:
            if failure == "timeout":
                return None
            raise PermissionError(13, "denied", str(path))
        return real(path, **kw)

    monkeypatch.setattr(pm.filesystem, "lock_path", setup_unavailable)
    root = reload._new_root()
    monkeypatch.setattr(pm.filesystem, "lock_path", real)

    assert root.name.startswith(reload._UNSWEPT_PREFIX)
    assert stray.is_dir()  # it reclaimed nothing without the lock
    reload._snapshot_root = None
    reload._new_root()  # a later sweep, holding the setup lock: never touches the unswept root
    assert root.is_dir() and not stray.exists()
    (release, args) = hooks[0]
    release(*args)  # a normal exit still removes it
    assert not root.exists()


def test_a_lock_failure_keeps_its_cause_and_leaves_no_root(loader, monkeypatch):
    import pm.filesystem

    reload, _hooks = loader
    real = pm.filesystem.lock_path

    def refuse_owner(path, **kw):
        if path.name == reload._OWNER_LOCK:
            raise PermissionError(13, "denied", str(path))
        return real(path, **kw)

    monkeypatch.setattr(pm.filesystem, "lock_path", refuse_owner)
    with pytest.raises(PermissionError):
        reload._new_root()
    assert not [e for e in _parent().iterdir() if e.name.startswith(reload._OWNED_PREFIX)]


@pytest.mark.platforms("posix")  # os.fork
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


def test_snapshots_live_under_the_root_even_for_a_named_profile(loader, tmp_path, monkeypatch):
    # Deleting a named profile removes its whole home while a multiplexed gateway launched in it may keep serving
    # other profiles; the copies that gateway imports from must not be inside that home.
    reload, _hooks = loader
    root_home = tmp_path / "home"
    named = root_home / "profiles" / "work"
    named.mkdir(parents=True)
    (named / "config.yaml").write_text("")
    monkeypatch.setenv("HERMES_HOME", str(named))

    root = reload._new_root()

    assert root.parent == root_home / "runtime" / "provider-plugins"


def test_backups_and_profile_clones_leave_the_snapshots_out(tmp_path):
    from hermes_cli.backup import _should_exclude
    from hermes_cli.profiles import _clone_all_copytree_ignore

    for rel in ("runtime/provider-plugins/x/owner.lock", "profiles/work/runtime/provider-plugins/x/a.py"):
        assert _should_exclude(Path(rel))
    assert not _should_exclude(Path("runtime/active_sessions.json"))

    source = tmp_path / "source"
    (source / "runtime" / "provider-plugins" / "owned-x").mkdir(parents=True)
    (source / "runtime" / "provider-plugins" / "owned-x" / "a.py").write_text("")
    (source / "runtime" / "active_sessions.json").write_text("{}")
    (source / "skills" / "runtime" / "provider-plugins").mkdir(parents=True)  # a skill's own data: copied
    clone = tmp_path / "clone"
    shutil.copytree(source, clone, ignore=_clone_all_copytree_ignore(source))

    assert not (clone / "runtime" / "provider-plugins").exists()
    assert (clone / "runtime" / "active_sessions.json").is_file()
    assert (clone / "skills" / "runtime" / "provider-plugins").is_dir()


def test_a_data_only_uninstall_keeps_the_snapshots_live_processes_import(loader, tmp_path, monkeypatch):
    from hermes_cli.data_cleanup import plan_data_removal, remove_data

    reload, _hooks = loader
    home = tmp_path / "home"
    root = reload._new_root()
    (home / "notes.md").write_text("user data", encoding="utf-8")
    (root / "pkg").mkdir()
    (root / "pkg" / "lazy.py").write_text("x = 1\n", encoding="utf-8")
    project = tmp_path / "project"
    (project / "hermes_cli").mkdir(parents=True)

    _removed, failures = remove_data(plan_data_removal(home, project))

    assert not failures and not (home / "notes.md").exists()
    assert (root / "pkg" / "lazy.py").is_file()
