"""Generation leases and install locks live beside the dependency generations, never inside them,
so a sandbox can keep ``installs/<key>/environments/`` and ``pm-runtime/generations/`` read-only
and grant write access to ``installs/<key>/leases/`` and ``locks/`` alone (#151951).

Processes started on an earlier release still lease ``<generation>/.leases/`` and lock
``installs/<key>/.install.lock`` / ``pm-runtime/.prepare.lock`` until they exit: collectors honour a
lease in either place, and installers take the old lock too, first, whenever they can open it.
"""

from __future__ import annotations

import contextlib
import errno
import json
import os
import sys
import time
from pathlib import Path

import pytest

from hermes_cli.runtime_state import collect_generations, lease_directory, leases_held, runtime_lock
from pm.environments import activate_dependencies, install_state_dir, runtime_facts_path, site_packages
from pm.filesystem import lock_fd
from pm.install_states import collect_orphan_install_states
from pm.runtime import collect_runtime_generations

GENERATIONS = {"environments": "environments", "pm-runtime": "pm-runtime/generations"}

non_root = pytest.mark.skipif(getattr(os, "geteuid", lambda: 1)() == 0, reason="root ignores file modes")


@pytest.fixture
def install(tmp_path, monkeypatch):
    project = tmp_path / "project"
    project.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "home"))
    return project, install_state_dir(project)


def _generation(state: Path, kind: str, name: str) -> Path:
    generation = state / GENERATIONS[kind] / name
    generation.mkdir(parents=True)
    (generation / ".lease-managed").touch()
    return generation


def _select_environment(project: Path, state: Path, name: str) -> Path:
    environment = _generation(state, "environments", name) / "venv"
    environment.mkdir()
    (environment / "pyvenv.cfg").write_text("version = 3.11\n", encoding="utf-8")
    site_packages(environment).mkdir(parents=True)
    runtime_facts_path(project).write_text(
        json.dumps({"packages": {"venv": {"environment": str(environment)}}}), encoding="utf-8")
    return environment


def _runtime_generation(state: Path, name: str, *, select: bool = False, inputs: str = "") -> Path:
    generation = _generation(state, "pm-runtime", name)
    (generation / "pm-runtime.json").write_text(json.dumps({"inputs": inputs}), encoding="utf-8")
    if select:
        (state / "pm-runtime" / "selected.json").write_text(
            json.dumps({"inputs": inputs, "generation": f"generations/{name}"}), encoding="utf-8")
    return generation


def _hold(path: Path) -> int:
    """A lease or lock another process holds: the file at *path*, kernel-locked."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(path, os.O_CREAT | os.O_RDWR, 0o600)
    assert lock_fd(fd, wait=False)
    return fd


@contextlib.contextmanager
def _read_only(*directories: Path):
    for directory in directories:
        directory.chmod(0o555)
    try:
        yield
    finally:
        for directory in directories:
            directory.chmod(0o755)


def _keep_process_paths(monkeypatch):
    monkeypatch.setattr(sys, "path", list(sys.path))
    for key in ("PYTHONPATH", "PATH", "VIRTUAL_ENV"):
        monkeypatch.setenv(key, os.environ.get(key, ""))


@pytest.mark.platforms("posix")
@non_root
@pytest.mark.parametrize("kind", sorted(GENERATIONS))
def test_read_only_generation_is_leased_from_beside_it(install, kind):
    _project, state = install
    generation = _generation(state, kind, "gen")
    before = sorted(os.listdir(generation))
    with _read_only(generation.parent, generation):
        release = lease_directory(generation)
        try:
            assert leases_held(generation)
        finally:
            release()
        assert not leases_held(generation)
    assert sorted(os.listdir(generation)) == before


@pytest.mark.platforms("posix")
@non_root
@pytest.mark.parametrize("old_lock_file", [False, True], ids=["fresh", "old-lock-file"])
def test_activation_writes_only_leases_and_locks(install, monkeypatch, old_lock_file):
    project, state = install
    environment = _select_environment(project, state, "gen")
    (state / "leases").mkdir()
    (state / "locks").mkdir()
    if old_lock_file:  # left by a launch on an earlier release; the sandbox sees it read-only
        (state / ".install.lock").touch()
        (state / ".install.lock").chmod(0o444)
    _keep_process_paths(monkeypatch)
    with _read_only(environment.parent, state / "environments", state):
        activate_dependencies(project)
        assert str(site_packages(environment.resolve())) in sys.path
        assert leases_held(environment.parent)


@pytest.mark.platforms("posix")
@non_root
def test_pm_runtime_launch_writes_only_leases_and_locks(install, monkeypatch):
    from pm import runtime

    project, state = install
    for name in ("pyproject.toml", "uv.lock"):
        (project / name).write_text("", encoding="utf-8")
    python = Path(sys.executable)
    root = state / "pm-runtime"
    generation = _runtime_generation(state, "gen", select=True, inputs=runtime._inputs(project, python))
    (state / "leases").mkdir()
    (state / "locks").mkdir()
    monkeypatch.setattr(runtime, "_validate", lambda *_: "")
    monkeypatch.setattr(runtime, "_HELD", {})
    try:
        with _read_only(generation, root / "generations", root, state):
            chosen = runtime.prepare_runtime(Path("uv"), python, root, project=project, bootstrap=False)
            assert chosen == runtime._python(generation)
            assert leases_held(generation)
    finally:
        for release in runtime._HELD.values():
            release()


@pytest.mark.platforms("any")
@pytest.mark.parametrize("where", ["beside", "inside"])
def test_dependency_collector_keeps_a_generation_leased_in_either_place(install, where):
    project, state = install
    _select_environment(project, state, "selected")
    superseded = _generation(state, "environments", "superseded")
    lease = (state / "leases" / "environments" / "superseded" / "reader" if where == "beside"
             else superseded / ".leases" / "reader")
    fd = _hold(lease)
    try:
        assert collect_generations(project, min_age_seconds=0) == []
        assert superseded.is_dir()
    finally:
        os.close(fd)
    assert collect_generations(project, min_age_seconds=0) == [superseded]


@pytest.mark.platforms("any")
@pytest.mark.parametrize("where", ["beside", "inside"])
def test_runtime_collector_keeps_a_generation_leased_in_either_place(install, where):
    _project, state = install
    _runtime_generation(state, "selected", select=True)
    superseded = _runtime_generation(state, "superseded")
    lease = (state / "leases" / "pm-runtime" / "superseded" / "reader" if where == "beside"
             else superseded / ".leases" / "reader")
    fd = _hold(lease)
    try:
        assert collect_runtime_generations(state / "pm-runtime") == []
        assert superseded.is_dir()
    finally:
        os.close(fd)
    assert collect_runtime_generations(state / "pm-runtime") == [superseded]


@pytest.mark.platforms("any")
@pytest.mark.parametrize("kind", sorted(GENERATIONS))
def test_collected_generations_take_their_lease_directories_with_them(install, kind):
    project, state = install
    if kind == "environments":
        _select_environment(project, state, "selected")
        superseded = _generation(state, kind, "superseded")
    else:
        _runtime_generation(state, "selected", select=True)
        superseded = _runtime_generation(state, "superseded")
    lease_directory(superseded)()
    (state / "leases" / kind / "removed-earlier").mkdir(parents=True)
    collected = (collect_generations(project, min_age_seconds=0) if kind == "environments"
                 else collect_runtime_generations(state / "pm-runtime"))
    assert collected == [superseded]
    assert list((state / "leases" / kind).iterdir()) == []


@pytest.mark.platforms("any")
@pytest.mark.parametrize("where", ["new", "old"])
def test_install_lock_held_in_either_place_excludes_an_installer(install, where):
    project, state = install
    fd = _hold(state / "locks" / "install.lock" if where == "new" else state / ".install.lock")
    try:
        with runtime_lock(project, timeout=0.2) as held:
            assert held is False
    finally:
        os.close(fd)
    with runtime_lock(project, timeout=0.2) as held:
        assert held is True


@pytest.mark.platforms("posix")
@non_root
def test_old_install_lock_still_excludes_where_it_can_only_be_read(install):
    """A sandbox sees the old lock file read-only, but an older process outside it still locks
    that file: the sandboxed installer waits on it rather than failing or skipping it."""
    project, state = install
    (state / "locks").mkdir(parents=True)
    old = state / ".install.lock"
    fd = _hold(old)
    old.chmod(0o444)
    try:
        with _read_only(state):
            with runtime_lock(project, timeout=0.2) as held:
                assert held is False
            os.close(fd)
            fd = -1
            with runtime_lock(project, timeout=0.2) as held:
                assert held is True
    finally:
        if fd >= 0:
            os.close(fd)


@pytest.mark.platforms("posix")
@non_root
def test_failure_to_lock_a_read_only_old_lock_reaches_the_caller(install, monkeypatch):
    """Only an old lock this process cannot open is skipped; one it opened but could not lock
    (ENOLCK on NFS) fails the installer rather than leaving it on the new lock alone."""
    from pm import filesystem

    project, state = install
    (state / "locks").mkdir(parents=True)
    old = state / ".install.lock"
    old.touch()
    old.chmod(0o444)

    def lock_fd(fd, **kwargs):
        if os.path.samestat(os.fstat(fd), os.stat(old)):
            raise OSError(errno.ENOLCK, "No locks available")
        return filesystem_lock_fd(fd, **kwargs)

    filesystem_lock_fd = filesystem.lock_fd
    monkeypatch.setattr(filesystem, "lock_fd", lock_fd)
    with _read_only(state), pytest.raises(OSError) as raised:
        with runtime_lock(project, timeout=0.2):
            pass
    assert raised.value.errno == errno.ENOLCK


@pytest.mark.platforms("posix")
@non_root
def test_old_install_lock_this_process_cannot_open_is_skipped(install):
    project, state = install
    (state / "locks").mkdir(parents=True)
    old = state / ".install.lock"
    old.touch()
    old.chmod(0o000)
    with _read_only(state):
        with runtime_lock(project, timeout=0.2) as held:
            assert held is True


@pytest.mark.platforms("any")
@pytest.mark.parametrize("where", ["new", "old"])
def test_prepare_lock_held_in_either_place_defers_runtime_collection(install, where):
    """An unpublished generation is a stage in flight while the prepare lock is held."""
    _project, state = install
    root = state / "pm-runtime"
    staging = _generation(state, "pm-runtime", "staging")
    fd = _hold(state / "locks" / "prepare.lock" if where == "new" else root / ".prepare.lock")
    try:
        assert collect_runtime_generations(root) == []
        assert staging.is_dir()
    finally:
        os.close(fd)
    assert collect_runtime_generations(root) == [staging]


def _orphan(tmp_path: Path) -> tuple[Path, Path]:
    installs = tmp_path / "home" / "installs"
    state = installs / "aaaa"
    (state / "inputs").mkdir(parents=True)
    (state / "inputs" / ".project-root").write_text(str(tmp_path / "home" / "cache" / "gone"), encoding="utf-8")
    return installs, state


@pytest.mark.platforms("any")
@pytest.mark.parametrize("held", ["environments-lease", "pm-runtime-lease", "install-lock"])
def test_orphan_reclaim_skips_a_state_held_beside_its_generations(tmp_path, held):
    installs, state = _orphan(tmp_path)
    _generation(state, "environments", "gen")
    _generation(state, "pm-runtime", "gen")
    path = {"environments-lease": state / "leases" / "environments" / "gen" / "reader",
            "pm-runtime-lease": state / "leases" / "pm-runtime" / "gen" / "reader",
            "install-lock": state / "locks" / "install.lock"}[held]
    fd = _hold(path)
    try:
        assert collect_orphan_install_states(installs) == []
        assert (state / "environments" / "gen").is_dir()
    finally:
        os.close(fd)
    assert collect_orphan_install_states(installs) == [state]
    assert not state.exists()


@pytest.mark.platforms("any")
def test_a_launch_beside_the_generations_counts_as_recent_use(tmp_path):
    """A checkout this process cannot see is reclaimed only once its state has gone unused;
    a lease taken and released under ``leases/`` today is use."""
    installs = tmp_path / "home" / "installs"
    state = installs / "aaaa"
    (state / "inputs").mkdir(parents=True)
    (state / "inputs" / ".project-root").write_text(str(tmp_path / "unmounted" / "checkout"), encoding="utf-8")
    generation = _generation(state, "environments", "gen")
    month_ago = time.time() - 30 * 86400
    for path in sorted(state.rglob("*"), reverse=True) + [state]:
        os.utime(path, (month_ago, month_ago))
    lease = state / "leases" / "environments" / "gen" / "reader"
    os.close(_hold(lease))
    lease.unlink()
    assert collect_orphan_install_states(installs) == []
    assert generation.is_dir()
