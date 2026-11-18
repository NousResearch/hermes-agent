"""PATH composition and shell discovery are memoized (pm.install, pm.user_facing, pm.shell).

The terminal composes the user's PATH and resolves bash before every command, and both
used to read the installed state and spawn a probe process each time. These tests pin the
memo's two halves: a steady state costs no syscall, and every way the answer can change —
recorded state, the resolvers around it, the binary itself — is still observed.
"""
from __future__ import annotations

import os
import time

import pytest

import pm.install
import pm.shell
import pm.user_facing
from pm import paths, registry
from pm.lock import Facts, Lockfile
from pm.store import current_target, tree_digest


def _record_tool(store, name: str, entry_name: str, lock: Lockfile) -> None:
    """Install *name* into *store* the way pm records one, without a network or a runner."""
    facts_path = store / "facts.json"
    if not facts_path.is_file():
        facts_path.write_text('{"schema": 1, "packages": {}}', encoding="utf-8")
    facts = Facts(facts_path)
    digest = "0" * 64
    if lock.version(name) is None:
        lock.set_pin(name, "1", {"any": {"url": "https://test.invalid/x.tar.gz", "sha256": digest}})
        lock.save()
    package = registry.get_package(name)
    entry = store / entry_name
    binary = package.binary(entry, current_target())
    binary.parent.mkdir(parents=True, exist_ok=True)
    binary.write_bytes(b"installed tool")
    facts.record(name, "1", entry_name, {"PATH": [str(entry)]}, store,
                 target=current_target(), artifacts=[digest], digest=tree_digest(entry))


@pytest.fixture
def store_env(tmp_path, monkeypatch):
    """A fabricated store with node and npm installed, and pm pointed at it."""
    store = tmp_path / "tools"
    store.mkdir()
    lock_path = tmp_path / "lock.json"
    monkeypatch.setattr(paths, "store_root", lambda: store)
    monkeypatch.setattr(paths, "writable_store_root", lambda: store)
    monkeypatch.setattr(paths, "facts_path", lambda: store / "facts.json")
    monkeypatch.setattr(paths, "lockfile_path", lambda: lock_path)
    lock = Lockfile(lock_path)
    for name in ("node", "npm"):
        _record_tool(store, name, f"{name}-1", lock)
    pm.install._forget_memo()
    return store


def test_store_path_dirs_are_composed_once_while_nothing_changes(store_env, monkeypatch):
    """A second ask is a dict lookup: composing parses the lock and the installed state."""
    composed = []
    real = pm.install._compose_store_path_dirs
    monkeypatch.setattr(pm.install, "_compose_store_path_dirs",
                        lambda: composed.append(1) or real())

    dirs = pm.install._store_path_dirs()
    assert any(d.endswith("node-1") for d in dirs), dirs
    assert len(composed) == 1

    assert pm.install._store_path_dirs() == dirs
    assert len(composed) == 1, "a warm ask re-ran the composition"


def test_store_path_dirs_drop_their_answer_when_a_resolver_is_replaced(store_env, monkeypatch):
    """A test (or other code) that swaps a resolver must see its own answer, not the memo's."""
    assert pm.install._store_path_dirs()

    monkeypatch.setattr(pm.install, "_installed_location", lambda *args, **kwargs: None)
    assert pm.install._store_path_dirs() == []


def test_store_path_dirs_follow_a_moved_store(store_env, monkeypatch, tmp_path):
    """``paths.store_root`` is part of the key, so a different root is a different answer."""
    assert pm.install._store_path_dirs()

    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.setattr(paths, "store_root", lambda: elsewhere)
    monkeypatch.setattr(paths, "writable_store_root", lambda: elsewhere)
    assert pm.install._store_path_dirs() == []


def test_store_path_dirs_observe_recorded_state_once_the_window_elapses(store_env, monkeypatch):
    """Facts pm records elsewhere in the process are picked up by the periodic re-check."""
    monkeypatch.setattr(pm.install, "_memo_ttl", 0.0)
    assert any(d.endswith("npm-1") for d in pm.install._store_path_dirs())

    _record_tool(store_env, "npm", "npm-2", Lockfile(paths.lockfile_path()))
    dirs = pm.install._store_path_dirs()
    assert any(d.endswith("npm-2") for d in dirs), dirs
    assert not any(d.endswith("npm-1") for d in dirs), dirs


def test_forget_memo_drops_both_the_paths_and_the_shell(store_env, monkeypatch):
    """An install drops the memo: the next ask recomposes instead of serving the old dirs."""
    from tools.environments.local import _find_bash

    composed = []
    real = pm.install._compose_store_path_dirs
    monkeypatch.setattr(pm.install, "_compose_store_path_dirs",
                        lambda: composed.append(1) or real())
    monkeypatch.delenv("HERMES_GIT_BASH_PATH", raising=False)
    pm.install._store_path_dirs()
    try:
        _find_bash()
    except RuntimeError:  # pragma: no cover - a host without a shell
        pytest.skip("this host has no bash to discover")
    assert pm.shell._bash_cache is not None

    pm.install._forget_memo()

    assert pm.shell._bash_cache is None, "the resolved shell outlived the store state"
    pm.install._store_path_dirs()
    assert len(composed) == 2


def test_python_dirs_are_composed_once(store_env, monkeypatch):
    """``env_for("python")`` walks the dependency chain; user-facing PATH asks per command."""
    asked = []
    real = pm.install.env_for
    monkeypatch.setattr(pm.install, "env_for",
                        lambda *args, **kwargs: asked.append(args) or real(*args, **kwargs))

    first = pm.user_facing._python_dirs()
    assert len(asked) == 1
    assert pm.user_facing._python_dirs() == first
    assert len(asked) == 1, "a warm ask re-ran env_for"


def test_bash_starts_probes_each_binary_once(tmp_path, monkeypatch):
    """The probe is a subprocess: it is paid for the file, not for the command."""
    import subprocess

    binary = tmp_path / "bash.exe"
    binary.write_bytes(b"first build")
    spawns = []

    def fake_run(cmd, **kwargs):
        spawns.append(cmd[0])
        return subprocess.CompletedProcess(cmd, 0)

    monkeypatch.setattr(pm.shell.subprocess, "run", fake_run)

    assert pm.shell._bash_starts(str(binary)) is True
    assert len(spawns) == 1
    assert pm.shell._bash_starts(str(binary)) is True
    assert len(spawns) == 1, "the probe re-ran for an unchanged binary"

    binary.write_bytes(b"a second build, longer than the first")
    assert pm.shell._bash_starts(str(binary)) is True
    assert len(spawns) == 2, "a rebuilt binary must be probed again"


def test_bash_starts_rejects_a_missing_binary_without_spawning(tmp_path, monkeypatch):
    monkeypatch.setattr(pm.shell.subprocess, "run",
                        lambda *args, **kwargs: pytest.fail("spawned a missing binary"))
    assert pm.shell._bash_starts(str(tmp_path / "absent-bash.exe")) is False


def test_bash_is_resolved_once(monkeypatch):
    """``shutil.which`` walks every PATH entry; the answer holds until the PATH changes."""
    monkeypatch.delenv("HERMES_GIT_BASH_PATH", raising=False)
    resolved = []
    real = pm.shell._resolve_bash
    monkeypatch.setattr(pm.shell, "_resolve_bash",
                        lambda: resolved.append(1) or real())
    pm.shell.forget_bash()

    first = pm.shell.bash()
    if first is None:  # pragma: no cover - a host without a shell
        pytest.skip("this host has no bash to discover")
    assert pm.shell.bash() == first
    assert len(resolved) == 1, "a warm ask re-resolved the shell"

    monkeypatch.setenv("PATH", os.environ["PATH"] + os.pathsep + "C:\\nowhere")
    pm.shell.bash()
    assert len(resolved) == 2, "a changed PATH must be resolved again"


def test_warm_path_composition_and_shell_discovery_stay_under_two_milliseconds(store_env, monkeypatch):
    """The measured regression: ~630ms per terminal command, now a dict lookup plus one stat."""
    from tools.environments.local import _find_bash, _user_tools_first

    monkeypatch.delenv("HERMES_GIT_BASH_PATH", raising=False)
    pm.shell.forget_bash()
    path = os.pathsep.join([str(store_env / "node-1"), str(store_env / "npm-1"), "/usr/bin"])

    def compose_and_find():
        _user_tools_first(path)
        _find_bash()

    try:
        compose_and_find()
    except RuntimeError:
        pytest.skip("this host has no bash to discover")

    samples = []
    for _ in range(5):
        started = time.perf_counter()
        for _ in range(20):
            compose_and_find()
        samples.append((time.perf_counter() - started) / 20 * 1000)
    assert min(samples) < 2.0, f"warm path composition took {min(samples):.2f}ms"
