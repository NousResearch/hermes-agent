"""Cooperating plugin writers and clone readers share a profile-local transaction."""
from __future__ import annotations

import json
import multiprocessing
from pathlib import Path
import subprocess
import threading

import pytest


def _snapshot_reader(home, ready, acquired, result):
    from hermes_cli.plugin_installation import plugin_installation_lock

    ready.set()
    with plugin_installation_lock(Path(home)):
        acquired.set()
        root = Path(home) / "plugins"
        target = root / "demo"
        result.send((
            (target / "marker.txt").read_text() if target.exists() else None,
            json.loads((root / ".install-metadata.json").read_text()) if target.exists() else {},
            json.loads((target / ".hermes-catalog.json").read_text()) if target.exists() else {},
        ))
    result.close()


def _reader(home):
    ctx = multiprocessing.get_context("spawn")
    ready, acquired = ctx.Event(), ctx.Event()
    receive, send = ctx.Pipe(duplex=False)
    process = ctx.Process(target=_snapshot_reader, args=(str(home), ready, acquired, send))
    process.start()
    send.close()
    assert ready.wait(15), "reader did not reach transaction entry"
    return process, acquired, receive


def _finish_reader(process, receive):
    try:
        assert receive.poll(15), "reader remained blocked after transaction exit"
        value = receive.recv()
        process.join(15)
        assert process.exitcode == 0
        return value
    finally:
        if process.is_alive():
            process.terminate()
            process.join(15)
        receive.close()


def _fork_probe(home, ready, acquired):
    from hermes_cli.plugin_installation import plugin_installation_lock
    ready.set()
    with plugin_installation_lock(Path(home)):
        acquired.set()


def _fork_invariant(home, guard=False):
    from hermes_cli import plugins_state
    from hermes_cli.plugin_installation import plugin_installation_lock

    held, release = threading.Event(), threading.Event()

    def holder():
        lock = plugins_state._PLUGIN_STATE_LOCKS_GUARD if guard else plugin_installation_lock(home)
        with lock:
            held.set()
            release.wait(20)

    thread = threading.Thread(target=holder)
    thread.start()
    assert held.wait(15)
    ctx = multiprocessing.get_context("fork")
    ready, acquired = ctx.Event(), ctx.Event()
    child = ctx.Process(target=_fork_probe, args=(str(home), ready, acquired))
    try:
        child.start()
        assert ready.wait(15)
        if not guard:
            assert not acquired.wait(2), "child unlocked the parent's OS lock"
        release.set()
        thread.join(15)
        assert acquired.wait(10), "child inherited an orphaned thread lock"
        child.join(15)
        assert child.exitcode == 0
    finally:
        release.set()
        thread.join(15)
        if child.is_alive():
            child.terminate()
            child.join(15)


@pytest.mark.parametrize("scenario", [
    "settings", "enabled", "consent", "cross-home",
    pytest.param("fork", marks=pytest.mark.macos_only, id="fork-macos"),
    pytest.param("fork", marks=pytest.mark.linux_only, id="fork-linux"),
    pytest.param("fork-guard", marks=pytest.mark.macos_only, id="fork-guard-macos"),
    pytest.param("fork-guard", marks=pytest.mark.linux_only, id="fork-guard-linux"),
])
def test_nested_profile_locks_exclude_other_threads_and_processes(tmp_path, monkeypatch, scenario):
    from contextvars import copy_context
    import yaml
    from hermes_cli.plugin_installation import plugin_installation_lock
    from hermes_cli.plugins_cmd import _set_plugin_enabled
    from hermes_cli.plugin_capabilities import record_consent
    from hermes_cli.plugins import PluginContext, PluginManifest, PluginManager

    monkeypatch.setenv("HOME", str(tmp_path))
    a, b = tmp_path / "a", tmp_path / "b"
    monkeypatch.setenv("HERMES_HOME", str(a))
    if scenario.startswith("fork"):
        _fork_invariant(a, guard=scenario == "fork-guard")
        return
    if scenario == "cross-home":
        barrier = threading.Barrier(2)
        results = []

        def cross(first, second):
            with plugin_installation_lock(first):
                barrier.wait(15)
                try:
                    with plugin_installation_lock(second):
                        results.append("accepted")
                except RuntimeError as exc:
                    results.append(str(exc))

        threads = [threading.Thread(target=cross, args=pair, daemon=True) for pair in ((a, b), (b, a))]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(5)
        assert all(not t.is_alive() for t in threads), "opposite-home nesting deadlocked"
        assert len(results) == 2 and all("distinct" in result for result in results)
        return

    ctx = PluginContext(PluginManifest(name="a-only"), PluginManager())
    writers = {
        "settings": lambda: ctx.set_config("endpoint", "new"),
        "enabled": lambda: _set_plugin_enabled("contender", enable=True),
        "consent": lambda: record_consent("a-only", ["tools.override"], ["tools.override"]),
    }
    thread_ready, thread_acquired = threading.Event(), threading.Event()
    errors = []

    def contender():
        thread_ready.set()
        try:
            writers[scenario]()
        except Exception as exc:
            errors.append(exc)
        finally:
            thread_acquired.set()

    with plugin_installation_lock(a):
        _set_plugin_enabled("a-only", enable=True)
        with plugin_installation_lock(a / "."):
            process, acquired, receive = _reader(a)
            other, other_acquired, other_receive = _reader(b)
            thread = threading.Thread(target=copy_context().run, args=(contender,))
            thread.start()
            assert thread_ready.wait(15)
            blocked = not thread_acquired.wait(2)
            assert not acquired.is_set(), "same-profile process escaped the lock"
        assert not acquired.is_set(), "nested exit released the outer transaction"
        assert _finish_reader(other, other_receive) == (None, {}, {})
        assert other_acquired.is_set(), "independent profile stayed locked by A"
    thread.join(15)
    assert thread_acquired.is_set() and not errors
    assert _finish_reader(process, receive) == (None, {}, {})
    assert blocked, "plugin settings/activation writer escaped the snapshot lock"
    a_config = yaml.safe_load((a / "config.yaml").read_text())["plugins"]
    assert "a-only" in a_config["enabled"]
    if scenario == "settings":
        assert a_config["entries"]["a-only"]["settings"]["endpoint"] == "new"
    assert not (b / "config.yaml").exists()


def _git(repo, *args):
    return subprocess.run(["git", *args], cwd=repo, check=True, capture_output=True, text=True).stdout.strip()


def _redirect_fixture(tmp_path, monkeypatch):
    from hermes_cli import plugins_cmd as cmd

    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "home"))
    monkeypatch.setenv("GIT_CONFIG_NOSYSTEM", "1")
    monkeypatch.setenv("GIT_CONFIG_GLOBAL", str(tmp_path / "no-global-config"))
    repo = tmp_path / "origin"
    repo.mkdir()
    _git(repo, "init", "-q")
    _git(repo, "config", "user.name", "Fixture")
    _git(repo, "config", "user.email", "fixture@example.invalid")
    (repo / "local.txt").write_text("baseline\n")
    revisions = []
    for value in ("old", "new"):
        (repo / "marker.txt").write_text(value)
        _git(repo, "add", ".")
        _git(repo, "commit", "-qm", value)
        revisions.append(_git(repo, "rev-parse", "HEAD"))
    target = tmp_path / "home" / "plugins" / "demo"
    target.parent.mkdir(parents=True)
    sibling = tmp_path / "sibling"
    for checkout in (target, sibling):
        _git(tmp_path, "clone", repo.as_uri(), str(checkout))
        _git(checkout, "config", "user.name", "Fixture")
        _git(checkout, "config", "user.email", "fixture@example.invalid")
        _git(checkout, "reset", "--hard", revisions[0])
        (checkout / "local.txt").write_text("stashed edit\n")
        _git(checkout, "stash", "push", "-m", "existing stash")
        (checkout / "local.txt").write_text("staged edit\n")
        _git(checkout, "add", "local.txt")
        (checkout / "local.txt").write_text("unstaged edit\n")
        (checkout / "untracked.txt").write_text("untracked\n")
    cmd._write_install_metadata({"demo": {
        "pinned": False, "revision": revisions[0], "source": repo.as_uri(),
    }})
    return target, sibling, revisions


def _checkout_bytes(root):
    # Includes HEAD, index, refs/reflogs, stash objects and every working file.
    return {str(p.relative_to(root)): p.read_bytes() for p in root.rglob("*") if p.is_file()}


@pytest.mark.parametrize("redirect", [
    "core.worktree", "config.worktree", "git-symlink", "commondir",
    "GIT_WORK_TREE", "GIT_DIR", "GIT_INDEX_FILE", "GIT_COMMON_DIR",
])
def test_staged_update_failure_cannot_write_outside_staging(tmp_path, monkeypatch, redirect):
    from hermes_cli import plugins_cmd as cmd

    target, sibling, revisions = _redirect_fixture(tmp_path, monkeypatch)
    if redirect == "core.worktree":
        _git(target, "config", "core.worktree", str(target))
    elif redirect == "config.worktree":
        _git(target, "config", "extensions.worktreeConfig", "true")
        _git(target, "config", "--worktree", "core.worktree", str(sibling))
    elif redirect == "git-symlink":
        (target / ".git" / "index").unlink()
        (target / ".git" / "index").symlink_to(sibling / ".git" / "index")
    elif redirect == "commondir":
        (target / ".git" / "commondir").write_text(str(sibling / ".git"))
    before = [_checkout_bytes(p) for p in (target, sibling)]
    metadata = cmd._install_metadata_path().read_bytes()
    write = cmd._write_install_metadata

    def fail_metadata(value):
        if value["demo"]["revision"] != revisions[0]:
            raise OSError("fixture metadata disk failure")
        write(value)

    with monkeypatch.context() as scoped:
        scoped.setattr(cmd, "_write_install_metadata", fail_metadata)
        if redirect.startswith("GIT_"):
            paths = {"GIT_WORK_TREE": sibling, "GIT_DIR": sibling / ".git",
                     "GIT_INDEX_FILE": sibling / ".git" / "index", "GIT_COMMON_DIR": sibling / ".git"}
            scoped.setenv(redirect, str(paths[redirect]))
        with pytest.raises((OSError, cmd.PluginOperationError), match="metadata disk failure|linked git"):
            cmd._pull_plugin_update(target, lambda rec: "pinned", lambda: "not git")
    assert [_checkout_bytes(p) for p in (target, sibling)] == before
    assert cmd._install_metadata_path().read_bytes() == metadata
    if redirect not in {"git-symlink", "commondir"}:
        cmd._pull_plugin_update(target, lambda rec: "pinned", lambda: "not git")
        assert (target / "marker.txt").read_text() == "new"
        assert cmd._read_install_metadata()["demo"]["revision"] == revisions[1]
        assert _checkout_bytes(sibling) == before[1]


@pytest.mark.parametrize("redirect", ["core.worktree", "GIT_WORK_TREE", "GIT_DIR"])
def test_exact_checkout_is_bound_to_requested_repository(tmp_path, monkeypatch, redirect):
    from hermes_cli import plugins_cmd as cmd

    target, sibling, revisions = _redirect_fixture(tmp_path, monkeypatch)
    # The sibling remains dirty; only the requested checkout is clean for checkout.
    _git(target, "reset", "--hard", "HEAD")
    if redirect == "core.worktree":
        _git(target, "config", "core.worktree", str(sibling))
    before = _checkout_bytes(sibling)
    git_exe = cmd._resolve_git_executable()
    assert git_exe is not None
    with monkeypatch.context() as scoped:
        if redirect.startswith("GIT_"):
            scoped.setenv(redirect, str(sibling / ".git" if redirect == "GIT_DIR" else sibling))
        cmd._checkout_exact_revision(target, git_exe, revisions[1])
    assert _checkout_bytes(sibling) == before
    assert (target / "marker.txt").read_text() == "new"


@pytest.mark.parametrize("operation", ["catalog", "pull-clean", "pull-dirty"])
def test_catalog_reinstall_publishes_code_metadata_and_provenance_together(tmp_path, monkeypatch, operation):
    from hermes_cli import plugins_cmd as cmd, plugins_cmd_catalog as catalog
    from hermes_cli.plugin_catalog import PluginCatalogEntry

    home = tmp_path / "home"
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("HERMES_HOME", str(home))
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-q")
    _git(repo, "config", "user.name", "Fixture")
    _git(repo, "config", "user.email", "fixture@example.invalid")
    (repo / "plugin.yaml").write_text("name: demo\nversion: '1'\n")
    revisions = []
    for marker in ("old", "new"):
        (repo / "marker.txt").write_text(marker)
        _git(repo, "add", ".")
        _git(repo, "commit", "-qm", marker)
        revisions.append(_git(repo, "rev-parse", "HEAD"))
    old, new = [PluginCatalogEntry(name="demo", repo=repo.as_uri(), sha=sha, description="fixture", maintainer="fixture") for sha in revisions]
    if operation.startswith("pull"):
        target = home / "plugins" / "demo"
        target.parent.mkdir(parents=True)
        _git(tmp_path, "clone", repo.as_uri(), str(target))
        _git(target, "config", "user.name", "Fixture")
        _git(target, "config", "user.email", "fixture@example.invalid")
        _git(target, "reset", "--hard", old.sha)
        metadata = {"demo": {"pinned": False, "source": repo.as_uri(), "revision": old.sha}}
        cmd._write_install_metadata(metadata)
        if operation == "pull-dirty":
            (target / "marker.txt").write_text("stashed edit")
            _git(target, "stash", "push", "-m", "user stash")
            (target / "marker.txt").write_text("local edit")
            _git(target, "add", "marker.txt")
            (target / "marker.txt").write_text("unstaged edit")
            (target / "untracked.txt").write_text("user untracked")
        before_status = _git(target, "status", "--porcelain=v1")
        before_stash = _git(target, "stash", "list", "--format=%H")
        before_diff = _git(target, "diff")
        before_index = _git(target, "diff", "--cached")
        before_marker = (target / "marker.txt").read_text()
        write_metadata = cmd._write_install_metadata

        def fail_new_metadata(value):
            if value["demo"]["revision"] != old.sha:
                raise OSError("fixture metadata disk failure")
            write_metadata(value)

        monkeypatch.setattr(cmd, "_write_install_metadata", fail_new_metadata)
        with pytest.raises((OSError, cmd.PluginOperationError), match="metadata disk failure"):
            cmd._pull_plugin_update(target, lambda rec: "pinned", lambda: "not git")
        assert _git(target, "rev-parse", "HEAD") == old.sha, "metadata failure left new code installed"
        assert cmd._read_install_metadata() == metadata
        assert _git(target, "status", "--porcelain=v1") == before_status
        assert _git(target, "stash", "list", "--format=%H") == before_stash
        assert _git(target, "diff") == before_diff
        assert _git(target, "diff", "--cached") == before_index
        assert (target / "marker.txt").read_text() == before_marker
        if operation == "pull-dirty":
            assert (target / "untracked.txt").read_text() == "user untracked"
        monkeypatch.setattr(cmd, "_write_install_metadata", write_metadata)
        cmd._pull_plugin_update(target, lambda rec: "pinned", lambda: "not git")
        assert _git(target, "rev-parse", "HEAD") == new.sha
        assert cmd._read_install_metadata()["demo"]["revision"] == new.sha
        return

    catalog.install_catalog_entry(old, force=False, allow_removed=True, python_deps=False)
    write = catalog.write_catalog_sidecar
    readers = []

    def paused_write(target, entry):
        process, acquired, receive = _reader(home)
        readers.append((process, receive))
        assert not acquired.wait(2), "clone observed installation before provenance was persisted"
        write(target, entry)

    monkeypatch.setattr(catalog, "write_catalog_sidecar", paused_write)
    catalog.install_catalog_entry(new, force=True, allow_removed=True, python_deps=False)
    marker, metadata, sidecar = _finish_reader(*readers.pop())
    assert marker == "new"
    assert metadata["demo"]["revision"] == sidecar["sha"] == new.sha

    # A failed provenance write must refuse BEFORE replacing the old coherent install.
    def refuse_write(target, entry):
        raise OSError("fixture sidecar disk failure")

    monkeypatch.setattr(catalog, "write_catalog_sidecar", refuse_write)
    with pytest.raises(cmd.PluginOperationError, match="sidecar"):
        catalog.install_catalog_entry(old, force=True, allow_removed=True, python_deps=False)
    target = home / "plugins" / "demo"
    assert (target / "marker.txt").read_text() == "new"
    assert json.loads((target.parent / ".install-metadata.json").read_text())["demo"]["revision"] == new.sha
    sidecar = catalog.read_catalog_sidecar(target)
    assert sidecar is not None and sidecar["sha"] == new.sha

    # A TUI request can arrive with provenance read before a competing removal.
    cmd._remove_plugin_core(target)
    monkeypatch.setattr(catalog, "get_live_catalog_entry", lambda name: new)
    with pytest.raises(cmd.PluginOperationError, match="changed"):
        catalog.repin_catalog_plugin(target, sidecar)
    assert not target.exists()
