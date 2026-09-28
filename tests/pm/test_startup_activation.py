"""Startup consumes one real PM verdict without activating a partial store."""

import importlib
import os
import sys
from pathlib import Path

import pytest

import pm
from pm import paths, registry
from pm.lock import Lockfile
from pm.packages import BinaryPackage
from tests.pm._fixtures import make_tar, served as served


@pytest.fixture
def checked_store(tmp_path, monkeypatch, served):
    engine = importlib.import_module("pm.install")
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "home"))
    monkeypatch.setenv("HERMES_RUNTIME_DIR", str(tmp_path / "store"))
    monkeypatch.setattr(paths, "repo_root", lambda: tmp_path / "repo")
    monkeypatch.setattr(paths, "lockfile_path", lambda: tmp_path / "lock.json")
    monkeypatch.setattr(registry, "_packages", {})
    docroot, url = served
    lock = Lockfile(paths.lockfile_path())
    binaries = []
    for name in ("first", "second"):
        package = BinaryPackage()
        package.name = name
        package.probe_version = False
        package.binary_rel = {"posix": f"bin/{name}", "win32": f"bin/{name}"}
        registry._packages[name] = package
        archive, digest = make_tar(docroot, f"{name}.tar.gz", {f"bin/{name}": "fixture tool"})
        lock.set_pin(name, "1.0", {"any": {"url": f"{url}/{archive}", "sha256": digest}})
        lock.save()
        engine.ensure(name, explicit=True)
        installed = pm.installed_package(name)
        assert installed is not None and installed.binary is not None
        binaries.append(installed.binary)

    checks = []
    real_check = engine.check

    def counted_check(**kwargs):
        problems = real_check(**kwargs)
        checks.append(problems)
        return problems

    monkeypatch.setattr(engine, "check", counted_check)
    monkeypatch.setattr(pm, "check", counted_check)
    return binaries, checks


def test_activate_returns_live_verdict_without_partial_environment(checked_store, monkeypatch):
    binaries, checks = checked_store
    original_path = os.environ["PATH"]
    monkeypatch.setenv("PATH", original_path)

    assert pm.activate() == []
    assert checks == [[]]
    active_path = os.environ["PATH"].split(os.pathsep)
    assert all(str(binary.parent) in active_path for binary in binaries)

    # A second invocation must observe fresh damage, even after healthy activation.
    monkeypatch.setenv("PATH", original_path)
    binaries[1].unlink()
    before = dict(os.environ)
    problems = pm.activate()
    assert problems == ["second: not installed or outdated"]
    assert checks == [[], problems]
    assert dict(os.environ) == before


@pytest.mark.parametrize("surface", ["gateway", "cli"])
@pytest.mark.parametrize("damaged", [False, True])
def test_startup_uses_one_verdict(checked_store, monkeypatch, capsys, caplog, surface, damaged):
    binaries, checks = checked_store
    if surface == "gateway":
        from gateway import run

        async def started(*args, **kwargs):
            return True

        monkeypatch.setattr(run, "start_gateway", started)
        monkeypatch.setattr(run, "_exit_after_graceful_shutdown", lambda code: None)
        monkeypatch.setattr("hermes_cli.boot_bootstrap.maybe_run_boot_bootstrap", lambda _root: None)
        monkeypatch.setattr(sys, "argv", ["gateway"])
        start = run.main
    else:
        from hermes_cli import main

        # Stop at --help: exercise the actual startup block, not an agent session.
        monkeypatch.setattr("hermes_cli.boot_bootstrap.maybe_run_boot_bootstrap", lambda _root: None)
        for name in ("_set_process_title", "_advertise_agent_env",
                     "_sweep_stale_bytecode_if_checkout_changed",
                     "_try_termux_fast_tui_launch", "_try_termux_fast_cli_launch",
                     "_try_fast_serve_launch", "_try_fast_chat_launch"):
            monkeypatch.setattr(main, name, lambda: None)
        monkeypatch.setattr(sys, "argv", ["hermes", "--help"])

        def start():
            with pytest.raises(SystemExit) as exit_info:
                main.main()
            assert exit_info.value.code == 0

    if damaged:
        binaries[1].unlink()
    original_path = os.environ["PATH"]
    monkeypatch.setenv("PATH", original_path)
    start()

    expected = ["second: not installed or outdated"] if damaged else []
    assert checks == [expected]
    diagnostics = capsys.readouterr().err + caplog.text
    if damaged:
        assert os.environ["PATH"] == original_path
        assert "install out of sync (second: not installed or outdated)" in diagnostics
    else:
        assert all(str(binary.parent) in os.environ["PATH"].split(os.pathsep) for binary in binaries)
        assert "install out of sync" not in diagnostics

def _fake_python_store(tmp_path, monkeypatch, names):
    """Lockfile + registry with fake packages whose env_for yields per-package bin dirs."""
    engine = importlib.import_module("pm.install")
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "home"))
    monkeypatch.setenv("HERMES_RUNTIME_DIR", str(tmp_path / "store"))
    monkeypatch.setattr(paths, "repo_root", lambda: tmp_path / "repo")
    monkeypatch.setattr(paths, "lockfile_path", lambda: tmp_path / "lock.json")
    monkeypatch.setattr(registry, "_packages", {})
    lock = Lockfile(paths.lockfile_path())
    for name in names:
        package = BinaryPackage()
        package.name = name
        package.probe_version = False
        package.internal = False
        package.on_path = True
        package.binary_rel = {"posix": f"bin/{name}", "win32": f"bin/{name}"}
        package.missing_reason = lambda target: None
        def env_for(n, root, _n=name):
            return {"PATH": [str(root / _n / "bin")]}
        package.env_for = env_for
        registry._packages[name] = package
        lock.set_pin(name, "1.0", {"any": {"url": "unused", "sha256": "0" * 64}})
        lock.save()

    def fake_location(package, lockfile, target):
        return (package, type("S", (), {"root": tmp_path / "store"})())
    monkeypatch.setattr(engine, "_installed_location", fake_location)
    return engine


def test_store_path_dirs_splits_python_tail(tmp_path, monkeypatch):
    """Regression for #125040: the python store package's bin dirs are split out
    of the prepend head — the running dependency venv IS the pinned interpreter's
    environment, so the standalone tool Python must never shadow the venv's
    python3 (skill scripts run through the terminal tool would lose every
    third-party module)."""
    engine = _fake_python_store(tmp_path, monkeypatch, ["first", "python"])
    head, tail = engine._store_path_dirs()
    assert head == [str(tmp_path / "store" / "first" / "bin")]
    assert tail == [str(tmp_path / "store" / "python" / "bin")]


def test_activate_appends_python_bin_after_ambient_path(tmp_path, monkeypatch):
    """The full contract: after activate(), a venv python3 already on the ambient
    PATH resolves ahead of the tool Python's bin, and unrelated store-tool
    resolution is preserved (tool dir first)."""
    engine = _fake_python_store(tmp_path, monkeypatch, ["first", "python"])
    venv_bin = tmp_path / "venv" / "bin"
    venv_bin.mkdir(parents=True)
    original_path = str(venv_bin)
    monkeypatch.setenv("PATH", original_path)
    assert engine.activate() == []
    parts = os.environ["PATH"].split(os.pathsep)
    tool_dir = str(tmp_path / "store" / "first" / "bin")
    python_dir = str(tmp_path / "store" / "python" / "bin")
    assert tool_dir in parts and python_dir in parts
    assert parts.index(original_path) < parts.index(python_dir), (
        "tool Python bin must not shadow the ambient (venv) python3")
    assert parts.index(tool_dir) < parts.index(python_dir)


def test_python_only_store_still_discoverable(tmp_path, monkeypatch):
    """Tail placement keeps the tool Python discoverable when nothing else
    provides python3 - it is appended, not dropped."""
    engine = _fake_python_store(tmp_path, monkeypatch, ["python"])
    original_path = os.environ["PATH"]
    monkeypatch.setenv("PATH", original_path)
    assert engine.activate() == []
    assert str(tmp_path / "store" / "python" / "bin") in os.environ["PATH"].split(os.pathsep)
